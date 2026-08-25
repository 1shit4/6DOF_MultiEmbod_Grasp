"""Executing a pick-and-place by editing the scene and re-rendering it.

No physics. The object is teleported by the plan's translation and the scene is
ray-cast again, which makes this deterministic, fast, dependency-free, and exact
about where things ended up — so the evaluator can be tested against ground
truth rather than against a second guess.

What it does *not* model is the interesting half of reality: objects do not
topple, roll, settle, or knock their neighbours over on their own. Those are
supplied by explicit **failure injection** instead. That is an honest trade
rather than a hidden one — every failure this backend produces is one somebody
wrote down, so the evaluator is only ever tested against failures we thought of.
A physics backend is the thing that would surprise us, and it is deferred.

Two decisions worth stating:

* **The grasped object is identified from the pose, not from the plan's id.**
  A real hand closes on whatever is between its fingers, so the executor looks
  up whichever object is nearest the fingertip midpoint. "The plan named obj_3
  but the pose was over obj_5" is therefore a reportable outcome instead of an
  invisible one.
* **A grasp that reaches nothing fails.** If the fingertips are not near any
  object the report is `failed` at the `grasp` stage, which is what a robot
  closing on air would tell you.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import synth_scene as ss
from perception3d import gripper_finger_points

from .base import ExecutionReport, Observation, PickPlacePlan

logger = logging.getLogger(__name__)

# How near the fingertip midpoint has to be to an object's centre for the hand
# to have grasped it. Generous, because the midpoint sits between the fingers
# rather than on the surface, and objects here are 5-20 cm across.
GRASP_REACH_M = 0.12

INJECTABLE = ("offset", "drop", "tip", "collateral", "wrong_object")


class MutationExecutor:
    """Moves objects by editing a `SceneSpec` and re-rendering it."""

    backend = "mutation"

    def __init__(
        self,
        spec: ss.SceneSpec,
        K: Optional[np.ndarray] = None,
        T_cam_world: Optional[np.ndarray] = None,
        height: int = 480,
        width: int = 640,
        inject: Sequence[str] = (),
        seed: int = 0,
        offset_m: float = 0.08,
        offset_dir: str = "random",
        inject_at: Optional[int] = 0,
    ):
        unknown = set(inject) - set(INJECTABLE)
        if unknown:
            raise ValueError(
                f"unknown failure mode(s) {sorted(unknown)}; have {list(INJECTABLE)}"
            )
        self._initial = spec
        self.spec = spec
        self.K = np.asarray(K if K is not None else ss.default_intrinsics(height, width))
        self.T_cam_world = np.asarray(
            T_cam_world if T_cam_world is not None else ss.default_camera()
        )
        self.height, self.width = int(height), int(width)
        self.inject = list(inject)
        # Which pick-and-place the fault fires on, counting from 0. A fault on
        # *every* execution only tests that the loop gives up: no object the
        # planner intends to move can ever move, so the task is impossible by
        # construction and the only correct outcome is failure. Injecting once
        # asks the question worth asking — does the system notice what went
        # wrong, and does it recover? `None` restores the persistent fault, which
        # is its own scenario (a gripper that is simply broken).
        self.inject_at = inject_at
        self.offset_m = float(offset_m)
        if offset_dir not in ("random", "short"):
            raise ValueError(
                f"offset_dir must be 'random' or 'short', got {offset_dir!r}"
            )
        self.offset_dir = offset_dir
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._executions = 0
        self.history: List[dict] = []

    @property
    def injecting(self) -> bool:
        """Whether a fault fires on the execution about to run."""
        if not self.inject:
            return False
        return self.inject_at is None or self._executions == self.inject_at

    # -- observation -------------------------------------------------------

    def capture(self, iteration: int = 0) -> Observation:
        out = ss.render(
            self.spec, self.K, self.T_cam_world, height=self.height, width=self.width
        )
        return Observation(
            rgb=out["rgb"],
            depth=out["depth"],
            K=self.K,
            seg=out["seg"],
            label_map=self.spec.label_map(),
            iteration=iteration,
        )

    def reset(self) -> None:
        self._executions = 0
        self.spec = self._initial
        self._rng = np.random.default_rng(self._seed)
        self.history = []

    # -- ground truth (for tests and the verification script) --------------

    def true_position(self, name: str) -> np.ndarray:
        """World-frame position of an object. Ground truth, not an estimate."""
        return self.spec.by_name(name).primitive.position.copy()

    def true_positions(self) -> Dict[str, np.ndarray]:
        return {o.name: o.primitive.position.copy() for o in self.spec.objects}

    # -- execution ---------------------------------------------------------

    def _world_from_camera(self, v: np.ndarray) -> np.ndarray:
        """Rotate a camera-frame displacement into the world frame."""
        R = np.linalg.inv(self.T_cam_world)[:3, :3]
        return R @ np.asarray(v, dtype=np.float64).reshape(3)

    def _object_at(self, pose: np.ndarray, gripper: str) -> Tuple[Optional[str], float]:
        """Which object the hand would actually close on, and how far away it is."""
        from scene_registry import _gripper_geometry

        width, fingertip = _gripper_geometry(gripper)
        tips = gripper_finger_points(pose, width, fingertip)
        midpoint_cam = tips.mean(axis=0)

        T_world_cam = np.linalg.inv(self.T_cam_world)
        midpoint_world = T_world_cam[:3, :3] @ midpoint_cam + T_world_cam[:3, 3]

        best, best_d = None, float("inf")
        for obj in self.spec.objects:
            d = float(np.linalg.norm(obj.primitive.position - midpoint_world))
            if d < best_d:
                best, best_d = obj.name, d
        return best, best_d

    def execute_pick_place(self, plan: PickPlacePlan) -> ExecutionReport:
        t0 = time.time()
        report = ExecutionReport(backend=self.backend)
        # Bound once per execution so every guard below agrees about this one
        # attempt, and counted even when the grasp reaches nothing — "the
        # attempt where the fault fires" has to mean the same thing regardless
        # of how far that attempt got.
        injecting = self.injecting
        self._executions += 1

        grasped, distance = self._object_at(plan.grasp_pose, plan.gripper)
        if grasped is None or distance > GRASP_REACH_M:
            report.status = "failed"
            report.stage_reached = "grasp"
            report.error = (
                f"the fingertips reached nothing: nearest object is {distance*100:.0f} cm away"
            )
            report.duration_s = time.time() - t0
            return report

        report.grasped_object = grasped

        if "wrong_object" in self.inject and injecting:
            others = [o.name for o in self.spec.objects if o.name != grasped]
            if others:
                grasped = str(self._rng.choice(others))
                report.grasped_object = grasped
                report.status = "partial"
                report.notes.append("injected: the hand closed on a neighbour")

        if "drop" in self.inject and injecting:
            report.status = "failed"
            report.stage_reached = "lift"
            report.error = "injected: the object slipped out of the jaw on lift"
            report.applied_translation = np.zeros(3)
            report.duration_s = time.time() - t0
            self.history.append({"plan": plan.describe(), "report": report.describe()})
            return report

        delta_world = self._world_from_camera(plan.translation)

        if "offset" in self.inject and injecting:
            if self.offset_dir == "short":
                # Released *early*, along the path rather than beside it. The
                # random slip always cleared the target in practice, so it only
                # ever exercised one half of the two-verdict design: an action
                # judged imperfect that nonetheless worked. Falling short leaves
                # the object between the camera and the target, which produces
                # the other half — a move the geometry calls a success while the
                # target stays blocked, so the planner has to re-plan rather
                # than accept it.
                # Measured in the table plane, not in 3D. Placement is 2.5D
                # onto the support surface and `resting()` re-seats Z anyway, so
                # a 3D direction that is then flattened lands short of the
                # requested miss by however much the path climbed — 7.6 mm on
                # this scene, enough to fail an exact assertion.
                planar = np.array([delta_world[0], delta_world[1], 0.0])
                travel = float(np.linalg.norm(planar))
                if travel > 1e-6:
                    back = min(self.offset_m, travel * 0.95)
                    slip = -back * (planar / travel)
                else:
                    slip = np.zeros(3)
            else:
                angle = float(self._rng.uniform(0, 2 * np.pi))
                slip = self.offset_m * np.array([np.cos(angle), np.sin(angle), 0.0])
            slip[2] = 0.0
            delta_world = delta_world + slip
            report.status = "partial"
            report.notes.append(
                f"injected: released {np.linalg.norm(slip)*100:.0f} cm "
                + ("short of" if self.offset_dir == "short" else "off")
                + " the intended spot"
            )

        prim = self.spec.by_name(grasped).primitive
        moved = ss.Primitive(
            kind=prim.kind,
            size=prim.size.copy(),
            position=prim.position + delta_world,
            yaw=prim.yaw,
            color=prim.color,
        )

        if "tip" in self.inject and injecting:
            # No orientation in the primitive model beyond yaw, so a topple is
            # represented the way it reads to a camera: the object's height and
            # footprint swap, and it settles back onto the table.
            size = moved.size.copy()
            size[0], size[2] = size[2], size[0]
            moved = ss.Primitive(moved.kind, size, moved.position, moved.yaw, moved.color)
            report.status = "partial"
            report.notes.append("injected: the object toppled on release")

        moved = moved.resting()
        self.spec = self.spec.replace(grasped, moved)

        if "collateral" in self.inject and injecting:
            others = [o.name for o in self.spec.objects if o.name != grasped]
            if others:
                victim = str(self._rng.choice(others))
                vp = self.spec.by_name(victim).primitive
                nudge = self._rng.normal(0, 0.05, 3)
                nudge[2] = 0.0
                self.spec = self.spec.replace(
                    victim,
                    ss.Primitive(
                        vp.kind, vp.size, vp.position + nudge, vp.yaw, vp.color
                    ).resting(),
                )
                report.disturbed.append(victim)
                report.status = "partial" if report.status == "ok" else report.status
                report.notes.append(f"injected: knocked {victim} while moving")

        report.applied_translation = delta_world
        report.duration_s = time.time() - t0
        self.history.append({"plan": plan.describe(), "report": report.describe()})
        return report


class ReplayExecutor:
    """Hands back pre-recorded observations, ignoring the plan.

    Exists so the loop can be exercised on *real* captures. The two sample
    scenes are the same tabletop rearranged, so feeding them in order is a
    genuine before/after pair with real depth, real noise and real segmentation
    — the one thing a synthetic scene cannot provide.

    It cannot honour a plan, so it never claims to: every report is `partial`
    with a note saying the motion was replayed, and the evaluator is left to
    work out what changed. That is exactly the situation it is meant to handle.
    """

    backend = "replay"

    def __init__(self, observations: Sequence[Observation], loop: bool = False):
        if not observations:
            raise ValueError("ReplayExecutor needs at least one observation")
        self._observations = list(observations)
        self.loop = bool(loop)
        self._index = 0

    def capture(self, iteration: int = 0) -> Observation:
        obs = self._observations[min(self._index, len(self._observations) - 1)]
        # The recording's own timestamp is kept: this observation was taken then,
        # not now, and the evaluator reasons about which capture it is looking at.
        return Observation(
            rgb=obs.rgb, depth=obs.depth, K=obs.K, seg=obs.seg,
            label_map=obs.label_map, iteration=iteration, timestamp=obs.timestamp,
        )

    def execute_pick_place(self, plan: PickPlacePlan) -> ExecutionReport:
        report = ExecutionReport(backend=self.backend, status="partial")
        if self._index + 1 < len(self._observations):
            self._index += 1
        elif self.loop:
            self._index = 0
        else:
            report.status = "failed"
            report.stage_reached = "grasp"
            report.error = "no further recorded observations"
            return report
        report.notes.append(
            "replayed a recorded capture; the plan was not executed, so any "
            "change in the scene is whatever the recording contains"
        )
        return report

    def reset(self) -> None:
        self._index = 0
