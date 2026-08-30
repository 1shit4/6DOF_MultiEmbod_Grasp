"""What an executor is, independent of what moves the objects.

Everything above this layer — the loop, the planner, the evaluator — depends
only on these three types. A simulator, a scene-mutation model and a real arm
are interchangeable behind them, which is the point: the loop should not be able
to tell, and the parts that would differ (physics, failure modes, latency) are
the executor's business, not the planner's.

The one contract that matters: **`execute_pick_place` must not report what it
was asked to do, it must report what happened.** An executor that echoes the
plan back as success makes the evaluator meaningless and the whole loop a
simulation of competence. Backends here identify the grasped object from the
pose geometrically, exactly as a real robot would grasp whatever is at that
pose, so "the grasp pointed at the wrong object" is a representable outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

# What execute_pick_place can report. Anything other than `ok` means the scene
# is not what the plan intended, and the evaluator has to work out what is.
STATUSES = ("ok", "partial", "failed", "error")

# The stages a pick-and-place passes through, in order. `stage_reached` names
# the last one completed, so a failure localises without a stack trace.
STAGES = ("pre_grasp", "grasp", "lift", "pre_place", "place", "retreat")


@dataclass
class Observation:
    """One RGB-D capture of the scene, plus whatever ground truth exists."""

    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    seg: Optional[np.ndarray] = None
    label_map: Dict[str, int] = field(default_factory=dict)
    iteration: int = 0
    timestamp: str = ""

    def __post_init__(self):
        self.depth = np.asarray(self.depth)
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def has_ground_truth(self) -> bool:
        return self.seg is not None

    def describe(self) -> dict:
        return {
            "iteration": self.iteration,
            "shape": list(np.shape(self.depth)),
            "has_seg": self.has_ground_truth,
            "n_labels": len(self.label_map),
            "timestamp": self.timestamp,
        }


@dataclass
class PickPlacePlan:
    """One object, where to grasp it, and where to let go.

    `grasp` and `place` are the dicts `Grasp6D.as_dict()` and
    `PlacePose.as_dict()` produce, so a plan is JSON-serialisable and lands in
    `progress.json` unchanged.
    """

    object_id: str
    grasp: dict
    place: dict
    gripper: str = "franka_panda"
    object_label: str = ""

    @property
    def grasp_pose(self) -> np.ndarray:
        return np.asarray(self.grasp["pose"], dtype=np.float64).reshape(4, 4)

    @property
    def place_pose(self) -> np.ndarray:
        return np.asarray(self.place["pose"], dtype=np.float64).reshape(4, 4)

    @property
    def translation(self) -> np.ndarray:
        """Camera-frame displacement the object is meant to undergo.

        Exact, because a place pose is a pure translation of the grasp pose —
        see `placement`. The object's orientation does not change, so this one
        vector fully describes the intended motion.
        """
        return self.place_pose[:3, 3] - self.grasp_pose[:3, 3]

    @property
    def waypoints(self) -> List[tuple]:
        return [
            (w["name"], np.asarray(w["pose"], dtype=np.float64).reshape(4, 4))
            for w in self.place.get("waypoints", [])
        ]

    def describe(self) -> dict:
        return {
            "object_id": self.object_id,
            "object_label": self.object_label,
            "gripper": self.gripper,
            "grasp_position_m": [round(float(v), 3) for v in self.grasp_pose[:3, 3]],
            "place_position_m": [round(float(v), 3) for v in self.place_pose[:3, 3]],
            "travel_cm": round(float(np.linalg.norm(self.translation)) * 100, 1),
        }


@dataclass
class ExecutionReport:
    """What actually happened. Never a restatement of the plan."""

    status: str = "ok"
    stage_reached: str = "retreat"
    error: Optional[str] = None
    grasped_object: Optional[str] = None  # what the hand actually closed on
    applied_translation: Optional[np.ndarray] = None
    disturbed: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    duration_s: float = 0.0
    backend: str = ""
    #: Why this outcome happened, when the backend happens to know. A real arm
    #: never does — it reports where the hand got to and what moved, not the
    #: cause. So this is written by simulated backends for **our** analysis and
    #: is deliberately excluded from `describe()`, which is what the loop and
    #: the planner see. A planner told "the object slipped out of the jaw on
    #: lift" is not inferring anything; it is reading the answer key.
    ground_truth: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}; got {self.status!r}")
        if self.stage_reached not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}; got {self.stage_reached!r}")

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def describe(self) -> dict:
        return {
            "status": self.status,
            "stage_reached": self.stage_reached,
            "error": self.error,
            "grasped_object": self.grasped_object,
            "applied_translation_cm": (
                [round(float(v) * 100, 1) for v in self.applied_translation]
                if self.applied_translation is not None
                else None
            ),
            "disturbed": list(self.disturbed),
            "notes": list(self.notes),
            "duration_s": round(self.duration_s, 3),
            "backend": self.backend,
            # `ground_truth` is intentionally absent. See the field's comment.
        }

    def describe_with_ground_truth(self) -> dict:
        """`describe()` plus the cause, for the analysis log only.

        Never feed this to an agent: it is the difference between measuring
        whether the loop can infer a failure and measuring whether it can read.
        """
        return {**self.describe(), "ground_truth": dict(self.ground_truth)}


@runtime_checkable
class Executor(Protocol):
    """The interface the loop programs against."""

    def capture(self, iteration: int = 0) -> Observation:
        """Look at the scene as it is now."""

    def execute_pick_place(self, plan: PickPlacePlan) -> ExecutionReport:
        """Attempt the plan. Report the outcome, not the intent."""

    def reset(self) -> None:
        """Return the scene to its initial state."""
