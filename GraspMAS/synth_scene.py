"""Synthetic cluttered tabletops with exact ground truth.

The two real sample scenes (`GraspGenX/assets/sample_data/real_world/{00,01}`)
are genuinely cluttered — eight labelled objects each, real depth, real
intrinsics — and they are the right thing to test *perception* against. But
their objects are well separated: nothing meaningfully occludes anything else,
so they cannot exercise a decluttering loop. They also come with no ground
truth for where an object *should* end up after a move.

This module fills both gaps. It composes primitives on a table, then renders
them by **ray casting**, which gives a hole-free depth image, a pixel-exact
segmentation map, and object poses known to machine precision. No meshes, no
`trimesh`, no simulator, no GPU. Deterministic under a seed.

Ray casting rather than point splatting because splatting leaves gaps that look
exactly like sensor dropout, and "unobserved" is load-bearing elsewhere in this
codebase — `placement.free_space` treats it as occupied. A renderer that
invents dropout would make those tests lie.

Frames: a world frame with +Z up and the table at z=0, and the usual OpenCV
camera frame (+X right, +Y down, +Z forward). Everything is metres.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BACKGROUND_ID = 0
TABLE_ID = 1
FIRST_OBJECT_ID = 100


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass
class Primitive:
    """A convex solid, positioned in the world frame.

    `size` is interpreted per kind:
      * ``box``      — full extents (x, y, z)
      * ``cylinder`` — (radius, radius, height); the axis is local +Z
      * ``sphere``   — (radius, radius, radius), only the first is read
    """

    kind: str
    size: np.ndarray
    position: np.ndarray  # centre of the solid, world frame
    yaw: float = 0.0
    color: Tuple[int, int, int] = (180, 180, 180)

    def __post_init__(self):
        if self.kind not in ("box", "cylinder", "sphere"):
            raise ValueError(f"unknown primitive kind {self.kind!r}")
        self.size = np.asarray(self.size, dtype=np.float64).reshape(3)
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.yaw = float(self.yaw)

    @property
    def height(self) -> float:
        return float(self.size[2]) if self.kind != "sphere" else float(self.size[0] * 2)

    @property
    def T_world_obj(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        T = np.eye(4)
        T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T[:3, 3] = self.position
        return T

    def resting(self) -> "Primitive":
        """The same solid, translated so it sits on the table (z=0)."""
        p = self.position.copy()
        p[2] = self.height / 2.0
        return Primitive(self.kind, self.size, p, self.yaw, self.color)

    def moved_to(self, xy: Sequence[float]) -> "Primitive":
        p = self.position.copy()
        p[0], p[1] = float(xy[0]), float(xy[1])
        return Primitive(self.kind, self.size, p, self.yaw, self.color)


@dataclass
class SceneObject:
    name: str
    primitive: Primitive
    label_id: int


@dataclass
class SceneSpec:
    """A table plus the things on it."""

    objects: List[SceneObject] = field(default_factory=list)
    table_extent: Tuple[float, float] = (0.9, 0.7)  # full width (x), depth (y)
    table_color: Tuple[int, int, int] = (150, 120, 90)
    background_color: Tuple[int, int, int] = (30, 30, 35)

    def by_name(self, name: str) -> SceneObject:
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(f"no object named {name!r}; have {[o.name for o in self.objects]}")

    def label_map(self) -> Dict[str, int]:
        return {o.name: o.label_id for o in self.objects}

    def replace(self, name: str, primitive: Primitive) -> "SceneSpec":
        """A copy with one object's primitive swapped — how a move is applied."""
        objects = [
            SceneObject(o.name, primitive if o.name == name else o.primitive, o.label_id)
            for o in self.objects
        ]
        return SceneSpec(objects, self.table_extent, self.table_color, self.background_color)


def add_objects(
    specs: Sequence[Tuple[str, Primitive]], table_extent=(0.9, 0.7)
) -> SceneSpec:
    """Build a SceneSpec, assigning label ids and resting everything on the table."""
    objects = [
        SceneObject(name, prim.resting(), FIRST_OBJECT_ID + i)
        for i, (name, prim) in enumerate(specs)
    ]
    return SceneSpec(objects=objects, table_extent=table_extent)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def look_at(
    eye: Sequence[float],
    target: Sequence[float] = (0.0, 0.0, 0.05),
    up: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """World -> camera transform for a camera at `eye` looking at `target`.

    Returns ``T_cam_world``. OpenCV axes: +Z is the viewing direction, +Y points
    down in the image, so the camera's +Y is the *downward* world direction
    projected into the image plane.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    forward = target - eye
    n = np.linalg.norm(forward)
    if n < 1e-9:
        raise ValueError("camera eye and target coincide")
    forward /= n

    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-9:
        raise ValueError("camera looks straight along `up`; pick another up vector")
    right /= rn
    down = np.cross(forward, right)

    T_world_cam = np.eye(4)
    T_world_cam[:3, 0] = right
    T_world_cam[:3, 1] = down
    T_world_cam[:3, 2] = forward
    T_world_cam[:3, 3] = eye

    R = T_world_cam[:3, :3]
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = R.T
    T_cam_world[:3, 3] = -R.T @ eye
    return T_cam_world


def table_plane_in_camera(T_cam_world: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ground-truth ``(normal, offset)`` of the table in the camera frame.

    The plane is ``normal . x + offset = 0``, with `normal` oriented toward the
    camera so that ``normal . p + offset`` is height above the table — the same
    convention `placement.SupportPlane` uses, so tests can compare directly.
    """
    T = np.asarray(T_cam_world, dtype=np.float64).reshape(4, 4)
    normal = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    origin_cam = T[:3, 3]  # the world origin, which lies on the table
    offset = -float(np.dot(normal, origin_cam))
    if offset < 0:
        normal, offset = -normal, -offset
    return normal, offset


# ---------------------------------------------------------------------------
# Ray casting
# ---------------------------------------------------------------------------


def _ray_box(o: np.ndarray, d: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Slab method. Returns entry distance per ray, inf where there is no hit."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t0 = (-half - o) * inv
        t1 = (half - o) * inv
    tmin = np.maximum.reduce(np.minimum(t0, t1), axis=-1)
    tmax = np.minimum.reduce(np.maximum(t0, t1), axis=-1)
    hit = (tmax >= np.maximum(tmin, 0.0)) & np.isfinite(tmin)
    return np.where(hit & (tmin > 0), tmin, np.inf)


def _ray_sphere(o: np.ndarray, d: np.ndarray, radius: float) -> np.ndarray:
    a = np.einsum("ij,ij->i", d, d)
    b = 2.0 * np.einsum("ij,ij->i", o, d)
    c = np.einsum("ij,ij->i", o, o) - radius * radius
    disc = b * b - 4 * a * c
    ok = disc >= 0
    sq = np.zeros_like(disc)
    sq[ok] = np.sqrt(disc[ok])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-b - sq) / (2 * a)
    return np.where(ok & (t > 0), t, np.inf)


def _ray_cylinder(
    o: np.ndarray, d: np.ndarray, radius: float, half_h: float
) -> np.ndarray:
    """Finite cylinder about local +Z, including both caps."""
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2.0 * (o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1])
    c = o[:, 0] ** 2 + o[:, 1] ** 2 - radius * radius

    best = np.full(len(o), np.inf)

    # Curved surface.
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = b * b - 4 * a * c
        ok = (disc >= 0) & (a > 1e-12)
        sq = np.zeros_like(disc)
        sq[ok] = np.sqrt(disc[ok])
        t = np.where(ok, (-b - sq) / (2 * a), np.inf)
    z = o[:, 2] + t * d[:, 2]
    side = ok & (t > 0) & (np.abs(z) <= half_h)
    best = np.where(side, t, best)

    # Caps.
    for cap_z in (-half_h, half_h):
        with np.errstate(divide="ignore", invalid="ignore"):
            tc = (cap_z - o[:, 2]) / d[:, 2]
        x = o[:, 0] + tc * d[:, 0]
        y = o[:, 1] + tc * d[:, 1]
        hit = np.isfinite(tc) & (tc > 0) & (x * x + y * y <= radius * radius)
        best = np.where(hit & (tc < best), tc, best)

    return best


def _ray_table(
    o: np.ndarray, d: np.ndarray, extent: Tuple[float, float]
) -> np.ndarray:
    """The table top: the z=0 plane, clipped to a rectangle."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -o[:, 2] / d[:, 2]
    x = o[:, 0] + t * d[:, 0]
    y = o[:, 1] + t * d[:, 1]
    inside = (np.abs(x) <= extent[0] / 2.0) & (np.abs(y) <= extent[1] / 2.0)
    return np.where(np.isfinite(t) & (t > 0) & inside, t, np.inf)


def render(
    spec: SceneSpec,
    K: np.ndarray,
    T_cam_world: np.ndarray,
    height: int = 480,
    width: int = 640,
) -> Dict[str, np.ndarray]:
    """Ray-cast the scene. Returns ``{"rgb", "depth", "seg"}``.

    `depth` is metres along the optical axis (i.e. the point's camera-frame Z),
    which is what `perception3d.unproject` expects; misses are 0.0, the same
    encoding real sensors use for "no return". `seg` carries `BACKGROUND_ID`,
    `TABLE_ID`, or the object's `label_id`.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    T_cam_world = np.asarray(T_cam_world, dtype=np.float64).reshape(4, 4)

    vs, us = np.mgrid[0:height, 0:width]
    # Unnormalised rays with d_z = 1, so the ray parameter *is* the depth.
    dirs = np.stack(
        [
            (us.ravel() - K[0, 2]) / K[0, 0],
            (vs.ravel() - K[1, 2]) / K[1, 1],
            np.ones(height * width),
        ],
        axis=-1,
    )

    best_t = np.full(height * width, np.inf)
    best_id = np.full(height * width, BACKGROUND_ID, dtype=np.int32)
    colors = {BACKGROUND_ID: spec.background_color, TABLE_ID: spec.table_color}

    T_world_cam = np.linalg.inv(T_cam_world)

    def local_ray(T_world_local: np.ndarray):
        T_local_cam = np.linalg.inv(T_world_local) @ T_world_cam
        R, t = T_local_cam[:3, :3], T_local_cam[:3, 3]
        return np.broadcast_to(t, dirs.shape).copy(), dirs @ R.T

    # Table.
    o, d = local_ray(np.eye(4))
    t_table = _ray_table(o, d, spec.table_extent)
    closer = t_table < best_t
    best_t = np.where(closer, t_table, best_t)
    best_id = np.where(closer, TABLE_ID, best_id)

    for obj in spec.objects:
        prim = obj.primitive
        o, d = local_ray(prim.T_world_obj)
        if prim.kind == "box":
            t = _ray_box(o, d, prim.size / 2.0)
        elif prim.kind == "sphere":
            t = _ray_sphere(o, d, float(prim.size[0]))
        else:
            t = _ray_cylinder(o, d, float(prim.size[0]), float(prim.size[2]) / 2.0)
        closer = t < best_t
        best_t = np.where(closer, t, best_t)
        best_id = np.where(closer, obj.label_id, best_id)
        colors[obj.label_id] = prim.color

    depth = np.where(np.isfinite(best_t), best_t, 0.0).reshape(height, width)
    seg = best_id.reshape(height, width)

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for label, color in colors.items():
        rgb[seg == label] = color

    return {
        "rgb": rgb,
        "depth": depth.astype(np.float32),
        "seg": seg.astype(np.int32),
    }


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def default_intrinsics(height: int = 480, width: int = 640) -> np.ndarray:
    from perception3d import default_intrinsics as _K

    return _K(height, width, fov_deg=55.0)


def default_camera() -> np.ndarray:
    """A tabletop viewpoint: 85 cm back, 55 cm up, looking down at the table.

    Roughly the geometry of the real sample scenes, so the fitted plane is
    tilted well away from any axis and the tests are not accidentally passing
    on a degenerate head-on view.
    """
    return look_at(eye=(0.0, -0.85, 0.55), target=(0.0, 0.0, 0.05))


def occluded_target_scene() -> SceneSpec:
    """The motivating case, with the two blocking reasons deliberately split.

    The banana lies at the back of the table with its long axis pointing away
    from the camera, which is the axis a parallel jaw would close across.

    A parallel jaw closes across an elongated object's *short* axis, so its wide
    20 cm span lies along the banana's short axis — pointing at the camera.
    Blockers therefore have to sit in front of the banana, not beside it, which
    is the non-obvious part: an object 13 cm to the left is harmless, and one
    9 cm in front is not.

    Measured on the rendered scene with a Panda: the banana is 19% visible, the
    bottle puts 391 points inside the gripper's swept volume and the mug 40, the
    box none, and the sweep becomes clear once both are gone. So it is a genuine
    two-step problem, and `test_synth_scene.py` asserts each of those.

    * The **bottle** is 10.5 cm away and 20 cm tall. It blocks by *proximity*
      and also hides most of the banana.
    * The **mug** is 11.5 cm away on the other side and only 10 cm tall. It
      blocks by proximity while occluding very little, so a purely visual notion
      of "in the way" would miss it.
    * The **box** is a distractor 35 cm away. A loop that moves it has failed
      even if it eventually reaches the banana.

    Both blockers clear the banana itself by ~2.7 cm, so nothing interpenetrates
    and each is independently graspable.
    """
    return add_objects(
        [
            ("banana", Primitive("box", (0.16, 0.045, 0.04), (0.0, 0.16, 0.0),
                                 yaw=0.05, color=(220, 200, 60))),
            ("bottle", Primitive("cylinder", (0.035, 0.035, 0.20), (-0.06, 0.075, 0.0),
                                 color=(200, 70, 60))),
            ("mug", Primitive("cylinder", (0.045, 0.045, 0.10), (0.05, 0.065, 0.0),
                              color=(70, 110, 200))),
            ("box", Primitive("box", (0.16, 0.06, 0.22), (0.30, 0.0, 0.0),
                              yaw=-0.2, color=(210, 120, 50))),
        ]
    )


def two_identical_bottles_scene() -> SceneSpec:
    """Two visually identical bottles, only one of which is in the way.

    Exists so the instance-identity path has something to fail on: a planner
    that says "the bottle" instead of naming an instance cannot get this right,
    and a registry that re-derives ids from labels alone will swap them.
    """
    return add_objects(
        [
            ("target_cup", Primitive("cylinder", (0.04, 0.04, 0.09), (0.0, 0.17, 0.0),
                                     color=(240, 230, 90))),
            ("bottle_a", Primitive("cylinder", (0.033, 0.033, 0.18), (-0.01, 0.02, 0.0),
                                   color=(60, 160, 90))),
            ("bottle_b", Primitive("cylinder", (0.033, 0.033, 0.18), (0.30, -0.10, 0.0),
                                   color=(60, 160, 90))),
        ]
    )


def open_table_scene() -> SceneSpec:
    """One small object on an otherwise empty table — lots of valid placements."""
    return add_objects(
        [("mug", Primitive("cylinder", (0.045, 0.045, 0.10), (0.0, 0.0, 0.0),
                           color=(70, 110, 200)))]
    )


def crowded_table_scene() -> SceneSpec:
    """A table packed edge to edge, so placement legitimately has no answer."""
    objects = []
    for i, x in enumerate(np.linspace(-0.36, 0.36, 7)):
        for j, y in enumerate(np.linspace(-0.24, 0.24, 5)):
            objects.append(
                (
                    f"blk_{i}_{j}",
                    Primitive("box", (0.09, 0.085, 0.10), (x, y, 0.0),
                              color=(120 + 8 * i, 100, 140)),
                )
            )
    return add_objects(objects)


def affordance_table_scene() -> SceneSpec:
    """Four objects with four different uses, so an abstract goal has an answer.

    Exists because the other scenarios are all *labelled* for the task — asking
    for "the banana" needs no reasoning about what a banana is. Here the goal is
    a need rather than a name, and exactly one object affords each need:

    * *"something to cut"* -> **knife** (nothing else has an edge)
    * *"I am hungry"*      -> **apple** (nothing else is food)
    * *"something to drink from"* -> **mug** (the bottle is closed)

    Each goal has a plausible-but-wrong neighbour, so a model choosing by size
    or proximity rather than by use gets it wrong.

    **Measured on the rendered scene, and asserted by `test_synth_scene.py`**, so
    the scenario cannot drift into being easy nor into claiming an obstruction it
    does not have:

    * the **knife** is 81% visible, blocked by the bottle at 21% occlusion, and
      its footprint is therefore *unreliable* — so only occlusion is reported,
      which is the documented gate in §7.4 rather than a gap;
    * the **apple** and the **mug** are 100% visible with **no blockers at all**.

    That asymmetry is the point. The right answer to "something to cut" costs an
    iteration to uncover and the wrong answers are free, so a model that
    retargets to dodge work is choosing measurably worse — and a run that
    reaches the knife had to do real work to get there. Scripted, with no LLM:
    two iterations, move the bottle, grasp the knife.

    The bottle sits **in front of** the knife rather than beside it because a
    parallel jaw closes across an elongated object's *short* axis, the same
    geometry `occluded_target` documents.
    """
    return add_objects(
        [
            ("knife", Primitive("box", (0.20, 0.025, 0.015), (-0.12, 0.18, 0.0),
                                yaw=0.03, color=(190, 190, 200))),
            ("bottle", Primitive("cylinder", (0.035, 0.035, 0.22), (-0.14, 0.07, 0.0),
                                 color=(60, 140, 80))),
            ("apple", Primitive("sphere", (0.037, 0.037, 0.037), (0.15, -0.02, 0.0),
                                color=(200, 60, 50))),
            ("mug", Primitive("cylinder", (0.045, 0.045, 0.10), (0.35, 0.15, 0.0),
                              color=(70, 110, 200))),
        ]
    )


def affordance_choice_scene() -> SceneSpec:
    """Two objects serve the same need; the better one is the harder one.

    `affordance_table` gives each need exactly one answer, which is the right
    test for *reading* a goal. This one exists to test what happens when the
    reading is right and the object is unreachable: **both** the knife and the
    scissors cut, the knife is blocked by a bottle standing in front of it, and
    the scissors are in the clear.

    So a run asked for "something to cut" should pick the knife or the scissors
    on the merits, and — only if the first choice proves genuinely unreachable —
    fall back to the other. That fallback is the one path in the whole loop a
    model has to volunteer, and CLAUDE.md §7.11 is a list of what happens to
    paths only a model can take when nothing has ever exercised them.

    Measured, and asserted by `test_synth_scene.py`: the knife is 81% visible
    with the bottle as its single blocker (occlusion); the scissors are 100%
    visible with **no blockers at all**, and directly graspable in one iteration
    against the knife's two.

    Two things had to be measured rather than assumed. A fourth object was
    removed because at 22 cm it fouled the scissors' jaw approach, and a
    scenario whose fallback is itself blocked tests nothing. And the scissors
    are 3 cm thick, not the 1.2 cm they started at: unblocked is not the same as
    graspable, and at 1.2 cm no grasp was ever found, so the run reached its
    iteration cap still looking. Both failures made the scene *look* right while
    testing nothing.
    """
    return add_objects(
        [
            ("knife", Primitive("box", (0.20, 0.025, 0.015), (-0.12, 0.18, 0.0),
                                yaw=0.03, color=(190, 190, 200))),
            ("bottle", Primitive("cylinder", (0.035, 0.035, 0.22), (-0.14, 0.07, 0.0),
                                 color=(60, 140, 80))),
            ("scissors", Primitive("box", (0.13, 0.040, 0.030), (0.24, 0.02, 0.0),
                                   yaw=-0.15, color=(160, 60, 160))),
        ]
    )


SCENARIOS = {
    "occluded_target": occluded_target_scene,
    "affordance_table": affordance_table_scene,
    "affordance_choice": affordance_choice_scene,
    "two_identical_bottles": two_identical_bottles_scene,
    "open_table": open_table_scene,
    "crowded_table": crowded_table_scene,
}


def build(
    scenario: str = "occluded_target",
    height: int = 480,
    width: int = 640,
) -> Dict[str, object]:
    """Render a named scenario. The one call most callers and tests need.

    Returns the render plus everything needed to reason about it: `K`, the
    camera extrinsic, the `SceneSpec`, the label map, and the ground-truth table
    plane in camera coordinates.
    """
    if scenario not in SCENARIOS:
        raise ValueError(
            f"unknown scenario {scenario!r}; have {sorted(SCENARIOS)}"
        )
    spec = SCENARIOS[scenario]()
    K = default_intrinsics(height, width)
    T_cam_world = default_camera()
    out = render(spec, K, T_cam_world, height=height, width=width)
    normal, offset = table_plane_in_camera(T_cam_world)
    return {
        **out,
        "K": K,
        "T_cam_world": T_cam_world,
        "spec": spec,
        "label_map": spec.label_map(),
        "plane_truth": {"normal": normal, "offset": offset},
    }
