"""Does the hand hit anything on its way in?

`grasp_detection` filters candidates by visibility and by the target's own mask,
but never against the rest of the scene. On an isolated object that is fine. In
clutter it is not: GraspGen-X sees only the cloud it was given, so given a
bottle's points it will happily propose a grasp whose fingers pass through the
mug standing next to it. Nothing downstream notices, because the mug was never
in the input.

This module supplies the missing test. It is deliberately cheap:

* The gripper geometry comes from the `points.json` every gripper description
  ships — a 10,500-point surface sample **in the gripper base frame**, the same
  frame our poses use (verified against `config.json["bbox"]`). So there is no
  mesh loading, and neither `trimesh` nor FCL is needed; neither is installed in
  the `graspmas` environment.
* Proximity is a `scipy.spatial.cKDTree` ball query rather than the
  `torch.cdist` sweep in `GraspGenX/graspgenx/utils/collision_filter.py`, which
  its own docstring calls "dramatically slower" on CPU.

Frames and units follow `perception3d`: metres, camera frame, pose at the
gripper base with +Z along the approach.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# How close a scene point may come to the gripper surface before we call it a
# collision. The same order as GraspGenX's own `collision_threshold=0.02`, but
# tighter: our clouds are a single view, so a generous margin rejects grasps
# that are actually fine.
DEFAULT_COLLISION_THRESH_M = 0.01

# A lone stray return should not veto a grasp. Depth edges produce flyer points
# exactly where two objects meet, which is precisely where grasps live.
DEFAULT_MIN_HITS = 3

# Sampling the full 10,500-point gripper for every candidate is waste, but
# undersampling is a correctness bug: proximity is measured *from* these points,
# so anything smaller than the gap between them can pass through the hand
# undetected. Measured 95th-percentile nearest-neighbour spacing on a Panda:
# 384 points -> 12.3 mm, 1024 -> 7.5 mm, 2048 -> 5.5 mm. The spacing has to stay
# under DEFAULT_COLLISION_THRESH_M (10 mm), so 384 is not enough and 1024 is.
# Cost for 200 candidates against a 15k-point scene: 116 ms -> 289 ms, which is
# noise next to the 2-13 s a grasp inference already takes.
DEFAULT_GRIPPER_POINTS = 1024


def gripper_asset_dir(gripper_name: str) -> Optional[str]:
    """Locate a gripper's description directory, or None if it is not there.

    `GRASPGENX_GRIPPER_CFG_DIR` points at the *checkout root*, and
    `get_gripper_descriptions_assets()` appends the rest — the same path shape
    `ImagePatch._gripper_geometry` uses.
    """
    cfg_root = os.environ.get("GRASPGENX_GRIPPER_CFG_DIR")
    if not cfg_root:
        return None
    path = os.path.join(
        cfg_root, "gripper_descriptions", "assets", "x_grippers", gripper_name
    )
    return path if os.path.isdir(path) else None


@lru_cache(maxsize=None)
def gripper_config(gripper_name: str) -> dict:
    """A gripper's `config.json`, or an empty dict if it is not installed.

    One reader for a file three subsystems need and each used to open its own
    way: the mask filter (jaw width, fingertip depth), `vis6d` (drawing a hand
    that is the right shape), and the Observer's grasp summary (morphology).
    Reading it three ways is how the overlay ended up drawing every gripper at
    the Franka's fingertip depth.

    Read-only: `lru_cache` hands the same dict to every caller.
    """
    asset_dir = gripper_asset_dir(gripper_name)
    if not asset_dir:
        return {}
    try:
        with open(os.path.join(asset_dir, "config.json")) as f:
            cfg = json.load(f)
    except Exception as exc:
        logger.debug("could not read config.json for %s: %s", gripper_name, exc)
        return {}
    return cfg if isinstance(cfg, dict) else {}


def gripper_morphology(gripper_name: str) -> dict:
    """What kind of hand this is, in terms the Observer can reason about.

    It was given a name string and a jaw width, which is not enough to judge
    "can this hand close on that object" for anything but the gripper it had
    already seen. `type` is the description's own label — `parallel_2f`,
    `revolute_2f`, `revolute_3f` — and the finger count falls out of it.
    """
    cfg = gripper_config(gripper_name)
    kind = str(cfg.get("type", "") or "")
    fingers = None
    if kind.endswith("f") and kind[-2:-1].isdigit():
        fingers = int(kind[-2:-1])

    out = {"name": gripper_name, "type": kind or "unknown", "n_fingers": fingers}
    bbox = cfg.get("bbox")
    try:
        lo, hi = bbox[0], bbox[1]
        out["extent_cm"] = [round((hi[i] - lo[i]) * 100, 1) for i in range(3)]
    except (TypeError, IndexError, KeyError):
        pass
    return out


def gripper_geometry(gripper_name: str) -> tuple:
    """``(jaw aperture, base->fingertip depth)`` in metres.

    Falls back to Franka-like numbers when the descriptions are not installed,
    which then only affect projection and drawing, never a returned pose.
    """
    cfg = gripper_config(gripper_name)
    try:
        return (
            float(cfg["sweep_volume"]["extents"][0]),
            float(cfg["fingertip"][-1]),
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.08, 0.11


def _fallback_gripper_points(n: int) -> np.ndarray:
    """A Franka-sized box hull, for when the descriptions are not installed.

    Deliberately coarse. It keeps the collision filter functional (and testable)
    without asset downloads, and it is bigger than a real Panda in every
    dimension, so it errs toward rejecting grasps rather than passing bad ones.
    """
    rng = np.random.default_rng(0)
    lo = np.array([-0.10, -0.032, -0.026])
    hi = np.array([0.104, 0.032, 0.113])
    pts = rng.uniform(lo, hi, size=(n, 3))
    # Push each point onto its nearest face so the sample is a surface, not a
    # solid — the same trick `scripts/bench_cpu.py:synthetic_box` uses.
    centre = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    rel = (pts - centre) / half
    axis = np.argmax(np.abs(rel), axis=1)
    rows = np.arange(len(pts))
    pts[rows, axis] = centre[axis] + np.sign(rel[rows, axis]) * half[axis]
    return pts


@lru_cache(maxsize=32)
def load_gripper_points(
    gripper_name: str,
    state: str = "open",
    max_points: int = DEFAULT_GRIPPER_POINTS,
    seed: int = 0,
) -> np.ndarray:
    """Surface points of a gripper in its own base frame, metres.

    `state` is "open" or "close". Approach and placement checks want "open",
    because that is the configuration the hand travels in.

    Cached: the file is 10,500 points of JSON and the outer loop asks for the
    same gripper on every candidate of every iteration.
    """
    if state not in ("open", "close"):
        raise ValueError(f"state must be 'open' or 'close'; got {state!r}")

    asset_dir = gripper_asset_dir(gripper_name)
    pts = None
    if asset_dir:
        try:
            with open(os.path.join(asset_dir, "points.json")) as f:
                pts = np.asarray(json.load(f)[state], dtype=np.float64)
        except Exception as exc:
            logger.debug("could not read points.json for %s: %s", gripper_name, exc)

    if pts is None or pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        logger.warning(
            "no gripper geometry for %r; using a conservative box hull", gripper_name
        )
        return _fallback_gripper_points(max_points)

    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[np.sort(rng.choice(len(pts), max_points, replace=False))]
    # Read-only: lru_cache hands the same array to every caller.
    pts.flags.writeable = False
    return pts


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to (N, 3) points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    return pts @ T[:3, :3].T + T[:3, 3]


def scene_cloud_excluding(
    depth: np.ndarray,
    K: np.ndarray,
    exclude_mask: Optional[np.ndarray] = None,
    max_depth: float = 3.0,
    max_points: int = 20000,
) -> np.ndarray:
    """The scene as a cloud, minus the object being grasped.

    Excluding the target is not optional: the gripper closes *around* it, so
    with its points left in, every grasp collides with the thing it is meant to
    pick up. This mirrors `build_scene_pc_excluding_object` in
    `GraspGenX/graspgenx/utils/scene_loaders.py`.

    `max_depth` defaults to 3 m rather than `perception3d.unproject`'s 5 m
    because obstacles that matter are on the table, and a laboratory background
    at 4 m contributes nothing but points.
    """
    from perception3d import downsample_cloud, unproject

    keep = None
    if exclude_mask is not None:
        keep = ~np.asarray(exclude_mask).astype(bool)

    cloud = unproject(depth, K, mask=keep, max_depth=max_depth)
    if len(cloud) > max_points:
        cloud = downsample_cloud(cloud, max_points=max_points)
    return cloud


def _build_tree(scene_cloud: np.ndarray):
    from scipy.spatial import cKDTree

    pts = np.asarray(scene_cloud, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return None
    return cKDTree(pts)


# Gripper trees are built once per gripper and reused across every candidate.
# Keyed on identity, with the array kept alive so the id cannot be recycled;
# `load_gripper_points` is itself cached, so this holds at most a few entries.
_GRIPPER_TREES: dict = {}


def _gripper_tree(gripper_points: np.ndarray):
    from scipy.spatial import cKDTree

    cached = _GRIPPER_TREES.get(id(gripper_points))
    if cached is not None and cached[0] is gripper_points:
        return cached[1]
    tree = cKDTree(np.asarray(gripper_points, dtype=np.float64).reshape(-1, 3))
    _GRIPPER_TREES[id(gripper_points)] = (gripper_points, tree)
    return tree


def pose_collides(
    pose: np.ndarray,
    scene_tree,
    gripper_points: np.ndarray,
    thresh: float = DEFAULT_COLLISION_THRESH_M,
    min_hits: int = DEFAULT_MIN_HITS,
) -> bool:
    """Does the hand at `pose` overlap the scene?

    `min_hits` counts **distinct scene points** inside the hand, which is the
    only reading that does what it is for. Querying the scene from each gripper
    point and counting the gripper points that found something looks equivalent
    and is not: with the gripper sampled every 3.4 mm and a 10 mm threshold, one
    stray depth flyer lands within reach of dozens of gripper points and trips
    any threshold instantly — the exact noise `min_hits` exists to absorb.

    So the scene is pruned to the hand's neighbourhood, transformed into the
    gripper frame, and queried against a cached tree of the gripper itself.

    `scene_tree` is a prebuilt cKDTree so the caller pays for it once across
    hundreds of candidates. A None tree means an empty scene: nothing to hit.
    """
    if scene_tree is None:
        return False

    T = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    gpts = np.asarray(gripper_points, dtype=np.float64).reshape(-1, 3)

    # Only scene points within the hand's own reach can possibly be inside it.
    radius = float(np.linalg.norm(gpts, axis=1).max()) + thresh
    near = scene_tree.query_ball_point(T[:3, 3], r=radius)
    if len(near) < min_hits:
        return False

    local = (np.asarray(scene_tree.data)[near] - T[:3, 3]) @ T[:3, :3]
    counts = _gripper_tree(gripper_points).query_ball_point(
        local, r=thresh, return_length=True
    )
    return int(np.count_nonzero(counts)) >= min_hits


def sweep_is_clear(
    pose: np.ndarray,
    scene_cloud: np.ndarray,
    gripper_points: np.ndarray,
    approach_len: float = 0.10,
    n_samples: int = 4,
    thresh: float = DEFAULT_COLLISION_THRESH_M,
    min_hits: int = DEFAULT_MIN_HITS,
    scene_tree=None,
) -> bool:
    """Is the whole approach corridor free, not just the final pose?

    The hand advances along +Z, so it occupies ``pose`` translated backwards
    along the approach axis for everything up to `approach_len` before arrival.
    Checking only the final pose misses the case where the hand has to pass
    through a neighbour to reach an otherwise clean grasp.
    """
    tree = scene_tree if scene_tree is not None else _build_tree(scene_cloud)
    if tree is None:
        return True

    approach = np.asarray(pose, dtype=np.float64).reshape(4, 4)[:3, 2]
    for t in np.linspace(0.0, approach_len, max(n_samples, 1)):
        probe = np.asarray(pose, dtype=np.float64).reshape(4, 4).copy()
        probe[:3, 3] = probe[:3, 3] - t * approach
        if pose_collides(probe, tree, gripper_points, thresh=thresh, min_hits=min_hits):
            return False
    return True


def filter_grasps_by_scene_collision(
    grasps: np.ndarray,
    scene_cloud: np.ndarray,
    gripper_points: np.ndarray,
    thresh: float = DEFAULT_COLLISION_THRESH_M,
    min_hits: int = DEFAULT_MIN_HITS,
    approach_len: float = 0.0,
    n_samples: int = 4,
) -> np.ndarray:
    """Indices of grasps whose hand does not overlap the scene.

    Set `approach_len` above zero to also require a clear corridor. That costs
    `n_samples` times as much, which is why it is opt-in: the final-pose check
    alone already removes the grasps that matter most, and the Observer sees the
    approach arrow drawn over the image either way.

    Returns all indices unchanged when the scene cloud is empty, so a caller
    with no depth outside the object degrades to today's behaviour rather than
    rejecting everything.
    """
    grasps = np.asarray(grasps, dtype=np.float64)
    if len(grasps) == 0:
        return np.zeros((0,), dtype=int)

    tree = _build_tree(scene_cloud)
    if tree is None:
        return np.arange(len(grasps), dtype=int)

    keep = []
    for i, pose in enumerate(grasps):
        if approach_len > 0:
            ok = sweep_is_clear(
                pose,
                scene_cloud,
                gripper_points,
                approach_len=approach_len,
                n_samples=n_samples,
                thresh=thresh,
                min_hits=min_hits,
                scene_tree=tree,
            )
        else:
            ok = not pose_collides(
                pose, tree, gripper_points, thresh=thresh, min_hits=min_hits
            )
        if ok:
            keep.append(i)
    return np.asarray(keep, dtype=int)


def points_inside_sweep(
    pose: np.ndarray,
    points: np.ndarray,
    gripper_points: np.ndarray,
    approach_len: float = 0.10,
    n_samples: int = 4,
    thresh: float = DEFAULT_COLLISION_THRESH_M,
) -> np.ndarray:
    """Which of `points` lie in the hand's swept volume.

    Used by `scene_registry.blocking_objects` to attribute an obstruction to a
    specific object: run this with one object's cloud and a candidate grasp on
    the target, and a non-empty result names that object as a blocker.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0,), dtype=int)

    tree = _build_tree(pts)
    if tree is None:
        return np.zeros((0,), dtype=int)

    approach = np.asarray(pose, dtype=np.float64).reshape(4, 4)[:3, 2]
    hit = np.zeros(len(pts), dtype=bool)
    for t in np.linspace(0.0, approach_len, max(n_samples, 1)):
        probe = np.asarray(pose, dtype=np.float64).reshape(4, 4).copy()
        probe[:3, 3] = probe[:3, 3] - t * approach
        world = transform_points(gripper_points, probe)
        for idx in tree.query_ball_point(world, r=thresh):
            if idx:
                hit[idx] = True
    return np.nonzero(hit)[0]
