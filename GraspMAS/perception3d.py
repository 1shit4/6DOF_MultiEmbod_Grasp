"""Depth, camera geometry and the 6-DoF grasp representation.

This module is the bridge between GraspMAS's 2D image world (masks, pixels) and
GraspGen-X's 3D world (metric point clouds, SE(3) poses).

Conventions, held everywhere:

* **Units are metres.** Depth, point clouds, gripper widths, translations.
* **Frame is the camera frame**, right-handed, +X right, +Y down, +Z forward
  (standard OpenCV pinhole). We hand GraspGen-X a camera-frame cloud, so the
  poses come back in the camera frame — no world transform is applied.
* **A grasp pose is anchored at the gripper base**, not the fingertips:
  ``pose[:3, 2]`` (+Z) is the approach axis, ``pose[:3, 0]`` (+X) is the
  closing direction. This is GraspGen-X's convention and we do not change it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Below this many valid points, GraspGen-X's own outlier removal and the PTv3
# encoder produce noise. The GraspGen-X scene loader uses the same floor.
MIN_OBJECT_POINTS = 100

# Hard cap on what we send over the wire. GraspGen-X's outlier removal is
# O(N^2) in the point count (torch.cdist(X, X) + torch.eye(N)), so an
# unthrottled 40k-point mask asks for a ~6 GB matrix and never returns. The
# model resamples to cfg.data.num_points (3500) anyway, so this costs nothing.
MAX_CLOUD_POINTS = 8192


# ---------------------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------------------


def default_intrinsics(height: int, width: int, fov_deg: float = 60.0) -> np.ndarray:
    """Synthesize a pinhole K when the real one is unknown.

    Used only on the RGB-only path, where we have no calibration. Assumes a
    centred principal point and a square pixel with `fov_deg` horizontal field
    of view — a reasonable stand-in for a consumer RGB camera.

    The absolute scale of the reconstruction depends on this guess, so grasps
    from this path carry real scale error. That is the documented cost of not
    supplying depth + intrinsics.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size {height}x{width}")
    f = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def load_intrinsics(source) -> np.ndarray:
    """Read a 3x3 K from a path (.npy / .json) or an array-like.

    JSON may be a bare 3x3 nested list, or a dict with an ``intrinsics`` /
    ``K`` / ``camera_matrix`` key — the last form is what GraspGen-X's
    ``meta_data.json`` sample scenes use.
    """
    import json
    import os

    if isinstance(source, (str, os.PathLike)):
        path = str(source)
        if path.endswith(".npy"):
            data = np.load(path)
        else:
            with open(path) as f:
                blob = json.load(f)
            if isinstance(blob, dict):
                for key in ("intrinsics", "K", "camera_matrix", "intrinsic"):
                    if key in blob:
                        blob = blob[key]
                        break
                else:
                    raise ValueError(
                        f"{path}: no intrinsics/K/camera_matrix key found"
                    )
            data = blob
    else:
        data = source

    K = np.asarray(data, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3; got {K.shape}")
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"intrinsics look invalid:\n{K}")
    return K


def unproject(
    depth: np.ndarray,
    K: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_depth: float = 5.0,
) -> np.ndarray:
    """Back-project masked pixels to a camera-frame point cloud in metres.

    Args:
        depth: (H, W) metric depth. Non-positive, non-finite and beyond
            `max_depth` values are treated as invalid and dropped — that is
            how depth sensors encode "no return".
        K: (3, 3) pinhole intrinsics.
        mask: optional (H, W) boolean/uint8 selecting the object. ``None``
            unprojects the whole image.
        max_depth: reject returns farther than this (metres).

    Returns:
        (N, 3) float32. May be empty; callers must handle that.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H, W); got {depth.shape}")
    K = np.asarray(K, dtype=np.float64)

    valid = np.isfinite(depth) & (depth > 0) & (depth < max_depth)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != depth.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match depth {depth.shape}"
            )
        valid &= mask.astype(bool)

    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth[ys, xs]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def downsample_cloud(
    points: np.ndarray, max_points: int = 8192, seed: int = 0
) -> np.ndarray:
    """Voxel-grid downsample to at most `max_points`.

    This is not an optimization, it is a correctness requirement. GraspGen-X's
    `point_cloud_outlier_removal` runs `torch.cdist(X, X)` plus a `torch.eye(N)`
    mask (utils/point_cloud.py), both O(N^2): a 40k-point mask asks for a
    40k x 40k matrix, which on CPU means ~6 GB and an effectively infinite wait.
    A close-up object mask on a 1280x720 depth image easily reaches that size.

    Downsampling costs nothing in quality either — the model resamples to
    `cfg.data.num_points` (3500) internally regardless.

    Voxel rather than random subsampling because perspective makes point density
    a function of distance: near surfaces get many more pixels than far ones, and
    random sampling would inherit that bias and skew the shape the model sees.
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) <= max_points:
        return pts

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    extent = float(np.max(hi - lo))
    if extent <= 0:
        return pts[:max_points]

    def occupied(voxel: float) -> np.ndarray:
        keys = np.floor((pts - lo) / voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return np.sort(idx)

    # A depth-image cloud is a *surface*, so occupied cells grow as
    # (extent / voxel)^2, not ^3. Sizing from the cube root (the volume
    # assumption) undershoots badly — on a cracker box it cut 16786 points to
    # 212, which is far too sparse for the encoder to see any shape.
    voxel = extent / max(np.sqrt(max_points), 2.0)

    # Bisect on voxel size so the result lands just under the cap from below,
    # rather than accepting whatever the first guess happens to give.
    lo_v, hi_v = voxel, voxel
    if len(occupied(voxel)) > max_points:
        for _ in range(24):                      # too many: coarsen
            hi_v *= 1.3
            if len(occupied(hi_v)) <= max_points:
                break
        lo_v = voxel
    else:
        for _ in range(24):                      # too few: refine
            lo_v /= 1.3
            if len(occupied(lo_v)) > max_points:
                break
        hi_v = voxel

    best = occupied(hi_v)
    for _ in range(20):
        mid = 0.5 * (lo_v + hi_v)
        if not np.isfinite(mid) or mid <= 0:
            break
        idx = occupied(mid)
        if len(idx) <= max_points:
            best, hi_v = idx, mid
        else:
            lo_v = mid
        if len(best) > max_points * 0.85:        # close enough to the cap
            break

    if len(best) > max_points:                   # pathological; fall back
        rng = np.random.default_rng(seed)
        return pts[np.sort(rng.choice(len(pts), max_points, replace=False))]
    return pts[best]


def project_points(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project camera-frame points to pixels. Points behind the camera get NaN."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64)
    z = pts[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pts[:, 0] / z + K[0, 2]
        v = K[1, 1] * pts[:, 1] / z + K[1, 2]
    uv = np.stack([u, v], axis=-1)
    uv[z <= 0] = np.nan
    return uv


# ---------------------------------------------------------------------------
# Metric monocular depth (RGB-only fallback)
# ---------------------------------------------------------------------------


class MetricDepthEstimator:
    """Lazy wrapper around Depth-Anything-V2-Metric-Indoor-Small.

    Only constructed when a query supplies no real depth. Unlike the MiDaS
    model GraspMAS used for `compute_depth`, this one predicts **metric** depth
    in metres, which is what makes unprojection meaningful. It is ~99 MB and
    runs in about a second on CPU.

    Accuracy caveat: it is a single-view indoor model. Absolute scale is a
    prior, not a measurement, so grasp width and approach distance inherit that
    error. Supply real depth when you have it.
    """

    DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

    def __init__(self, model_id: Optional[str] = None, device: str = "cpu"):
        self.model_id = model_id or self.DEFAULT_MODEL
        self.device = device
        self._model = None
        self._processor = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        logger.info("Loading metric depth model %s on %s", self.model_id, self.device)
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(
            self.device
        )
        self._model.eval()
        self._torch = torch

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Estimate metric depth (metres) for an RGB uint8 (H, W, 3) image."""
        self._ensure()
        torch = self._torch

        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"expected RGB (H, W, 3); got {img.shape}")
        h, w = img.shape[:2]

        inputs = self._processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        depth = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return depth.squeeze().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Per-query scene context
# ---------------------------------------------------------------------------


@dataclass
class SceneContext:
    """Metric depth + intrinsics for the image currently being reasoned about.

    Published as a module global on `image_patch` for the duration of a query,
    because the LLM-generated `execute_command(image)` only receives the RGB
    array — there is nowhere else to thread the calibration through.

    ``depth`` may be None on construction; the first call to `get_depth` then
    estimates it monocularly and caches the result, so the 99 MB model loads at
    most once per process and runs at most once per image.
    """

    depth: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    image_shape: Optional[tuple] = None  # (H, W)
    depth_scale: float = 1.0
    fov_deg: float = 60.0
    estimator: Optional[MetricDepthEstimator] = None
    _estimated: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.depth is not None:
            self.depth = np.asarray(self.depth, dtype=np.float32)
            if self.depth.ndim == 3 and self.depth.shape[2] == 1:
                self.depth = self.depth[:, :, 0]
            if self.depth_scale != 1.0:
                self.depth = self.depth * float(self.depth_scale)
            if self.image_shape is None:
                self.image_shape = self.depth.shape[:2]
        if self.intrinsics is not None:
            self.intrinsics = load_intrinsics(self.intrinsics)

    @property
    def has_real_depth(self) -> bool:
        return self.depth is not None and not self._estimated

    @property
    def depth_source(self) -> str:
        if self.depth is None:
            return "none"
        return "estimated" if self._estimated else "sensor"

    def get_depth(self, image: np.ndarray) -> np.ndarray:
        """Metric depth for `image`, estimating it once if none was supplied."""
        if self.depth is not None:
            if self.depth.shape[:2] != image.shape[:2]:
                raise ValueError(
                    f"depth {self.depth.shape[:2]} does not match image "
                    f"{image.shape[:2]}"
                )
            return self.depth
        if self.estimator is None:
            self.estimator = MetricDepthEstimator()
        logger.info("No sensor depth supplied; estimating metric depth monocularly")
        self.depth = self.estimator(image)
        self._estimated = True
        return self.depth

    def get_intrinsics(self, image: np.ndarray) -> np.ndarray:
        """Real K if supplied, else a synthesized one for this image size."""
        if self.intrinsics is not None:
            return self.intrinsics
        h, w = image.shape[:2]
        logger.info(
            "No intrinsics supplied; assuming %.0f deg horizontal FOV at %dx%d",
            self.fov_deg,
            w,
            h,
        )
        self.intrinsics = default_intrinsics(h, w, self.fov_deg)
        return self.intrinsics

    def describe(self) -> dict:
        K = self.intrinsics
        return {
            "depth_source": self.depth_source,
            "has_intrinsics": K is not None,
            "fx": float(K[0, 0]) if K is not None else None,
            "fy": float(K[1, 1]) if K is not None else None,
            "cx": float(K[0, 2]) if K is not None else None,
            "cy": float(K[1, 2]) if K is not None else None,
            "image_shape": list(self.image_shape) if self.image_shape else None,
        }


# ---------------------------------------------------------------------------
# The 6-DoF grasp
# ---------------------------------------------------------------------------


@dataclass
class Grasp6D:
    """A single 6-DoF grasp, in the camera frame, metres.

    `rect_2d` is the back-projection of the gripper's closing segment into the
    image as `[score, cx, cy, w, h, angle_deg]`. It keeps the original 2D
    tooling working — the OCID-VLG IoU/angle metric and the rectangle overlay —
    so 6-DoF output stays comparable against the planar baseline.
    """

    pose: np.ndarray  # (4, 4)
    score: float
    gripper: str
    width: float  # jaw aperture, metres
    rect_2d: Optional[list] = None
    branch: Optional[str] = None  # "diff" | "obb" when the server reports it

    def __post_init__(self):
        self.pose = np.asarray(self.pose, dtype=np.float64).reshape(4, 4)
        self.score = float(self.score)
        self.width = float(self.width)

    @property
    def position(self) -> np.ndarray:
        return self.pose[:3, 3]

    @property
    def approach(self) -> np.ndarray:
        """+Z of the gripper frame: the direction the hand advances along."""
        return self.pose[:3, 2]

    @property
    def closing(self) -> np.ndarray:
        """+X of the gripper frame: the direction the fingers close along."""
        return self.pose[:3, 0]

    def as_dict(self) -> dict:
        """JSON/msgpack-safe dict — this is what `grasp_detection` returns."""
        return {
            "pose": self.pose.tolist(),
            "score": self.score,
            "gripper": self.gripper,
            "width": self.width,
            "position": self.position.tolist(),
            "approach": self.approach.tolist(),
            "closing": self.closing.tolist(),
            "rect_2d": list(self.rect_2d) if self.rect_2d is not None else None,
            "branch": self.branch,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Grasp6D":
        return cls(
            pose=np.asarray(d["pose"], dtype=np.float64),
            score=d["score"],
            gripper=d.get("gripper", "unknown"),
            width=d.get("width", 0.08),
            rect_2d=d.get("rect_2d"),
            branch=d.get("branch"),
        )

    def summary(self) -> str:
        p, a = self.position, self.approach
        return (
            f"{self.gripper} score={self.score:.3f} "
            f"pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})m "
            f"approach=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f}) "
            f"width={self.width*100:.1f}cm"
        )


def gripper_finger_points(
    pose: np.ndarray, width: float, fingertip_depth: float
) -> np.ndarray:
    """The two fingertip centres of a parallel jaw at `pose`, camera frame.

    The pose is at the gripper base, so the fingertips sit `fingertip_depth`
    along +Z and are separated by `width` along +X.
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    local = np.array(
        [[-width / 2.0, 0.0, fingertip_depth], [width / 2.0, 0.0, fingertip_depth]],
        dtype=np.float64,
    )
    return (pose[:3, :3] @ local.T).T + pose[:3, 3]


def grasp_to_rect2d(
    pose: np.ndarray,
    K: np.ndarray,
    width: float,
    score: float = 1.0,
    fingertip_depth: float = 0.0,
    finger_thickness: float = 0.02,
) -> Optional[list]:
    """Back-project a 6-DoF grasp to `[score, cx, cy, w, h, angle_deg]`.

    The rectangle's long axis is the projected closing direction and its length
    is the projected jaw aperture — the same semantics as the planar grasp
    rectangles GraspMAS used to emit, so the OCID-VLG metric still applies.

    `angle_deg` follows OpenCV's rotated-rect convention (degrees,
    counter-clockwise from the image +X axis), matching `cv2.boxPoints`.

    Returns None if the grasp is behind the camera or projects to nothing.
    """
    tips = gripper_finger_points(pose, width, fingertip_depth)
    uv = project_points(tips, K)
    if not np.isfinite(uv).all():
        return None

    (u0, v0), (u1, v1) = uv
    cx, cy = (u0 + u1) / 2.0, (v0 + v1) / 2.0
    du, dv = u1 - u0, v1 - v0
    w_px = float(math.hypot(du, dv))
    if w_px < 1e-6:
        return None
    angle = float(math.degrees(math.atan2(dv, du)))

    # Finger thickness in pixels, using the focal length at the grasp depth.
    z = float(np.mean(tips[:, 2]))
    fy = float(np.asarray(K, dtype=np.float64)[1, 1])
    h_px = float(finger_thickness * fy / z) if z > 0 else w_px * 0.5

    return [float(score), float(cx), float(cy), w_px, h_px, angle]


def filter_grasps_by_mask(
    grasps: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    width: float,
    fingertip_depth: float = 0.0,
    dilate_px: int = 12,
) -> np.ndarray:
    """Indices of grasps whose fingertip midpoint projects inside `mask`.

    This is what keeps the language constraint attached to the 6-DoF output.
    GraspGen-X only sees geometry: given a knife-handle cloud it will happily
    propose a grasp that reaches back over the blade. Requiring the grasp
    centre to land on the referred region enforces the referring expression.

    `dilate_px` tolerates the fact that a grasp centre legitimately sits a
    little off the visible surface (mask edges are noisy, and the centre is
    between the fingers, not on the object).
    """
    if len(grasps) == 0:
        return np.zeros((0,), dtype=int)

    mask = np.asarray(mask).astype(bool)
    if dilate_px > 0:
        try:
            import cv2

            k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
        except ImportError:  # pragma: no cover - cv2 is a hard dep in practice
            logger.warning("cv2 unavailable; skipping mask dilation")

    h, w = mask.shape
    keep = []
    for i, pose in enumerate(grasps):
        tips = gripper_finger_points(pose, width, fingertip_depth)
        centre = tips.mean(axis=0, keepdims=True)
        uv = project_points(centre, K)[0]
        if not np.isfinite(uv).all():
            continue
        u, v = int(round(uv[0])), int(round(uv[1]))
        if 0 <= u < w and 0 <= v < h and mask[v, u]:
            keep.append(i)
    return np.asarray(keep, dtype=int)


def filter_grasps_by_visibility(
    grasps: np.ndarray,
    max_backward_deg: float = 100.0,
) -> np.ndarray:
    """Indices of grasps that do not approach from the object's unobserved side.

    A single depth image sees one surface. GraspGen-X reconstructs a plausible
    full object from that partial view, and GraspMoE's OBB branch then sweeps
    candidate poses over *every* face of the fitted box — including the far side
    the camera never saw. Measured on the sample scene, that floods the
    candidate set: median approach elevation 180 degrees, i.e. a hand travelling
    from behind the object toward the camera, which for a tabletop means
    reaching through the table or through the object itself.

    The discriminator does rank those down (top-20 by score had a median
    elevation of 76 degrees), but the mask-containment filter does not remove
    them, so an argmax over in-mask candidates can still land on one.

    The test is against the **viewing ray**, not the optical axis: a grasp is
    rejected when its approach axis points back along the ray to the camera by
    more than `max_backward_deg`. Side approaches (~90 degrees) stay, because
    those are legitimate and are much of the point of going 6-DoF.
    """
    if len(grasps) == 0:
        return np.zeros((0,), dtype=int)

    positions = grasps[:, :3, 3]
    approach = grasps[:, :3, 2]

    ray_norm = np.linalg.norm(positions, axis=1)
    a_norm = np.linalg.norm(approach, axis=1)
    valid = (ray_norm > 1e-9) & (a_norm > 1e-9)

    cos = np.zeros(len(grasps))
    cos[valid] = np.einsum(
        "ij,ij->i", approach[valid] / a_norm[valid, None],
        positions[valid] / ray_norm[valid, None],
    )
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.nonzero(valid & (angle <= max_backward_deg))[0]


def approach_elevation_deg(approach: Sequence[float]) -> float:
    """Angle between the approach axis and the camera's optical axis (+Z).

    0 deg is a head-on approach straight down the camera ray (what the old
    2D pipeline implicitly assumed); larger values mean the 6-DoF model chose
    a genuinely oblique approach. Useful as a diversity statistic.
    """
    a = np.asarray(approach, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-9:
        return float("nan")
    return float(math.degrees(math.acos(np.clip(a[2] / n, -1.0, 1.0))))
