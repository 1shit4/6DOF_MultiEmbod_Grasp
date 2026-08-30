"""Where to put an object down.

`perception3d` answers "where do I grasp this?". This module answers the other
half of a pick-and-place: given a grasp on an object and a view of the scene,
where on the support surface can the object be released without hitting
anything and without still obstructing the target?

Conventions inherited from `perception3d` — metres, camera frame, grasp poses
anchored at the gripper base with +Z approach — plus one of our own:

* **The table frame.** A right-handed frame whose +Z is the support-surface
  normal pointing back toward the camera, whose origin is the point of the
  plane closest to the camera centre, and whose +X is the camera's +X projected
  onto the plane. Heights are then simply the table-frame Z, and free-space
  search becomes a 2D problem in table-frame XY.

* **A place pose is a pure translation of the grasp pose.** The object is set
  down in exactly the orientation it was picked up in, so
  ``T_place = Translation(delta) @ T_grasp`` and only ``delta`` has to be
  solved. This is not a simplification we are apologising for: it means the
  object's footprint is unchanged between pick and place, so the free-space
  test is exact rather than an estimate over an unknown post-rotation shape,
  and it removes any question of whether the object is stable in a new pose.

* **Unobserved space is occupied.** A single depth image sees no surface behind
  an object. A cell with no returns is not known to be empty, and placing into
  one is the mistake that puts a bottle on top of whatever was hiding there.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# A plane fitted from fewer points than this is not evidence of a support
# surface, it is noise. Matches perception3d.MIN_OBJECT_POINTS in spirit.
MIN_PLANE_POINTS = 200

# Default grid resolution for the support-surface height map. 5 mm is well
# below the placement tolerances that matter (centimetres) and keeps a
# 1.5 m x 1.5 m table at 300x300 cells, which every operation here handles in
# milliseconds.
DEFAULT_CELL_M = 0.005

# How far above the plane a cell has to rise before it counts as an obstacle.
# Below this is table texture, depth noise and the odd stray return.
DEFAULT_CLEARANCE_M = 0.01

# Slack required around a placement, on top of the object's own radius.
# Two things live in this number:
#   * ~1.7 cm because `object_footprint` is biased small — a single depth view
#     sees only the front of an object, measured on the synthetic scenes.
#   * ~1.3 cm of genuine safety, for release jitter and depth noise.
# Kept as one constant so the bias is corrected once rather than at each site.
DEFAULT_MARGIN_M = 0.03


def gripper_clearance_m(gripper_name: str) -> float:
    """Space to leave around a placed object, for the hand that must reach it.

    `DEFAULT_MARGIN_M` corrects the footprint bias; it says nothing about the
    hand. But a placement is only useful if the gripper can get to the object
    afterwards, and the room needed for that is a property of the gripper —
    measured as the half-width of its own surface geometry.

    Falls back to `DEFAULT_MARGIN_M` when the descriptions are not installed,
    which keeps every offline test and the no-asset path behaving as before.
    """
    try:
        import numpy as _np

        import collision as _col

        pts = _col.load_gripper_points(gripper_name)
        half_width = float(_np.abs(pts[:, 0]).max())
    except Exception:
        return DEFAULT_MARGIN_M
    return max(DEFAULT_MARGIN_M, half_width + DEFAULT_MARGIN_M)


# ---------------------------------------------------------------------------
# Support plane
# ---------------------------------------------------------------------------


@dataclass
class SupportPlane:
    """The table, as a plane plus a frame anchored to it.

    The plane is ``normal . x + offset = 0`` in the camera frame, with `normal`
    a unit vector pointing back toward the camera — so the signed distance
    ``normal . p + offset`` of any point is its height above the table, positive
    on the side the objects are on.
    """

    normal: np.ndarray  # (3,) unit, camera frame, pointing toward the camera
    offset: float
    T_cam_table: np.ndarray  # (4, 4) table frame -> camera frame
    inlier_ratio: float
    n_inliers: int = 0

    def __post_init__(self):
        self.normal = np.asarray(self.normal, dtype=np.float64).reshape(3)
        self.offset = float(self.offset)
        self.T_cam_table = np.asarray(self.T_cam_table, dtype=np.float64).reshape(4, 4)
        self.inlier_ratio = float(self.inlier_ratio)

    @property
    def origin(self) -> np.ndarray:
        """Table-frame origin, in camera coordinates."""
        return self.T_cam_table[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        """Table -> camera rotation. Columns are the table axes in camera coords."""
        return self.T_cam_table[:3, :3]

    def height_of(self, points: np.ndarray) -> np.ndarray:
        """Signed height above the plane, metres. Positive is toward the camera."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return pts @ self.normal + self.offset

    def to_table(self, points: np.ndarray) -> np.ndarray:
        """Camera-frame points -> table-frame points."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return (pts - self.origin) @ self.rotation

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        """Table-frame points -> camera-frame points."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return pts @ self.rotation.T + self.origin

    def camera_xy(self) -> np.ndarray:
        """The camera centre projected into the table's XY plane.

        This is where the viewing rays converge from, so it anchors the
        occlusion keep-out in `projected_occlusion_keep_out`.
        """
        return self.to_table(np.zeros((1, 3)))[0, :2]

    def describe(self) -> dict:
        return {
            "normal": [round(float(v), 4) for v in self.normal],
            "offset_m": round(self.offset, 4),
            "inlier_ratio": round(self.inlier_ratio, 3),
            "n_inliers": int(self.n_inliers),
            "tilt_deg": round(
                float(np.degrees(np.arccos(np.clip(-self.normal[2], -1.0, 1.0)))), 1
            ),
        }


def _frame_from_plane(normal: np.ndarray, offset: float) -> np.ndarray:
    """Build the table frame from a plane, deterministically.

    +Z is the plane normal. The origin is the plane point closest to the camera
    centre, which for ``n.x + d = 0`` is ``-d * n``. +X is the camera's +X
    projected onto the plane, so the table frame's XY axes stay visually aligned
    with the image and a debug print of a table-frame coordinate is readable.
    """
    z_axis = normal / np.linalg.norm(normal)

    # Camera +X, orthogonalised against the normal. A tabletop viewed by a
    # camera that is not rolled 90 degrees always leaves this well conditioned;
    # the fallback covers the degenerate case rather than trusting it not to
    # happen.
    x_ref = np.array([1.0, 0.0, 0.0])
    x_axis = x_ref - np.dot(x_ref, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        x_ref = np.array([0.0, 0.0, 1.0])
        x_axis = x_ref - np.dot(x_ref, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = -offset * z_axis
    return T


def support_cloud(
    depth: np.ndarray,
    K: np.ndarray,
    object_mask: Optional[np.ndarray] = None,
    margin_m: float = 0.5,
    max_depth: float = 3.0,
    max_points: int = 40000,
) -> np.ndarray:
    """Unproject the working volume — the objects and what they rest on.

    Fitting a support plane to a *whole* real capture does not work, and it
    should not: `real_world/00` contains a floor, a far wall, another bench and
    a robot, and more than 5% of that cloud lies below any candidate plane, so
    `fit_support_plane` correctly refuses. Measured on that scene, cutting the
    cloud at 2 m instead of 5 m takes it from "no plane found" to a 59%-inlier
    fit with every object sitting 0.8-15 cm above the surface.

    Rather than make callers guess a cutoff, derive it: everything of interest
    is at most `margin_m` behind the furthest object we can see. With no object
    mask this falls back to a fixed `max_depth`, which is still tighter than
    `perception3d.unproject`'s 5 m default.
    """
    from perception3d import downsample_cloud, unproject

    limit = max_depth
    if object_mask is not None:
        m = np.asarray(object_mask).astype(bool)
        obj_depth = np.asarray(depth)[m]
        obj_depth = obj_depth[np.isfinite(obj_depth) & (obj_depth > 0)]
        if len(obj_depth):
            limit = float(np.percentile(obj_depth, 95)) + margin_m

    cloud = unproject(depth, K, max_depth=limit)
    if len(cloud) > max_points:
        cloud = downsample_cloud(cloud, max_points=max_points)
    return cloud


def fit_support_plane(
    cloud: np.ndarray,
    thresh: float = 0.01,
    iters: int = 400,
    min_inlier_ratio: float = 0.15,
    max_below: float = 0.05,
    max_points: int = 20000,
    seed: int = 0,
) -> SupportPlane:
    """RANSAC the support surface out of a full-scene point cloud.

    Plain "biggest plane wins" is not safe on a cluttered table. Measured on the
    `crowded_table` scenario, where boxes cover most of the surface, it fits a
    plane tilted 9.9 degrees through the box fronts and misses the table by
    8.4 cm — the tabletop simply has fewer visible points than the clutter
    standing on it.

    What separates a support surface from any other plane is not size, it is
    that **nothing is underneath it**. On the same scene the true tabletop has
    0.0% of the cloud below it and the wrong plane has 6.9%. So candidates with
    more than `max_below` of the cloud beneath them are discarded outright, and
    only then does inlier count decide. `max_below` is not zero because a real
    scene shows floor past the table edge and the odd depth flyer.

    Raises:
        ValueError: if the cloud is too small, or nothing survives — better to
            fail loudly than to hand back a wall and call it a table.
    """
    pts = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < MIN_PLANE_POINTS:
        raise ValueError(
            f"need at least {MIN_PLANE_POINTS} points to fit a support plane; "
            f"got {len(pts)}"
        )

    rng = np.random.default_rng(seed)

    # Every candidate is scored against the whole cloud, so a full-resolution
    # 1280x720 scene would cost 400 x 900k dot products for no extra accuracy —
    # a plane is over-determined by a few thousand points. Subsample uniformly,
    # which preserves the inlier *proportions* the scoring depends on.
    if len(pts) > max_points:
        pts = pts[np.sort(rng.choice(len(pts), max_points, replace=False))]
    n = len(pts)
    best_score = -1.0
    best: Optional[Tuple[np.ndarray, float, np.ndarray]] = None

    # Sampling all three points uniformly wastes iterations on triples that
    # straddle an object. Drawing them from the same neighbourhood would be
    # better still, but uniform with this many iterations is already reliable
    # at these cloud sizes and keeps the function dependency-free.
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        offset = -float(np.dot(normal, p0))

        dist = pts @ normal + offset
        inliers = np.abs(dist) <= thresh
        n_in = int(inliers.sum())
        if n_in < MIN_PLANE_POINTS:
            continue

        # Orient toward the camera (the origin), so "above the table" is
        # positive and "below" means underneath the surface.
        if offset < 0:
            normal, offset, dist = -normal, -offset, -dist

        if float((dist < -thresh).mean()) > max_below:
            continue  # something is under it, so it is not what things rest on

        if n_in > best_score:
            best_score = n_in
            best = (normal, offset, inliers)

    if best is None:
        raise ValueError(
            "no plane had a clear underside; either this is not a scene resting "
            f"on a surface, or more than {max_below:.0%} of it lies below every "
            "candidate (try restricting the cloud to the working volume)"
        )

    normal, offset, inliers = best

    # Least-squares refit on the inliers. RANSAC picks the right *set*; three
    # points are a poor estimate of the plane those points define.
    inlier_pts = pts[inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = vh[2]
    offset = -float(np.dot(normal, centroid))
    if offset < 0:
        normal, offset = -normal, -offset

    dist = pts @ normal + offset
    n_inliers = int((np.abs(dist) <= thresh).sum())
    ratio = n_inliers / float(n)
    if ratio < min_inlier_ratio:
        raise ValueError(
            f"best plane holds only {ratio:.1%} of the cloud "
            f"(need {min_inlier_ratio:.0%}); no support surface found"
        )

    return SupportPlane(
        normal=normal,
        offset=offset,
        T_cam_table=_frame_from_plane(normal, offset),
        inlier_ratio=ratio,
        n_inliers=n_inliers,
    )


# ---------------------------------------------------------------------------
# Height map over the support surface
# ---------------------------------------------------------------------------


@dataclass
class HeightMap:
    """A 2.5D occupancy grid in table-frame XY.

    `heights[r, c]` is the tallest thing observed over that cell, in metres
    above the plane. **NaN means unobserved**, which is a different and more
    dangerous thing than zero: see the module docstring.
    """

    heights: np.ndarray  # (rows, cols) float32, NaN where nothing was seen
    origin: np.ndarray  # (2,) table-frame XY of the (0, 0) cell's corner
    cell_m: float

    def __post_init__(self):
        self.heights = np.asarray(self.heights, dtype=np.float32)
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(2)
        self.cell_m = float(self.cell_m)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.heights.shape

    def to_cell(self, xy: np.ndarray) -> np.ndarray:
        """Table-frame XY -> integer (row, col). Not bounds-checked."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        rc = np.floor((xy - self.origin) / self.cell_m).astype(np.int64)
        return rc[:, ::-1]  # (x, y) -> (row=y, col=x)

    def to_xy(self, rc: np.ndarray) -> np.ndarray:
        """Integer (row, col) -> table-frame XY of the cell centre."""
        rc = np.asarray(rc, dtype=np.float64).reshape(-1, 2)
        xy = rc[:, ::-1]  # (row, col) -> (x=col, y=row)
        return self.origin + (xy + 0.5) * self.cell_m

    def in_bounds(self, rc: np.ndarray) -> np.ndarray:
        rc = np.asarray(rc).reshape(-1, 2)
        rows, cols = self.shape
        return (
            (rc[:, 0] >= 0) & (rc[:, 0] < rows) & (rc[:, 1] >= 0) & (rc[:, 1] < cols)
        )

    @property
    def observed(self) -> np.ndarray:
        return np.isfinite(self.heights)


def build_height_map(
    scene_cloud: np.ndarray,
    plane: SupportPlane,
    cell_m: float = DEFAULT_CELL_M,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    max_height_m: float = 0.6,
) -> HeightMap:
    """Rasterize a scene cloud into a table-frame height map.

    `bounds` is ``(xmin, ymin, xmax, ymax)`` in table-frame metres; the cloud's
    own extent is used when it is omitted. `max_height_m` drops points far above
    the surface — on a real scene that is the wall, the robot arm and the
    ceiling, none of which are things you can put a mug on.
    """
    pts = np.asarray(scene_cloud, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        raise ValueError("cannot build a height map from an empty cloud")

    tab = plane.to_table(pts)
    xy, z = tab[:, :2], tab[:, 2]

    keep = np.isfinite(xy).all(axis=1) & np.isfinite(z) & (z < max_height_m)
    # Points below the plane are sensor noise on the surface itself or returns
    # from beyond the table edge. Neither is an obstacle; clamping rather than
    # dropping keeps those cells marked *observed*, which matters because
    # unobserved cells are treated as occupied.
    xy, z = xy[keep], np.maximum(z[keep], 0.0)
    if len(xy) == 0:
        raise ValueError("no scene points fall within the height-map limits")

    if bounds is None:
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
    else:
        lo = np.array([bounds[0], bounds[1]], dtype=np.float64)
        hi = np.array([bounds[2], bounds[3]], dtype=np.float64)
        inside = ((xy >= lo) & (xy <= hi)).all(axis=1)
        xy, z = xy[inside], z[inside]
        if len(xy) == 0:
            raise ValueError("no scene points fall inside the requested bounds")

    # Round before the ceiling: an extent that is an exact multiple of the cell
    # size lands on 40.000000000000014 in floating point and would otherwise
    # buy a whole extra row.
    cols = max(int(np.ceil(round((hi[0] - lo[0]) / cell_m, 6))), 1)
    rows = max(int(np.ceil(round((hi[1] - lo[1]) / cell_m, 6))), 1)

    ci = np.clip(((xy[:, 0] - lo[0]) / cell_m).astype(np.int64), 0, cols - 1)
    ri = np.clip(((xy[:, 1] - lo[1]) / cell_m).astype(np.int64), 0, rows - 1)

    # -inf as the identity for a running maximum, converted to NaN afterwards
    # so "unobserved" is representable and cannot be confused with "flat".
    heights = np.full((rows, cols), -np.inf, dtype=np.float64)
    np.maximum.at(heights, (ri, ci), z)
    heights[~np.isfinite(heights)] = np.nan

    return HeightMap(heights=heights.astype(np.float32), origin=lo, cell_m=cell_m)


def free_space(
    hmap: HeightMap,
    clearance_m: float = DEFAULT_CLEARANCE_M,
    unknown_is_free: bool = False,
) -> np.ndarray:
    """Boolean grid of cells that are flat, observed, and therefore placeable.

    `unknown_is_free` exists for tests and for synthetic scenes with complete
    coverage. On real sensor data it must stay False.
    """
    h = hmap.heights
    free = np.isfinite(h) & (h <= clearance_m)
    if unknown_is_free:
        free |= ~np.isfinite(h)
    return free


# ---------------------------------------------------------------------------
# Object footprint
# ---------------------------------------------------------------------------


@dataclass
class Footprint:
    """An object's shadow on the support surface, plus how tall it stands."""

    centroid_xy: np.ndarray  # (2,) table frame
    half_extent: np.ndarray  # (2,) OBB half-extents along its own axes
    yaw: float  # OBB rotation from table +X, radians
    top_m: float  # highest observed point above the plane
    bottom_m: float  # lowest observed point above the plane
    n_points: int = 0

    def __post_init__(self):
        self.centroid_xy = np.asarray(self.centroid_xy, dtype=np.float64).reshape(2)
        self.half_extent = np.asarray(self.half_extent, dtype=np.float64).reshape(2)
        self.yaw = float(self.yaw)
        self.top_m = float(self.top_m)
        self.bottom_m = float(self.bottom_m)

    @property
    def radius_m(self) -> float:
        """Circumscribed radius — the clearance a placement must guarantee.

        Using the circumscribed rather than the inscribed radius means the
        placement is safe at *any* yaw, which matters because the object keeps
        its pick orientation and we do not want that orientation to change
        whether the answer is valid.
        """
        return float(np.hypot(*self.half_extent))

    @property
    def height_m(self) -> float:
        return self.top_m - self.bottom_m

    def corners(self) -> np.ndarray:
        """(4, 2) OBB corners in table-frame XY, counter-clockwise."""
        hx, hy = self.half_extent
        local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        R = np.array([[c, -s], [s, c]])
        return local @ R.T + self.centroid_xy

    def describe(self) -> dict:
        return {
            "centroid_xy_m": [round(float(v), 3) for v in self.centroid_xy],
            "extent_cm": [round(float(v) * 200, 1) for v in self.half_extent],
            "radius_cm": round(self.radius_m * 100, 1),
            "height_cm": round(self.height_m * 100, 1),
            "yaw_deg": round(math.degrees(self.yaw), 1),
        }


def _min_area_rect_angle(pts_xy: np.ndarray) -> float:
    """Rotating-calipers minimum-area rectangle; returns its axis angle.

    Ported from GraspGenX's `graspgenx/samplers/graspmoe.py:_min_area_rect_xy`,
    which is pure numpy/scipy. The minimum-area rectangle of a convex polygon
    always has a side flush with one of its edges, so trying every hull edge is
    exact, not a heuristic.
    """
    from scipy.spatial import ConvexHull, QhullError

    if len(pts_xy) < 3:
        raise ValueError(f"need >=3 points for a min-area rect; got {len(pts_xy)}")
    try:
        hull_pts = pts_xy[ConvexHull(pts_xy).vertices]
    except (QhullError, ValueError):
        # Collinear or otherwise degenerate: fall back to the principal axis.
        cov = np.cov(pts_xy, rowvar=False)
        _, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, -1]
        return float(np.arctan2(principal[1], principal[0]))

    best_area, best_angle = np.inf, 0.0
    for i in range(len(hull_pts)):
        edge = hull_pts[(i + 1) % len(hull_pts)] - hull_pts[i]
        if np.linalg.norm(edge) < 1e-9:
            continue
        angle = float(np.arctan2(edge[1], edge[0]))
        c, s = math.cos(-angle), math.sin(-angle)
        rotated = hull_pts @ np.array([[c, -s], [s, c]]).T
        span = rotated.max(axis=0) - rotated.min(axis=0)
        area = float(span[0] * span[1])
        if area < best_area:
            best_area, best_angle = area, angle
    return best_angle


def object_footprint(
    object_cloud: np.ndarray,
    plane: SupportPlane,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
) -> Footprint:
    """Fit an oriented footprint to an object's points.

    Extents come from the 2nd/98th percentiles rather than the min/max, which is
    what GraspGen-X's own OBB helper does: a handful of stray returns at a mask
    edge would otherwise inflate the footprint by centimetres and rule out
    perfectly good placements.

    **This footprint is biased small, and deliberately not corrected here.** A
    single depth view sees only the front of an object, so measured on the
    synthetic scenes the fit comes out 1.0-1.7 cm under the true extent even
    with nothing occluding it. Too small is the dangerous direction, since it
    invites placing an object into a gap it does not fit.

    Mirroring the points about their centroid along the viewing ray does shrink
    that to under a centimetre, and an earlier version did exactly that — but it
    makes the reported centroid *view-dependent*, so the same object measured
    before and after a move appears to shift by a millimetre or two for no
    physical reason. `evaluator` compares centroids across observations to
    decide whether a move succeeded, and a stable centroid is worth more there
    than a tighter extent. The bias is absorbed instead by
    `find_placement`'s `margin_m`, in one place, with the number written down.

    **Known limitation.** Nothing recovers an object hidden behind *another*
    object. The banana in the `occluded_target` scenario is 19% visible and
    comes out 11.8 cm short on its long axis. `n_points` and the occlusion
    fraction `scene_registry` computes are the signals for that; a footprint
    from a heavily occluded mask must not be trusted for placement. Placing the
    *target* is never required, and a blocker has to be visible enough to grasp
    in the first place, so this is a caveat rather than a hole — but a real one.
    """
    pts = np.asarray(object_cloud, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        raise ValueError(f"need >=3 object points for a footprint; got {len(pts)}")

    tab = plane.to_table(pts)
    xy, z = tab[:, :2], tab[:, 2]

    yaw = _min_area_rect_angle(xy)
    c, s = math.cos(-yaw), math.sin(-yaw)
    local = xy @ np.array([[c, -s], [s, c]]).T

    lo = np.percentile(local, lo_pct, axis=0)
    hi = np.percentile(local, hi_pct, axis=0)
    half_extent = (hi - lo) / 2.0
    centre_local = (hi + lo) / 2.0

    c, s = math.cos(yaw), math.sin(yaw)
    centroid_xy = np.array([[c, -s], [s, c]]) @ centre_local

    return Footprint(
        centroid_xy=centroid_xy,
        half_extent=half_extent,
        yaw=yaw,
        top_m=float(np.percentile(z, hi_pct)),
        bottom_m=float(np.percentile(z, lo_pct)),
        n_points=len(pts),
    )


# ---------------------------------------------------------------------------
# Keep-out regions
# ---------------------------------------------------------------------------


def _fill_polygon(hmap: HeightMap, polygon_xy: np.ndarray, dilate_m: float) -> np.ndarray:
    """Rasterize a table-frame polygon onto the height-map grid."""
    import cv2

    poly = np.asarray(polygon_xy, dtype=np.float64).reshape(-1, 2)
    grid = np.zeros(hmap.shape, dtype=np.uint8)
    if len(poly) < 3:
        return grid.astype(bool)

    cells = np.stack(
        [
            (poly[:, 0] - hmap.origin[0]) / hmap.cell_m,
            (poly[:, 1] - hmap.origin[1]) / hmap.cell_m,
        ],
        axis=-1,
    )
    cv2.fillPoly(grid, [np.round(cells).astype(np.int32)], 1)

    if dilate_m > 0:
        r = int(math.ceil(dilate_m / hmap.cell_m))
        if r > 0:
            k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
            grid = cv2.dilate(grid, k)
    return grid.astype(bool)


def footprint_keep_out(
    hmap: HeightMap, footprints: Sequence[Footprint], dilate_m: float = 0.03
) -> np.ndarray:
    """Cells covered by any of `footprints`, dilated by a safety margin."""
    out = np.zeros(hmap.shape, dtype=bool)
    for fp in footprints:
        out |= _fill_polygon(hmap, fp.corners(), dilate_m)
    return out


def projected_occlusion_keep_out(
    hmap: HeightMap,
    plane: SupportPlane,
    K: np.ndarray,
    target_mask: np.ndarray,
    target_depth_m: float,
    object_height_m: float,
    object_radius_m: float = 0.0,
    dilate_px: int = 12,
    n_samples: int = 4,
) -> np.ndarray:
    """Cells where putting the object down would put it back in front of the target.

    Moving a bottle out of the way and setting it down between the target and
    the camera achieves nothing, so those cells have to be excluded. The obvious
    way to find them is a wedge on the table between the target's footprint and
    the camera — and that was the first implementation, and it was wrong in
    exactly the case that matters.

    The target is *occluded*; that is why it needs clearing. So its fitted
    footprint is a fragment (measured: a 16 cm banana fits to 4 cm), the wedge
    built from it is a narrow sliver, and a bottle released 5 cm to the side
    falls outside it and lands back in front of the target. Observed: the object
    was placed 30 cm away and still hid 19% of the banana.

    Occlusion is a property of the image, so it is decided in the image. Each
    candidate cell is treated as a column of the object's height and radius, its
    samples are projected through `K`, and the cell is rejected if any sample
    lands inside the dilated target mask while being **nearer to the camera**
    than the target. That depends only on the target's *mask*, which occlusion
    does not corrupt, and not at all on its footprint.

    `object_radius_m` matters more than it looks. Sampling only the centre line
    lets a wide object be released just to the side of the forbidden column and
    still hide the target with its body — measured, the bottle was placed 20 cm
    away and the banana went to 40% visible anyway. The ring is what makes the
    test about the object rather than about its axis.
    """
    import cv2

    from perception3d import project_points

    mask = np.asarray(target_mask).astype(np.uint8)
    if dilate_px > 0:
        k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        mask = cv2.dilate(mask, k)
    mask = mask.astype(bool)
    h, w = mask.shape

    rows, cols = hmap.shape
    rr, cc = np.mgrid[0:rows, 0:cols]
    centres = hmap.to_xy(np.stack([rr.ravel(), cc.ravel()], axis=-1))  # (N, 2)

    r = float(object_radius_m)
    offsets = [(0.0, 0.0)]
    if r > 0:
        offsets += [(r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)]

    blocked = np.zeros(rows * cols, dtype=bool)
    heights = np.linspace(0.0, max(object_height_m, 1e-3), max(n_samples, 1))
    for dx, dy in offsets:
        shifted = centres + np.array([dx, dy])
        for z in heights:
            table_pts = np.column_stack([shifted, np.full(len(shifted), z)])
            cam = plane.to_camera(table_pts)
            uv = project_points(cam, K)
            ok = np.isfinite(uv).all(axis=1)
            u = np.clip(np.round(uv[:, 0]), 0, w - 1).astype(np.int64)
            v = np.clip(np.round(uv[:, 1]), 0, h - 1).astype(np.int64)
            # In front of the target, not behind it: a cell further away
            # projects onto the same pixels without hiding anything.
            nearer = cam[:, 2] < target_depth_m
            blocked |= ok & nearer & mask[v, u]

    return blocked.reshape(rows, cols)


# ---------------------------------------------------------------------------
# The placement search
# ---------------------------------------------------------------------------


@dataclass
class Placement:
    """A chosen spot on the table, in table-frame XY."""

    xy: np.ndarray
    clearance_m: float  # distance from this cell to the nearest obstacle
    travel_m: float  # how far the object moves in the plane
    cell: Tuple[int, int]
    n_candidates: int = 0

    def __post_init__(self):
        self.xy = np.asarray(self.xy, dtype=np.float64).reshape(2)

    def describe(self) -> dict:
        return {
            "xy_m": [round(float(v), 3) for v in self.xy],
            "clearance_cm": round(self.clearance_m * 100, 1),
            "travel_cm": round(self.travel_m * 100, 1),
            "n_candidates": int(self.n_candidates),
        }


def find_placement(
    hmap: HeightMap,
    footprint: Footprint,
    keep_out: Optional[np.ndarray] = None,
    prefer_near_xy: Optional[np.ndarray] = None,
    clearance_m: float = DEFAULT_CLEARANCE_M,
    margin_m: float = DEFAULT_MARGIN_M,
    workspace: Optional[Tuple[np.ndarray, float]] = None,
    unknown_is_free: bool = False,
) -> Optional[Placement]:
    """Find where the object can go, or return None if nowhere can.

    A cell is a candidate when it is free, at least ``footprint.radius +
    margin_m`` from the nearest obstacle, outside every keep-out region, and
    inside the workspace. Among the candidates we take the one **closest to
    `prefer_near_xy`** (the pick location by default): the shortest move is the
    one with the least opportunity to knock something over, and it keeps the
    object near where a human would look for it.

    Returning None is a real answer, not a failure — it means this object cannot
    be moved anywhere useful, and the caller should choose a different one.

    Note the object being moved is still present in `hmap`, so its current
    position reads as occupied. That is deliberate and gives two properties for
    free: the object is never "moved" back onto its own spot, and the clearance
    around its old position stays conservative.
    """
    from scipy.ndimage import distance_transform_edt

    free = free_space(hmap, clearance_m, unknown_is_free=unknown_is_free)
    if not free.any():
        logger.info("placement: no free cells at all on the support surface")
        return None

    # Distance from every free cell to the nearest obstacle, in metres. EDT
    # measures distance to the nearest zero, so this is exactly what we want
    # with `free` as the input.
    dist_m = distance_transform_edt(free) * hmap.cell_m

    required = footprint.radius_m + margin_m
    valid = free & (dist_m >= required)

    if keep_out is not None:
        valid &= ~np.asarray(keep_out, dtype=bool)

    rows, cols = hmap.shape
    rr, cc = np.mgrid[0:rows, 0:cols]
    centres = hmap.to_xy(np.stack([rr.ravel(), cc.ravel()], axis=-1)).reshape(rows, cols, 2)

    if workspace is not None:
        centre_xy, radius = workspace
        d = np.linalg.norm(centres - np.asarray(centre_xy, dtype=np.float64), axis=-1)
        valid &= d <= float(radius)

    n_candidates = int(valid.sum())
    if n_candidates == 0:
        logger.info(
            "placement: no cell clears %.1f cm from obstacles inside the allowed region",
            required * 100,
        )
        return None

    anchor = (
        np.asarray(prefer_near_xy, dtype=np.float64).reshape(2)
        if prefer_near_xy is not None
        else footprint.centroid_xy
    )
    travel = np.linalg.norm(centres - anchor, axis=-1)
    travel_masked = np.where(valid, travel, np.inf)

    flat = int(np.argmin(travel_masked))
    r, c = divmod(flat, cols)

    return Placement(
        xy=centres[r, c],
        clearance_m=float(dist_m[r, c]),
        travel_m=float(travel[r, c]),
        cell=(r, c),
        n_candidates=n_candidates,
    )


# ---------------------------------------------------------------------------
# From a placement to a gripper pose
# ---------------------------------------------------------------------------


@dataclass
class PlacePose:
    """Where the hand must be to release the object at the chosen spot."""

    pose: np.ndarray  # (4, 4) camera frame; same rotation as the grasp
    width: float
    gripper: str
    clearance_m: float
    travel_m: float
    place_xy: np.ndarray  # table frame, for logging and evaluation
    waypoints: list = field(default_factory=list)

    def __post_init__(self):
        self.pose = np.asarray(self.pose, dtype=np.float64).reshape(4, 4)
        self.width = float(self.width)
        self.place_xy = np.asarray(self.place_xy, dtype=np.float64).reshape(2)

    @property
    def position(self) -> np.ndarray:
        return self.pose[:3, 3]

    @property
    def approach(self) -> np.ndarray:
        return self.pose[:3, 2]

    def as_dict(self) -> dict:
        """JSON/msgpack-safe. Mirrors `Grasp6D.as_dict` so the two travel together."""
        return {
            "pose": self.pose.tolist(),
            "width": self.width,
            "gripper": self.gripper,
            "position": self.position.tolist(),
            "approach": self.approach.tolist(),
            "clearance_m": self.clearance_m,
            "travel_m": self.travel_m,
            "place_xy": self.place_xy.tolist(),
            "waypoints": [
                {"name": n, "pose": np.asarray(p, dtype=np.float64).tolist()}
                for n, p in self.waypoints
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlacePose":
        return cls(
            pose=np.asarray(d["pose"], dtype=np.float64),
            width=d.get("width", 0.08),
            gripper=d.get("gripper", "unknown"),
            clearance_m=float(d.get("clearance_m", 0.0)),
            travel_m=float(d.get("travel_m", 0.0)),
            place_xy=np.asarray(d.get("place_xy", [0.0, 0.0]), dtype=np.float64),
            waypoints=[
                (w["name"], np.asarray(w["pose"], dtype=np.float64))
                for w in d.get("waypoints", [])
            ],
        )

    def summary(self) -> str:
        p = self.position
        return (
            f"place {self.gripper} at ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})m "
            f"travel={self.travel_m*100:.0f}cm clearance={self.clearance_m*100:.1f}cm"
        )


def place_delta(
    plane: SupportPlane,
    footprint: Footprint,
    target_xy: np.ndarray,
    release_gap_m: float = 0.005,
) -> np.ndarray:
    """The camera-frame translation that moves the object to `target_xy`.

    Because the object keeps its orientation, this is exact: the XY component is
    the difference in footprint centroids and the Z component is whatever puts
    the object's lowest observed point back on the surface, plus a small gap so
    the release is not a scrape.
    """
    target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
    d_xy = target_xy - footprint.centroid_xy
    d_z = -footprint.bottom_m + release_gap_m
    delta_table = np.array([d_xy[0], d_xy[1], d_z])
    return plane.rotation @ delta_table


def pick_place_waypoints(
    grasp_pose: np.ndarray,
    place_pose: np.ndarray,
    plane: SupportPlane,
    lift_m: float = 0.12,
    retreat_m: float = 0.10,
) -> list:
    """The six-pose motion sketch an executor replays.

    Lift and descend along the **table normal**, not the gripper's approach
    axis: a side grasp retreating along its own -Z would drag the object
    sideways across the table.
    """
    grasp_pose = np.asarray(grasp_pose, dtype=np.float64).reshape(4, 4)
    place_pose = np.asarray(place_pose, dtype=np.float64).reshape(4, 4)
    up = plane.normal

    def shifted(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
        out = pose.copy()
        out[:3, 3] = out[:3, 3] + delta
        return out

    return [
        ("pre_grasp", shifted(grasp_pose, -retreat_m * grasp_pose[:3, 2])),
        ("grasp", grasp_pose.copy()),
        ("lift", shifted(grasp_pose, lift_m * up)),
        ("pre_place", shifted(place_pose, lift_m * up)),
        ("place", place_pose.copy()),
        ("retreat", shifted(place_pose, lift_m * up)),
    ]


def plan_place(
    grasp_pose: np.ndarray,
    object_cloud: np.ndarray,
    scene_cloud: np.ndarray,
    plane: SupportPlane,
    gripper: str = "unknown",
    width: float = 0.08,
    keep_out: Optional[np.ndarray] = None,
    hmap: Optional[HeightMap] = None,
    clearance_m: float = DEFAULT_CLEARANCE_M,
    margin_m: Optional[float] = None,
    release_gap_m: float = 0.005,
    workspace: Optional[Tuple[np.ndarray, float]] = None,
    lift_m: float = 0.12,
    retreat_m: float = 0.10,
    cell_m: float = DEFAULT_CELL_M,
) -> Optional[PlacePose]:
    """End-to-end: grasp + clouds in, a release pose out. None if nowhere fits.

    This is the one function callers normally need; everything above is exposed
    so the individual steps stay testable and so `scene_registry` can reuse the
    plane and the height map it has already built.

    `margin_m` defaults to a **gripper-derived** clearance rather than a
    constant. Setting an object down 3 cm from its neighbour is legal by the old
    rule and useless in practice: the hand needs room to get in afterwards, and
    how much room depends entirely on which hand it is — 7.4 cm for a Robotiq,
    16.6 cm for a Barrett. A fixed number either wastes table space for small
    hands or leaves big ones unable to reach what it just put down.

    The consequence is real and worth stating: on a crowded table a large hand
    may now find nowhere legal to put anything. That returns None, which the
    caller reports honestly, and is the right failure — better than a placement
    that has to be undone.
    """
    footprint = object_footprint(object_cloud, plane)
    if hmap is None:
        hmap = build_height_map(scene_cloud, plane, cell_m=cell_m)

    # Ask for enough room that the hand can come back for it, then settle for
    # enough room that it is safely down. Preference, not requirement: a
    # Franka wants 13.4 cm of clearance and a Barrett 19.6 cm, which on a
    # cluttered table is often nowhere at all — and refusing to put an object
    # down because the *next* grasp might be awkward is the wrong trade. If the
    # tighter placement does foul a later grasp, the collision filter names it
    # and the loop moves it again, which costs one iteration instead of
    # deadlocking the run.
    attempts = [gripper_clearance_m(gripper)] if margin_m is None else [margin_m]
    if margin_m is None and attempts[0] > DEFAULT_MARGIN_M:
        attempts.append(DEFAULT_MARGIN_M)

    placement = None
    for i, want in enumerate(attempts):
        placement = find_placement(
            hmap,
            footprint,
            keep_out=keep_out,
            prefer_near_xy=footprint.centroid_xy,
            clearance_m=clearance_m,
            margin_m=want,
            workspace=workspace,
        )
        if placement is not None:
            if i:
                logger.info(
                    "plan_place: no spot with %.1f cm of gripper clearance; "
                    "placed with %.1f cm instead", attempts[0] * 100, want * 100,
                )
            break
    if placement is None:
        return None

    delta = place_delta(plane, footprint, placement.xy, release_gap_m=release_gap_m)
    pose = np.asarray(grasp_pose, dtype=np.float64).reshape(4, 4).copy()
    pose[:3, 3] = pose[:3, 3] + delta

    return PlacePose(
        pose=pose,
        width=width,
        gripper=gripper,
        clearance_m=placement.clearance_m,
        travel_m=placement.travel_m,
        place_xy=placement.xy,
        waypoints=pick_place_waypoints(
            grasp_pose, pose, plane, lift_m=lift_m, retreat_m=retreat_m
        ),
    )
