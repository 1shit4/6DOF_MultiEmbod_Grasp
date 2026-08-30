"""Render a 6-DoF grasp back into the image plane.

Two audiences, and they pull in different directions:

1. **The Observer agent** (a vision LLM) looks at this PNG and judges whether
   the grasp is sensible. It needs an unambiguous picture of *where* the hand
   is and *which way it comes in* — so the drawing is schematic and
   high-contrast rather than photorealistic.
2. **A human** debugging a run.

The old 2D pipeline drew a rotated rectangle (`grasp/utils.py`); a rectangle
cannot express approach direction, which is precisely what 6-DoF adds. So we
draw a projected gripper: the outline of the hand, the line its jaw closes
along, and an arrow showing the approach axis.

The outline is the gripper's **own** surface geometry, not a generic sketch.
Four of the shipped grippers have three fingers, and drawing all of them as a
parallel jaw meant the Observer was asked to judge a hand that did not exist.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from perception3d import Grasp6D, approach_elevation_deg, project_points

logger = logging.getLogger(__name__)

# BGR, chosen to survive JPEG compression and stay distinguishable in the
# thumbnail a vision model actually sees.
COLOR_CLOSING = (0, 255, 255)    # yellow  - the line the fingers close along
COLOR_FINGER = (0, 165, 255)     # orange  - finger shanks
COLOR_APPROACH = (255, 100, 0)   # blue    - approach axis, base -> fingertips
COLOR_MASK = (255, 0, 255)       # magenta - the referred region
COLOR_TEXT = (255, 255, 255)


def _gripper_wireframe(
    pose: np.ndarray, width: float, fingertip_depth: float, finger_len: float
) -> dict:
    """Gripper keypoints in the camera frame.

    Built in the gripper's own frame (+Z approach, +X closing, origin at the
    base) and then transformed, so the geometry is easy to reason about.
    """
    hw = width / 2.0
    knuckle_z = max(fingertip_depth - finger_len, 0.0)
    local = {
        "base": [0.0, 0.0, 0.0],
        "wrist": [0.0, 0.0, knuckle_z],
        "left_knuckle": [-hw, 0.0, knuckle_z],
        "right_knuckle": [hw, 0.0, knuckle_z],
        "left_tip": [-hw, 0.0, fingertip_depth],
        "right_tip": [hw, 0.0, fingertip_depth],
        "centre": [0.0, 0.0, fingertip_depth],
    }
    R, t = pose[:3, :3], pose[:3, 3]
    return {k: R @ np.asarray(v, dtype=np.float64) + t for k, v in local.items()}


def _gripper_silhouette(
    pose: np.ndarray,
    K: np.ndarray,
    gripper_name: str,
    shape: tuple,
    max_points: int = 1024,
) -> Optional[list]:
    """Outline of the *real* hand projected into the image, or None.

    The seven-point wireframe above is a parallel jaw, and four of the shipped
    grippers have three fingers — so the Observer was shown a hand that does
    not exist while being told "the orange lines are the two fingers". Every
    gripper description carries 10,500 surface points in the same frame the
    pose uses; `collision.py` has been loading them all along for the collision
    filter.

    Traced from a rasterised point cloud rather than a convex hull on purpose:
    the hull fills in the gap between the fingers, and that gap is the one part
    of the picture the Observer is asked to read.
    """
    try:
        from collision import load_gripper_points, transform_points

        pts = load_gripper_points(gripper_name, state="open", max_points=max_points)
    except Exception as exc:  # missing assets must not cost us the picture
        logger.debug("no gripper geometry for %r: %s", gripper_name, exc)
        return None

    uv = project_points(transform_points(pts, pose), K)
    uv = uv[np.isfinite(uv).all(axis=1)]
    if len(uv) < 16:
        return None

    h, w = int(shape[0]), int(shape[1])
    # Pad so a hand half out of frame still traces a sane outline instead of
    # being clipped into a straight edge along the image border.
    pad = 64
    grid = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    ij = np.round(uv + pad).astype(np.int64)
    keep = (
        (ij[:, 0] >= 0) & (ij[:, 0] < grid.shape[1])
        & (ij[:, 1] >= 0) & (ij[:, 1] < grid.shape[0])
    )
    ij = ij[keep]
    if len(ij) < 16:
        return None

    # Dot radius from the projected point density, so the outline closes at any
    # viewing distance without bridging the jaw opening — the samples land a few
    # pixels apart while the opening is tens of pixels wide.
    spread = ij.max(axis=0) - ij.min(axis=0)
    area = max(float(spread[0]) * float(spread[1]), 1.0)
    radius = int(np.clip(round(0.6 * math.sqrt(area / len(ij))), 2, 10))
    for x, y in ij:
        cv2.circle(grid, (int(x), int(y)), radius, 255, -1)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = [c - pad for c in contours if cv2.contourArea(c) > 20.0]
    return out or None


def visualize_grasp_6dof(
    image: np.ndarray,
    grasp,
    K: np.ndarray,
    save_folder: str | os.PathLike = "imgs/",
    filename: str = "grasp_pose_visualization.png",
    mask: Optional[np.ndarray] = None,
    fingertip_depth: Optional[float] = None,
    finger_len: float = 0.05,
    draw_annotation: bool = True,
) -> Path:
    """Draw `grasp` on `image` (RGB uint8) and write a PNG. Returns its path.

    Always writes a file, even when the grasp cannot be projected — the
    Observer reads `execute_results["image"]` unconditionally and would crash
    on a missing path.

    `fingertip_depth` defaults to the grasp's own gripper rather than to a
    constant. It used to default to 0.11 m for everything, which is 3.4 cm out
    on a Robotiq and 6 cm on an Inspire hand — and it is not even the Franka's
    number (0.1034). The fingertip line is what the Observer is told to check
    the object sits between, so drawing it in the wrong place quietly corrupted
    every judgement on any hand but one.
    """
    if isinstance(grasp, dict):
        grasp = Grasp6D.from_dict(grasp)

    if fingertip_depth is None:
        try:
            from collision import gripper_geometry

            fingertip_depth = gripper_geometry(getattr(grasp, "gripper", "") or "")[1]
        except Exception:
            fingertip_depth = 0.11

    canvas = np.ascontiguousarray(np.asarray(image)[:, :, ::-1].copy())  # RGB->BGR
    h, w = canvas.shape[:2]

    if mask is not None:
        m = np.asarray(mask).astype(bool)
        if m.shape == (h, w) and m.any():
            tint = canvas.copy()
            tint[m] = COLOR_MASK
            canvas = cv2.addWeighted(canvas, 0.75, tint, 0.25, 0)
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, COLOR_MASK, 1, cv2.LINE_AA)

    ok = grasp is not None
    if ok:
        pts3 = _gripper_wireframe(grasp.pose, grasp.width, fingertip_depth, finger_len)
        names = list(pts3.keys())
        uv = project_points(np.stack([pts3[n] for n in names]), K)
        p = {n: uv[i] for i, n in enumerate(names)}
        ok = bool(np.isfinite(uv).all())

    if ok:
        def xy(name):
            return (int(round(p[name][0])), int(round(p[name][1])))

        base, centre = xy("base"), xy("centre")
        # A grasp aimed straight down the optical axis projects the base and the
        # fingertips onto the same pixel, so an arrow would be invisible exactly
        # when the approach is "toward the viewer". Draw the standard
        # into-the-page glyph (circle with a cross) instead, so the Observer can
        # still tell which way the hand comes in.
        if math.hypot(centre[0] - base[0], centre[1] - base[1]) < 6.0:
            cx, cy = centre
            cv2.circle(canvas, (cx, cy), 11, COLOR_APPROACH, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx - 8, cy - 8), (cx + 8, cy + 8), COLOR_APPROACH, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx - 8, cy + 8), (cx + 8, cy - 8), COLOR_APPROACH, 2, cv2.LINE_AA)
        else:
            cv2.arrowedLine(canvas, base, centre, COLOR_APPROACH, 3,
                            cv2.LINE_AA, tipLength=0.18)

        # The hand itself: the real surface outline when the gripper
        # descriptions are installed, the parallel-jaw wireframe otherwise.
        outline = _gripper_silhouette(
            grasp.pose, K, getattr(grasp, "gripper", "") or "", (h, w)
        )
        if outline:
            cv2.drawContours(canvas, outline, -1, COLOR_FINGER, 3, cv2.LINE_AA)
        else:
            cv2.line(canvas, xy("left_knuckle"), xy("left_tip"), COLOR_FINGER, 5, cv2.LINE_AA)
            cv2.line(canvas, xy("right_knuckle"), xy("right_tip"), COLOR_FINGER, 5, cv2.LINE_AA)
            cv2.line(canvas, xy("left_knuckle"), xy("right_knuckle"), COLOR_FINGER, 5, cv2.LINE_AA)

        # The closing line between the fingertips — the "grasp" itself. Drawn
        # thinner than the orange hand so that when the view is nearly head-on
        # and the two become collinear, the orange still reads underneath.
        # Spanned by the jaw aperture the model actually conditioned on, at the
        # gripper's real fingertip depth. Measuring the extremes of the surface
        # cloud instead looks more principled and is not: it returns the outer
        # edge of the finger bodies (11.7 cm on a Panda whose jaw opens 8 cm),
        # and on a three-finger hand the whole 33 cm width of the palm.
        cv2.line(canvas, xy("left_tip"), xy("right_tip"), COLOR_CLOSING, 2, cv2.LINE_AA)
        for n in ("left_tip", "right_tip"):
            cv2.circle(canvas, xy(n), 5, COLOR_CLOSING, -1, cv2.LINE_AA)
        cv2.circle(canvas, xy("centre"), 4, (255, 255, 255), -1, cv2.LINE_AA)

        if draw_annotation:
            elev = approach_elevation_deg(grasp.approach)
            pos = grasp.position
            lines = [
                f"{grasp.gripper}  score {grasp.score:.2f}",
                f"pos ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}) m",
                f"approach {elev:.0f}deg off optical axis",
                f"jaw {grasp.width*100:.1f} cm",
            ]
            _draw_panel(canvas, lines)
    else:
        _draw_panel(canvas, ["NO VALID 6-DoF GRASP", "(nothing to project)"])

    out_dir = Path(save_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), canvas)
    return out_path


def _draw_panel(canvas: np.ndarray, lines: list[str]) -> None:
    """Legible text box in the top-left, sized to its content."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    pads, lh = 8, 20
    widths = [cv2.getTextSize(t, font, scale, thick)[0][0] for t in lines]
    box_w = max(widths) + 2 * pads
    box_h = lh * len(lines) + pads
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (pads, pads + lh * i + 8), font, scale,
                    COLOR_TEXT, thick, cv2.LINE_AA)


def visualize_grasp_candidates(
    image: np.ndarray,
    grasps: np.ndarray,
    scores: np.ndarray,
    K: np.ndarray,
    save_folder: str | os.PathLike,
    filename: str = "grasp_candidates.png",
    width: float = 0.08,
    fingertip_depth: float = 0.11,
    max_draw: int = 60,
) -> Path:
    """Overlay the whole candidate set, coloured red (low) to green (high).

    Diagnostic only — it never goes to the Observer, which should judge one
    decision, not a cloud of options. Useful in `outputs/` for seeing whether
    the model was confident-and-focused or scattered.
    """
    canvas = np.ascontiguousarray(np.asarray(image)[:, :, ::-1].copy())
    if len(grasps):
        order = np.argsort(scores)[-max_draw:]  # draw best last, on top
        lo, hi = float(np.min(scores)), float(np.max(scores))
        span = max(hi - lo, 1e-6)
        for i in order:
            pts = _gripper_wireframe(grasps[i], width, fingertip_depth, 0.05)
            uv = project_points(np.stack([pts["left_tip"], pts["right_tip"]]), K)
            if not np.isfinite(uv).all():
                continue
            t = (float(scores[i]) - lo) / span
            color = (0, int(255 * t), int(255 * (1 - t)))  # BGR red->green
            a = (int(round(uv[0][0])), int(round(uv[0][1])))
            b = (int(round(uv[1][0])), int(round(uv[1][1])))
            cv2.line(canvas, a, b, color, 2, cv2.LINE_AA)
    _draw_panel(canvas, [f"{len(grasps)} candidates", "red=low  green=high score"])

    out_dir = Path(save_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), canvas)
    return out_path


def depth_colormap(depth: np.ndarray, save_path: str | os.PathLike) -> Path:
    """Write a colourised depth image so bad depth is visible at a glance."""
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    vis = np.zeros(d.shape, dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
        norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        vis = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), colored)
    return out_path
