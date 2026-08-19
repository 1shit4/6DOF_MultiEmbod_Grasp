"""Render a 6-DoF grasp back into the image plane.

Two audiences, and they pull in different directions:

1. **The Observer agent** (a vision LLM) looks at this PNG and judges whether
   the grasp is sensible. It needs an unambiguous picture of *where* the hand
   is and *which way it comes in* — so the drawing is schematic and
   high-contrast rather than photorealistic.
2. **A human** debugging a run.

The old 2D pipeline drew a rotated rectangle (`grasp/utils.py`); a rectangle
cannot express approach direction, which is precisely what 6-DoF adds. So we
draw a projected gripper: the two fingers, the line they close along, and an
arrow showing the approach axis.
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


def visualize_grasp_6dof(
    image: np.ndarray,
    grasp,
    K: np.ndarray,
    save_folder: str | os.PathLike = "imgs/",
    filename: str = "grasp_pose_visualization.png",
    mask: Optional[np.ndarray] = None,
    fingertip_depth: float = 0.11,
    finger_len: float = 0.05,
    draw_annotation: bool = True,
) -> Path:
    """Draw `grasp` on `image` (RGB uint8) and write a PNG. Returns its path.

    Always writes a file, even when the grasp cannot be projected — the
    Observer reads `execute_results["image"]` unconditionally and would crash
    on a missing path.
    """
    if isinstance(grasp, dict):
        grasp = Grasp6D.from_dict(grasp)

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

        # Finger shanks and the knuckle bar (back of the hand).
        cv2.line(canvas, xy("left_knuckle"), xy("left_tip"), COLOR_FINGER, 5, cv2.LINE_AA)
        cv2.line(canvas, xy("right_knuckle"), xy("right_tip"), COLOR_FINGER, 5, cv2.LINE_AA)
        cv2.line(canvas, xy("left_knuckle"), xy("right_knuckle"), COLOR_FINGER, 5, cv2.LINE_AA)
        # The closing line between the fingertips — the "grasp" itself. Drawn
        # thinner than the orange hand so that when the view is nearly head-on
        # and the two become collinear, the orange still reads underneath.
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
