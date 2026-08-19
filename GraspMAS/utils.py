"""Rectangle-space utilities: 2D overlay and the OCID-VLG grasp metric.

The pipeline is 6-DoF now, but the OCID-VLG ground truth is planar rectangles.
`Grasp6D.rect_2d` back-projects a 6-DoF grasp into exactly that form, so these
functions still apply and let 6-DoF results be scored against the same
IoU@0.25 / angle@30 metric the 2D literature reports.
"""

import os
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon


def visualize_grasp_pose(image, grasp_pose, save_folder="imgs/"):
    """Draw a `[quality, x, y, w, h, angle]` rectangle on the image.

    Moved here from `grasp/utils.py` — `main_batch.py` imported it from `utils`,
    which meant the module could not import at all as shipped.

    Parameters
    ----------
    image : np.ndarray
        RGB image.
    grasp_pose : sequence
        (quality, x, y, w, h, angle); x, y is the rectangle centre, angle in degrees.

    Returns
    -------
    Path to the written PNG.
    """
    point_color1 = (255, 255, 0)  # BGR
    point_color2 = (255, 0, 255)  # BGR
    thickness = 2
    lineType = 4

    image = np.ascontiguousarray(image.copy())
    x, y, w, h, angle = grasp_pose[1:]
    box = cv2.boxPoints(((x, y), (w, h), angle))
    box = np.int64(box)

    cv2.line(image, box[0], box[3], point_color1, thickness, lineType)
    cv2.line(image, box[3], box[2], point_color2, thickness, lineType)
    cv2.line(image, box[2], box[1], point_color1, thickness, lineType)
    cv2.line(image, box[1], box[0], point_color2, thickness, lineType)

    os.makedirs(save_folder, exist_ok=True)
    output_path = os.path.join(save_folder, "grasp_pose_visualization.png")
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return Path(output_path)


def rotated_rect_to_polygon(x, y, w, h, angle):
    """Convert a rotated rectangle to a polygon."""
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    corners = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2],
    ])
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = np.dot(corners, rotation_matrix.T)
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y
    return Polygon(rotated_corners)


def calculate_iou_and_angle(rect1, rect2, dataset_type="OCID"):
    """IoU and absolute angle difference between a prediction and a ground truth.

    `rect1` (the prediction) is `[x, y, w, h, angle]` — no score.

    `dataset_type` selects the ground-truth layout, and the default is "OCID"
    rather than upstream's "GraspAnything". OCID-VLG rects are
    `[cx, cy, w, h, theta, target_id]` (target LAST, see OCID_VLG/dataset.py),
    whereas the GraspAnything layout puts a quality value FIRST. Using the wrong
    one silently shifts every field by a position and produces meaningless IoU —
    which is what the shipped default did for the only dataset in the repo.
    """
    x1, y1, w1, h1, angle1 = rect1
    if dataset_type == "GraspAnything":
        _, x2, y2, w2, h2, angle2 = rect2
    else:  # OCID-VLG: [cx, cy, w, h, theta, target]
        x2, y2, w2, h2, angle2, _ = rect2

    poly1 = rotated_rect_to_polygon(x1, y1, w1, h1, angle1)
    poly2 = rotated_rect_to_polygon(x2, y2, w2, h2, angle2)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = intersection / union if union != 0 else 0

    angle_diff = abs(angle1 - angle2)
    angle_diff = min(angle_diff, 180 - angle_diff)
    return iou, angle_diff


def eval_grasp(
    predict,
    ground_truth,
    iou_threshold=0.25,
    angle_threshold=30,
    dataset_type="OCID",
    return_details=False,
):
    """Standard rectangle grasp metric: a hit needs IoU >= 0.25 AND angle <= 30 deg.

    `predict` is `[x, y, w, h, angle]` or `[score, x, y, w, h, angle]` — the
    leading score is dropped if present, so `Grasp6D.rect_2d` can be passed
    straight in.
    """
    predict = list(predict)
    if len(predict) == 6:
        predict = predict[1:]

    max_iou = 0.0
    best_angle = float("inf")
    for gt in ground_truth:
        gt = gt.tolist() if hasattr(gt, "tolist") else list(gt)
        iou_score, angle_diff = calculate_iou_and_angle(predict, gt, dataset_type)
        if angle_diff > angle_threshold:
            continue
        if iou_score > max_iou:
            max_iou = iou_score
            best_angle = angle_diff

    success = max_iou >= iou_threshold
    if return_details:
        return success, float(max_iou), float(best_angle)
    return success
