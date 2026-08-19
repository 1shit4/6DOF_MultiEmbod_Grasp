"""vis6d: the rendering the Observer agent actually looks at.

The load-bearing property is that a PNG *always* exists afterwards —
`agents/observer.py` calls `encode_image` on the path unconditionally, so a
missing file crashes the round rather than degrading it.
"""

import numpy as np
import pytest

from perception3d import Grasp6D
from vis6d import (
    depth_colormap,
    visualize_grasp_6dof,
    visualize_grasp_candidates,
)


class TestVisualizeGrasp:
    def test_writes_png_of_the_right_size(self, rgb, K, identity_grasp, tmp_path):
        import cv2

        path = visualize_grasp_6dof(rgb, identity_grasp, K, save_folder=tmp_path)
        assert path.exists()
        img = cv2.imread(str(path))
        assert img.shape == rgb.shape

    def test_accepts_a_dict(self, rgb, K, identity_grasp, tmp_path):
        """graspmas passes the dict form straight through."""
        path = visualize_grasp_6dof(
            rgb, identity_grasp.as_dict(), K, save_folder=tmp_path
        )
        assert path.exists()

    def test_writes_a_png_even_with_no_grasp(self, rgb, K, tmp_path):
        # This is the case that would otherwise crash the Observer.
        path = visualize_grasp_6dof(rgb, None, K, save_folder=tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_writes_a_png_for_a_grasp_behind_the_camera(self, rgb, K, tmp_path):
        pose = np.eye(4)
        pose[:3, 3] = [0, 0, -1.0]  # unprojectable
        grasp = Grasp6D(pose=pose, score=0.5, gripper="x", width=0.08)
        path = visualize_grasp_6dof(rgb, grasp, K, save_folder=tmp_path)
        assert path.exists()

    @staticmethod
    def _oblique_grasp():
        """A 40-degree side approach — the realistic 6-DoF case, and one where
        every part of the gripper projects to a distinct place."""
        a = np.deg2rad(40.0)
        pose = np.eye(4)
        pose[:3, :3] = np.array([[np.cos(a), 0, np.sin(a)],
                                 [0, 1, 0],
                                 [-np.sin(a), 0, np.cos(a)]])
        pose[:3, 3] = [-0.05, 0.0, 0.55]
        return Grasp6D(pose=pose, score=0.8, gripper="franka_panda", width=0.08)

    def test_draws_all_gripper_parts(self, rgb, K, tmp_path):
        """A blank or partial overlay gives the Observer nothing to judge."""
        import cv2

        path = visualize_grasp_6dof(rgb, self._oblique_grasp(), K, save_folder=tmp_path)
        out = cv2.imread(str(path))
        assert not np.array_equal(out, rgb[:, :, ::-1])
        assert (np.all(out == np.array([0, 255, 255]), axis=-1)).any(), "no yellow closing line"
        assert (np.all(out == np.array([0, 165, 255]), axis=-1)).any(), "no orange fingers"
        assert (np.all(out == np.array([255, 100, 0]), axis=-1)).any(), "no blue approach"

    def test_head_on_grasp_still_shows_an_approach_marker(self, rgb, K,
                                                          identity_grasp, tmp_path):
        """A grasp straight down the optical axis projects base and fingertips
        onto one pixel, so an arrow would vanish exactly when the approach is
        'toward the viewer'. The into-the-page glyph must appear instead."""
        import cv2

        path = visualize_grasp_6dof(rgb, identity_grasp, K, save_folder=tmp_path)
        out = cv2.imread(str(path))
        assert (np.all(out == np.array([255, 100, 0]), axis=-1)).any(), \
            "head-on grasp drew no approach indicator"
        assert (np.all(out == np.array([0, 255, 255]), axis=-1)).any(), "no yellow"

    def test_mask_overlay_is_drawn(self, rgb, K, identity_grasp, box_mask, tmp_path):
        import cv2

        plain = visualize_grasp_6dof(rgb, identity_grasp, K, save_folder=tmp_path,
                                     filename="plain.png")
        masked = visualize_grasp_6dof(rgb, identity_grasp, K, save_folder=tmp_path,
                                      filename="masked.png", mask=box_mask)
        assert not np.array_equal(cv2.imread(str(plain)), cv2.imread(str(masked)))

    def test_creates_missing_directory(self, rgb, K, identity_grasp, tmp_path):
        target = tmp_path / "deep" / "nested"
        path = visualize_grasp_6dof(rgb, identity_grasp, K, save_folder=target)
        assert path.exists()

    def test_does_not_mutate_the_input_image(self, rgb, K, identity_grasp, tmp_path):
        original = rgb.copy()
        visualize_grasp_6dof(rgb, identity_grasp, K, save_folder=tmp_path)
        assert np.array_equal(rgb, original)

    def test_custom_filename_is_used(self, rgb, K, identity_grasp, tmp_path):
        path = visualize_grasp_6dof(rgb, identity_grasp, K, save_folder=tmp_path,
                                    filename="round3_overlay.png")
        assert path.name == "round3_overlay.png"


class TestCandidateOverlay:
    def test_draws_all_candidates(self, rgb, K, tmp_path):
        grasps = np.tile(np.eye(4), (10, 1, 1))
        for i in range(10):
            grasps[i, :3, 3] = [0.01 * i - 0.05, 0.0, 0.6]
        scores = np.linspace(0.1, 0.9, 10)
        path = visualize_grasp_candidates(rgb, grasps, scores, K, tmp_path)
        assert path.exists()

    def test_handles_empty_candidate_set(self, rgb, K, tmp_path):
        path = visualize_grasp_candidates(
            rgb, np.zeros((0, 4, 4)), np.zeros(0), K, tmp_path
        )
        assert path.exists()


class TestDepthColormap:
    def test_writes_colourised_depth(self, depth_map, tmp_path):
        import cv2

        path = depth_colormap(depth_map, tmp_path / "d.png")
        assert path.exists()
        img = cv2.imread(str(path))
        assert img.shape[:2] == depth_map.shape

    def test_invalid_depth_renders_black(self, depth_map, tmp_path):
        import cv2

        img = cv2.imread(str(depth_colormap(depth_map, tmp_path / "d.png")))
        # The fixture zeroes out the left 40 columns.
        assert (img[:, :40] == 0).all()

    def test_all_invalid_does_not_crash(self, tmp_path):
        path = depth_colormap(np.zeros((10, 10), np.float32), tmp_path / "d.png")
        assert path.exists()
