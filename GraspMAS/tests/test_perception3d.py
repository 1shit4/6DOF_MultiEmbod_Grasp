"""perception3d: camera geometry, the 2D<->3D bridge.

Every downstream number depends on this module being right, and a sign error
here produces grasps that look plausible but are metrically wrong — so these
tests check the maths against closed-form expectations, not just shapes.
"""

import math

import numpy as np
import pytest

from perception3d import (
    downsample_cloud,
    Grasp6D,
    SceneContext,
    approach_elevation_deg,
    default_intrinsics,
    filter_grasps_by_mask,
    filter_grasps_by_visibility,
    gripper_finger_points,
    grasp_to_rect2d,
    load_intrinsics,
    project_points,
    unproject,
)


class TestIntrinsics:
    def test_default_intrinsics_fov_math(self):
        K = default_intrinsics(480, 640, fov_deg=90.0)
        # At 90 deg horizontal FOV, f = (W/2) / tan(45) = W/2.
        assert K[0, 0] == pytest.approx(320.0)
        assert K[0, 2] == pytest.approx(320.0)
        assert K[1, 2] == pytest.approx(240.0)

    def test_narrower_fov_gives_longer_focal_length(self):
        wide = default_intrinsics(480, 640, fov_deg=90.0)
        narrow = default_intrinsics(480, 640, fov_deg=45.0)
        assert narrow[0, 0] > wide[0, 0]

    def test_rejects_bad_size(self):
        with pytest.raises(ValueError):
            default_intrinsics(0, 640)

    def test_load_from_nested_list(self, K):
        assert np.allclose(load_intrinsics(K.tolist()), K)

    def test_load_from_meta_data_json(self, tmp_path, K):
        # The shape GraspGen-X's sample scenes ship.
        p = tmp_path / "meta_data.json"
        p.write_text(f'{{"intrinsics": {K.tolist()}, "camera_pose": []}}')
        assert np.allclose(load_intrinsics(str(p)), K)

    def test_load_from_npy(self, tmp_path, K):
        p = tmp_path / "K.npy"
        np.save(p, K)
        assert np.allclose(load_intrinsics(str(p)), K)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            load_intrinsics(np.eye(4))

    def test_rejects_degenerate_focal_length(self):
        bad = np.array([[0.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
        with pytest.raises(ValueError):
            load_intrinsics(bad)


class TestUnproject:
    def test_round_trip_is_subpixel(self, depth_map, K, box_mask):
        """unproject then project must land back on the originating pixels."""
        cloud = unproject(depth_map, K, box_mask)
        assert len(cloud) > 0
        uv = project_points(cloud, K)

        ys, xs = np.nonzero(box_mask & (depth_map > 0))
        expected = np.stack([xs, ys], axis=-1).astype(float)
        # unproject returns pixels in row-major order, so compare directly.
        # Tolerance is 1e-3 px because depth is stored float32; that is still
        # three orders of magnitude finer than the sub-pixel claim.
        assert np.abs(uv[:, 0] - expected[:, 0]).max() < 1e-3
        assert np.abs(uv[:, 1] - expected[:, 1]).max() < 1e-3

    def test_known_point_geometry(self, K):
        """A pixel one focal length right of centre at 1 m is at x = 1 m."""
        # Wide synthetic image so column cx + fx = 920 exists.
        depth = np.zeros((480, 1000), np.float32)
        depth[240, 920] = 1.0
        cloud = unproject(depth, K)
        assert len(cloud) == 1
        assert cloud[0] == pytest.approx([1.0, 0.0, 1.0], abs=1e-5)

    def test_centre_pixel_maps_to_optical_axis(self, K):
        depth = np.zeros((480, 640), np.float32)
        depth[240, 320] = 2.0
        cloud = unproject(depth, K)
        assert cloud[0] == pytest.approx([0.0, 0.0, 2.0], abs=1e-6)

    def test_drops_zero_depth(self, depth_map, K):
        """Zero is a sensor's 'no return' — it must never become a 3D point."""
        cloud = unproject(depth_map, K)
        assert len(cloud) == depth_map.size - np.sum(depth_map == 0)
        assert (cloud[:, 2] > 0).all()

    def test_drops_nan_and_inf(self, K):
        depth = np.full((10, 10), 1.0, np.float32)
        depth[0, 0] = np.nan
        depth[0, 1] = np.inf
        cloud = unproject(depth, K)
        assert len(cloud) == 98
        assert np.isfinite(cloud).all()

    def test_respects_max_depth(self, K):
        depth = np.full((10, 10), 1.0, np.float32)
        depth[0, 0] = 99.0
        assert len(unproject(depth, K, max_depth=5.0)) == 99

    def test_empty_mask_returns_empty_not_error(self, depth_map, K):
        cloud = unproject(depth_map, K, np.zeros_like(depth_map, dtype=bool))
        assert cloud.shape == (0, 3)

    def test_mask_shape_mismatch_raises(self, depth_map, K):
        with pytest.raises(ValueError):
            unproject(depth_map, K, np.zeros((10, 10), bool))

    def test_rejects_non_2d_depth(self, K):
        with pytest.raises(ValueError):
            unproject(np.zeros((4, 4, 4)), K)


class TestDownsample:
    """Guards a correctness bug, not a performance nicety: GraspGen-X's outlier
    removal allocates an N x N matrix, so an un-capped 40k-point mask hangs the
    server for minutes and then fails."""

    def test_small_clouds_pass_through_untouched(self, synthetic_box_cloud):
        out = downsample_cloud(synthetic_box_cloud, max_points=8192)
        assert out.shape == synthetic_box_cloud.shape
        assert np.array_equal(out, synthetic_box_cloud)

    def test_large_cloud_is_capped(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(-0.1, 0.1, (60_000, 3)).astype(np.float32)
        out = downsample_cloud(pts, max_points=4096)
        assert len(out) <= 4096
        assert out.shape[1] == 3

    def test_preserves_the_bounding_box(self):
        """Shape must survive: the model conditions on object geometry."""
        rng = np.random.default_rng(1)
        pts = rng.uniform(-1, 1, (50_000, 3)).astype(np.float32) * np.array(
            [0.03, 0.03, 0.12], np.float32
        )
        out = downsample_cloud(pts, max_points=3000)
        assert np.allclose(out.min(axis=0), pts.min(axis=0), atol=0.01)
        assert np.allclose(out.max(axis=0), pts.max(axis=0), atol=0.01)

    def test_thins_dense_regions_more_than_sparse_ones(self):
        """Voxel, not random: perspective makes near surfaces far denser, and
        random sampling would inherit that bias."""
        rng = np.random.default_rng(2)
        dense = rng.uniform(0.0, 0.01, (40_000, 3)).astype(np.float32)
        sparse = rng.uniform(0.5, 0.6, (2_000, 3)).astype(np.float32)
        out = downsample_cloud(np.vstack([dense, sparse]), max_points=4000)
        n_dense = int((out[:, 0] < 0.25).sum())
        n_sparse = len(out) - n_dense
        # Sparse region is 10x the linear extent, so it must not be swamped.
        assert n_sparse > n_dense * 0.5

    def test_output_is_float32_and_finite(self):
        rng = np.random.default_rng(3)
        pts = rng.uniform(-1, 1, (30_000, 3)).astype(np.float32)
        out = downsample_cloud(pts, max_points=2000)
        assert out.dtype == np.float32
        assert np.isfinite(out).all()

    def test_degenerate_zero_extent_cloud(self):
        pts = np.zeros((20_000, 3), np.float32)
        out = downsample_cloud(pts, max_points=500)
        assert len(out) <= 500


class TestProject:
    def test_points_behind_camera_are_nan(self, K):
        uv = project_points(np.array([[0.0, 0.0, -1.0]]), K)
        assert np.isnan(uv).all()

    def test_z_zero_is_nan(self, K):
        uv = project_points(np.array([[0.1, 0.1, 0.0]]), K)
        assert np.isnan(uv).all()


class TestGripperGeometry:
    def test_finger_points_straddle_the_axis(self):
        pose = np.eye(4)
        pose[:3, 3] = [0, 0, 0.5]
        tips = gripper_finger_points(pose, width=0.08, fingertip_depth=0.1)
        # +X is the closing direction, so the tips are +/- w/2 apart in x ...
        assert tips[0] == pytest.approx([-0.04, 0.0, 0.6])
        assert tips[1] == pytest.approx([0.04, 0.0, 0.6])
        # ... and both sit fingertip_depth along +Z from the base.
        assert np.linalg.norm(tips[1] - tips[0]) == pytest.approx(0.08)

    def test_finger_points_follow_rotation(self):
        # Rotate 90 deg about Z: the closing direction becomes +Y.
        pose = np.eye(4)
        pose[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
        tips = gripper_finger_points(pose, width=0.08, fingertip_depth=0.0)
        assert tips[0] == pytest.approx([0.0, -0.04, 0.0], abs=1e-9)
        assert tips[1] == pytest.approx([0.0, 0.04, 0.0], abs=1e-9)


class TestGraspToRect2d:
    def test_axis_aligned_grasp_gives_horizontal_rect(self, K):
        pose = np.eye(4)
        pose[:3, 3] = [0.0, 0.0, 1.0]
        rect = grasp_to_rect2d(pose, K, width=0.08, score=0.7)
        score, cx, cy, w, h, angle = rect
        assert score == pytest.approx(0.7)
        assert cx == pytest.approx(320.0)
        assert cy == pytest.approx(240.0)
        # 8 cm at 1 m with f=600 projects to 0.08 * 600 / 1.0 = 48 px.
        assert w == pytest.approx(48.0, abs=1e-3)
        assert angle == pytest.approx(0.0, abs=1e-6)

    def test_rotated_grasp_reports_rotated_angle(self, K):
        pose = np.eye(4)
        pose[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
        pose[:3, 3] = [0.0, 0.0, 1.0]
        rect = grasp_to_rect2d(pose, K, width=0.08)
        assert abs(rect[5]) == pytest.approx(90.0, abs=1e-6)

    def test_width_scales_inversely_with_distance(self, K):
        near, far = np.eye(4), np.eye(4)
        near[:3, 3] = [0, 0, 0.5]
        far[:3, 3] = [0, 0, 1.0]
        w_near = grasp_to_rect2d(near, K, 0.08)[3]
        w_far = grasp_to_rect2d(far, K, 0.08)[3]
        assert w_near == pytest.approx(2 * w_far, rel=1e-6)

    def test_behind_camera_returns_none(self, K):
        pose = np.eye(4)
        pose[:3, 3] = [0.0, 0.0, -1.0]
        assert grasp_to_rect2d(pose, K, 0.08) is None


class TestMaskFilter:
    def _pose_at(self, x, y, z):
        pose = np.eye(4)
        pose[:3, 3] = [x, y, z]
        return pose

    def test_keeps_grasp_inside_mask_drops_outside(self, K, box_mask):
        # Mask covers rows 200:280, cols 280:360 -> centred near (320, 240).
        inside = self._pose_at(0.0, 0.0, 0.6)
        outside = self._pose_at(0.5, 0.5, 0.6)  # projects far away
        grasps = np.stack([inside, outside])
        scores = np.array([0.5, 0.99])

        keep = filter_grasps_by_mask(
            grasps, scores, box_mask, K, width=0.08, fingertip_depth=0.0, dilate_px=0
        )
        assert 0 in keep
        assert 1 not in keep

    def test_returns_empty_for_no_grasps(self, K, box_mask):
        keep = filter_grasps_by_mask(
            np.zeros((0, 4, 4)), np.zeros(0), box_mask, K, 0.08
        )
        assert len(keep) == 0

    def test_containment_is_tested_at_the_fingertips_not_the_base(self, K, box_mask):
        """The pose is anchored at the gripper BASE, ~10 cm behind the contact.

        Testing containment there checks a point well short of where the hand
        actually closes. Measured on the sample scene that rejected roughly 90%
        of valid grasps (yellow cup: 6 of 58 kept at the base vs 58 of 58 at the
        fingertips), so the depth must be threaded through.
        """
        # Hand sits back and to the side; its fingertips reach the mask centre.
        pose = np.eye(4)
        pose[:3, 3] = [-0.10, 0.0, 0.6]     # base projects left of the mask
        pose[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)  # +Z -> +X
        grasps, scores = pose[None], np.array([0.9])

        at_base = filter_grasps_by_mask(grasps, scores, box_mask, K, width=0.08,
                                        fingertip_depth=0.0, dilate_px=0)
        at_tips = filter_grasps_by_mask(grasps, scores, box_mask, K, width=0.08,
                                        fingertip_depth=0.10, dilate_px=0)
        assert len(at_base) == 0, "base projects outside the mask, as constructed"
        assert len(at_tips) == 1, "fingertips reach the mask and must be kept"

    def test_dilation_admits_near_misses(self, K, box_mask):
        # Just outside the mask edge: rejected tight, accepted with dilation.
        edge = self._pose_at((365 - 320) * 0.6 / 600, 0.0, 0.6)
        grasps = edge[None]
        scores = np.array([0.5])
        tight = filter_grasps_by_mask(grasps, scores, box_mask, K, 0.08,
                                      fingertip_depth=0.0, dilate_px=0)
        loose = filter_grasps_by_mask(grasps, scores, box_mask, K, 0.08,
                                      fingertip_depth=0.0, dilate_px=20)
        assert len(tight) == 0
        assert len(loose) == 1


class TestVisibilityFilter:
    """Single-view grasping: the far side of the object was never observed, so a
    hand approaching from behind it would have to pass through the object or the
    table. GraspMoE's OBB sweep produces many such candidates."""

    @staticmethod
    def _grasp(position, approach):
        """A pose at `position` whose +Z is `approach` (other axes arbitrary)."""
        a = np.asarray(approach, float)
        a = a / np.linalg.norm(a)
        tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(tmp, a); x /= np.linalg.norm(x)
        y = np.cross(a, x)
        pose = np.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2] = x, y, a
        pose[:3, 3] = position
        return pose

    def test_keeps_forward_approach(self):
        g = self._grasp([0, 0, 1.0], [0, 0, 1.0])[None]
        assert len(filter_grasps_by_visibility(g)) == 1

    def test_keeps_side_approach(self):
        """~90 degrees is legitimate and is much of the point of 6-DoF."""
        g = self._grasp([0, 0, 1.0], [1.0, 0, 0])[None]
        assert len(filter_grasps_by_visibility(g)) == 1

    def test_rejects_approach_from_behind_the_object(self):
        g = self._grasp([0, 0, 1.0], [0, 0, -1.0])[None]
        assert len(filter_grasps_by_visibility(g)) == 0

    def test_threshold_is_against_the_viewing_ray_not_the_optical_axis(self):
        """An off-axis object: 'backwards' means back along ITS ray."""
        pos = [1.0, 0.0, 1.0]                 # 45 degrees off the optical axis
        along = self._grasp(pos, pos)[None]    # travelling away from the camera
        back = self._grasp(pos, [-1.0, 0.0, -1.0])[None]
        assert len(filter_grasps_by_visibility(along)) == 1
        assert len(filter_grasps_by_visibility(back)) == 0

    def test_threshold_is_configurable(self):
        g = self._grasp([0, 0, 1.0], [1.0, 0, 0])[None]   # exactly 90 degrees
        assert len(filter_grasps_by_visibility(g, max_backward_deg=80.0)) == 0
        assert len(filter_grasps_by_visibility(g, max_backward_deg=100.0)) == 1

    def test_empty_input(self):
        assert len(filter_grasps_by_visibility(np.zeros((0, 4, 4)))) == 0

    def test_mixed_batch_returns_only_visible_indices(self):
        good = self._grasp([0, 0, 1.0], [0, 0, 1.0])
        bad = self._grasp([0, 0, 1.0], [0, 0, -1.0])
        keep = filter_grasps_by_visibility(np.stack([bad, good, bad, good]))
        assert set(keep.tolist()) == {1, 3}


class TestApproachElevation:
    def test_straight_down_optical_axis_is_zero(self):
        assert approach_elevation_deg([0, 0, 1]) == pytest.approx(0.0)

    def test_perpendicular_is_ninety(self):
        assert approach_elevation_deg([1, 0, 0]) == pytest.approx(90.0)

    def test_backwards_is_one_eighty(self):
        assert approach_elevation_deg([0, 0, -1]) == pytest.approx(180.0)

    def test_zero_vector_is_nan(self):
        assert math.isnan(approach_elevation_deg([0, 0, 0]))


class TestSceneContext:
    def test_uses_supplied_depth_without_estimator(self, depth_map, K, rgb):
        scene = SceneContext(depth=depth_map, intrinsics=K)
        assert scene.has_real_depth
        assert scene.depth_source == "sensor"
        assert np.allclose(scene.get_depth(rgb), depth_map)
        assert scene.estimator is None  # never constructed

    def test_depth_scale_converts_millimetres(self, K):
        mm = np.full((10, 10), 600.0, np.float32)
        scene = SceneContext(depth=mm, intrinsics=K, depth_scale=0.001)
        assert scene.depth.max() == pytest.approx(0.6)

    def test_synthesizes_intrinsics_when_absent(self, rgb, depth_map):
        scene = SceneContext(depth=depth_map)
        K = scene.get_intrinsics(rgb)
        assert K.shape == (3, 3)
        assert K[0, 2] == pytest.approx(rgb.shape[1] / 2)

    def test_shape_mismatch_raises(self, K):
        scene = SceneContext(depth=np.zeros((10, 10), np.float32), intrinsics=K)
        with pytest.raises(ValueError):
            scene.get_depth(np.zeros((480, 640, 3), np.uint8))

    def test_describe_reports_source(self, depth_map, K):
        d = SceneContext(depth=depth_map, intrinsics=K).describe()
        assert d["depth_source"] == "sensor"
        assert d["fx"] == pytest.approx(600.0)


class TestGrasp6D:
    def test_accessors_match_pose_columns(self):
        pose = np.eye(4)
        pose[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
        pose[:3, 3] = [1, 2, 3]
        g = Grasp6D(pose=pose, score=0.5, gripper="x", width=0.08)
        assert np.allclose(g.position, [1, 2, 3])
        assert np.allclose(g.approach, pose[:3, 2])
        assert np.allclose(g.closing, pose[:3, 0])

    def test_dict_round_trip(self, identity_grasp):
        d = identity_grasp.as_dict()
        back = Grasp6D.from_dict(d)
        assert np.allclose(back.pose, identity_grasp.pose)
        assert back.score == identity_grasp.score
        assert back.gripper == identity_grasp.gripper
        assert back.width == identity_grasp.width

    def test_as_dict_is_json_serializable(self, identity_grasp):
        import json
        json.dumps(identity_grasp.as_dict())  # must not raise

    def test_summary_is_readable(self, identity_grasp):
        s = identity_grasp.summary()
        assert "franka_panda" in s and "score" in s

    def test_pose_is_reshaped_from_flat_input(self):
        g = Grasp6D(pose=np.eye(4).ravel(), score=1.0, gripper="x", width=0.08)
        assert g.pose.shape == (4, 4)
