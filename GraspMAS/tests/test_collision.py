"""Scene collision checking.

The gripper geometry comes from the installed gripper descriptions when they
are present and from a conservative box hull when they are not, so the whole
module is exercised either way; the tests that genuinely need the real asset say
so and skip.
"""

import numpy as np
import pytest

import collision as col
import placement as pl
import synth_scene as ss
from perception3d import unproject

HAVE_ASSETS = col.gripper_asset_dir("franka_panda") is not None
needs_assets = pytest.mark.skipif(
    not HAVE_ASSETS, reason="gripper_descriptions not installed"
)


@pytest.fixture
def gripper_pts():
    return col.load_gripper_points("franka_panda")


def _blob(centre, half=0.03, n=600, seed=0):
    """A small flat patch of surface — a stand-in for the face of an obstacle.

    Deliberately *not* a shifted copy of the gripper: the gripper is long in Z
    and wide in X, so a translated copy of it overlaps itself for any shift
    smaller than its own extent, which makes distance assertions meaningless.

    Sized like a real object face rather than a pinpoint, because proximity is
    measured from the gripper's sampled surface and something much smaller than
    the sample spacing can legitimately pass between the points.
    """
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-half, half, (n, 2))
    return np.asarray(centre, dtype=np.float64) + np.column_stack([xy, np.zeros(n)])


def _pose_at(xyz, approach=(0.0, 0.0, 1.0)):
    """A pose with +Z along `approach` and its origin at `xyz`."""
    z = np.asarray(approach, dtype=np.float64)
    z = z / np.linalg.norm(z)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, np.cross(z, x), z
    T[:3, 3] = xyz
    return T


class TestLoadGripperPoints:
    def test_returns_points_in_the_gripper_frame(self, gripper_pts):
        assert gripper_pts.ndim == 2 and gripper_pts.shape[1] == 3
        assert gripper_pts[:, 2].max() > 0.05, "fingertips should be along +Z"

    def test_respects_the_point_cap(self):
        assert len(col.load_gripper_points("franka_panda", max_points=64)) <= 64

    def test_is_cached(self):
        a = col.load_gripper_points("franka_panda")
        b = col.load_gripper_points("franka_panda")
        assert a is b

    def test_result_is_read_only(self, gripper_pts):
        """Caching hands the same array to every caller, so it must not be writable."""
        with pytest.raises(ValueError):
            gripper_pts[0, 0] = 99.0

    def test_rejects_a_bad_state(self):
        with pytest.raises(ValueError, match="open"):
            col.load_gripper_points("franka_panda", state="ajar")

    def test_unknown_gripper_falls_back(self):
        pts = col.load_gripper_points("no_such_gripper_xyz")
        assert len(pts) == col.DEFAULT_GRIPPER_POINTS
        assert np.isfinite(pts).all()

    @needs_assets
    def test_matches_the_declared_bounding_box(self):
        """Confirms points.json really is in the pose frame our grasps use."""
        import json
        import os

        d = col.gripper_asset_dir("franka_panda")
        with open(os.path.join(d, "config.json")) as f:
            bbox = np.array(json.load(f)["bbox"])
        pts = col.load_gripper_points("franka_panda", max_points=10500)
        assert np.all(pts.min(axis=0) >= bbox[0] - 1e-3)
        assert np.all(pts.max(axis=0) <= bbox[1] + 1e-3)

    @needs_assets
    def test_open_and_close_differ(self):
        a = col.load_gripper_points("franka_panda", state="open", max_points=10500)
        b = col.load_gripper_points("franka_panda", state="close", max_points=10500)
        assert not np.array_equal(a, b)


class TestTransformPoints:
    def test_identity_is_a_no_op(self, gripper_pts):
        assert col.transform_points(gripper_pts, np.eye(4)) == pytest.approx(gripper_pts)

    def test_translation_shifts(self, gripper_pts):
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        moved = col.transform_points(gripper_pts, T)
        delta = moved - gripper_pts
        assert delta == pytest.approx(np.broadcast_to([1.0, 2.0, 3.0], delta.shape))

    def test_rotation_preserves_pairwise_distance(self, gripper_pts):
        T = _pose_at([0, 0, 0], approach=[1, 1, 1])
        moved = col.transform_points(gripper_pts, T)
        d0 = np.linalg.norm(gripper_pts[0] - gripper_pts[1])
        d1 = np.linalg.norm(moved[0] - moved[1])
        assert d1 == pytest.approx(d0)


class TestPoseCollides:
    def test_empty_scene_never_collides(self, gripper_pts):
        assert not col.pose_collides(np.eye(4), None, gripper_pts)
        assert not col.pose_collides(np.eye(4), col._build_tree(np.zeros((0, 3))), gripper_pts)

    def test_points_inside_the_hand_collide(self, gripper_pts):
        scene = gripper_pts + np.array([0.0, 0.0, 0.0])  # exactly on the surface
        assert col.pose_collides(np.eye(4), col._build_tree(scene), gripper_pts)

    def test_distant_points_do_not_collide(self, gripper_pts):
        scene = gripper_pts + np.array([5.0, 0.0, 0.0])
        assert not col.pose_collides(np.eye(4), col._build_tree(scene), gripper_pts)

    def test_a_single_stray_point_is_ignored(self, gripper_pts):
        """Depth flyers cluster at object boundaries, which is where grasps are."""
        stray = gripper_pts[:1].copy()
        tree = col._build_tree(stray)
        assert not col.pose_collides(np.eye(4), tree, gripper_pts, min_hits=3)
        assert col.pose_collides(np.eye(4), tree, gripper_pts, min_hits=1)

    def test_threshold_controls_sensitivity(self, gripper_pts):
        """A blob 3 cm off the hand's side: inside a 5 cm margin, outside a 5 mm one."""
        side = gripper_pts[:, 1].max() + 0.03
        tree = col._build_tree(_blob([0.0, side, 0.05]))
        assert not col.pose_collides(np.eye(4), tree, gripper_pts, thresh=0.005)
        assert col.pose_collides(np.eye(4), tree, gripper_pts, thresh=0.05)

    def test_pose_is_applied(self, gripper_pts):
        scene = gripper_pts + np.array([1.0, 0.0, 0.0])
        tree = col._build_tree(scene)
        assert not col.pose_collides(np.eye(4), tree, gripper_pts)
        shifted = np.eye(4)
        shifted[:3, 3] = [1.0, 0.0, 0.0]
        assert col.pose_collides(shifted, tree, gripper_pts)


class TestSweep:
    def test_clear_corridor_is_clear(self, gripper_pts):
        far = np.array([[3.0, 3.0, 3.0]])
        assert col.sweep_is_clear(np.eye(4), far, gripper_pts)

    def test_empty_scene_is_clear(self, gripper_pts):
        assert col.sweep_is_clear(np.eye(4), np.zeros((0, 3)), gripper_pts)

    def test_obstacle_behind_the_grasp_is_caught(self, gripper_pts):
        """Only the swept volume sees this; the final pose alone would pass.

        The blob sits a clear 10 cm back along -Z, so at the grasp pose the hand
        misses it entirely and only the approach runs into it.
        """
        pose = _pose_at([0.0, 0.0, 0.0], approach=[0, 0, 1])
        back = gripper_pts[:, 2].min() - 0.10
        obstacle = _blob([0.0, 0.0, back])
        tree = col._build_tree(obstacle)
        assert not col.pose_collides(pose, tree, gripper_pts, thresh=0.005)
        assert not col.sweep_is_clear(
            pose, obstacle, gripper_pts, approach_len=0.15, thresh=0.005
        )

    def test_zero_length_sweep_is_just_the_final_pose(self, gripper_pts):
        pose = _pose_at([0.0, 0.0, 0.0], approach=[0, 0, 1])
        obstacle = _blob([0.0, 0.0, gripper_pts[:, 2].min() - 0.10])
        assert col.sweep_is_clear(
            pose, obstacle, gripper_pts, approach_len=0.0, thresh=0.005
        )


class TestFilterGrasps:
    def test_empty_grasp_set(self, gripper_pts):
        out = col.filter_grasps_by_scene_collision(
            np.zeros((0, 4, 4)), np.ones((10, 3)), gripper_pts
        )
        assert len(out) == 0

    def test_empty_scene_keeps_everything(self, gripper_pts):
        grasps = np.stack([np.eye(4)] * 5)
        out = col.filter_grasps_by_scene_collision(grasps, np.zeros((0, 3)), gripper_pts)
        assert list(out) == [0, 1, 2, 3, 4]

    def test_separates_colliding_from_free(self, gripper_pts):
        clean = np.eye(4)
        dirty = np.eye(4)
        dirty[:3, 3] = [1.0, 0.0, 0.0]
        grasps = np.stack([clean, dirty])
        scene = gripper_pts + np.array([1.0, 0.0, 0.0])
        out = col.filter_grasps_by_scene_collision(grasps, scene, gripper_pts)
        assert list(out) == [0]

    def test_sweep_mode_is_stricter(self, gripper_pts):
        pose = _pose_at([0.0, 0.0, 0.0], approach=[0, 0, 1])
        grasps = pose[None]
        scene = _blob([0.0, 0.0, gripper_pts[:, 2].min() - 0.10])
        final_only = col.filter_grasps_by_scene_collision(
            grasps, scene, gripper_pts, thresh=0.005
        )
        swept = col.filter_grasps_by_scene_collision(
            grasps, scene, gripper_pts, thresh=0.005, approach_len=0.15
        )
        assert len(final_only) == 1 and len(swept) == 0


class TestSceneCloudExcluding:
    def test_excludes_the_masked_object(self, tabletop):
        label = tabletop["label_map"]["bottle"]
        full = col.scene_cloud_excluding(tabletop["depth"], tabletop["K"])
        without = col.scene_cloud_excluding(
            tabletop["depth"], tabletop["K"], exclude_mask=(tabletop["seg"] == label)
        )
        assert len(without) < len(full)

    def test_excluded_points_are_really_gone(self, tabletop, tabletop_plane):
        label = tabletop["label_map"]["bottle"]
        bottle = unproject(tabletop["depth"], tabletop["K"], mask=(tabletop["seg"] == label))
        without = col.scene_cloud_excluding(
            tabletop["depth"], tabletop["K"], exclude_mask=(tabletop["seg"] == label)
        )
        tree = col._build_tree(without)
        # No remaining point should sit on the bottle's surface.
        near = tree.query_ball_point(bottle[::20], r=0.002, return_length=True)
        assert np.mean(near > 0) < 0.05

    def test_respects_the_point_cap(self, tabletop):
        out = col.scene_cloud_excluding(tabletop["depth"], tabletop["K"], max_points=2000)
        assert len(out) <= 2000

    def test_max_depth_clips(self, tabletop):
        near = col.scene_cloud_excluding(tabletop["depth"], tabletop["K"], max_depth=0.9)
        assert near[:, 2].max() < 0.9


class TestPointsInsideSweep:
    @pytest.fixture
    def banana_grasp(self, tabletop, tabletop_plane, object_cloud_fn):
        """A top-down grasp closing across the banana's short axis."""
        fp = pl.object_footprint(object_cloud_fn("banana"), tabletop_plane)
        short = np.array([-np.sin(fp.yaw), np.cos(fp.yaw)])
        gx = tabletop_plane.rotation[:, :2] @ short
        T = np.eye(4)
        T[:3, 2] = -tabletop_plane.normal
        T[:3, 0] = gx / np.linalg.norm(gx)
        T[:3, 1] = np.cross(T[:3, 2], T[:3, 0])
        T[:3, 3] = tabletop_plane.to_camera(
            np.array([[*fp.centroid_xy, fp.top_m + 0.005]])
        )[0] + 0.1034 * tabletop_plane.normal
        return T

    def test_names_the_blocking_objects(self, banana_grasp, gripper_pts, object_cloud_fn):
        """Both blockers must be attributed; the distractor must not be."""
        hits = {
            name: len(
                col.points_inside_sweep(
                    banana_grasp, object_cloud_fn(name), gripper_pts, approach_len=0.12
                )
            )
            for name in ("bottle", "mug", "box")
        }
        assert hits["bottle"] > 0, "the bottle is 10.5 cm away and must block"
        assert hits["mug"] > 0, "the mug is 11.5 cm away and must block"
        assert hits["box"] == 0, "the distractor is 35 cm away and must not block"

    def test_empty_for_no_points(self, banana_grasp, gripper_pts):
        assert len(col.points_inside_sweep(banana_grasp, np.zeros((0, 3)), gripper_pts)) == 0

    def test_returns_indices_into_the_input(self, banana_grasp, gripper_pts, object_cloud_fn):
        pts = object_cloud_fn("bottle")
        idx = col.points_inside_sweep(banana_grasp, pts, gripper_pts, approach_len=0.12)
        assert idx.max() < len(pts)


class TestDeclutteringActuallyHelps:
    """The end the whole module exists for: removing blockers frees the grasp."""

    def test_sweep_is_blocked_before_and_clear_after(self, tabletop, gripper_pts):
        spec, K, T = tabletop["spec"], tabletop["K"], tabletop["T_cam_world"]
        plane = pl.fit_support_plane(unproject(tabletop["depth"], K, max_depth=5.0))
        label = tabletop["label_map"]["banana"]

        fp = pl.object_footprint(
            unproject(tabletop["depth"], K, mask=(tabletop["seg"] == label)), plane
        )
        short = np.array([-np.sin(fp.yaw), np.cos(fp.yaw)])
        gx = plane.rotation[:, :2] @ short
        grasp = np.eye(4)
        grasp[:3, 2] = -plane.normal
        grasp[:3, 0] = gx / np.linalg.norm(gx)
        grasp[:3, 1] = np.cross(grasp[:3, 2], grasp[:3, 0])
        grasp[:3, 3] = plane.to_camera(
            np.array([[*fp.centroid_xy, fp.top_m + 0.005]])
        )[0] + 0.1034 * plane.normal

        before = col.scene_cloud_excluding(
            tabletop["depth"], K, exclude_mask=(tabletop["seg"] == label)
        )
        assert not col.sweep_is_clear(grasp, before, gripper_pts, approach_len=0.12)

        cleared = ss.SceneSpec(
            [o for o in spec.objects if o.name in ("banana", "box")], spec.table_extent
        )
        r = ss.render(cleared, K, T)
        after = col.scene_cloud_excluding(r["depth"], K, exclude_mask=(r["seg"] == label))
        assert col.sweep_is_clear(grasp, after, gripper_pts, approach_len=0.12)
