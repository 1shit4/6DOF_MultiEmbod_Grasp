"""Placement geometry, checked against ground truth rather than against itself.

The synthetic scenes are ray-cast, so the table plane and every object's extent
are known exactly. That lets these tests assert real numbers — "the fitted
normal is within 0.1 degrees of the true one", "the object's bottom lands on the
surface to within a millimetre" — instead of the weaker "it returned something".
"""

import numpy as np
import pytest

import placement as pl
import synth_scene as ss
from perception3d import unproject


def _scene_cloud(scene):
    return unproject(scene["depth"], scene["K"], max_depth=5.0)


def _object_cloud(scene, name):
    return unproject(
        scene["depth"], scene["K"], mask=(scene["seg"] == scene["label_map"][name])
    )


def _angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))


# ---------------------------------------------------------------------------


class TestSupportPlane:
    def test_recovers_the_ground_truth_plane(self, tabletop, tabletop_plane):
        truth = tabletop["plane_truth"]
        assert _angle_deg(tabletop_plane.normal, truth["normal"]) < 0.1
        assert tabletop_plane.offset == pytest.approx(truth["offset"], abs=2e-3)

    def test_normal_points_toward_the_camera(self, tabletop_plane):
        """Height must be positive on the side the objects are on."""
        assert tabletop_plane.height_of(np.zeros((1, 3)))[0] > 0

    def test_frame_is_orthonormal_and_right_handed(self, tabletop_plane):
        R = tabletop_plane.rotation
        assert R.T @ R == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)

    def test_frame_z_axis_is_the_normal(self, tabletop_plane):
        assert tabletop_plane.rotation[:, 2] == pytest.approx(tabletop_plane.normal)

    def test_origin_lies_on_the_plane(self, tabletop_plane):
        assert tabletop_plane.height_of(tabletop_plane.origin[None, :])[0] == pytest.approx(
            0.0, abs=1e-12
        )

    def test_table_points_have_zero_height(self, tabletop, tabletop_plane):
        pts = unproject(
            tabletop["depth"], tabletop["K"], mask=(tabletop["seg"] == ss.TABLE_ID)
        )
        assert np.abs(tabletop_plane.height_of(pts)).max() < 5e-3

    def test_table_frame_z_equals_height(self, tabletop, tabletop_plane, object_cloud_fn):
        pts = object_cloud_fn("mug")
        assert tabletop_plane.to_table(pts)[:, 2] == pytest.approx(
            tabletop_plane.height_of(pts), abs=1e-9
        )

    def test_to_table_round_trips(self, tabletop_plane, synthetic_box_cloud):
        pts = synthetic_box_cloud.astype(np.float64)
        back = tabletop_plane.to_camera(tabletop_plane.to_table(pts))
        assert back == pytest.approx(pts, abs=1e-9)

    def test_camera_xy_is_the_projected_camera_centre(self, tabletop_plane):
        xy = tabletop_plane.camera_xy()
        assert xy == pytest.approx(tabletop_plane.to_table(np.zeros((1, 3)))[0, :2])

    def test_survives_a_table_hidden_under_clutter(self):
        """The case that broke inlier-count-only fitting.

        On `crowded_table` the boxes cover most of the surface, so "biggest
        plane wins" fits a plane tilted 9.9 degrees through the box fronts,
        8.4 cm off. Requiring a clear underside recovers it to 0.05 degrees.
        """
        scene = ss.build("crowded_table")
        plane = pl.fit_support_plane(_scene_cloud(scene))
        truth = scene["plane_truth"]
        assert _angle_deg(plane.normal, truth["normal"]) < 0.5
        assert plane.offset == pytest.approx(truth["offset"], abs=5e-3)

    def test_rejects_a_cloud_with_no_support_surface(self):
        rng = np.random.default_rng(0)
        blob = rng.normal(0, 0.3, (4000, 3)) + np.array([0, 0, 1.0])
        with pytest.raises(ValueError, match="clear underside|holds only"):
            pl.fit_support_plane(blob)

    def test_rejects_too_small_a_cloud(self):
        with pytest.raises(ValueError, match="at least"):
            pl.fit_support_plane(np.zeros((10, 3)))

    def test_is_deterministic_under_a_seed(self, tabletop_cloud):
        a = pl.fit_support_plane(tabletop_cloud, seed=7)
        b = pl.fit_support_plane(tabletop_cloud, seed=7)
        assert a.normal == pytest.approx(b.normal)
        assert a.offset == pytest.approx(b.offset)

    def test_describe_is_json_safe(self, tabletop_plane):
        import json

        assert json.loads(json.dumps(tabletop_plane.describe()))["n_inliers"] > 0


class TestSupportCloud:
    def test_object_mask_tightens_the_depth_limit(self, tabletop):
        wide = pl.support_cloud(tabletop["depth"], tabletop["K"], max_depth=5.0)
        near = pl.support_cloud(
            tabletop["depth"],
            tabletop["K"],
            object_mask=(tabletop["seg"] >= ss.FIRST_OBJECT_ID),
            margin_m=0.05,
        )
        assert near[:, 2].max() < wide[:, 2].max()

    def test_limit_tracks_the_furthest_object(self, tabletop):
        objects = unproject(
            tabletop["depth"], tabletop["K"],
            mask=(tabletop["seg"] >= ss.FIRST_OBJECT_ID), max_depth=5.0,
        )
        margin = 0.1
        near = pl.support_cloud(
            tabletop["depth"], tabletop["K"],
            object_mask=(tabletop["seg"] >= ss.FIRST_OBJECT_ID), margin_m=margin,
        )
        assert near[:, 2].max() <= np.percentile(objects[:, 2], 95) + margin + 1e-6

    def test_respects_the_point_cap(self, tabletop):
        cloud = pl.support_cloud(tabletop["depth"], tabletop["K"], max_points=5000)
        assert len(cloud) <= 5000

    def test_works_without_a_mask(self, tabletop):
        assert len(pl.support_cloud(tabletop["depth"], tabletop["K"])) > 1000


class TestHeightMap:
    @pytest.fixture
    def hmap(self, tabletop_cloud, tabletop_plane):
        return pl.build_height_map(tabletop_cloud, tabletop_plane, cell_m=0.005)

    def test_object_cells_report_the_object_height(self, hmap, tabletop, tabletop_plane):
        mug = tabletop["spec"].by_name("mug").primitive
        centre_xy = tabletop_plane.to_table(
            np.array([[mug.position[0], mug.position[1], mug.position[2]]])
        )
        # The mug centre in table XY is not the world XY; go through the cloud.
        pts = unproject(
            tabletop["depth"], tabletop["K"],
            mask=(tabletop["seg"] == tabletop["label_map"]["mug"]),
        )
        xy = tabletop_plane.to_table(pts)[:, :2].mean(axis=0)
        r, c = hmap.to_cell(xy)[0]
        assert hmap.heights[r, c] == pytest.approx(mug.height, abs=0.02)
        assert centre_xy.shape == (1, 3)  # sanity on the helper above

    def test_table_cells_are_flat(self, hmap):
        flat = hmap.heights[np.isfinite(hmap.heights)]
        assert np.percentile(flat, 20) < 0.01

    def test_unobserved_cells_are_nan(self, hmap):
        assert not hmap.observed.all(), "a single view must leave shadows"
        assert np.isnan(hmap.heights[~hmap.observed]).all()

    def test_to_cell_and_to_xy_round_trip(self, hmap):
        rc = np.array([[10, 20], [3, 4]])
        assert hmap.to_cell(hmap.to_xy(rc)) == pytest.approx(rc)

    def test_to_xy_returns_cell_centres(self, hmap):
        xy = hmap.to_xy(np.array([[0, 0]]))[0]
        assert xy == pytest.approx(hmap.origin + hmap.cell_m / 2.0)

    def test_in_bounds_flags_outside_cells(self, hmap):
        rows, cols = hmap.shape
        rc = np.array([[0, 0], [rows, 0], [-1, 0], [0, cols]])
        assert list(hmap.in_bounds(rc)) == [True, False, False, False]

    def test_bounds_restrict_the_grid(self, tabletop_cloud, tabletop_plane):
        small = pl.build_height_map(
            tabletop_cloud, tabletop_plane, cell_m=0.01, bounds=(-0.2, 0.7, 0.2, 1.1)
        )
        assert small.shape == (40, 40)

    def test_rejects_an_empty_cloud(self, tabletop_plane):
        with pytest.raises(ValueError, match="empty cloud"):
            pl.build_height_map(np.zeros((0, 3)), tabletop_plane)

    def test_points_below_the_plane_are_clamped_not_dropped(self, tabletop_plane):
        """Noise under the surface must still mark the cell observed."""
        table_pt = tabletop_plane.to_camera(np.array([[0.0, 0.0, -0.004]]))
        cloud = np.repeat(table_pt, 500, axis=0)
        cloud = cloud + np.random.default_rng(0).normal(0, 1e-4, cloud.shape)
        hm = pl.build_height_map(cloud, tabletop_plane, cell_m=0.01)
        assert hm.observed.any()
        assert np.nanmin(hm.heights) >= 0.0


class TestFreeSpace:
    @pytest.fixture
    def hmap(self, tabletop_cloud, tabletop_plane):
        return pl.build_height_map(tabletop_cloud, tabletop_plane, cell_m=0.005)

    def test_flat_observed_cells_are_free(self, hmap):
        free = pl.free_space(hmap)
        assert free.sum() > 0
        assert np.all(hmap.heights[free] <= pl.DEFAULT_CLEARANCE_M)

    def test_tall_cells_are_not_free(self, hmap):
        free = pl.free_space(hmap)
        tall = np.isfinite(hmap.heights) & (hmap.heights > 0.05)
        assert not free[tall].any()

    def test_unknown_is_occupied_by_default(self, hmap):
        assert not pl.free_space(hmap)[~hmap.observed].any()

    def test_unknown_can_be_opted_into(self, hmap):
        assert pl.free_space(hmap, unknown_is_free=True)[~hmap.observed].all()

    def test_clearance_threshold_is_respected(self, hmap):
        assert pl.free_space(hmap, clearance_m=0.001).sum() <= pl.free_space(
            hmap, clearance_m=0.05
        ).sum()


class TestFootprint:
    def test_matches_a_known_box(self, tabletop, tabletop_plane, object_cloud_fn):
        truth = tabletop["spec"].by_name("box").primitive
        fp = pl.object_footprint(object_cloud_fn("box"), tabletop_plane)
        extents = np.sort(fp.half_extent * 2)
        assert extents == pytest.approx(np.sort(truth.size[:2]), abs=0.015)

    def test_matches_a_known_cylinder(self, tabletop, tabletop_plane, object_cloud_fn):
        truth = tabletop["spec"].by_name("mug").primitive
        fp = pl.object_footprint(object_cloud_fn("mug"), tabletop_plane)
        diameter = float(truth.size[0]) * 2
        assert fp.half_extent * 2 == pytest.approx([diameter, diameter], abs=0.015)

    def test_is_biased_small_by_less_than_the_placement_margin(
        self, tabletop, tabletop_plane, object_cloud_fn
    ):
        """A single view sees only the front, so the fit is always undersized.

        That bias is not corrected in the footprint (it would make the centroid
        view-dependent); `DEFAULT_MARGIN_M` absorbs it instead. This test is
        what ties the two together: if the bias ever grows past the margin,
        placements stop being safe and this fails.
        """
        for name in ("bottle", "mug", "box"):
            truth = tabletop["spec"].by_name(name).primitive
            expected = np.sort(
                truth.size[:2]
                if truth.kind != "cylinder"
                else np.array([truth.size[0] * 2] * 2)
            )
            got = np.sort(pl.object_footprint(object_cloud_fn(name), tabletop_plane).half_extent * 2)
            assert np.all(got <= expected + 1e-9), f"{name} is not undersized"
            assert np.all(expected - got < pl.DEFAULT_MARGIN_M), f"{name} bias exceeds margin"

    def test_centroid_does_not_depend_on_the_viewpoint(self, tabletop_plane, object_cloud_fn):
        """The evaluator compares centroids across observations, so they must be stable."""
        cloud = object_cloud_fn("mug")
        near = pl.object_footprint(cloud, tabletop_plane).centroid_xy
        shifted = pl.object_footprint(cloud + np.array([0.0, 0.0, 0.0]), tabletop_plane)
        assert shifted.centroid_xy == pytest.approx(near, abs=1e-12)

    def test_a_heavily_occluded_object_is_underestimated(
        self, tabletop, tabletop_plane, object_cloud_fn
    ):
        """A documented limitation, asserted so it cannot regress into a surprise.

        The banana is 19% visible; nothing recovers the part another object
        hides. Callers must gate on visibility, not on the footprint looking
        plausible.
        """
        truth = tabletop["spec"].by_name("banana").primitive
        fp = pl.object_footprint(object_cloud_fn("banana"), tabletop_plane)
        longest = float(np.max(fp.half_extent * 2))
        assert longest < 0.5 * float(np.max(truth.size[:2]))

    def test_height_matches_the_primitive(self, tabletop, tabletop_plane, object_cloud_fn):
        for name in ("mug", "bottle", "banana"):
            fp = pl.object_footprint(object_cloud_fn(name), tabletop_plane)
            expected = tabletop["spec"].by_name(name).primitive.height
            assert fp.top_m == pytest.approx(expected, abs=0.01), name

    def test_bottom_is_on_the_table(self, tabletop_plane, object_cloud_fn):
        fp = pl.object_footprint(object_cloud_fn("mug"), tabletop_plane)
        assert fp.bottom_m == pytest.approx(0.0, abs=0.01)

    def test_radius_is_circumscribed(self, tabletop_plane, object_cloud_fn):
        fp = pl.object_footprint(object_cloud_fn("banana"), tabletop_plane)
        assert fp.radius_m == pytest.approx(np.hypot(*fp.half_extent))
        assert fp.radius_m >= fp.half_extent.max()

    def test_corners_are_consistent_with_extent(self, tabletop_plane, object_cloud_fn):
        fp = pl.object_footprint(object_cloud_fn("mug"), tabletop_plane)
        corners = fp.corners()
        assert corners.shape == (4, 2)
        assert corners.mean(axis=0) == pytest.approx(fp.centroid_xy)
        side = np.linalg.norm(corners[1] - corners[0])
        assert side == pytest.approx(2 * fp.half_extent[0], abs=1e-9)

    def test_percentiles_absorb_a_flyer_point(self, tabletop_plane, object_cloud_fn):
        clean = object_cloud_fn("mug")
        fp_clean = pl.object_footprint(clean, tabletop_plane)
        noisy = np.vstack([clean, clean[:1] + np.array([0.5, 0.5, 0.0])])
        fp_noisy = pl.object_footprint(noisy, tabletop_plane)
        assert fp_noisy.radius_m == pytest.approx(fp_clean.radius_m, abs=0.01)

    def test_rejects_too_few_points(self, tabletop_plane):
        with pytest.raises(ValueError, match=">=3"):
            pl.object_footprint(np.zeros((2, 3)), tabletop_plane)

    def test_handles_collinear_points(self, tabletop_plane):
        line = tabletop_plane.to_camera(
            np.stack([np.linspace(0, 0.1, 50), np.zeros(50), np.zeros(50)], axis=-1)
        )
        fp = pl.object_footprint(line, tabletop_plane)
        assert np.isfinite(fp.radius_m)

    def test_describe_is_json_safe(self, tabletop_plane, object_cloud_fn):
        import json

        fp = pl.object_footprint(object_cloud_fn("mug"), tabletop_plane)
        assert "radius_cm" in json.loads(json.dumps(fp.describe()))


class TestKeepOut:
    @pytest.fixture
    def hmap(self, tabletop_cloud, tabletop_plane):
        return pl.build_height_map(tabletop_cloud, tabletop_plane, cell_m=0.005)

    def test_footprint_keep_out_covers_the_object(self, hmap, tabletop_plane, object_cloud_fn):
        fp = pl.object_footprint(object_cloud_fn("mug"), tabletop_plane)
        blocked = pl.footprint_keep_out(hmap, [fp], dilate_m=0.0)
        r, c = hmap.to_cell(fp.centroid_xy)[0]
        assert blocked[r, c]

    def test_dilation_grows_the_region(self, hmap, tabletop_plane, object_cloud_fn):
        fp = pl.object_footprint(object_cloud_fn("mug"), tabletop_plane)
        small = pl.footprint_keep_out(hmap, [fp], dilate_m=0.0).sum()
        big = pl.footprint_keep_out(hmap, [fp], dilate_m=0.05).sum()
        assert big > small

    def test_multiple_footprints_union(self, hmap, tabletop_plane, object_cloud_fn):
        a = pl.object_footprint(object_cloud_fn("mug"), tabletop_plane)
        b = pl.object_footprint(object_cloud_fn("box"), tabletop_plane)
        both = pl.footprint_keep_out(hmap, [a, b])
        assert both.sum() >= pl.footprint_keep_out(hmap, [a]).sum()

    def test_empty_footprint_list_blocks_nothing(self, hmap):
        assert not pl.footprint_keep_out(hmap, []).any()

    def _projected(self, hmap, tabletop, tabletop_plane, **kw):
        mask = tabletop["seg"] == tabletop["label_map"]["banana"]
        depth = float(np.median(tabletop["depth"][mask]))
        return pl.projected_occlusion_keep_out(
            hmap, tabletop_plane, tabletop["K"], mask,
            target_depth_m=depth, object_height_m=0.20, **kw
        )

    def test_projected_keep_out_blocks_the_camera_side(
        self, hmap, tabletop, tabletop_plane, object_cloud_fn
    ):
        """Anywhere an object could plausibly be released and still hide the target.

        Bounded at 30 cm deliberately. Further toward the camera than that, a
        20 cm column projects *below* the target rather than over it and stops
        occluding, so blocking it would be over-blocking — the region is a
        truncated cone, not a half-plane.
        """
        keep = self._projected(hmap, tabletop, tabletop_plane, object_radius_m=0.04)
        target = pl.object_footprint(object_cloud_fn("banana"), tabletop_plane)
        toward = tabletop_plane.camera_xy() - target.centroid_xy
        toward = toward / np.linalg.norm(toward)

        for distance in (0.05, 0.10, 0.15, 0.20, 0.30):
            rc = hmap.to_cell(target.centroid_xy + toward * distance)
            assert hmap.in_bounds(rc)[0]
            r, c = rc[0]
            assert keep[r, c], f"{distance*100:.0f} cm in front of the target is not blocked"

    def test_projected_keep_out_leaves_the_far_side_open(
        self, hmap, tabletop, tabletop_plane, object_cloud_fn
    ):
        """Behind the target projects onto it but hides nothing."""
        keep = self._projected(hmap, tabletop, tabletop_plane)
        target = pl.object_footprint(object_cloud_fn("banana"), tabletop_plane)
        away = target.centroid_xy - tabletop_plane.camera_xy()
        away = away / max(np.linalg.norm(away), 1e-9)
        rc = hmap.to_cell(target.centroid_xy + away * 0.25)
        if hmap.in_bounds(rc)[0]:
            r, c = rc[0]
            assert not keep[r, c]

    def test_object_radius_widens_the_blocked_region(
        self, hmap, tabletop, tabletop_plane
    ):
        """Sampling only the centre line lets a wide object slip past it."""
        thin = self._projected(hmap, tabletop, tabletop_plane, object_radius_m=0.0)
        wide = self._projected(hmap, tabletop, tabletop_plane, object_radius_m=0.06)
        assert wide.sum() > thin.sum()

    def test_a_taller_object_is_blocked_over_more_ground(
        self, hmap, tabletop, tabletop_plane
    ):
        short = self._projected(hmap, tabletop, tabletop_plane)
        mask = tabletop["seg"] == tabletop["label_map"]["banana"]
        tall = pl.projected_occlusion_keep_out(
            hmap, tabletop_plane, tabletop["K"], mask,
            target_depth_m=float(np.median(tabletop["depth"][mask])),
            object_height_m=0.02,
        )
        assert short.sum() >= tall.sum()


class TestFindPlacement:
    def _setup(self, scenario, obj):
        scene = ss.build(scenario)
        cloud = _scene_cloud(scene)
        plane = pl.fit_support_plane(cloud)
        hmap = pl.build_height_map(cloud, plane, cell_m=0.005)
        fp = pl.object_footprint(_object_cloud(scene, obj), plane)
        return scene, cloud, plane, hmap, fp

    def test_finds_a_spot_on_an_open_table(self):
        _, _, _, hmap, fp = self._setup("open_table", "mug")
        placement = pl.find_placement(hmap, fp)
        assert placement is not None
        assert placement.clearance_m >= fp.radius_m + pl.DEFAULT_MARGIN_M

    def test_returns_none_on_a_crowded_table(self):
        _, _, _, hmap, fp = self._setup("crowded_table", "blk_3_2")
        assert pl.find_placement(hmap, fp) is None

    def test_returns_none_for_an_object_bigger_than_any_gap(self):
        _, _, _, hmap, fp = self._setup("open_table", "mug")
        huge = pl.Footprint(
            centroid_xy=fp.centroid_xy,
            half_extent=np.array([2.0, 2.0]),
            yaw=0.0,
            top_m=0.1,
            bottom_m=0.0,
        )
        assert pl.find_placement(hmap, huge) is None

    def test_chosen_cell_really_has_the_required_clearance(self):
        from scipy.ndimage import distance_transform_edt

        _, _, _, hmap, fp = self._setup("occluded_target", "mug")
        placement = pl.find_placement(hmap, fp, margin_m=pl.DEFAULT_MARGIN_M)
        assert placement is not None
        dist = distance_transform_edt(pl.free_space(hmap)) * hmap.cell_m
        r, c = placement.cell
        assert dist[r, c] >= fp.radius_m + pl.DEFAULT_MARGIN_M

    def test_respects_keep_out(self):
        _, _, plane, hmap, fp = self._setup("open_table", "mug")
        free = pl.find_placement(hmap, fp)
        assert free is not None
        everywhere = np.ones(hmap.shape, dtype=bool)
        assert pl.find_placement(hmap, fp, keep_out=everywhere) is None

    def test_keep_out_pushes_the_answer_elsewhere(self):
        _, _, _, hmap, fp = self._setup("open_table", "mug")
        base = pl.find_placement(hmap, fp)
        block = np.zeros(hmap.shape, dtype=bool)
        r, c = base.cell
        block[max(r - 30, 0):r + 30, max(c - 30, 0):c + 30] = True
        moved = pl.find_placement(hmap, fp, keep_out=block)
        assert moved is not None and moved.cell != base.cell

    def test_prefers_the_nearest_valid_cell(self):
        _, _, _, hmap, fp = self._setup("occluded_target", "mug")
        near = np.array([0.0, 0.9])
        placement = pl.find_placement(hmap, fp, prefer_near_xy=near)
        assert placement is not None
        # No other valid cell may be closer to the anchor than the one chosen.
        from scipy.ndimage import distance_transform_edt

        free = pl.free_space(hmap)
        dist = distance_transform_edt(free) * hmap.cell_m
        valid = free & (dist >= fp.radius_m + pl.DEFAULT_MARGIN_M)
        rows, cols = hmap.shape
        rr, cc = np.mgrid[0:rows, 0:cols]
        centres = hmap.to_xy(np.stack([rr.ravel(), cc.ravel()], -1)).reshape(rows, cols, 2)
        d = np.linalg.norm(centres - near, axis=-1)
        assert placement.travel_m == pytest.approx(np.min(d[valid]), abs=1e-9)

    def test_respects_the_workspace_radius(self):
        _, _, _, hmap, fp = self._setup("open_table", "mug")
        far = (np.array([9.0, 9.0]), 0.05)
        assert pl.find_placement(hmap, fp, workspace=far) is None

    def test_reports_candidate_count(self):
        _, _, _, hmap, fp = self._setup("open_table", "mug")
        placement = pl.find_placement(hmap, fp)
        assert placement.n_candidates > 1

    def test_describe_is_json_safe(self):
        import json

        _, _, _, hmap, fp = self._setup("open_table", "mug")
        placement = pl.find_placement(hmap, fp)
        assert "clearance_cm" in json.loads(json.dumps(placement.describe()))


class TestPlacePose:
    @pytest.fixture
    def setup(self, tabletop, tabletop_cloud, tabletop_plane, object_cloud_fn):
        plane = tabletop_plane
        obj = object_cloud_fn("bottle")
        fp = pl.object_footprint(obj, plane)
        # A top-down grasp above the bottle, expressed in the camera frame.
        grasp = np.eye(4)
        grasp[:3, :3] = np.column_stack(
            [plane.rotation[:, 0], -plane.rotation[:, 1], -plane.rotation[:, 2]]
        )
        grasp[:3, 3] = plane.to_camera(
            np.array([[fp.centroid_xy[0], fp.centroid_xy[1], fp.top_m + 0.02]])
        )[0]
        return plane, obj, fp, grasp

    def test_place_is_a_pure_translation(self, setup, tabletop_cloud):
        plane, obj, fp, grasp = setup
        place = pl.plan_place(grasp, obj, tabletop_cloud, plane)
        assert place is not None
        assert place.pose[:3, :3] == pytest.approx(grasp[:3, :3], abs=1e-12)

    def test_object_bottom_lands_on_the_surface(self, setup, tabletop_cloud):
        plane, obj, fp, grasp = setup
        place = pl.plan_place(grasp, obj, tabletop_cloud, plane, release_gap_m=0.005)
        delta = place.pose[:3, 3] - grasp[:3, 3]
        heights = plane.height_of(obj + delta)
        assert np.percentile(heights, 2) == pytest.approx(0.005, abs=1e-3)

    def test_object_actually_moves(self, setup, tabletop_cloud):
        plane, obj, fp, grasp = setup
        place = pl.plan_place(grasp, obj, tabletop_cloud, plane)
        assert place.travel_m > 0.02

    def test_place_delta_is_exact_for_a_requested_target(self, setup):
        plane, obj, fp, _ = setup
        target = fp.centroid_xy + np.array([0.1, -0.05])
        delta = pl.place_delta(plane, fp, target, release_gap_m=0.0)
        moved_xy = plane.to_table(obj + delta)[:, :2]
        centre = pl.object_footprint(obj + delta, plane).centroid_xy
        assert centre == pytest.approx(target, abs=1e-6)
        assert moved_xy.shape[1] == 2

    def test_waypoints_are_named_and_ordered(self, setup, tabletop_cloud):
        plane, obj, fp, grasp = setup
        place = pl.plan_place(grasp, obj, tabletop_cloud, plane)
        names = [n for n, _ in place.waypoints]
        assert names == ["pre_grasp", "grasp", "lift", "pre_place", "place", "retreat"]

    def test_lift_is_along_the_table_normal(self, setup, tabletop_cloud):
        plane, obj, fp, grasp = setup
        place = pl.plan_place(grasp, obj, tabletop_cloud, plane, lift_m=0.12)
        wp = dict(place.waypoints)
        rise = wp["lift"][:3, 3] - wp["grasp"][:3, 3]
        assert rise == pytest.approx(0.12 * plane.normal, abs=1e-9)

    def test_pre_grasp_retreats_along_the_approach(self, setup, tabletop_cloud):
        plane, obj, fp, grasp = setup
        place = pl.plan_place(grasp, obj, tabletop_cloud, plane, retreat_m=0.10)
        wp = dict(place.waypoints)
        back = wp["pre_grasp"][:3, 3] - wp["grasp"][:3, 3]
        assert back == pytest.approx(-0.10 * grasp[:3, 2], abs=1e-9)

    def test_as_dict_round_trips(self, setup, tabletop_cloud):
        plane, obj, fp, grasp = setup
        place = pl.plan_place(grasp, obj, tabletop_cloud, plane, gripper="franka_panda")
        back = pl.PlacePose.from_dict(place.as_dict())
        assert back.pose == pytest.approx(place.pose)
        assert back.gripper == "franka_panda"
        assert len(back.waypoints) == len(place.waypoints)

    def test_as_dict_is_json_serialisable(self, setup, tabletop_cloud):
        import json

        plane, obj, fp, grasp = setup
        place = pl.plan_place(grasp, obj, tabletop_cloud, plane)
        assert json.loads(json.dumps(place.as_dict()))["width"] == place.width

    def test_summary_mentions_travel(self, setup, tabletop_cloud):
        plane, obj, fp, grasp = setup
        place = pl.plan_place(grasp, obj, tabletop_cloud, plane)
        assert "travel" in place.summary()

    def test_returns_none_when_there_is_nowhere_to_put_it(self):
        scene = ss.build("crowded_table")
        cloud = _scene_cloud(scene)
        plane = pl.fit_support_plane(cloud)
        obj = _object_cloud(scene, "blk_3_2")
        grasp = np.eye(4)
        grasp[:3, 3] = obj.mean(axis=0)
        assert pl.plan_place(grasp, obj, cloud, plane) is None

    def test_keep_out_is_honoured_end_to_end(self, setup, tabletop_cloud, tabletop_plane):
        plane, obj, fp, grasp = setup
        hmap = pl.build_height_map(tabletop_cloud, plane, cell_m=0.005)
        blocked = np.ones(hmap.shape, dtype=bool)
        assert (
            pl.plan_place(grasp, obj, tabletop_cloud, plane, keep_out=blocked, hmap=hmap)
            is None
        )
