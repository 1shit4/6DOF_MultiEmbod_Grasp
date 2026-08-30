"""Instance identity and obstruction reasoning.

Two things are being defended here. That an object keeps its id when the scene
is re-perceived, including when a look-alike sits next to it — a loop whose ids
drift will move the wrong object and never notice. And that "in the way" covers
all three ways an object can be in the way, not just the visible one.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import collision as col
import placement as pl
import scene_registry as sr
import synth_scene as ss

REAL = Path(__file__).resolve().parents[2] / "GraspGenX/assets/sample_data/real_world"
needs_real = pytest.mark.skipif(not REAL.is_dir(), reason="sample scenes not present")


def _registry(scene, iteration=0, registry=None):
    reg = registry or sr.SceneRegistry()
    reg.update_from_segmentation(
        scene["depth"], scene["K"], scene["seg"], iteration, scene["label_map"]
    )
    return reg


def _render(spec, scene, keep):
    """Re-render `scene`'s spec keeping only the named objects."""
    sub = ss.SceneSpec([o for o in spec.objects if o.name in keep], spec.table_extent)
    out = ss.render(sub, scene["K"], scene["T_cam_world"])
    return {**out, "K": scene["K"], "T_cam_world": scene["T_cam_world"],
            "spec": sub, "label_map": scene["label_map"]}


@pytest.fixture(scope="module")
def registry(tabletop):
    return _registry(tabletop)


@pytest.fixture(scope="module")
def target_id(registry):
    inst = registry.resolve_target("banana")
    assert inst is not None
    return inst.id


class TestBuilding:
    def test_finds_every_object(self, registry, tabletop):
        assert len(registry.instances) == len(tabletop["label_map"])

    def test_ids_are_well_formed(self, registry):
        assert all(i.startswith("obj_") for i in registry.instances)

    def test_labels_come_from_the_label_map(self, registry):
        assert {i.label for i in registry.instances.values()} == {
            "banana", "bottle", "mug", "box"
        }

    def test_fits_a_plane_and_a_height_map(self, registry):
        assert registry.plane is not None and registry.plane.inlier_ratio > 0.3
        assert registry.hmap is not None

    def test_centroids_are_consistent_between_frames(self, registry):
        for inst in registry.instances.values():
            back = registry.plane.to_camera(inst.centroid_table[None, :])[0]
            assert back == pytest.approx(inst.centroid_cam, abs=1e-9)

    def test_drops_fragments_below_the_point_floor(self, tabletop):
        reg = sr.SceneRegistry(min_points=10_000_000)
        reg.update_from_segmentation(
            tabletop["depth"], tabletop["K"], tabletop["seg"], 0
        )
        assert reg.instances == {}
        assert reg.plane is not None, "the table is still there even with no objects"

    def test_unnamed_labels_get_a_fallback_name(self, tabletop):
        reg = sr.SceneRegistry()
        reg.update_from_segmentation(tabletop["depth"], tabletop["K"], tabletop["seg"], 0)
        assert all(i.label.startswith("label_") for i in reg.instances.values())

    def test_from_patches_matches_from_segmentation(self, tabletop):
        patches = [
            (name, tabletop["seg"] == label)
            for name, label in tabletop["label_map"].items()
        ]
        reg = sr.SceneRegistry()
        reg.update_from_patches(tabletop["depth"], tabletop["K"], patches, 0)
        assert len(reg.instances) == len(tabletop["label_map"])

    def test_describe_is_json_safe(self, registry):
        assert json.loads(json.dumps(registry.describe()))["n_instances"] == 4


class TestIdentity:
    def test_ids_survive_re_perception(self, tabletop):
        reg = _registry(tabletop, 0)
        before = {i.id: i.label for i in reg.instances.values()}
        _registry(tabletop, 1, registry=reg)
        assert {i.id: i.label for i in reg.instances.values()} == before

    def test_first_seen_is_preserved(self, tabletop):
        reg = _registry(tabletop, 0)
        _registry(tabletop, 5, registry=reg)
        assert all(i.first_seen == 0 for i in reg.instances.values())
        assert all(i.last_seen == 5 for i in reg.instances.values())

    def test_a_small_move_keeps_the_id(self, tabletop):
        reg = _registry(tabletop, 0)
        mug_id = next(i.id for i in reg.instances.values() if i.label == "mug")
        spec = tabletop["spec"]
        nudged = spec.replace("mug", spec.by_name("mug").primitive.moved_to((0.08, 0.065)))
        _registry(
            {**tabletop, "spec": nudged,
             **ss.render(nudged, tabletop["K"], tabletop["T_cam_world"])},
            1, registry=reg,
        )
        assert mug_id in reg.instances
        assert reg.instances[mug_id].label == "mug"

    def test_a_large_move_creates_a_new_id(self, tabletop):
        reg = _registry(tabletop, 0)
        mug_id = next(i.id for i in reg.instances.values() if i.label == "mug")
        spec = tabletop["spec"]
        moved = spec.replace("mug", spec.by_name("mug").primitive.moved_to((-0.32, -0.22)))
        _registry(
            {**tabletop, "spec": moved,
             **ss.render(moved, tabletop["K"], tabletop["T_cam_world"])},
            1, registry=reg,
        )
        assert mug_id not in reg.instances, "a 40 cm jump is a new object, not a nudge"

    def test_removing_an_object_drops_its_id(self, tabletop):
        reg = _registry(tabletop, 0)
        bottle_id = next(i.id for i in reg.instances.values() if i.label == "bottle")
        _registry(_render(tabletop["spec"], tabletop, {"banana", "mug", "box"}), 1, reg)
        assert bottle_id not in reg.instances
        assert len(reg.instances) == 3

    def test_two_identical_objects_do_not_swap(self):
        """The case that makes name-based planning unusable."""
        scene = ss.build("two_identical_bottles")
        reg = _registry(scene, 0)
        ids = {i.label: i.id for i in reg.instances.values()}
        positions = {i.id: i.centroid_table[:2].copy() for i in reg.instances.values()}

        spec = scene["spec"]
        nudged = spec.replace(
            "bottle_a", spec.by_name("bottle_a").primitive.moved_to((0.01, 0.04))
        )
        _registry(
            {**scene, "spec": nudged,
             **ss.render(nudged, scene["K"], scene["T_cam_world"])},
            1, registry=reg,
        )
        assert {i.label: i.id for i in reg.instances.values()} == ids
        # bottle_a moved; bottle_b did not.
        assert np.linalg.norm(
            reg.instances[ids["bottle_a"]].centroid_table[:2] - positions[ids["bottle_a"]]
        ) > 0.01
        assert np.linalg.norm(
            reg.instances[ids["bottle_b"]].centroid_table[:2] - positions[ids["bottle_b"]]
        ) < 0.005

    def test_new_ids_never_reuse_a_retired_one(self, tabletop):
        reg = _registry(tabletop, 0)
        seen = set(reg.instances)
        _registry(_render(tabletop["spec"], tabletop, {"banana"}), 1, reg)
        _registry(tabletop, 2, registry=reg)
        assert len(set(reg.instances) - seen) >= 1

    @needs_real
    def test_tracks_a_static_object_across_two_real_captures(self):
        """Real data, where label ids are not stable but positions are.

        The Cheez-It box is `obj_2` in scene 00 and `obj_1` in scene 01 while
        sitting a few millimetres from where it was. Any identity scheme keyed on
        the label gets this wrong; keying on position gets it right.
        """
        from PIL import Image

        def load(n):
            d = REAL / n
            meta = json.loads((d / "meta_data.json").read_text())
            return {
                "depth": np.load(d / "depth.npy"),
                "K": np.array(meta["intrinsics"]),
                "seg": np.array(Image.open(d / "seg.png")),
                "label_map": {},
            }

        a, b = load("00"), load("01")
        reg = sr.SceneRegistry()
        reg.update_from_segmentation(a["depth"], a["K"], a["seg"], 0)
        before = reg.positions()
        reg.update_from_segmentation(b["depth"], b["K"], b["seg"], 1)

        kept = set(before) & set(reg.instances)
        assert kept, "no object was tracked between the two real captures"
        static = [
            pid for pid in kept
            if np.linalg.norm(reg.instances[pid].centroid_table[:2] - before[pid][:2]) < 0.02
        ]
        assert static, "objects that did not move must keep their ids"


class TestVisibility:
    def test_an_occluded_target_scores_low(self, registry, target_id):
        assert registry.get(target_id).visibility < 0.6

    def test_unoccluded_objects_score_high(self, registry, target_id):
        for inst in registry.instances.values():
            if inst.id != target_id:
                assert inst.visibility > 0.9, inst.label

    def test_clearing_the_blockers_restores_visibility(self, tabletop, target_id):
        reg = _registry(_render(tabletop["spec"], tabletop, {"banana", "box"}), 0)
        banana = next(i for i in reg.instances.values() if i.label == "banana")
        assert banana.visibility > 0.9

    def test_occluded_footprint_is_flagged_unreliable(self, registry, target_id):
        assert not registry.get(target_id).footprint_is_reliable

    def test_a_cleared_target_has_a_correct_footprint(self, tabletop):
        """The measure has to actually gate something: 4x3 cm becomes 15x4 cm."""
        reg = _registry(_render(tabletop["spec"], tabletop, {"banana", "box"}), 0)
        banana = next(i for i in reg.instances.values() if i.label == "banana")
        assert banana.footprint_is_reliable
        truth = tabletop["spec"].by_name("banana").primitive.size[:2]
        got = np.sort(banana.footprint.half_extent * 2)
        assert got == pytest.approx(np.sort(truth), abs=0.02)


class TestBlocking:
    def test_reports_occlusion_while_the_target_is_hidden(self, registry, target_id):
        blockers = registry.blocking_objects(target_id)
        assert {b.object_id for b in blockers} == {
            i.id for i in registry.instances.values() if i.label in ("bottle", "mug")
        }
        assert all("occlusion" in b.reasons for b in blockers)

    def test_the_distractor_never_blocks(self, registry, target_id):
        blockers = registry.blocking_objects(target_id)
        box_id = next(i.id for i in registry.instances.values() if i.label == "box")
        assert box_id not in {b.object_id for b in blockers}

    def test_partial_clearing_still_reports_the_remaining_occluder(self, tabletop):
        """Clearing the first blocker must not end the loop while a second remains."""
        reg = _registry(_render(tabletop["spec"], tabletop, {"banana", "mug", "box"}), 0)
        banana = next(i for i in reg.instances.values() if i.label == "banana")
        mug_id = next(i.id for i in reg.instances.values() if i.label == "mug")

        assert not banana.footprint_is_reliable, "the mug still cuts the banana off"
        assert mug_id in {b.object_id for b in reg.blocking_objects(banana.id)}

    def test_no_blockers_once_the_table_is_clear(self, tabletop):
        reg = _registry(_render(tabletop["spec"], tabletop, {"banana", "box"}), 0)
        banana = next(i for i in reg.instances.values() if i.label == "banana")
        assert reg.blocking_objects(banana.id) == []

    def test_severity_orders_the_worst_first(self, tabletop):
        reg = _registry(_render(tabletop["spec"], tabletop, {"banana", "mug", "box"}), 0)
        banana = next(i for i in reg.instances.values() if i.label == "banana")
        blockers = reg.blocking_objects(banana.id)
        severities = [b.severity for b in blockers]
        assert severities == sorted(severities, reverse=True)

    def test_blocker_describe_is_json_safe(self, registry, target_id):
        for b in registry.blocking_objects(target_id):
            assert "reasons" in json.loads(json.dumps(b.describe()))


class TestNominalGrasp:
    def test_closes_across_the_short_axis(self, tabletop):
        reg = _registry(_render(tabletop["spec"], tabletop, {"banana", "box"}), 0)
        banana = next(i for i in reg.instances.values() if i.label == "banana")
        pose = reg.nominal_grasp(banana.id)
        assert pose is not None

        fp = banana.footprint
        long_axis = np.array([np.cos(fp.yaw), np.sin(fp.yaw)])
        if fp.half_extent[0] < fp.half_extent[1]:
            long_axis = np.array([-np.sin(fp.yaw), np.cos(fp.yaw)])
        closing_table = reg.plane.rotation[:, :2].T @ pose[:3, 0]
        cos = abs(float(np.dot(closing_table / np.linalg.norm(closing_table), long_axis)))
        assert cos < 0.3, "the jaw must close across the object, not along it"

    def test_approaches_along_the_table_normal(self, tabletop):
        reg = _registry(_render(tabletop["spec"], tabletop, {"banana", "box"}), 0)
        banana = next(i for i in reg.instances.values() if i.label == "banana")
        pose = reg.nominal_grasp(banana.id)
        assert pose[:3, 2] == pytest.approx(-reg.plane.normal, abs=1e-9)

    def test_is_a_valid_rigid_transform(self, registry, target_id):
        pose = registry.nominal_grasp(target_id)
        R = pose[:3, :3]
        assert R.T @ R == pytest.approx(np.eye(3), abs=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)

    def test_fingertips_land_on_the_object(self, tabletop):
        """The pose is at the base, so the fingertips are a hand-length ahead."""
        from perception3d import gripper_finger_points

        reg = _registry(_render(tabletop["spec"], tabletop, {"banana", "box"}), 0)
        banana = next(i for i in reg.instances.values() if i.label == "banana")
        pose = reg.nominal_grasp(banana.id)
        width, fingertip = sr._gripper_geometry("franka_panda")
        tips = gripper_finger_points(pose, width, fingertip)
        heights = reg.plane.height_of(tips)
        assert np.all(heights < banana.footprint.top_m + 0.02)
        assert np.all(heights > -0.01)


class TestSupportingQueries:
    def test_positions_and_moved_since(self, tabletop):
        reg = _registry(tabletop, 0)
        before = reg.positions()
        spec = tabletop["spec"]
        moved = spec.replace("mug", spec.by_name("mug").primitive.moved_to((0.09, 0.05)))
        _registry(
            {**tabletop, "spec": moved,
             **ss.render(moved, tabletop["K"], tabletop["T_cam_world"])},
            1, registry=reg,
        )
        shifted = dict(reg.moved_since(before, thresh_m=0.02))
        mug_id = next(i.id for i in reg.instances.values() if i.label == "mug")
        assert mug_id in shifted

    def test_moved_since_can_exclude_untrustworthy_centroids(self, tabletop):
        """Uncovering an object shifts its apparent centre without moving it.

        Measured at 2.3 cm on the banana when the mug beside it moved — past any
        sane movement threshold, so the evaluator would report phantom
        collateral damage without the visibility filter.
        """
        reg = _registry(tabletop, 0)
        before = reg.positions()
        spec = tabletop["spec"]
        moved = spec.replace("mug", spec.by_name("mug").primitive.moved_to((0.09, 0.05)))
        _registry(
            {**tabletop, "spec": moved,
             **ss.render(moved, tabletop["K"], tabletop["T_cam_world"])},
            1, registry=reg,
        )
        banana_id = next(i.id for i in reg.instances.values() if i.label == "banana")
        assert banana_id in dict(reg.moved_since(before, thresh_m=0.02))
        assert banana_id not in dict(
            reg.moved_since(before, thresh_m=0.02, min_visibility=0.95)
        )

    def test_moved_since_ignores_vanished_objects(self, tabletop):
        reg = _registry(tabletop, 0)
        before = reg.positions()
        _registry(_render(tabletop["spec"], tabletop, {"banana"}), 1, reg)
        assert reg.moved_since(before) == []

    def test_scene_cloud_excluding_drops_the_object(self, registry, target_id):
        full = registry.scene_cloud_excluding()
        without = registry.scene_cloud_excluding(target_id)
        assert len(without) < len(full)

    def test_scene_cloud_requires_an_observation(self):
        with pytest.raises(RuntimeError, match="no observation"):
            sr.SceneRegistry().scene_cloud_excluding()

    def test_keep_out_covers_the_target_and_its_approach(self, registry, target_id):
        keep = registry.keep_out_for(target_id)
        assert keep.shape == registry.hmap.shape
        assert keep.any()
        r, c = registry.hmap.to_cell(registry.get(target_id).footprint.centroid_xy)[0]
        assert keep[r, c]

    def test_keep_out_also_blocks_the_moving_object_origin(self, registry, target_id):
        mover = next(i.id for i in registry.instances.values() if i.label == "box")
        base = registry.keep_out_for(target_id)
        with_mover = registry.keep_out_for(target_id, moving_id=mover)
        assert with_mover.sum() > base.sum()

    def test_get_reports_what_exists(self, registry):
        with pytest.raises(KeyError, match="obj_999"):
            registry.get("obj_999")


class TestDescriptors:
    def test_unique_labels_get_a_plain_descriptor(self, registry):
        assert registry.get("obj_001").descriptor.startswith("the ")

    def test_duplicates_are_disambiguated(self):
        reg = _registry(ss.build("two_identical_bottles"))
        # bottle_a and bottle_b carry different labels from the seg map, so force
        # the collision the registry is meant to handle.
        for inst in reg.instances.values():
            if inst.label.startswith("bottle"):
                inst.label = "bottle"
        reg._assign_descriptors()
        bottles = [i for i in reg.instances.values() if i.label == "bottle"]
        assert len({b.descriptor for b in bottles}) == len(bottles)
        assert any("leftmost" in b.descriptor for b in bottles)
        assert any("rightmost" in b.descriptor for b in bottles)

    def test_prompt_table_lists_every_instance(self, registry, target_id):
        table = registry.as_prompt_table(target_id)
        for pid in registry.instances:
            assert pid in table
        assert "(target)" in table

    def test_prompt_table_handles_an_empty_scene(self):
        assert "no objects" in sr.SceneRegistry().as_prompt_table()


class TestResolveTarget:
    def test_resolves_an_unambiguous_name(self, registry):
        assert registry.resolve_target("banana").label == "banana"

    def test_returns_none_for_an_absent_object(self, registry):
        assert registry.resolve_target("elephant") is None

    def test_spatial_words_break_a_tie(self):
        reg = _registry(ss.build("two_identical_bottles"))
        for inst in reg.instances.values():
            if inst.label.startswith("bottle"):
                inst.label = "bottle"
        left = reg.resolve_target("the left bottle")
        right = reg.resolve_target("the right bottle")
        assert left is not None and right is not None and left.id != right.id
        assert left.centroid_table[0] < right.centroid_table[0]

    def test_falls_back_to_the_largest_when_ambiguous(self):
        reg = _registry(ss.build("two_identical_bottles"))
        for inst in reg.instances.values():
            if inst.label.startswith("bottle"):
                inst.label = "bottle"
        assert reg.resolve_target("bottle") is not None


class TestCollisionAttribution:
    """Which object is in the way, not merely that something is.

    `scene_cloud_excluding` flattens the scene into an anonymous point array, so
    the collision filter knew a grasp was fouled and could never say by what.
    That is the difference between "no grasp found for obj_002" — which the
    planner cannot act on — and "obj_003 is in the way of obj_002", which names
    the thing to move.

    Note what this measures: proximity to the hand's *surface*. A grasp
    deliberately does not touch the object it holds — a Panda's open jaw clears
    the banana by 2.4 cm — so an object being grasped correctly registers
    nothing. That is the behaviour that makes this a collision test rather than
    a containment test.
    """

    @staticmethod
    def _hand_at(inst):
        """A hand driven onto `inst`, i.e. a definite collision with it."""
        pose = np.eye(4)
        pose[:3, 3] = inst.centroid_cam
        return pose

    def test_it_names_the_obstructing_objects_worst_first(self, registry):
        """The mug's neighbourhood contains all three small objects."""
        mug = next(i for i in registry.instances.values() if i.label == "mug")
        named = registry.colliding_instances(
            self._hand_at(mug), approach_len=0.0
        )
        labels = [label for _, label, _ in named]
        assert labels[0] == "mug", named
        assert {"bottle", "banana"} <= set(labels), named
        counts = [n for _, _, n in named]
        assert counts == sorted(counts, reverse=True)

    def test_an_isolated_object_names_only_itself(self, registry):
        """The distractor box stands alone, and the measure has to show that.

        If everything named everything, attribution would be worthless — the
        planner would be told to move the whole table.
        """
        box = next(i for i in registry.instances.values() if i.label == "box")
        named = registry.colliding_instances(
            self._hand_at(box), approach_len=0.0
        )
        assert [label for _, label, _ in named] == ["box"], named

    def test_exclude_drops_the_object_being_grasped(self, registry):
        bottle = next(i for i in registry.instances.values() if i.label == "bottle")
        pose = self._hand_at(bottle)
        with_it = registry.colliding_instances(pose, approach_len=0.0)
        without = registry.colliding_instances(
            pose, exclude=(bottle.id,), approach_len=0.0
        )
        assert bottle.id in [oid for oid, _, _ in with_it]
        assert bottle.id not in [oid for oid, _, _ in without]
        # And the neighbour it actually fouls survives the exclusion.
        assert without, "excluding the grasped object hid the real obstruction"

    def test_a_clear_pose_names_nobody(self, registry):
        """No attribution means the failure was not an obstruction.

        This is the gate that stops the object-B cascade firing on failures
        nothing can be moved to fix — a jaw too small for the object, or a
        grasp that keeps slipping. Neither produces an attribution.
        """
        pose = np.eye(4)
        pose[:3, 3] = [5.0, 5.0, 5.0]  # nowhere near the table
        assert registry.colliding_instances(pose) == []


class TestReachableGrasp:
    """Ask the question the run exists to answer, instead of inferring it."""

    def test_a_clear_target_reports_a_grasp(self, tabletop):
        reg = _registry(_render(tabletop["spec"], tabletop, {"banana", "box"}), 0)
        banana = next(i for i in reg.instances.values() if i.label == "banana")
        assert reg.reachable_grasp(banana.id) is not None

    def test_it_is_a_lower_bound_not_a_verdict(self, registry, target_id):
        """One synthesised pose against the real cloud; the sampler tries 416.

        So None means "the obvious grasp is fouled", never "the target is
        unreachable" — which is the right way round, because the result is only
        ever used to talk the loop INTO trying a grasp, never out of clearing.
        """
        got = registry.reachable_grasp(target_id)
        assert got is None or got.shape == (4, 4)

    def test_an_unknown_object_does_not_raise(self, registry):
        import pytest as _pytest

        with _pytest.raises(KeyError):
            registry.reachable_grasp("obj_does_not_exist")


class TestContactReplacesProximity:
    """"In the way" now means hiding it, or touching it. Nothing else.

    *approach* swept one synthesised top-down pose and fired **zero** times
    across every recorded run. *proximity* flagged anything within half the
    width of the whole hand, on the assumption the hand must come in broadside.
    Both were deleted rather than tuned. What is left are two things whose
    removal demonstrably helps: an occluder buys perception, and something
    touching the target may resist the pick or topple when it is lifted away.
    """

    @staticmethod
    def _two(gap_y):
        spec = ss.add_objects([
            ("banana", ss.Primitive("box", (0.16, 0.045, 0.04), (0.0, 0.0, 0.0))),
            ("behind", ss.Primitive("cylinder", (0.04, 0.04, 0.09), (0.0, gap_y, 0.0))),
        ])
        K, T = ss.default_intrinsics(), ss.default_camera()
        out = ss.render(spec, K, T)
        reg = sr.SceneRegistry()
        reg.update_from_segmentation(out["depth"], K, out["seg"], 0, spec.label_map())
        banana = next(i for i in reg.instances.values() if i.label == "banana")
        behind = next(i for i in reg.instances.values() if i.label == "behind")
        return reg, banana, behind

    def test_a_separated_object_is_not_flagged(self):
        """Occludes nothing, touches nothing — so nothing to gain by moving it.

        The old approach test flagged exactly this case. Whether it actually
        blocks a grasp is now settled by the grasp search, which tries every
        direction rather than assuming top-down.
        """
        reg, banana, behind = self._two(0.10)
        assert reg.occlusion_of(banana.id).get(behind.id, 0.0) == 0.0
        assert behind.id not in {b.object_id for b in reg.blocking_objects(banana.id)}

    def test_a_touching_object_is_flagged(self):
        reg, banana, behind = self._two(0.065)
        bl = {b.object_id: b for b in reg.blocking_objects(banana.id)}
        assert behind.id in bl
        assert bl[behind.id].reasons == ["contact"]

    def test_the_gap_is_measured_between_surfaces_not_radii(self):
        """Circumscribed radii make a 16 cm banana a circle of radius 8 cm.

        By that measure a cylinder 3.75 cm away scores a gap of -3.5 cm —
        apparently interpenetrating — which is the same over-flagging the
        proximity test was deleted for.
        """
        reg, banana, behind = self._two(0.10)
        got = sr.SceneRegistry._surface_gap(banana, behind)
        assert got is not None and got > 0.0, got

    def test_no_deleted_reasons_can_come_back(self):
        reg, banana, _ = self._two(0.065)
        for b in reg.blocking_objects(banana.id):
            assert set(b.reasons) <= {"occlusion", "contact", "fouls_grasp"}

    def test_occlusion_alone_decides_while_the_target_is_hidden(self, registry, target_id):
        """The contact test needs a footprint, and a fragment has a bad one."""
        assert not registry.get(target_id).footprint_is_reliable
        for b in registry.blocking_objects(target_id):
            assert b.reasons == ["occlusion"]
