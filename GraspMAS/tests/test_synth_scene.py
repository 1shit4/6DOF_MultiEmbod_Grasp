"""The synthetic scene generator must be exact, or every test built on it lies.

These tests check the renderer against closed-form geometry rather than against
a stored image, so a change in resolution or colour scheme cannot mask a real
regression in the depth or the segmentation.
"""

import numpy as np
import pytest

import synth_scene as ss
from perception3d import unproject


class TestPrimitive:
    def test_resting_puts_the_base_on_the_table(self):
        p = ss.Primitive("cylinder", (0.04, 0.04, 0.12), (0.1, 0.2, 99.0)).resting()
        assert p.position[2] == pytest.approx(0.06)

    def test_resting_preserves_xy_and_yaw(self):
        p = ss.Primitive("box", (0.1, 0.1, 0.2), (0.3, -0.2, 5.0), yaw=0.4).resting()
        assert p.position[0] == pytest.approx(0.3)
        assert p.position[1] == pytest.approx(-0.2)
        assert p.yaw == pytest.approx(0.4)

    def test_sphere_height_is_the_diameter(self):
        assert ss.Primitive("sphere", (0.05, 0.05, 0.05), (0, 0, 0)).height == pytest.approx(0.1)

    def test_moved_to_changes_only_xy(self):
        p = ss.Primitive("box", (0.1, 0.1, 0.2), (0.0, 0.0, 0.1)).moved_to((0.25, -0.1))
        assert p.position == pytest.approx([0.25, -0.1, 0.1])

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="unknown primitive"):
            ss.Primitive("torus", (1, 1, 1), (0, 0, 0))


class TestLookAt:
    def test_is_a_rigid_transform(self):
        T = ss.look_at((0.0, -0.8, 0.5))
        R = T[:3, :3]
        assert R @ R.T == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)

    def test_target_lands_on_the_optical_axis(self):
        target = np.array([0.05, 0.1, 0.05])
        T = ss.look_at((0.0, -0.8, 0.5), target)
        p = T[:3, :3] @ target + T[:3, 3]
        assert p[0] == pytest.approx(0.0, abs=1e-9)
        assert p[1] == pytest.approx(0.0, abs=1e-9)
        assert p[2] > 0, "target must be in front of the camera"

    def test_camera_sees_the_table_from_above(self):
        T = ss.look_at((0.0, -0.85, 0.55))
        normal, offset = ss.table_plane_in_camera(T)
        # The camera centre is the origin of the camera frame, so its height
        # above the table is just the offset.
        assert offset == pytest.approx(0.55, abs=1e-9)

    def test_rejects_degenerate_up(self):
        with pytest.raises(ValueError, match="up vector"):
            ss.look_at((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))

    def test_rejects_coincident_eye_and_target(self):
        with pytest.raises(ValueError, match="coincide"):
            ss.look_at((0.1, 0.1, 0.1), (0.1, 0.1, 0.1))


class TestRender:
    def test_table_pixels_lie_exactly_on_the_ground_truth_plane(self, tabletop):
        n, off = tabletop["plane_truth"]["normal"], tabletop["plane_truth"]["offset"]
        pts = unproject(
            tabletop["depth"], tabletop["K"], mask=(tabletop["seg"] == ss.TABLE_ID)
        )
        heights = pts @ n + off
        assert np.abs(heights).max() < 1e-6

    def test_object_heights_match_their_primitives(self, tabletop):
        n, off = tabletop["plane_truth"]["normal"], tabletop["plane_truth"]["offset"]
        spec = tabletop["spec"]
        for name, label in tabletop["label_map"].items():
            pts = unproject(tabletop["depth"], tabletop["K"], mask=(tabletop["seg"] == label))
            heights = pts @ n + off
            expected = spec.by_name(name).primitive.height
            assert heights.min() >= -1e-6, f"{name} dips below the table"
            assert heights.max() == pytest.approx(expected, abs=2e-3), name

    def test_every_object_is_visible(self, tabletop):
        for name, label in tabletop["label_map"].items():
            assert (tabletop["seg"] == label).sum() > 100, f"{name} is not visible"

    def test_depth_and_segmentation_agree_on_misses(self, tabletop):
        background = tabletop["seg"] == ss.BACKGROUND_ID
        assert np.all(tabletop["depth"][background] == 0.0)
        assert np.all(tabletop["depth"][~background] > 0.0)

    def test_no_holes_inside_an_object(self, tabletop):
        """Ray casting must produce solid regions; splatting would not."""
        import cv2

        for label in tabletop["label_map"].values():
            m = (tabletop["seg"] == label).astype(np.uint8)
            filled = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            assert filled.sum() == m.sum(), "object mask has interior holes"

    def test_rgb_is_consistent_with_segmentation(self, tabletop):
        spec = tabletop["spec"]
        for name, label in tabletop["label_map"].items():
            sel = tabletop["seg"] == label
            colors = np.unique(tabletop["rgb"][sel].reshape(-1, 3), axis=0)
            assert len(colors) == 1
            assert tuple(colors[0]) == spec.by_name(name).primitive.color

    def test_occluded_target_really_is_occluded(self):
        """The scenario is worthless if the target is fully visible anyway."""
        spec = ss.occluded_target_scene()
        K, T = ss.default_intrinsics(), ss.default_camera()

        full = ss.render(spec, K, T)
        banana_id = spec.label_map()["banana"]
        occluded_px = int((full["seg"] == banana_id).sum())

        alone = ss.SceneSpec([spec.by_name("banana")], spec.table_extent)
        clear_px = int((ss.render(alone, K, T)["seg"] == banana_id).sum())

        assert occluded_px < 0.75 * clear_px, "the blockers do not actually block"

    def test_occluded_target_stays_detectable(self):
        """Fully hidden is the wrong kind of hard — the loop could never start."""
        spec = ss.occluded_target_scene()
        seg = ss.render(spec, ss.default_intrinsics(), ss.default_camera())["seg"]
        assert (seg == spec.label_map()["banana"]).sum() > 300

    def test_nothing_interpenetrates_in_any_scenario(self):
        """Overlapping solids would make every downstream footprint wrong."""
        for name, factory in ss.SCENARIOS.items():
            objs = factory().objects
            for i in range(len(objs)):
                for j in range(i + 1, len(objs)):
                    a, b = objs[i].primitive, objs[j].primitive
                    # Compare inscribed radii: a conservative lower bound on the
                    # separation two convex solids need.
                    ra = float(np.min(a.size[:2])) / 2.0
                    rb = float(np.min(b.size[:2])) / 2.0
                    d = float(np.linalg.norm(a.position[:2] - b.position[:2]))
                    assert d >= ra + rb - 1e-9, f"{name}: {objs[i].name}/{objs[j].name}"

    def test_is_deterministic(self):
        a = ss.build("occluded_target", height=120, width=160)
        b = ss.build("occluded_target", height=120, width=160)
        assert np.array_equal(a["depth"], b["depth"])
        assert np.array_equal(a["seg"], b["seg"])

    def test_resolution_is_honoured(self):
        s = ss.build("open_table", height=120, width=160)
        assert s["depth"].shape == (120, 160)
        assert s["rgb"].shape == (120, 160, 3)


class TestScenarios:
    def test_all_scenarios_build(self):
        for name in ss.SCENARIOS:
            s = ss.build(name, height=120, width=160)
            assert s["depth"].shape == (120, 160)

    def test_unknown_scenario_is_rejected(self):
        with pytest.raises(ValueError, match="unknown scenario"):
            ss.build("no_such_scene")

    def test_identical_bottles_are_geometrically_identical(self):
        spec = ss.two_identical_bottles_scene()
        a = spec.by_name("bottle_a").primitive
        b = spec.by_name("bottle_b").primitive
        assert a.size == pytest.approx(b.size)
        assert a.color == b.color

    def test_label_ids_are_unique(self):
        for name in ss.SCENARIOS:
            spec = ss.SCENARIOS[name]()
            ids = [o.label_id for o in spec.objects]
            assert len(ids) == len(set(ids))
            assert ss.TABLE_ID not in ids and ss.BACKGROUND_ID not in ids


class TestSceneSpec:
    def test_replace_swaps_one_object_only(self):
        spec = ss.occluded_target_scene()
        moved = spec.replace("mug", spec.by_name("mug").primitive.moved_to((0.3, -0.2)))
        assert moved.by_name("mug").primitive.position[:2] == pytest.approx([0.3, -0.2])
        assert moved.by_name("bottle").primitive.position == pytest.approx(
            spec.by_name("bottle").primitive.position
        )

    def test_replace_does_not_mutate_the_original(self):
        spec = ss.occluded_target_scene()
        before = spec.by_name("mug").primitive.position.copy()
        spec.replace("mug", spec.by_name("mug").primitive.moved_to((0.3, -0.2)))
        assert spec.by_name("mug").primitive.position == pytest.approx(before)

    def test_by_name_reports_what_exists(self):
        with pytest.raises(KeyError, match="banana"):
            ss.open_table_scene().by_name("banana")


class TestAffordanceTable:
    """Ground truth for the abstract-goal scenario.

    These numbers are the scenario's contract. A scene whose docstring claims an
    obstruction that measurement does not support is the mistake `occluded_target`
    already made once (CLAUDE.md §7.10), so every claim there is pinned here.
    """

    @pytest.fixture(scope="class")
    def registry(self):
        import scene_registry as sr

        c = ss.build("affordance_table")
        reg = sr.SceneRegistry()
        reg.update_from_segmentation(
            c["depth"], c["K"], c["seg"], iteration=0, label_map=c.get("label_map")
        )
        return reg

    @staticmethod
    def _id(reg, label):
        return next(i.id for i in reg.instances.values() if i.label == label)

    def test_all_four_objects_are_detected(self, registry):
        assert {i.label for i in registry.instances.values()} == {
            "knife", "bottle", "apple", "mug"
        }

    def test_the_knife_is_blocked_by_the_bottle(self, registry):
        """The right answer to "something to cut" costs work to reach."""
        blockers = registry.blocking_objects(
            self._id(registry, "knife"), gripper_name="franka_panda"
        )
        assert len(blockers) == 1
        assert registry.get(blockers[0].object_id).label == "bottle"
        assert "occlusion" in blockers[0].reasons
        assert blockers[0].occlusion_frac == pytest.approx(0.21, abs=0.06)

    def test_the_knifes_footprint_is_unreliable_while_occluded(self, registry):
        """So only occlusion is measurable — the documented gate, not a gap."""
        knife = registry.get(self._id(registry, "knife"))
        assert knife.visibility == pytest.approx(0.81, abs=0.06)
        assert not knife.footprint_is_reliable

    @pytest.mark.parametrize("label", ["apple", "mug"])
    def test_the_alternatives_are_completely_clear(self, registry, label):
        """The wrong answers are free. That asymmetry is what the scene is for."""
        inst = registry.get(self._id(registry, label))
        assert inst.visibility == pytest.approx(1.0, abs=0.02)
        assert registry.blocking_objects(inst.id, gripper_name="franka_panda") == []

    def test_every_object_is_dense_enough_to_be_a_target(self, registry):
        import scene_registry as sr

        for inst in registry.instances.values():
            assert inst.n_points >= sr.MIN_INSTANCE_POINTS


class TestAffordanceChoice:
    """Two cutters, one blocked — the scene the retarget path needs.

    A fallback that is itself obstructed tests nothing, so "the scissors are
    clear" is asserted rather than assumed.
    """

    @pytest.fixture(scope="class")
    def registry(self):
        import scene_registry as sr

        c = ss.build("affordance_choice")
        reg = sr.SceneRegistry()
        reg.update_from_segmentation(
            c["depth"], c["K"], c["seg"], iteration=0, label_map=c.get("label_map")
        )
        return reg

    @staticmethod
    def _id(reg, label):
        return next(i.id for i in reg.instances.values() if i.label == label)

    def test_both_cutters_are_present(self, registry):
        labels = {i.label for i in registry.instances.values()}
        assert labels == {"knife", "bottle", "scissors"}

    def test_the_knife_is_blocked(self, registry):
        blockers = registry.blocking_objects(
            self._id(registry, "knife"), gripper_name="franka_panda"
        )
        assert [registry.get(b.object_id).label for b in blockers] == ["bottle"]

    def test_the_scissors_are_the_free_alternative(self, registry):
        """If this ever acquires a blocker, the retarget scenario is dead."""
        scissors = registry.get(self._id(registry, "scissors"))
        assert scissors.visibility == pytest.approx(1.0, abs=0.02)
        assert registry.blocking_objects(scissors.id, gripper_name="franka_panda") == []


class TestAffordanceChoiceIsSolvable:
    """Unblocked is not the same as graspable, and the difference cost a scene.

    At 1.2 cm thick the scissors had no blockers and no reachable grasp either,
    so a run retargeting to them looked correct and then span until the
    iteration cap. Both halves are asserted.
    """

    def _run(self, target):
        import asyncio
        import tempfile
        from pathlib import Path

        from declutter import DeclutterLoop
        from execution import MutationExecutor
        from session_state import SessionState

        ex = MutationExecutor(ss.SCENARIOS["affordance_choice"]())
        state = SessionState(Path(tempfile.mkdtemp()))
        loop = DeclutterLoop(executor=ex, state=state, max_iterations=6)
        result = asyncio.run(loop.run(f"pick up the {target}", target))
        return result, state

    def test_the_scissors_need_no_clearing(self):
        result, state = self._run("scissors")
        assert result.status == "success"
        assert len(state.iterations) == 1

    def test_the_knife_costs_one_removal(self):
        result, state = self._run("knife")
        assert result.status == "success"
        assert len(state.iterations) == 2
