"""Executors.

The load-bearing property is that a report describes what happened rather than
what was asked for. Most of these tests therefore set up a plan that cannot
succeed, or inject a failure, and check the report says so.
"""

import numpy as np
import pytest

import placement as pl
import scene_registry as sr
import synth_scene as ss
from execution import (
    ExecutionReport,
    Executor,
    MutationExecutor,
    Observation,
    PickPlacePlan,
    ReplayExecutor,
    RobotExecutor,
)
from perception3d import unproject


@pytest.fixture
def executor():
    return MutationExecutor(ss.occluded_target_scene())


def _registry_for(executor, iteration=0):
    obs = executor.capture(iteration)
    reg = sr.SceneRegistry()
    reg.update_from_segmentation(obs.depth, obs.K, obs.seg, iteration, obs.label_map)
    return reg


def _plan_for(executor, name, keep_clear=None, gripper="franka_panda"):
    """Build a real pick-and-place plan for `name`, using the actual geometry.

    `keep_clear` names the target that must stay unobstructed, which is what the
    loop always passes. Leaving it out is what a naive implementation would do,
    and `TestKeepOutMatters` shows what that costs.
    """
    reg = _registry_for(executor)
    inst = next(i for i in reg.instances.values() if i.label == name)
    grasp_pose = reg.nominal_grasp(inst.id, gripper_name=gripper)

    keep_out = None
    if keep_clear is not None:
        target = next(i for i in reg.instances.values() if i.label == keep_clear)
        keep_out = reg.keep_out_for(target.id, moving_id=inst.id)

    place = pl.plan_place(
        grasp_pose, inst.cloud, reg.scene_cloud_excluding(), reg.plane,
        gripper=gripper, keep_out=keep_out, hmap=reg.hmap,
    )
    assert place is not None, f"no placement available for {name}"

    return PickPlacePlan(
        object_id=inst.id,
        grasp={"pose": grasp_pose.tolist(), "score": 0.9, "gripper": gripper, "width": 0.08},
        place=place.as_dict(),
        gripper=gripper,
        object_label=name,
    )


class TestProtocol:
    def test_mutation_executor_satisfies_the_interface(self, executor):
        assert isinstance(executor, Executor)

    def test_replay_executor_satisfies_the_interface(self, executor):
        assert isinstance(ReplayExecutor([executor.capture()]), Executor)

    def test_report_validates_its_status(self):
        with pytest.raises(ValueError, match="status must be"):
            ExecutionReport(status="probably fine")

    def test_report_validates_its_stage(self):
        with pytest.raises(ValueError, match="stage must be"):
            ExecutionReport(stage_reached="somewhere")

    def test_report_describe_is_json_safe(self):
        import json

        r = ExecutionReport(applied_translation=np.array([0.1, 0.0, 0.0]))
        assert json.loads(json.dumps(r.describe()))["status"] == "ok"


class TestCapture:
    def test_returns_a_complete_observation(self, executor):
        obs = executor.capture(iteration=3)
        assert obs.rgb.shape[:2] == obs.depth.shape
        assert obs.has_ground_truth
        assert obs.iteration == 3
        assert set(obs.label_map) == {"banana", "bottle", "mug", "box"}

    def test_capture_reflects_the_current_scene(self, executor):
        before = executor.capture()
        plan = _plan_for(executor, "bottle")
        executor.execute_pick_place(plan)
        after = executor.capture()
        assert not np.array_equal(before.depth, after.depth)

    def test_describe_is_json_safe(self, executor):
        import json

        assert json.loads(json.dumps(executor.capture().describe()))["has_seg"] is True


class TestPickPlacePlan:
    def test_translation_is_the_pose_difference(self, executor):
        plan = _plan_for(executor, "bottle")
        assert plan.translation == pytest.approx(
            plan.place_pose[:3, 3] - plan.grasp_pose[:3, 3]
        )

    def test_place_preserves_orientation(self, executor):
        plan = _plan_for(executor, "bottle")
        assert plan.place_pose[:3, :3] == pytest.approx(plan.grasp_pose[:3, :3])

    def test_waypoints_survive_serialisation(self, executor):
        plan = _plan_for(executor, "bottle")
        assert [n for n, _ in plan.waypoints] == [
            "pre_grasp", "grasp", "lift", "pre_place", "place", "retreat"
        ]


class TestExecution:
    def test_moves_the_intended_object(self, executor):
        before = executor.true_position("bottle")
        plan = _plan_for(executor, "bottle")
        report = executor.execute_pick_place(plan)
        assert report.ok
        assert report.grasped_object == "bottle"
        after = executor.true_position("bottle")
        assert np.linalg.norm(after[:2] - before[:2]) > 0.02

    def test_moves_it_where_the_plan_said(self, executor):
        """Ground truth, so the executor and the planner can be checked against
        each other rather than only against themselves."""
        plan = _plan_for(executor, "bottle")
        before = executor.true_position("bottle")
        executor.execute_pick_place(plan)
        after = executor.true_position("bottle")
        expected = executor._world_from_camera(plan.translation)
        assert (after - before)[:2] == pytest.approx(expected[:2], abs=2e-3)

    def test_leaves_the_object_on_the_table(self, executor):
        plan = _plan_for(executor, "bottle")
        executor.execute_pick_place(plan)
        prim = executor.spec.by_name("bottle").primitive
        assert prim.position[2] == pytest.approx(prim.height / 2.0)

    def test_does_not_disturb_other_objects(self, executor):
        before = executor.true_positions()
        executor.execute_pick_place(_plan_for(executor, "bottle"))
        after = executor.true_positions()
        for name in ("banana", "mug", "box"):
            assert after[name] == pytest.approx(before[name])

    def test_identifies_the_object_from_the_pose_not_the_id(self, executor):
        """A hand closes on whatever is between its fingers."""
        plan = _plan_for(executor, "bottle")
        mislabelled = PickPlacePlan(
            object_id="obj_999", grasp=plan.grasp, place=plan.place,
            gripper=plan.gripper, object_label="banana",
        )
        report = executor.execute_pick_place(mislabelled)
        assert report.grasped_object == "bottle", "the pose is over the bottle"

    def test_a_grasp_that_reaches_nothing_fails(self, executor):
        plan = _plan_for(executor, "bottle")
        empty = np.asarray(plan.grasp["pose"], dtype=float)
        empty[:3, 3] += np.array([3.0, 0.0, 0.0])
        report = executor.execute_pick_place(
            PickPlacePlan("obj_001", {"pose": empty.tolist()}, plan.place)
        )
        assert report.status == "failed"
        assert report.stage_reached == "grasp"
        assert "reached nothing" in report.error

    def test_records_history(self, executor):
        executor.execute_pick_place(_plan_for(executor, "bottle"))
        assert len(executor.history) == 1
        assert executor.history[0]["report"]["grasped_object"] == "bottle"

    def test_reset_restores_the_scene(self, executor):
        before = executor.true_position("bottle")
        executor.execute_pick_place(_plan_for(executor, "bottle"))
        executor.reset()
        assert executor.true_position("bottle") == pytest.approx(before)
        assert executor.history == []

    def test_is_deterministic_under_a_seed(self):
        results = []
        for _ in range(2):
            ex = MutationExecutor(ss.occluded_target_scene(), inject=["offset"], seed=7)
            ex.execute_pick_place(_plan_for(ex, "bottle"))
            results.append(ex.true_position("bottle"))
        assert results[0] == pytest.approx(results[1])


class TestFailureInjection:
    def test_rejects_an_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown failure mode"):
            MutationExecutor(ss.occluded_target_scene(), inject=["explode"])

    def test_drop_leaves_the_object_where_it_was(self):
        ex = MutationExecutor(ss.occluded_target_scene(), inject=["drop"], seed=0)
        before = ex.true_position("bottle")
        report = ex.execute_pick_place(_plan_for(ex, "bottle"))
        assert report.status == "failed"
        assert report.stage_reached == "lift"
        assert ex.true_position("bottle") == pytest.approx(before)

    def test_offset_lands_the_object_off_target(self):
        ex = MutationExecutor(ss.occluded_target_scene(), inject=["offset"], seed=1, offset_m=0.09)
        plan = _plan_for(ex, "bottle")
        before = ex.true_position("bottle")
        report = ex.execute_pick_place(plan)
        assert report.status == "partial"
        intended = before + ex._world_from_camera(plan.translation)
        assert np.linalg.norm(ex.true_position("bottle")[:2] - intended[:2]) == pytest.approx(
            0.09, abs=1e-3
        )

    def test_short_offset_releases_early_along_the_path(self):
        """`short` misses *backwards*, not sideways.

        The random slip always cleared the target in the recorded runs, so it
        only ever exercised one half of the two-verdict design. Falling short
        leaves the object between the camera and the target, which is the other
        half: a move geometry calls a success while the target stays blocked.
        """
        # 10 cm against a ~20 cm planned travel, so the 95%-of-travel cap that
        # keeps this from becoming a `drop` is not what is being measured here.
        ex = MutationExecutor(ss.occluded_target_scene(), inject=["offset"],
                              seed=1, offset_m=0.10, offset_dir="short")
        plan = _plan_for(ex, "bottle")
        before = ex.true_position("bottle")
        report = ex.execute_pick_place(plan)

        assert report.status == "partial"
        intended = before + ex._world_from_camera(plan.translation)
        actual = ex.true_position("bottle")
        travelled = actual - before
        planned = intended - before

        # Short, not sideways: the object moves along the planned direction...
        cos = float(
            travelled[:2] @ planned[:2]
            / (np.linalg.norm(travelled[:2]) * np.linalg.norm(planned[:2]))
        )
        assert cos > 0.99
        # ...and stops 10 cm before the mark.
        assert np.linalg.norm(actual[:2] - intended[:2]) == pytest.approx(0.10, abs=1e-3)

    def test_short_offset_never_reverses_past_the_start(self):
        """Capped at 95% of the travel, so it is a short throw, not a drop.

        Landing back where it started would be indistinguishable from `drop`,
        and the evaluator would read `not_moved` instead of the intended
        "moved, and still in the way".
        """
        ex = MutationExecutor(ss.occluded_target_scene(), inject=["offset"],
                              seed=1, offset_m=5.0, offset_dir="short")
        before = ex.true_position("bottle")
        plan = _plan_for(ex, "bottle")
        ex.execute_pick_place(plan)
        travel = np.linalg.norm(ex._world_from_camera(plan.translation)[:2])
        moved = np.linalg.norm(ex.true_position("bottle")[:2] - before[:2])
        assert 0.0 < moved <= travel * 0.06, f"moved {moved:.3f} of {travel:.3f}" 

    def test_short_offset_stays_on_the_table(self):
        ex = MutationExecutor(ss.occluded_target_scene(), inject=["offset"],
                              seed=1, offset_m=0.10, offset_dir="short")
        before = ex.true_position("bottle")
        ex.execute_pick_place(_plan_for(ex, "bottle"))
        assert ex.true_position("bottle")[2] == pytest.approx(before[2], abs=1e-6)

    def test_unknown_offset_dir_is_rejected(self):
        with pytest.raises(ValueError, match="offset_dir"):
            MutationExecutor(ss.occluded_target_scene(), offset_dir="sideways")

    def test_tip_changes_the_footprint_and_height(self):
        ex = MutationExecutor(ss.occluded_target_scene(), inject=["tip"], seed=0)
        before = ex.spec.by_name("bottle").primitive
        report = ex.execute_pick_place(_plan_for(ex, "bottle"))
        after = ex.spec.by_name("bottle").primitive
        assert report.status == "partial"
        assert after.height != pytest.approx(before.height)

    def test_collateral_disturbs_a_neighbour(self):
        ex = MutationExecutor(ss.occluded_target_scene(), inject=["collateral"], seed=3)
        before = ex.true_positions()
        report = ex.execute_pick_place(_plan_for(ex, "bottle"))
        assert report.disturbed
        victim = report.disturbed[0]
        assert np.linalg.norm(
            ex.true_position(victim)[:2] - before[victim][:2]
        ) > 1e-6

    def test_wrong_object_moves_a_neighbour_instead(self):
        ex = MutationExecutor(ss.occluded_target_scene(), inject=["wrong_object"], seed=2)
        before = ex.true_positions()
        report = ex.execute_pick_place(_plan_for(ex, "bottle"))
        assert report.status == "partial"
        assert report.grasped_object != "bottle"
        assert ex.true_position("bottle") == pytest.approx(before["bottle"])
        assert np.linalg.norm(
            ex.true_position(report.grasped_object)[:2] - before[report.grasped_object][:2]
        ) > 0.01

    def test_injection_is_always_announced(self):
        for mode in ("offset", "drop", "tip", "collateral", "wrong_object"):
            ex = MutationExecutor(ss.occluded_target_scene(), inject=[mode], seed=5)
            report = ex.execute_pick_place(_plan_for(ex, "bottle"))
            text = " ".join(report.notes) + (report.error or "")
            assert "injected" in text, mode
            assert not report.ok, mode


def _declutter(executor, target_label, use_keep_out, max_steps=5):
    """Run the geometric core of the loop: clear blockers until none remain.

    No LLM, no agents — just registry, placement, executor. Returns the objects
    moved in order and the final registry.
    """
    moved = []
    for step in range(max_steps):
        reg = _registry_for(executor, step)
        target = next(i for i in reg.instances.values() if i.label == target_label)
        blockers = reg.blocking_objects(target.id)
        if not blockers:
            return moved, reg, target
        inst = reg.get(blockers[0].object_id)
        grasp_pose = reg.nominal_grasp(inst.id)
        keep_out = reg.keep_out_for(target.id, moving_id=inst.id) if use_keep_out else None
        place = pl.plan_place(
            grasp_pose, inst.cloud, reg.scene_cloud_excluding(), reg.plane,
            keep_out=keep_out, hmap=reg.hmap,
        )
        if place is None:
            moved.append(f"{inst.label}:nowhere-to-put-it")
            break
        executor.execute_pick_place(
            PickPlacePlan(inst.id, {"pose": grasp_pose.tolist()}, place.as_dict(),
                          object_label=inst.label)
        )
        moved.append(inst.label)

    reg = _registry_for(executor, max_steps)
    target = next(i for i in reg.instances.values() if i.label == target_label)
    return moved, reg, target


class TestDeclutteringEndToEnd:
    def test_clearing_the_blockers_frees_the_target(self):
        """The loop's whole premise, executed rather than asserted."""
        ex = MutationExecutor(ss.occluded_target_scene())
        moved, reg, banana = _declutter(ex, "banana", use_keep_out=True)

        assert sorted(moved) == ["bottle", "mug"], f"moved {moved}"
        assert reg.blocking_objects(banana.id) == []
        assert banana.visibility > 0.95
        assert banana.footprint_is_reliable

    def test_the_distractor_is_never_touched(self):
        ex = MutationExecutor(ss.occluded_target_scene())
        before = ex.true_position("box")
        _declutter(ex, "banana", use_keep_out=True)
        assert ex.true_position("box") == pytest.approx(before)

    def test_the_target_itself_is_never_moved(self):
        ex = MutationExecutor(ss.occluded_target_scene())
        before = ex.true_position("banana")
        _declutter(ex, "banana", use_keep_out=True)
        assert ex.true_position("banana") == pytest.approx(before)


class TestKeepOutMatters:
    """What happens when placement ignores where the target is.

    This is the difference between a loop that terminates and one that does not,
    so it is worth an explicit before/after rather than a comment.
    """

    def test_without_keep_out_the_loop_puts_the_blocker_back(self):
        ex = MutationExecutor(ss.occluded_target_scene())
        moved, reg, banana = _declutter(ex, "banana", use_keep_out=False, max_steps=4)

        assert moved.count("bottle") > 1, (
            "without a keep-out region the bottle is released back in front of "
            "the banana and has to be moved again"
        )
        assert reg.blocking_objects(banana.id), "the target is still not reachable"

    def test_with_keep_out_one_move_per_blocker_suffices(self):
        ex = MutationExecutor(ss.occluded_target_scene())
        moved, reg, banana = _declutter(ex, "banana", use_keep_out=True, max_steps=4)

        assert len(moved) == len(set(moved)) == 2, f"moved {moved}"
        assert reg.blocking_objects(banana.id) == []


class TestReplayExecutor:
    def test_walks_through_the_recording(self, executor):
        a, b = executor.capture(0), executor.capture(1)
        rep = ReplayExecutor([a, b])
        assert rep.capture().timestamp == a.timestamp
        rep.execute_pick_place(PickPlacePlan("x", {"pose": np.eye(4).tolist()}, {"pose": np.eye(4).tolist()}))
        assert rep.capture().timestamp == b.timestamp

    def test_never_claims_to_have_executed_the_plan(self, executor):
        rep = ReplayExecutor([executor.capture(), executor.capture()])
        report = rep.execute_pick_place(
            PickPlacePlan("x", {"pose": np.eye(4).tolist()}, {"pose": np.eye(4).tolist()})
        )
        assert report.status == "partial"
        assert "not executed" in " ".join(report.notes)

    def test_running_out_of_recording_fails(self, executor):
        rep = ReplayExecutor([executor.capture()])
        report = rep.execute_pick_place(
            PickPlacePlan("x", {"pose": np.eye(4).tolist()}, {"pose": np.eye(4).tolist()})
        )
        assert report.status == "failed"
        assert "no further" in report.error

    def test_can_loop(self, executor):
        rep = ReplayExecutor([executor.capture()], loop=True)
        plan = PickPlacePlan("x", {"pose": np.eye(4).tolist()}, {"pose": np.eye(4).tolist()})
        assert rep.execute_pick_place(plan).status == "partial"

    def test_rejects_an_empty_recording(self):
        with pytest.raises(ValueError, match="at least one"):
            ReplayExecutor([])

    def test_reset_rewinds(self, executor):
        a, b = executor.capture(0), executor.capture(1)
        rep = ReplayExecutor([a, b])
        rep.execute_pick_place(PickPlacePlan("x", {"pose": np.eye(4).tolist()}, {"pose": np.eye(4).tolist()}))
        rep.reset()
        assert rep.capture().timestamp == a.timestamp


class TestRobotExecutor:
    def test_refuses_to_pretend(self):
        with pytest.raises(NotImplementedError, match="documented contract"):
            RobotExecutor()

    def test_documents_what_a_backend_must_provide(self):
        doc = RobotExecutor.__doc__
        for topic in ("waypoints", "fingertip", "metres", "reset"):
            assert topic in doc


class TestInjectionScope:
    """A fault fires once by default, because recovery is what is under test.

    Injecting on every attempt makes the task impossible by construction — no
    object the planner intends to move can ever move — so the only thing such a
    run can demonstrate is that the loop gives up. A one-shot fault asks the
    question worth asking: does the system notice, and does it carry on?
    """

    @staticmethod
    def _plan_for(ex, name="mug"):
        obj = ex.spec.by_name(name)
        pose = np.eye(4)
        pose[:3, 3] = ex._camera_from_world(obj.primitive.position)
        return pose

    def test_default_fires_only_on_the_first_attempt(self):
        ex = MutationExecutor(ss.occluded_target_scene(), inject=["drop"], seed=0)
        assert ex.injecting is True
        assert ex.inject_at == 0

    def test_injecting_turns_off_after_the_first_execution(self):
        ex = MutationExecutor(ss.occluded_target_scene(), inject=["drop"], seed=0)
        ex._executions = 1
        assert ex.injecting is False

    def test_a_later_index_can_be_chosen(self):
        ex = MutationExecutor(ss.occluded_target_scene(), inject=["drop"],
                              inject_at=2, seed=0)
        assert ex.injecting is False
        ex._executions = 2
        assert ex.injecting is True
        ex._executions = 3
        assert ex.injecting is False

    def test_none_means_every_attempt(self):
        ex = MutationExecutor(ss.occluded_target_scene(), inject=["drop"],
                              inject_at=None, seed=0)
        for n in range(4):
            ex._executions = n
            assert ex.injecting is True

    def test_no_injection_configured_never_fires(self):
        ex = MutationExecutor(ss.occluded_target_scene(), seed=0)
        assert ex.injecting is False

    def test_reset_rearms_the_fault(self):
        ex = MutationExecutor(ss.occluded_target_scene(), inject=["drop"], seed=0)
        ex._executions = 3
        assert ex.injecting is False
        ex.reset()
        assert ex.injecting is True
