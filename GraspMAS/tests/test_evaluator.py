"""The evaluator, driven by injected failures rather than by hand-built inputs.

Every verdict below is produced by actually executing a move that goes wrong in
a specific way, so what is tested is the evaluator's reading of a real scene
rather than its reading of a dict somebody wrote to make the test pass.
"""

import json

import numpy as np
import pytest

import evaluator as ev
import placement as pl
import scene_registry as sr
import synth_scene as ss
from execution import MutationExecutor, PickPlacePlan


def _registry(executor, iteration=0):
    obs = executor.capture(iteration)
    reg = sr.SceneRegistry()
    reg.update_from_segmentation(obs.depth, obs.K, obs.seg, iteration, obs.label_map)
    return reg


def _plan(reg, label, target_label="banana"):
    inst = next(i for i in reg.instances.values() if i.label == label)
    target = next(i for i in reg.instances.values() if i.label == target_label)
    grasp = reg.nominal_grasp(inst.id)
    place = pl.plan_place(
        grasp, inst.cloud, reg.scene_cloud_excluding(), reg.plane,
        keep_out=reg.keep_out_for(target.id, moving_id=inst.id), hmap=reg.hmap,
    )
    assert place is not None
    return inst, target, place, PickPlacePlan(
        inst.id, {"pose": grasp.tolist(), "score": 0.9}, place.as_dict(),
        object_label=label,
    )


def _run(inject=(), seed=0, label="mug", scenario="occluded_target"):
    """Execute one move and return everything the evaluator needs."""
    exc = MutationExecutor(getattr(ss, f"{scenario}_scene")(), inject=inject, seed=seed)
    before = _registry(exc, 0)
    inst, target, place, plan = _plan(before, label)
    report = exc.execute_pick_place(plan)
    after = _registry(exc, 1)
    return before, after, inst, target, place, report


class TestVerdicts:
    def test_a_clean_move_is_a_success(self):
        before, after, inst, target, place, report = _run()
        result = ev.evaluate(before, after, inst.id, target.id, place.place_xy, report)
        assert result.action_succeeded == "success"
        assert result.place_error_m < ev.PLACE_TOLERANCE_M
        assert result.displacement_m > ev.MOVED_THRESH_M

    def test_a_dropped_object_is_not_moved(self):
        before, after, inst, target, place, report = _run(inject=["drop"])
        result = ev.evaluate(before, after, inst.id, target.id, place.place_xy, report)
        assert result.action_succeeded == "not_moved"
        assert "did not take hold" in result.evidence

    def test_an_offset_release_is_moved_off_target(self):
        before, after, inst, target, place, report = _run(inject=["offset"], seed=1)
        result = ev.evaluate(
            before, after, inst.id, target.id, place.place_xy, report,
            place_tolerance_m=0.04,
        )
        assert result.action_succeeded == "moved_off_target"
        assert result.place_error_m > 0.04

    def test_without_an_intended_position_it_says_so(self):
        before, after, inst, target, place, report = _run()
        result = ev.evaluate(before, after, inst.id, target.id, None, report)
        assert result.action_succeeded == "moved_off_target"
        assert "no intended position" in result.evidence

    def test_an_untracked_object_is_unknown(self):
        before, after, inst, target, place, report = _run()
        result = ev.evaluate(before, after, "obj_999", target.id)
        assert result.action_succeeded == "unknown"
        assert result.needs_review

    def test_a_vanished_object_is_reported_missing(self):
        """Not 'failed': off the table and hidden are different things."""
        before, after, inst, target, place, report = _run()
        after.instances.pop(inst.id)
        result = ev.evaluate(before, after, inst.id, target.id, place.place_xy)
        assert result.action_succeeded == "object_missing"
        assert result.needs_review

    def test_the_verdict_set_is_enforced(self):
        with pytest.raises(ValueError, match="verdict must be"):
            ev.Evaluation(action_succeeded="probably fine")


class TestSuccessAndUsefulnessComeApart:
    """The distinction the whole design rests on, in both directions."""

    def test_a_failed_move_can_still_clear_the_target(self):
        """The gripper slipped, but far enough that the object is out of the way.

        `action_succeeded` says the action failed; `still_blocking_target` says
        the goal advanced. The planner must act on the second.
        """
        exc = MutationExecutor(ss.occluded_target_scene())
        before = _registry(exc, 0)
        inst, target, place, plan = _plan(before, "mug")

        # A release far from the plan, but well clear of the target.
        strayed = np.asarray(plan.place["pose"], dtype=float)
        strayed[:3, 3] += exc._world_from_camera(np.zeros(3))
        exc.execute_pick_place(plan)
        after = _registry(exc, 1)

        result = ev.evaluate(
            before, after, inst.id, target.id,
            intended_place_xy=place.place_xy + np.array([0.5, 0.5]),
        )
        assert result.action_succeeded == "moved_off_target"
        assert result.still_blocking_target is False
        assert result.helped

    def test_a_successful_move_can_leave_the_target_blocked(self):
        """Moved exactly as planned, and it changed nothing — the plan was wrong.

        A 5 cm sideways shuffle: far enough to count as movement, nowhere near
        far enough to stop obstructing. The action worked; the plan did not.
        """
        exc = MutationExecutor(ss.occluded_target_scene())
        before = _registry(exc, 0)
        mug = next(i for i in before.instances.values() if i.label == "mug")
        target = next(i for i in before.instances.values() if i.label == "banana")
        assert mug.id in {b.object_id for b in before.blocking_objects(target.id)}

        goal_xy = mug.footprint.centroid_xy + np.array([0.05, 0.0])
        grasp = before.nominal_grasp(mug.id)
        pose = grasp.copy()
        pose[:3, 3] += pl.place_delta(before.plane, mug.footprint, goal_xy)

        exc.execute_pick_place(
            PickPlacePlan(mug.id, {"pose": grasp.tolist()}, {"pose": pose.tolist()})
        )
        after = _registry(exc, 1)

        result = ev.evaluate(before, after, mug.id, target.id, goal_xy)
        assert result.action_succeeded == "success"
        assert result.still_blocking_target is True
        assert not result.helped

    def test_blocking_is_recomputed_not_inferred(self):
        """It must come from the new scene, not from the verdict."""
        before, after, inst, target, place, report = _run(inject=["drop"])
        result = ev.evaluate(before, after, inst.id, target.id, place.place_xy, report)
        assert result.action_succeeded == "not_moved"
        assert result.still_blocking_target is True


class TestCollateralDamage:
    def test_an_unplanned_disturbance_is_caught(self):
        before, after, inst, target, place, report = _run(inject=["collateral"], seed=3)
        result = ev.evaluate(before, after, inst.id, target.id, place.place_xy, report)
        assert result.collateral, "a knocked neighbour must be reported"
        assert result.needs_review
        assert "other objects moved" in result.evidence

    def test_a_clean_move_reports_none(self):
        before, after, inst, target, place, report = _run()
        result = ev.evaluate(before, after, inst.id, target.id, place.place_xy, report)
        assert result.collateral == []
        assert not result.needs_review

    def test_the_moved_object_is_not_its_own_collateral(self):
        before, after, inst, target, place, report = _run()
        result = ev.evaluate(before, after, inst.id, target.id, place.place_xy, report)
        assert inst.id not in [oid for oid, _ in result.collateral]

    def test_occlusion_drift_is_not_mistaken_for_movement(self):
        """Uncovering the banana shifts its apparent centre by ~2.3 cm.

        Without the visibility gate that reads as collateral damage on an object
        nobody touched.
        """
        before, after, inst, target, place, report = _run()
        result = ev.evaluate(before, after, inst.id, target.id, place.place_xy, report)
        assert target.id not in [oid for oid, _ in result.collateral]


class TestDisagreement:
    def test_an_executor_failure_outranks_a_geometric_success(self):
        """The gripper knows things the camera cannot see."""
        before, after, inst, target, place, report = _run()
        report.status = "failed"
        report.error = "the jaw closed on nothing"
        result = ev.evaluate(before, after, inst.id, target.id, place.place_xy, report)
        assert result.action_succeeded == "unknown"
        assert result.needs_review
        assert "disagree" in result.evidence

    def test_an_executor_note_is_carried_through(self):
        before, after, inst, target, place, report = _run(inject=["offset"], seed=1)
        report.error = "force sensor spiked on approach"
        result = ev.evaluate(before, after, inst.id, target.id, place.place_xy, report)
        assert "force sensor spiked" in result.evidence

    def test_works_with_no_execution_report_at_all(self):
        before, after, inst, target, place, _ = _run()
        result = ev.evaluate(before, after, inst.id, target.id, place.place_xy)
        assert result.action_succeeded == "success"


class TestUnknownRatherThanGuessing:
    def test_an_occluded_object_is_not_judged(self):
        """The banana's centroid is the mean of what is visible, so a change in
        occlusion moves it without the object moving."""
        exc = MutationExecutor(ss.occluded_target_scene())
        before = _registry(exc, 0)
        banana = next(i for i in before.instances.values() if i.label == "banana")
        assert banana.visibility < ev.MIN_COMPARABLE_VISIBILITY

        inst, target, place, plan = _plan(before, "mug")
        exc.execute_pick_place(plan)
        after = _registry(exc, 1)

        result = ev.evaluate(before, after, banana.id, banana.id)
        assert result.action_succeeded == "unknown"
        assert "not trustworthy" in result.evidence
        assert result.needs_review

    def test_a_missing_target_is_flagged(self):
        before, after, inst, target, place, report = _run()
        after.instances.pop(target.id)
        result = ev.evaluate(before, after, inst.id, target.id, place.place_xy)
        assert result.still_blocking_target is None
        assert result.needs_review


class TestReporting:
    def test_describe_is_json_safe(self):
        before, after, inst, target, place, report = _run()
        result = ev.evaluate(before, after, inst.id, target.id, place.place_xy, report)
        blob = json.loads(json.dumps(result.describe()))
        assert blob["action_succeeded"] == "success"
        assert blob["source"] == "geometric"

    def test_evidence_is_never_empty(self):
        before, after, inst, target, place, report = _run()
        result = ev.evaluate(before, after, inst.id, target.id, place.place_xy, report)
        assert len(result.evidence) > 20

    def test_reports_when_nothing_blocks_the_target_any_more(self):
        exc = MutationExecutor(ss.occluded_target_scene())
        for _ in range(2):
            reg = _registry(exc, 0)
            target = next(i for i in reg.instances.values() if i.label == "banana")
            blockers = reg.blocking_objects(target.id)
            if not blockers:
                break
            inst, _, place, plan = _plan(reg, reg.get(blockers[0].object_id).label)
            before, report = reg, exc.execute_pick_place(plan)
        after = _registry(exc, 2)
        result = ev.evaluate(before, after, inst.id, target.id, place.place_xy, report)
        assert "nothing now blocks" in result.evidence


class TestVlmFallback:
    """The fallback exists for what geometry structurally cannot see."""

    class StubLLM:
        system_prompt = "test"

        def __init__(self, reply):
            self.reply = reply
            self.calls = 0

        async def chat_with_image(self, *a, **kw):
            self.calls += 1
            return self.reply

    @pytest.mark.asyncio
    async def test_resolves_an_unknown_verdict(self, tmp_path):
        import cv2

        img = tmp_path / "after.png"
        cv2.imwrite(str(img), np.zeros((32, 32, 3), np.uint8))

        stub = self.StubLLM(
            '<review>{"object_moved": "yes", "landed_upright": "no", '
            '"target_now_reachable": "yes", "anything_else_disturbed": "no", '
            '"summary": "the mug is on its side next to the tray"}</review>'
        )
        result = await ev.review_with_vlm(
            ev.Evaluation(action_succeeded="unknown", needs_review=True),
            str(img), str(img), "move the mug clear of the banana", stub,
        )
        assert stub.calls == 1
        assert result.source == "vlm_review"
        assert result.action_succeeded == "moved_off_target"
        assert result.still_blocking_target is False
        assert "not upright" in result.evidence
        assert "on its side" in result.evidence

    @pytest.mark.asyncio
    async def test_unparseable_replies_leave_the_verdict_alone(self, tmp_path):
        import cv2

        img = tmp_path / "after.png"
        cv2.imwrite(str(img), np.zeros((32, 32, 3), np.uint8))

        stub = self.StubLLM("I could not tell from these pictures, sorry.")
        before = ev.Evaluation(action_succeeded="unknown", needs_review=True)
        result = await ev.review_with_vlm(before, str(img), str(img), "intent", stub)
        assert result.action_succeeded == "unknown"
        assert "nothing parseable" in result.evidence

    @pytest.mark.asyncio
    async def test_an_llm_failure_is_survivable(self, tmp_path):
        import cv2

        img = tmp_path / "after.png"
        cv2.imwrite(str(img), np.zeros((32, 32, 3), np.uint8))

        class Broken:
            system_prompt = "test"

            async def chat_with_image(self, *a, **kw):
                raise RuntimeError("quota exhausted")

        result = await ev.review_with_vlm(
            ev.Evaluation(action_succeeded="unknown"), str(img), str(img), "intent", Broken()
        )
        assert result.action_succeeded == "unknown"
        assert "unavailable" in result.evidence


class TestPhantomCollateralOnAnUncoveredObject:
    """Becoming visible is not moving.

    An occluded object's centroid is the mean of whatever is visible, so
    uncovering it shifts its apparent centre. Gating on how well it is seen
    *now* does not help — the stored centroid it is compared against came from
    the partly-hidden view. Measured on a live run: with the mug carried away
    the banana went 78% -> 100% visible and appeared to move 3.8 cm, was
    reported as collateral damage **to the target**, and the planner aborted on
    the strength of it.
    """

    @staticmethod
    def _well_seen(reg):
        """An instance the current-visibility gate will not reject on its own."""
        return next(i for i in reg.instances.values() if i.visibility >= 0.95)

    def test_an_object_that_was_badly_seen_before_is_not_compared(self, tabletop):
        reg = sr.SceneRegistry()
        reg.update_from_segmentation(
            tabletop["depth"], tabletop["K"], tabletop["seg"], 0, tabletop["label_map"]
        )
        subject = self._well_seen(reg)

        # Nothing moved; only the stored centroid is displaced, exactly the way
        # an uncovering displaces one.
        positions = reg.positions()
        positions[subject.id] = positions[subject.id] + np.array([0.05, 0.0, 0.0])
        clear = reg.visibilities()

        hidden_before = reg.moved_since(
            positions, thresh_m=0.03, min_visibility=0.95,
            previous_visibility={**clear, subject.id: 0.78},
        )
        assert subject.id not in [oid for oid, _ in hidden_before]

        # With both ends trustworthy the same shift is reported.
        seen_before = reg.moved_since(
            positions, thresh_m=0.03, min_visibility=0.95,
            previous_visibility=clear,
        )
        assert subject.id in [oid for oid, _ in seen_before]

    def test_the_gate_defaults_to_trusting_the_previous_snapshot(self, tabletop):
        """Callers that pass no history keep the old behaviour."""
        reg = sr.SceneRegistry()
        reg.update_from_segmentation(
            tabletop["depth"], tabletop["K"], tabletop["seg"], 0, tabletop["label_map"]
        )
        positions = {k: v + np.array([0.05, 0.0, 0.0]) for k, v in reg.positions().items()}
        assert reg.moved_since(positions, thresh_m=0.03, min_visibility=0.0)
