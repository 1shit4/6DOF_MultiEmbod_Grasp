"""The state files, and the guardrails on them.

Most of what is defended here is refusal: the grand plan must not be editable
into something the run trivially satisfies, and the history must not be
rewritable. The rest is that a run killed part-way through can be picked up
again without losing or inventing anything.
"""

import json
from pathlib import Path

import pytest

import session_state as st


@pytest.fixture
def state(tmp_path):
    s = st.SessionState(tmp_path / "run")
    s.start("grasp the banana", {"id": "obj_001", "label": "banana"})
    return s


@pytest.fixture
def planned(state):
    state.set_grand_plan(
        removal_order=[
            {"object_id": "obj_002", "label": "bottle", "reason": "occludes the banana"},
            {"object_id": "obj_003", "label": "mug", "reason": "inside the jaw"},
        ],
        success_criterion="banana graspable with a clear approach",
    )
    return state


def _finish_iteration(state, index, object_id="obj_002", succeeded="success",
                      still_blocking=False, action="remove"):
    state.begin_iteration(index, subgoal=f"move {object_id}", action=action,
                          object_id=object_id, object_label="bottle")
    state.record_evaluation(
        {"action_succeeded": succeeded, "still_blocking_target": still_blocking}
    )
    return state.end_iteration()


class TestLifecycle:
    def test_start_records_the_goal_and_target(self, state):
        assert state.progress["goal"] == "grasp the banana"
        assert state.progress["target"]["id"] == "obj_001"
        assert state.progress["status"] == "in_progress"

    def test_files_are_written_on_change(self, state, tmp_path):
        assert (tmp_path / "run" / st.PROGRESS_FILE).is_file()
        assert not (tmp_path / "run" / st.GRAND_PLAN_FILE).exists()

    def test_grand_plan_is_written_once_set(self, planned, tmp_path):
        blob = json.loads((tmp_path / "run" / st.GRAND_PLAN_FILE).read_text())
        assert [s["object_id"] for s in blob["removal_order"]] == ["obj_002", "obj_003"]

    def test_autosave_can_be_disabled(self, tmp_path):
        s = st.SessionState(tmp_path / "quiet", autosave=False)
        s.start("goal")
        assert not (tmp_path / "quiet" / st.PROGRESS_FILE).exists()
        s.save()
        assert (tmp_path / "quiet" / st.PROGRESS_FILE).is_file()

    def test_finish_requires_a_terminal_status(self, state):
        with pytest.raises(ValueError, match="status must be"):
            state.finish("nearly")

    def test_finish_closes_an_open_iteration(self, state):
        state.begin_iteration(0, action="remove", object_id="obj_002")
        state.finish("aborted")
        assert len(state.progress["iterations"]) == 1
        assert "still open" in state.progress["iterations"][0]["notes"][0]

    def test_finish_records_the_outcome(self, state):
        state.finish("success", {"grasp": {"score": 0.9}})
        assert state.progress["outcome"]["grasp"]["score"] == 0.9


class TestGrandPlanGuards:
    def test_goal_cannot_be_changed(self, planned):
        assert not planned.amend_grand_plan({"goal": "grasp the mug"}, "changed my mind")
        assert planned.grand_plan["goal"] == "grasp the banana"
        assert "goal" in planned.last_refusal

    def test_target_cannot_be_changed(self, planned):
        assert not planned.amend_grand_plan(
            {"target": {"id": "obj_009"}}, "the banana looks hard"
        )
        assert planned.grand_plan["target"]["id"] == "obj_001"

    def test_amendment_without_a_reason_is_refused(self, planned):
        assert not planned.amend_grand_plan({"removal_order": []}, "")
        assert not planned.amend_grand_plan({"removal_order": []}, "   ")
        assert "state why" in planned.last_refusal

    def test_empty_amendment_is_refused(self, planned):
        assert not planned.amend_grand_plan({}, "nothing to do")

    def test_unknown_fields_are_refused(self, planned):
        assert not planned.amend_grand_plan({"shortcut": True}, "faster")
        assert "unknown field" in planned.last_refusal

    def test_removal_order_can_be_amended_with_a_reason(self, planned):
        ok = planned.amend_grand_plan(
            {"removal_order": [{"object_id": "obj_003", "label": "mug"}]},
            reason="obj_002 no longer occludes the banana after it was moved",
            iteration=1,
        )
        assert ok
        assert [s["object_id"] for s in planned.grand_plan["removal_order"]] == ["obj_003"]

    def test_revisions_record_before_and_after(self, planned):
        planned.amend_grand_plan({"removal_order": []}, "everything is clear", iteration=2)
        rev = planned.grand_plan["revisions"][-1]
        assert rev["iteration"] == 2
        assert rev["reason"] == "everything is clear"
        assert len(rev["before"]["removal_order"]) == 2
        assert rev["after"]["removal_order"] == []

    def test_revision_budget_is_enforced(self, planned):
        for i in range(st.MAX_REVISIONS):
            assert planned.amend_grand_plan({"reasoning": f"take {i}"}, f"reason {i}")
        assert planned.revisions_left == 0
        assert not planned.amend_grand_plan({"reasoning": "again"}, "one more")
        assert "oscillating" in planned.last_refusal

    def test_amending_before_planning_is_refused(self, state):
        assert not state.amend_grand_plan({"removal_order": []}, "reason")
        assert "no grand plan" in state.last_refusal

    def test_setting_the_plan_twice_is_an_error(self, planned):
        with pytest.raises(RuntimeError, match="already set"):
            planned.set_grand_plan([], "")


class TestIterations:
    def test_records_accumulate(self, state):
        _finish_iteration(state, 0)
        _finish_iteration(state, 1, object_id="obj_003")
        assert state.next_index == 2
        assert [r.object_id for r in state.iterations] == ["obj_002", "obj_003"]

    def test_cannot_open_two_at_once(self, state):
        state.begin_iteration(0)
        with pytest.raises(RuntimeError, match="still open"):
            state.begin_iteration(1)

    def test_recording_without_an_open_iteration_is_an_error(self, state):
        with pytest.raises(RuntimeError, match="no iteration is open"):
            state.record_plan(None, None)

    def test_captures_the_full_record(self, state):
        state.begin_iteration(0, subgoal="move the bottle", action="remove",
                              object_id="obj_002", object_label="bottle",
                              rationale="it occludes the banana",
                              blockers=[{"object_id": "obj_002"}])
        state.record_plan({"score": 0.8}, {"travel_m": 0.1})
        state.record_observer({"verdict": "VALID"})
        state.record_execution({"status": "ok"})
        state.record_evaluation({"action_succeeded": "success"})
        state.record_artifacts(overlay="a.png", after="b.png")
        state.note("gripper slipped once")
        rec = state.end_iteration()

        assert rec.planned["grasp"]["score"] == 0.8
        assert rec.observer["verdict"] == "VALID"
        assert rec.artifacts == {"overlay": "a.png", "after": "b.png"}
        assert rec.notes == ["gripper slipped once"]
        assert rec.started_at and rec.ended_at

    def test_accepts_dataclass_like_reports(self, state):
        class Report:
            def describe(self):
                return {"status": "ok", "stage": "place"}

        state.begin_iteration(0)
        state.record_execution(Report())
        rec = state.end_iteration()
        assert rec.execution == {"status": "ok", "stage": "place"}

    def test_numpy_values_survive_serialisation(self, state, tmp_path):
        import numpy as np

        state.begin_iteration(0)
        state.record_plan({"pose": np.eye(4), "score": np.float32(0.5)}, None)
        state.end_iteration()
        blob = json.loads((tmp_path / "run" / st.PROGRESS_FILE).read_text())
        assert blob["iterations"][0]["planned"]["grasp"]["score"] == pytest.approx(0.5)

    def test_attempts_on_filters_by_object(self, state):
        _finish_iteration(state, 0, object_id="obj_002")
        _finish_iteration(state, 1, object_id="obj_003")
        _finish_iteration(state, 2, object_id="obj_002")
        assert len(state.attempts_on("obj_002")) == 2

    def test_moved_objects_lists_only_successes(self, state):
        _finish_iteration(state, 0, object_id="obj_002", succeeded="success")
        _finish_iteration(state, 1, object_id="obj_003", succeeded="not_moved")
        assert state.moved_objects() == ["obj_002"]


class TestStallDetection:
    def test_a_fresh_run_is_not_stalled(self, state):
        assert not state.is_stalled()

    def test_progress_clears_the_stall(self, state):
        _finish_iteration(state, 0, succeeded="not_moved", still_blocking=True)
        _finish_iteration(state, 1, succeeded="success", still_blocking=False)
        assert not state.is_stalled()

    def test_repeated_failure_is_a_stall(self, state):
        _finish_iteration(state, 0, succeeded="not_moved", still_blocking=True)
        _finish_iteration(state, 1, succeeded="not_moved", still_blocking=True)
        assert state.is_stalled()

    def test_a_failed_move_that_helped_is_not_a_stall(self, state):
        """The distinction the user asked for: success and usefulness differ.

        The gripper slipped, so the action failed — but the object was nudged
        far enough that it no longer blocks the target. Nothing needs redoing.
        """
        _finish_iteration(state, 0, succeeded="moved_off_target", still_blocking=False)
        _finish_iteration(state, 1, succeeded="moved_off_target", still_blocking=False)
        assert not state.is_stalled()

    def test_a_successful_move_that_did_not_help_is_a_stall(self, state):
        """And the converse: the object moved exactly as planned and it changed nothing."""
        _finish_iteration(state, 0, succeeded="success", still_blocking=True)
        _finish_iteration(state, 1, succeeded="success", still_blocking=True)
        assert state.is_stalled()

    def test_window_is_respected(self, state):
        _finish_iteration(state, 0, succeeded="not_moved", still_blocking=True)
        assert not state.is_stalled(window=2)
        assert state.is_stalled(window=1)


class TestPlannerContext:
    def test_reports_no_plan_before_one_is_set(self, state):
        assert "not set yet" in state.planner_context()

    def test_says_plainly_when_nothing_has_been_tried(self, planned):
        ctx = planned.planner_context()
        assert "nothing attempted yet" in ctx

    def test_lists_the_removal_order(self, planned):
        ctx = planned.planner_context()
        assert "obj_002" in ctx and "obj_003" in ctx
        assert "banana graspable" in ctx

    def test_history_shows_outcome_and_blocking_status(self, planned):
        _finish_iteration(planned, 0, succeeded="not_moved", still_blocking=True)
        ctx = planned.planner_context()
        assert "not_moved" in ctx and "still blocking" in ctx

    def test_unknown_blocking_status_is_stated_not_guessed(self, planned):
        planned.begin_iteration(0, action="remove", object_id="obj_002")
        planned.record_evaluation({"action_succeeded": "unknown"})
        planned.end_iteration()
        assert "blocking status unknown" in planned.planner_context()

    def test_revisions_are_surfaced_with_their_reasons(self, planned):
        planned.amend_grand_plan({"removal_order": []}, "the mug fell off the table", 1)
        ctx = planned.planner_context()
        assert "the mug fell off the table" in ctx
        assert "revisions left" in ctx

    def test_long_histories_are_truncated_from_the_front(self, planned):
        for i in range(10):
            _finish_iteration(planned, i, object_id=f"obj_{i:03d}")
        ctx = planned.planner_context(max_iterations=3)
        assert "earlier iteration(s) omitted" in ctx
        assert "obj_009" in ctx and "obj_000" not in ctx

    def test_notes_appear(self, planned):
        planned.begin_iteration(0, action="remove", object_id="obj_002")
        planned.note("server timed out once")
        planned.record_evaluation({"action_succeeded": "success"})
        planned.end_iteration()
        assert "server timed out once" in planned.planner_context()


class TestPersistence:
    def test_round_trips(self, planned, tmp_path):
        _finish_iteration(planned, 0)
        planned.amend_grand_plan({"reasoning": "revised"}, "new evidence", 0)

        back = st.SessionState.resume(tmp_path / "run")
        assert back.progress["goal"] == "grasp the banana"
        assert back.next_index == 1
        assert back.grand_plan["reasoning"] == "revised"
        assert len(back.grand_plan["revisions"]) == 1

    def test_resume_continues_the_numbering(self, planned, tmp_path):
        _finish_iteration(planned, 0)
        _finish_iteration(planned, 1)
        back = st.SessionState.resume(tmp_path / "run")
        _finish_iteration(back, back.next_index)
        assert [r.index for r in back.iterations] == [0, 1, 2]

    def test_resume_preserves_the_revision_budget(self, planned, tmp_path):
        planned.amend_grand_plan({"reasoning": "a"}, "reason a")
        planned.amend_grand_plan({"reasoning": "b"}, "reason b")
        back = st.SessionState.resume(tmp_path / "run")
        assert back.revisions_left == st.MAX_REVISIONS - 2

    def test_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=st.PROGRESS_FILE):
            st.SessionState.resume(tmp_path / "nothing_here")

    def test_a_future_schema_is_refused(self, planned, tmp_path):
        path = tmp_path / "run" / st.PROGRESS_FILE
        blob = json.loads(path.read_text())
        blob["schema_version"] = 99
        path.write_text(json.dumps(blob))
        with pytest.raises(ValueError, match="schema version 99"):
            st.SessionState.resume(tmp_path / "run")

    def test_writes_are_atomic(self, planned, tmp_path):
        """A reader must never see a partial file, so no temp files may linger."""
        _finish_iteration(planned, 0)
        leftovers = list((tmp_path / "run").glob(".*tmp"))
        assert leftovers == []
        json.loads((tmp_path / "run" / st.PROGRESS_FILE).read_text())

    def test_resume_survives_a_run_killed_mid_iteration(self, planned, tmp_path):
        """The open iteration was never appended, so the file stays consistent."""
        _finish_iteration(planned, 0)
        planned.begin_iteration(1, action="remove", object_id="obj_003")
        back = st.SessionState.resume(tmp_path / "run")
        assert back.next_index == 1
        assert back.progress["status"] == "in_progress"

    def test_describe_is_json_safe(self, planned):
        assert json.loads(json.dumps(planned.describe()))["has_grand_plan"] is True


class TestAmendmentShapeValidation:
    """Guarding *which* fields may change is not enough; the value must be sane.

    A live planner amended `removal_order` to `["obj_002"]` — plain strings
    rather than step objects. Nothing rejected it, the state file was written
    that way, and every later `planner_context()` raised
    `AttributeError: 'str' object has no attribute 'get'`. The run then aborted
    claiming the planner was unreachable, which is not what happened.
    """

    def test_a_bare_id_is_upgraded_rather_than_refused(self, state):
        state.set_grand_plan(removal_order=[{"object_id": "obj_003"}])

        assert state.amend_grand_plan(
            {"removal_order": ["obj_002"]}, "obj_003 is done", iteration=1
        )
        assert state.grand_plan["removal_order"] == [{"object_id": "obj_002"}]

    def test_the_plan_still_renders_after_such_an_amendment(self, state):
        """The actual regression: the next prompt build must not raise."""
        state.set_grand_plan(removal_order=[{"object_id": "obj_003"}])
        state.amend_grand_plan(
            {"removal_order": ["obj_002"]}, "obj_003 is done", iteration=1
        )
        assert "obj_002" in state.planner_context()

    def test_nonsense_entries_are_refused_with_a_reason(self, state):
        state.set_grand_plan(removal_order=[{"object_id": "obj_003"}])

        assert not state.amend_grand_plan(
            {"removal_order": [42]}, "because", iteration=1
        )
        assert "removal_order" in state.last_refusal
        # The refusal must leave the previous plan intact, not half-applied.
        assert state.grand_plan["removal_order"] == [{"object_id": "obj_003"}]

    def test_a_step_without_an_id_is_refused(self, state):
        state.set_grand_plan(removal_order=[])
        assert not state.amend_grand_plan(
            {"removal_order": [{"label": "bottle"}]}, "because", iteration=1
        )

    def test_a_non_list_is_refused(self, state):
        state.set_grand_plan(removal_order=[])
        assert not state.amend_grand_plan(
            {"removal_order": "obj_002"}, "because", iteration=1
        )

    def test_the_initial_plan_normalises_too(self, state):
        state.set_grand_plan(removal_order=["obj_003"])
        assert state.grand_plan["removal_order"] == [{"object_id": "obj_003"}]

    def test_the_initial_plan_rejects_nonsense(self, state):
        with pytest.raises(ValueError, match="removal_order"):
            state.set_grand_plan(removal_order=[None])


class TestStallDetection:
    """Progress is asked of the goal, not of the object the loop chose.

    Measured on `occluded_target` under the `wrong_object` fault: the hand
    closed on a real blocker by accident and carried it away, taking the target
    from 32% visible with two blockers to 78% with one. Every *chosen* object
    was still blocking, so the old test called that "no progress" and stopped a
    run that was in fact winning.
    """

    @staticmethod
    def _iteration(state, index, object_id, still_blocking, blockers):
        state.begin_iteration(index, action="remove", object_id=object_id,
                              object_label=object_id)
        state.record_execution({"status": "partial"})
        state.record_evaluation({
            "action_succeeded": "not_moved",
            "still_blocking_target": still_blocking,
            "target_blockers": blockers,
        })
        state.end_iteration()

    def test_stalls_when_the_scene_never_changes(self, state):
        """The `drop` fault: nothing moves, ever. Giving up is correct."""
        for i in range(2):
            self._iteration(state, i, "obj_002", True, ["obj_002", "obj_003"])
        assert state.is_stalled(window=2)

    def test_not_stalled_when_the_target_lost_a_blocker(self, state):
        """Progress the loop did not plan is still progress."""
        self._iteration(state, 0, "obj_002", True, ["obj_002", "obj_003"])
        self._iteration(state, 1, "obj_002", True, ["obj_002"])
        assert not state.is_stalled(window=2)

    def test_stalls_again_once_the_blocker_count_stops_falling(self, state):
        """The progress itself stays inside the window for one more iteration.

        "No progress in the last 2" is false while one of those 2 made progress,
        so the stall is declared on the second barren iteration after it — the
        same semantics the check has always had.
        """
        self._iteration(state, 0, "obj_002", True, ["obj_002", "obj_003"])
        self._iteration(state, 1, "obj_002", True, ["obj_002"])
        assert not state.is_stalled(window=2)
        self._iteration(state, 2, "obj_002", True, ["obj_002"])
        assert not state.is_stalled(window=2)
        self._iteration(state, 3, "obj_002", True, ["obj_002"])
        assert state.is_stalled(window=2)

    def test_a_growing_blocker_set_is_not_progress(self, state):
        self._iteration(state, 0, "obj_002", True, ["obj_002"])
        self._iteration(state, 1, "obj_002", True, ["obj_002", "obj_004"])
        assert state.is_stalled(window=2)

    def test_the_chosen_object_clearing_still_counts(self, state):
        """The original signal must keep working."""
        self._iteration(state, 0, "obj_002", True, ["obj_002", "obj_003"])
        self._iteration(state, 1, "obj_003", False, ["obj_002"])
        assert not state.is_stalled(window=2)

    def test_missing_blocker_data_falls_back_to_the_old_signal(self, state):
        for i in range(2):
            state.begin_iteration(i, action="remove", object_id="obj_002")
            state.record_evaluation({"action_succeeded": "not_moved",
                                     "still_blocking_target": True})
            state.end_iteration()
        assert state.is_stalled(window=2)


class TestStallDiagnosis:
    """A stalled run should say which failure it hit, when the reports agree."""

    @staticmethod
    def _attempt(state, index, *, status, stage=None, grasped=None,
                 label="mug", error=None):
        state.begin_iteration(index, action="remove", object_id=f"obj_{index}",
                              object_label=label)
        state.record_execution({"status": status, "stage_reached": stage,
                                "grasped_object": grasped, "error": error})
        state.record_evaluation({"action_succeeded": "not_moved",
                                 "still_blocking_target": True})
        state.end_iteration()

    def test_names_a_repeated_grasp_failure(self, state):
        """The `drop` fault: every attempt dies at the same stage."""
        for i in range(2):
            self._attempt(state, i, status="failed", stage="lift",
                          error="the object slipped out of the jaw on lift")
        text = state.stall_diagnosis()
        assert "same stage" in text and "lift" in text
        assert "will not help" in text

    def test_names_a_targeting_failure(self, state):
        """The `wrong_object` fault: the hand grasps something else each time."""
        self._attempt(state, 0, status="partial", stage="retreat",
                      grasped="box", label="mug")
        self._attempt(state, 1, status="partial", stage="retreat",
                      grasped="mug", label="bottle")
        text = state.stall_diagnosis()
        assert "different object than planned" in text
        assert "targeting or calibration" in text

    def test_names_an_exhausted_table(self, state):
        for i in range(2):
            self._attempt(state, i, status="failed",
                          error="no placement available")
        assert "nowhere left" in state.stall_diagnosis()

    def test_says_nothing_when_the_attempts_disagree(self, state):
        """Inventing a story from inconsistent evidence is worse than silence."""
        self._attempt(state, 0, status="failed", stage="lift", error="slipped")
        self._attempt(state, 1, status="partial", stage="retreat", grasped="mug",
                      label="mug")
        assert state.stall_diagnosis() == ""

    def test_a_lone_error_does_not_speak_for_every_attempt(self, state):
        """The overclaim this check was shipped with, and must not regress.

        Filtering empty errors before `all()` makes the test vacuously true, so
        one "no grasp" among three attempts was reported as all three.
        """
        self._attempt(state, 0, status="partial", stage="retreat",
                      grasped="mug", label="mug")
        self._attempt(state, 1, status="partial", stage="retreat",
                      grasped="mug", label="mug")
        self._attempt(state, 2, status="failed", error="no grasp found for obj_003")
        assert "3 of the last 3" not in state.stall_diagnosis()

    def test_counts_are_stated_and_true(self, state):
        """Two of three grasped the wrong object; the text must say two."""
        self._attempt(state, 0, status="partial", grasped="box", label="mug")
        self._attempt(state, 1, status="partial", grasped="box", label="mug")
        self._attempt(state, 2, status="failed", error="no grasp found")
        text = state.stall_diagnosis()
        assert "2 of the last 3 attempts" in text
        assert "different object than planned" in text

    def test_says_nothing_from_a_single_attempt(self, state):
        self._attempt(state, 0, status="failed", stage="lift", error="slipped")
        assert state.stall_diagnosis() == ""


# ---------------------------------------------------------------------------
# Retargeting: the one thing about the objective that may change
# ---------------------------------------------------------------------------


class TestRetarget:
    """The goal is the person's words and never moves.

    The *target* is the system's own inference about what those words referred
    to, so it may be revised — once, on evidence, by Python. A model that can
    re-pick its objective whenever the current one gets hard has no objective,
    which is the same argument `amend_grand_plan` rests on.
    """

    def _state(self, tmp_path, candidates=("obj_001", "obj_002")):
        state = st.SessionState(tmp_path)
        state.start(
            "something to cut",
            {"id": "obj_001", "label": "knife"},
            target_choice={"candidates": [{"object_id": c} for c in candidates]},
        )
        state.set_grand_plan([{"object_id": "obj_003"}], "knife in hand", "clear it")
        return state

    def test_ranked_candidates_are_remembered(self, tmp_path):
        assert self._state(tmp_path).target_candidates == ["obj_001", "obj_002"]

    def test_a_retarget_switches_the_target_and_records_why(self, tmp_path):
        state = self._state(tmp_path)
        assert state.retarget({"id": "obj_002", "label": "scissors"}, "knife unreachable", 2)
        assert state.target_id == "obj_002"
        rec = state.progress["retargets"][0]
        assert rec["from"]["id"] == "obj_001" and rec["to"]["id"] == "obj_002"
        assert rec["reason"] == "knife unreachable" and rec["iteration"] == 2

    def test_the_goal_is_untouched(self, tmp_path):
        state = self._state(tmp_path)
        state.retarget({"id": "obj_002", "label": "scissors"}, "because", 1)
        assert state.progress["goal"] == "something to cut"
        assert state.grand_plan["goal"] == "something to cut"

    def test_the_old_plan_is_archived_not_edited(self, tmp_path):
        state = self._state(tmp_path)
        state.retarget({"id": "obj_002", "label": "scissors"}, "knife unreachable", 1)
        archive = state.grand_plan["superseded"]
        assert len(archive) == 1
        assert archive[0]["target"]["id"] == "obj_001"
        assert archive[0]["superseded_because"] == "knife unreachable"
        assert state.grand_plan["target"]["id"] == "obj_002"

    def test_the_budget_is_one(self, tmp_path):
        state = self._state(tmp_path, candidates=("obj_001", "obj_002", "obj_003"))
        assert state.retarget({"id": "obj_002", "label": "b"}, "first", 1)
        assert not state.can_retarget()
        assert not state.retarget({"id": "obj_003", "label": "c"}, "second", 2)
        assert "limit is" in state.last_refusal
        assert state.target_id == "obj_002"

    def test_a_retarget_must_state_why(self, tmp_path):
        state = self._state(tmp_path)
        assert not state.retarget({"id": "obj_002", "label": "b"}, "  ", 1)
        assert "state why" in state.last_refusal

    def test_a_retarget_needs_an_id(self, tmp_path):
        state = self._state(tmp_path)
        assert not state.retarget({"label": "b"}, "reason", 1)
        assert state.target_id == "obj_001"

    def test_retargeting_to_the_current_target_is_refused(self, tmp_path):
        state = self._state(tmp_path)
        assert not state.retarget({"id": "obj_001", "label": "knife"}, "reason", 1)
        assert "already the target" in state.last_refusal

    def test_amend_still_refuses_the_target_outright(self, tmp_path):
        """A retarget is not an amendment, and the amendment path stays shut."""
        state = self._state(tmp_path)
        assert not state.amend_grand_plan({"target": {"id": "obj_002"}}, "because", 1)
        assert "cannot be changed" in state.last_refusal

    def test_it_survives_a_reload(self, tmp_path):
        state = self._state(tmp_path)
        state.retarget({"id": "obj_002", "label": "scissors"}, "knife unreachable", 1)
        back = st.SessionState.resume(tmp_path)
        assert back.target_id == "obj_002"
        assert back.retargets_used == 1
        assert not back.can_retarget()
