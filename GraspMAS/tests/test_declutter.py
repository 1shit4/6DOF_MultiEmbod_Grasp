"""The outer loop and its planner.

The loop is driven with a real executor and a real registry throughout — only
the LLM is stubbed — so what is tested is the machinery reaching a real
conclusion on a real scene, not a mock agreeing with itself.

The properties that matter most are the negative ones: it must terminate, it
must not move the target, and it must not claim success it did not achieve.
"""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

import synth_scene as ss
from agents.prompt import task_planner_prompt
from agents.task_planner import Decision, TaskPlanner, format_blocking
from declutter import DeclutterLoop
from execution import MutationExecutor
from scene_registry import SceneRegistry
from session_state import SessionState


class StubLLM:
    """Replays scripted replies, recording what it was asked."""

    system_prompt = "test"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    async def chat(self, system, user, **kw):
        self.prompts.append(user)
        return self.replies.pop(0) if self.replies else ""

    async def chat_with_image(self, system, user, image, **kw):
        return await self.chat(system, user, **kw)


def _decision_json(**kw):
    payload = {
        "action": "remove", "object_id": "obj_002", "subgoal": "move it",
        "rationale": "it is in the way", "grand_plan_update": None,
    }
    payload.update(kw)
    return f"<decision>{json.dumps(payload)}</decision>"


def _loop(tmp_path, planner=None, inject=(), max_iterations=6, scenario="occluded_target",
          inject_at=0):
    executor = MutationExecutor(
        ss.SCENARIOS[scenario](), inject=inject, inject_at=inject_at, seed=0
    )
    state = SessionState(tmp_path / "run")
    return DeclutterLoop(
        executor=executor, state=state, planner=planner,
        max_iterations=max_iterations,
    ), executor, state


# ---------------------------------------------------------------------------


class TestScriptedLoop:
    """The no-LLM path: the whole loop, spending zero requests."""

    @pytest.mark.asyncio
    async def test_reaches_the_target(self, tmp_path):
        loop, executor, state = _loop(tmp_path)
        result = await loop.run("pick up the banana", "banana")

        assert result.status == "success", result.reason
        assert sorted(result.moved) == sorted(set(result.moved))
        assert len(result.moved) == 2, f"expected two blockers, moved {result.moved}"
        assert result.grasp is not None

    @pytest.mark.asyncio
    async def test_never_moves_the_target(self, tmp_path):
        loop, executor, state = _loop(tmp_path)
        before = executor.true_position("banana")
        await loop.run("pick up the banana", "banana")
        assert executor.true_position("banana") == pytest.approx(before)

    @pytest.mark.asyncio
    async def test_never_moves_the_distractor(self, tmp_path):
        loop, executor, state = _loop(tmp_path)
        before = executor.true_position("box")
        await loop.run("pick up the banana", "banana")
        assert executor.true_position("box") == pytest.approx(before)

    @pytest.mark.asyncio
    async def test_writes_a_faithful_record(self, tmp_path):
        loop, executor, state = _loop(tmp_path)
        await loop.run("pick up the banana", "banana")

        blob = json.loads((tmp_path / "run" / "progress.json").read_text())
        assert blob["status"] == "success"
        assert len(blob["iterations"]) == 3  # two removals plus the final grasp
        for record in blob["iterations"][:2]:
            assert record["action"] == "remove"
            assert record["planned"]["grasp"] is not None
            assert record["planned"]["place"] is not None
            assert record["evaluation"]["action_succeeded"] == "success"
            assert record["evaluation"]["still_blocking_target"] is False

    @pytest.mark.asyncio
    async def test_writes_a_grand_plan(self, tmp_path):
        loop, executor, state = _loop(tmp_path)
        await loop.run("pick up the banana", "banana")
        blob = json.loads((tmp_path / "run" / "grand_plan.json").read_text())
        assert len(blob["removal_order"]) == 2
        assert blob["goal"] == "pick up the banana"

    @pytest.mark.asyncio
    async def test_an_absent_target_fails_immediately(self, tmp_path):
        loop, executor, state = _loop(tmp_path)
        result = await loop.run("pick up the elephant", "elephant")
        assert result.status == "failed"
        assert "not in the scene" in result.reason
        assert result.iterations == 0


class TestTermination:
    """Every path out of the loop, because the one that matters is failure."""

    @pytest.mark.asyncio
    async def test_a_grasp_that_never_takes_hold_stalls_out(self, tmp_path):
        # `inject_at=None` is the point of this test: a fault that never lifts.
        # A one-shot drop is recoverable and is covered by
        # TestRecoversFromAOneShotFault.
        loop, executor, state = _loop(tmp_path, inject=["drop"], inject_at=None)
        result = await loop.run("pick up the banana", "banana")

        assert result.status == "failed"
        assert "no progress" in result.reason
        assert result.iterations <= 3, "stall detection must fire before the cap"
        assert executor.true_position("banana") is not None

    @pytest.mark.asyncio
    async def test_the_iteration_cap_is_honoured(self, tmp_path):
        loop, executor, state = _loop(tmp_path, inject=["drop"], max_iterations=1)
        result = await loop.run("pick up the banana", "banana")
        assert result.status == "failed"
        assert result.iterations <= 1

    @pytest.mark.asyncio
    async def test_an_imprecise_release_still_counts_as_moved(self, tmp_path):
        """Landing 8 cm off plan is not success, but it is not 'nothing moved'."""
        loop, executor, state = _loop(tmp_path, inject=["offset"])
        result = await loop.run("pick up the banana", "banana")
        assert result.moved, "objects clearly relocated must be reported as moved"

    @pytest.mark.asyncio
    async def test_collateral_damage_does_not_derail_the_run(self, tmp_path):
        loop, executor, state = _loop(tmp_path, inject=["collateral"])
        result = await loop.run("pick up the banana", "banana")
        assert result.status in ("success", "failed")
        assert state.progress["status"] in ("success", "failed", "aborted")

    @pytest.mark.asyncio
    async def test_grasping_the_wrong_object_is_caught(self, tmp_path):
        loop, executor, state = _loop(tmp_path, inject=["wrong_object"])
        result = await loop.run("pick up the banana", "banana")
        assert result.status != "success" or result.grasp is not None

    @pytest.mark.asyncio
    async def test_a_crowded_table_aborts_rather_than_spinning(self, tmp_path):
        """Nowhere to put anything is a real answer, and must be reached quickly."""
        loop, executor, state = _loop(tmp_path, scenario="crowded_table")
        result = await loop.run("pick up a block", "blk_3_2")
        assert result.status in ("failed", "aborted")
        assert result.iterations <= 4


class TestIdentityAcrossIterations:
    @pytest.mark.asyncio
    async def test_a_deliberately_moved_object_keeps_its_id(self, tmp_path):
        """A pick-and-place moves things 20-30 cm, far past the match radius.

        Without the destination hint the object is re-registered under a new id,
        reads as 'missing', and the loop cannot tell a successful move from a
        vanished object.
        """
        loop, executor, state = _loop(tmp_path)
        await loop.run("pick up the banana", "banana")

        for record in state.iterations:
            if record.action == "remove":
                assert record.evaluation["action_succeeded"] != "object_missing", (
                    "a moved object must be recognised where it was put"
                )

    @pytest.mark.asyncio
    async def test_a_failed_move_also_keeps_its_id(self, tmp_path):
        """The hint must be additive: a dropped object never left its old spot."""
        loop, executor, state = _loop(tmp_path, inject=["drop"], inject_at=None)
        await loop.run("pick up the banana", "banana")

        verdicts = [
            r.evaluation.get("action_succeeded")
            for r in state.iterations if r.action == "remove"
        ]
        assert verdicts, "the loop should have attempted at least one removal"
        assert all(v == "not_moved" for v in verdicts), verdicts


class TestTaskPlannerDecisions:
    @pytest.mark.asyncio
    async def test_parses_a_well_formed_decision(self, tmp_path):
        llm = StubLLM(_decision_json(object_id="obj_002", rationale="tallest blocker"))
        planner = TaskPlanner(llm)
        d = await planner("goal", "obj_001", "table", "blocking", "history")
        assert d.action == "remove" and d.object_id == "obj_002"
        assert d.rationale == "tallest blocker"

    @pytest.mark.asyncio
    async def test_tolerates_markdown_fences(self, tmp_path):
        llm = StubLLM("```json\n" + _decision_json() + "\n```")
        d = await TaskPlanner(llm)("g", "obj_001", "t", "b", "h")
        assert d.action == "remove"

    @pytest.mark.asyncio
    async def test_reprompts_once_then_aborts_safely(self):
        llm = StubLLM("not json at all", "still not json")
        d = await TaskPlanner(llm)("g", "obj_001", "t", "b", "h")
        assert len(llm.prompts) == 2, "exactly one corrective re-prompt"
        assert d.action == "abort", "guessing an object to move would be worse"
        assert "could not be parsed" in d.rationale

    @pytest.mark.asyncio
    async def test_a_recovered_reprompt_is_used(self):
        llm = StubLLM("garbage", _decision_json(object_id="obj_004"))
        d = await TaskPlanner(llm)("g", "obj_001", "t", "b", "h")
        assert d.object_id == "obj_004"

    @pytest.mark.asyncio
    async def test_grand_plan_degrades_to_empty(self):
        llm = StubLLM("no plan here")
        plan = await TaskPlanner(llm).make_grand_plan("g", "obj_001", "t", "b")
        assert plan["removal_order"] == []

    @pytest.mark.asyncio
    async def test_grand_plan_is_parsed(self):
        llm = StubLLM(
            '<plan>{"removal_order": [{"object_id": "obj_002", "label": "bottle"}], '
            '"success_criterion": "clear", "reasoning": "tallest first"}</plan>'
        )
        plan = await TaskPlanner(llm).make_grand_plan("g", "obj_001", "t", "b")
        assert plan["removal_order"][0]["object_id"] == "obj_002"
        assert plan["reasoning"] == "tallest first"


class TestDecisionValidation:
    """A planner's decision is checked against the scene, never trusted."""

    @pytest.fixture
    def registry(self, tabletop):
        reg = SceneRegistry()
        reg.update_from_segmentation(
            tabletop["depth"], tabletop["K"], tabletop["seg"], 0, tabletop["label_map"]
        )
        return reg

    def test_a_valid_removal_passes(self, registry):
        oid = next(i for i in sorted(registry.instances) if i != "obj_001")
        d = TaskPlanner.validate(Decision("remove", oid), registry, "obj_001")
        assert d.action == "remove" and not d.corrections

    def test_an_unknown_object_aborts(self, registry):
        d = TaskPlanner.validate(Decision("remove", "obj_999"), registry, "obj_001")
        assert d.action == "abort"
        assert "not in the scene" in d.corrections[0]

    def test_removing_the_target_is_refused(self, registry):
        d = TaskPlanner.validate(Decision("remove", "obj_001"), registry, "obj_001")
        assert d.action == "abort"
        assert "the target itself" in d.corrections[0]

    def test_removal_without_an_object_aborts(self, registry):
        d = TaskPlanner.validate(Decision("remove", None), registry, "obj_001")
        assert d.action == "abort"

    def test_an_unknown_action_aborts(self, registry):
        d = TaskPlanner.validate(Decision("teleport", "obj_002"), registry, "obj_001")
        assert d.action == "abort"
        assert "unknown action" in d.corrections[0]

    def test_terminal_actions_drop_any_object_id(self, registry):
        d = TaskPlanner.validate(Decision("grasp_target", "obj_002"), registry, "obj_001")
        assert d.action == "grasp_target" and d.object_id is None

    def test_two_failures_on_one_object_stop_a_third(self, registry, tmp_path):
        state = SessionState(tmp_path / "run", autosave=False)
        state.start("goal")
        oid = sorted(registry.instances)[1]
        for i in range(2):
            state.begin_iteration(i, action="remove", object_id=oid)
            state.record_evaluation({"action_succeeded": "not_moved"})
            state.end_iteration()

        d = TaskPlanner.validate(Decision("remove", oid), registry, "obj_001", state)
        assert d.action == "abort"
        assert "already failed" in d.corrections[0]

    def test_one_failure_does_not_stop_a_retry(self, registry, tmp_path):
        state = SessionState(tmp_path / "run", autosave=False)
        state.start("goal")
        oid = sorted(registry.instances)[1]
        state.begin_iteration(0, action="remove", object_id=oid)
        state.record_evaluation({"action_succeeded": "not_moved"})
        state.end_iteration()

        d = TaskPlanner.validate(Decision("remove", oid), registry, "obj_001", state)
        assert d.action == "remove"

    def test_validation_does_not_mutate_the_input(self, registry):
        original = Decision("remove", "obj_999")
        TaskPlanner.validate(original, registry, "obj_001")
        assert original.action == "remove"


class TestLlmDrivenLoop:
    @pytest.mark.asyncio
    async def test_follows_the_planner_and_finishes(self, tmp_path, tabletop):
        """A scripted planner, so the loop's plumbing is what is under test."""
        reg = SceneRegistry()
        reg.update_from_segmentation(
            tabletop["depth"], tabletop["K"], tabletop["seg"], 0, tabletop["label_map"]
        )
        ids = {i.label: i.id for i in reg.instances.values()}

        llm = StubLLM(
            '<plan>{"removal_order": [], "success_criterion": "clear", "reasoning": ""}</plan>',
            _decision_json(object_id=ids["bottle"]),
            _decision_json(object_id=ids["mug"]),
            _decision_json(action="grasp_target", object_id=None),
        )
        loop, executor, state = _loop(tmp_path, planner=TaskPlanner(llm))
        result = await loop.run("pick up the banana", "banana")

        assert result.status == "success", result.reason
        assert sorted(result.moved) == sorted([ids["bottle"], ids["mug"]])

    @pytest.mark.asyncio
    async def test_an_aborting_planner_is_obeyed(self, tmp_path):
        llm = StubLLM(
            '<plan>{"removal_order": [], "success_criterion": "", "reasoning": ""}</plan>',
            _decision_json(action="abort", object_id=None,
                           rationale="the table is unworkable"),
        )
        loop, executor, state = _loop(tmp_path, planner=TaskPlanner(llm))
        result = await loop.run("pick up the banana", "banana")
        assert result.status == "aborted"
        assert "unworkable" in result.reason

    @pytest.mark.asyncio
    async def test_a_rejected_plan_amendment_is_recorded(self, tmp_path):
        llm = StubLLM(
            '<plan>{"removal_order": [], "success_criterion": "", "reasoning": ""}</plan>',
            _decision_json(
                action="abort", object_id=None,
                grand_plan_update={"goal": "something easier", "reason": "too hard"},
            ),
        )
        loop, executor, state = _loop(tmp_path, planner=TaskPlanner(llm))
        await loop.run("pick up the banana", "banana")

        notes = " ".join(n for r in state.iterations for n in r.notes)
        assert "rejected" in notes
        assert state.grand_plan["goal"] == "pick up the banana"

    @pytest.mark.asyncio
    async def test_planner_corrections_land_in_the_record(self, tmp_path):
        llm = StubLLM(
            '<plan>{"removal_order": [], "success_criterion": "", "reasoning": ""}</plan>',
            _decision_json(object_id="obj_404"),
        )
        loop, executor, state = _loop(tmp_path, planner=TaskPlanner(llm))
        result = await loop.run("pick up the banana", "banana")

        assert result.status == "aborted"
        notes = " ".join(n for r in state.iterations for n in r.notes)
        assert "planner corrected" in notes


class TestInnerLoopInstruction:
    """The instance id must reach the Coder, or `find_by_id` is unreachable.

    Found by running the loop against a live model: the Coder emitted
    `image_patch.find("mug")` because the query it was handed said "the mug".
    With unique labels that is merely redundant; with two bottles it silently
    grasps whichever one detection happens to list first.
    """

    def test_instruction_names_the_instance_id(self):
        inst = SimpleNamespace(id="obj_003", label="bottle", descriptor="the left bottle")
        text = DeclutterLoop._instruction(inst)
        assert "obj_003" in text
        assert "the left bottle" in text

    def test_instruction_falls_back_to_the_label(self):
        inst = SimpleNamespace(id="obj_007", label="mug", descriptor=None)
        assert DeclutterLoop._instruction(inst) == "obj_007 (mug)"

    def test_instruction_is_just_the_id_when_nothing_else_is_known(self):
        inst = SimpleNamespace(id="obj_009", label="", descriptor="")
        assert DeclutterLoop._instruction(inst) == "obj_009"

    @pytest.mark.asyncio
    async def test_the_loop_asks_the_agents_for_an_id(self, tmp_path, tabletop):
        """End to end: whatever reaches GraspMAS.query must carry the id."""
        reg = SceneRegistry()
        reg.update_from_segmentation(
            tabletop["depth"], tabletop["K"], tabletop["seg"], 0, tabletop["label_map"]
        )
        ids = {i.label: i.id for i in reg.instances.values()}

        asked = []

        class RecordingGraspMAS:
            def reset(self):
                pass

            async def query(self, query, *args, **kwargs):
                asked.append(query)
                return None, None  # force the nominal-grasp fallback

        llm = StubLLM(
            '<plan>{"removal_order": [], "success_criterion": "c", "reasoning": ""}</plan>',
            _decision_json(object_id=ids["bottle"]),
            _decision_json(action="abort", object_id=None),
        )
        loop, _executor, _state = _loop(tmp_path, planner=TaskPlanner(llm))
        loop.graspmas = RecordingGraspMAS()
        await loop.run("pick up the banana", "banana")

        assert asked, "the inner loop was never consulted"
        assert all(q.startswith("obj_") for q in asked), asked


class TestArtifactScoping:
    """Long-horizon runs must not overwrite their own visual record.

    The inner loop names files by inner round (`round0_overlay.png`), which is
    unique within one query and repeats on every outer iteration.
    """

    def test_scope_namespaces_images_and_arrays(self, tmp_path):
        from run_artifacts import RunRecorder

        rec = RunRecorder(name="scoping", root=tmp_path)
        unscoped = rec.image_path("round0_overlay.png")
        rec.set_scope("iter1")
        scoped = rec.image_path("round0_overlay.png")

        assert unscoped != scoped
        assert scoped.parent.name == "iter1"

        rec.save_mask(np.ones((4, 4), bool), "bottle", round_idx=0)
        assert (rec.dir / "masks" / "iter1_round0_bottle.npy").is_file()

    def test_clearing_the_scope_restores_the_flat_layout(self, tmp_path):
        from run_artifacts import RunRecorder

        rec = RunRecorder(name="scoping", root=tmp_path)
        rec.set_scope("iter2")
        rec.set_scope(None)
        assert rec.image_path("x.png") == rec.dir / "images" / "x.png"

    @pytest.mark.asyncio
    async def test_each_iteration_gets_its_own_image_directory(self, tmp_path, tabletop):
        """The LLM path photographs the scene every iteration; keep all of them."""
        from run_artifacts import RunRecorder

        reg = SceneRegistry()
        reg.update_from_segmentation(
            tabletop["depth"], tabletop["K"], tabletop["seg"], 0, tabletop["label_map"]
        )
        ids = {i.label: i.id for i in reg.instances.values()}

        llm = StubLLM(
            '<plan>{"removal_order": [], "success_criterion": "c", "reasoning": ""}</plan>',
            _decision_json(object_id=ids["bottle"]),
            _decision_json(object_id=ids["mug"]),
            _decision_json(action="grasp_target", object_id=None),
        )
        rec = RunRecorder(name="declutter", root=tmp_path)
        loop, _executor, _state = _loop(tmp_path, planner=TaskPlanner(llm))
        loop.recorder = rec
        result = await loop.run("pick up the banana", "banana")

        scopes = sorted(d.name for d in (rec.dir / "images").iterdir() if d.is_dir())
        assert scopes == [f"iter{i}" for i in range(result.iterations)], scopes
        # Every iteration kept its own photograph rather than overwriting one.
        before = list((rec.dir / "images").rglob("scene_iter*_before.png"))
        assert len(before) == result.iterations
        # And every iteration that actually moved something photographed the
        # result, which is the only view of what the action did.
        after = list((rec.dir / "images").rglob("scene_iter*_after.png"))
        assert len(after) >= 1
        assert len(after) < len(before), "the final grasp_target has nothing to execute"

    @pytest.mark.asyncio
    async def test_state_is_snapshotted_per_iteration(self, tmp_path):
        """`progress.json` is rewritten in place, so the end state is all that
        survives without a per-iteration freeze."""
        loop, _executor, state = _loop(tmp_path)
        result = await loop.run("pick up the banana", "banana")

        snaps = sorted(d.name for d in (state.run_dir / "states").iterdir())
        assert snaps == [f"iter{i}" for i in range(result.iterations)], snaps

        # Each snapshot holds the history as it stood *then*, not at the end.
        for i, name in enumerate(snaps):
            data = json.loads(
                (state.run_dir / "states" / name / "progress.json").read_text()
            )
            assert len(data["iterations"]) == i + 1


class TestSurvivesAnLlmOutage:
    """A provider outage must end the run, not crash it.

    Found by running against a live free tier: the daily quota was exhausted
    mid-run, the 429 propagated out of `run()`, and the process died with
    `progress.json` still reading "in_progress" — no verdict, no reason, and a
    half-open iteration for `--resume` to trip over.
    """

    class _DeadLLM:
        system_prompt = "s"

        async def chat(self, *a, **k):
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

        async def chat_with_image(self, *a, **k):
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

    @pytest.mark.asyncio
    async def test_run_aborts_cleanly_and_states_why(self, tmp_path):
        loop, _executor, state = _loop(
            tmp_path, planner=TaskPlanner(self._DeadLLM())
        )
        result = await loop.run("pick up the banana", "banana")

        assert result.status == "aborted"
        assert "could not be reached" in result.reason
        # The record must be closed and readable, not left mid-iteration.
        progress = json.loads((state.run_dir / "progress.json").read_text())
        assert progress["status"] == "aborted"
        assert all(rec.get("ended_at") for rec in progress["iterations"])

    @pytest.mark.asyncio
    async def test_a_failed_grand_plan_falls_back_to_geometry(self, tmp_path):
        """The plan is guidance; losing it must not stop the run starting."""
        loop, _executor, state = _loop(
            tmp_path, planner=TaskPlanner(self._DeadLLM())
        )
        await loop.run("pick up the banana", "banana")

        assert state.grand_plan is not None
        assert state.grand_plan["removal_order"], "blockers should still be listed"
        assert "unreachable" in state.grand_plan.get("reasoning", "")

    @pytest.mark.asyncio
    async def test_the_run_is_resumable_afterwards(self, tmp_path):
        loop, _executor, state = _loop(
            tmp_path, planner=TaskPlanner(self._DeadLLM())
        )
        await loop.run("pick up the banana", "banana")
        reloaded = SessionState.resume(state.run_dir)
        assert reloaded.progress["status"] == "aborted"


class TestReportCapture:
    """Everything a run needs to be explained afterwards, not just summarised."""

    @pytest.mark.asyncio
    async def test_the_outer_decision_is_recorded(self, tmp_path, tabletop):
        reg = SceneRegistry()
        reg.update_from_segmentation(
            tabletop["depth"], tabletop["K"], tabletop["seg"], 0, tabletop["label_map"]
        )
        ids = {i.label: i.id for i in reg.instances.values()}
        llm = StubLLM(
            '<plan>{"removal_order": [], "success_criterion": "c", "reasoning": ""}</plan>',
            _decision_json(object_id=ids["bottle"]),
            _decision_json(action="abort", object_id=None),
        )
        loop, _executor, state = _loop(tmp_path, planner=TaskPlanner(llm))
        await loop.run("pick up the banana", "banana")

        first = state.iterations[0]
        assert first.decision["action"] == "remove"
        assert first.decision["object_id"] == ids["bottle"]
        assert first.decision["rationale"]

    @pytest.mark.asyncio
    async def test_inner_rounds_are_kept_not_overwritten(self, tmp_path):
        """`GraspMAS` overwrites thought/plan/code each round; the record must
        keep every round or a two-round query looks like a one-round one."""
        class TwoRoundGraspMAS:
            def __init__(self):
                self.rounds = []
                self.observation_json = {"verdict": "VALID"}

            def reset(self):
                self.rounds = []

            async def query(self, *a, **k):
                self.rounds = [
                    {"round": 0, "thought": "first look", "plan": "find it",
                     "code": "def execute_command(image): ...", "grasp": None},
                    {"round": 1, "thought": "better now", "plan": "return to user",
                     "concluded": True},
                ]
                return None, None

        loop, _executor, state = _loop(tmp_path)
        loop.graspmas = TwoRoundGraspMAS()
        await loop.run("pick up the banana", "banana")

        rounds = state.iterations[0].agent_rounds
        assert [r["round"] for r in rounds] == [0, 1]
        assert rounds[0]["thought"] == "first look"
        assert rounds[1]["concluded"] is True

    def test_llm_calls_carry_the_iteration_they_belong_to(self, tmp_path):
        """Without this the trace is flat and no report can attribute a call."""
        from run_artifacts import RunRecorder

        rec = RunRecorder(name="trace", root=tmp_path)
        rec.set_scope("iter2")
        rec.log_llm_call(agent="coder", model="m", provider="p",
                         prompt="p", response="r", latency_s=0.1)
        line = json.loads((rec.dir / "llm_trace.jsonl").read_text().splitlines()[0])
        assert line["iteration"] == "iter2"

    def test_calls_before_any_iteration_are_marked_as_such(self, tmp_path):
        from run_artifacts import RunRecorder

        rec = RunRecorder(name="trace", root=tmp_path)
        rec.log_llm_call(agent="task_planner", model="m", provider="p",
                         prompt="p", response="r", latency_s=0.1)
        line = json.loads((rec.dir / "llm_trace.jsonl").read_text().splitlines()[0])
        assert line["iteration"] is None


class TestRecoversFromAOneShotFault:
    """The point of injection: does the loop notice and carry on?

    Measured on `occluded_target` with the scripted planner — every mode
    recovers and still reaches the target, costing at most one extra iteration.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["drop", "offset", "collateral", "wrong_object", "tip"])
    async def test_recovers_and_reaches_the_target(self, tmp_path, mode):
        executor = MutationExecutor(ss.occluded_target_scene(), inject=[mode], seed=0)
        state = SessionState(tmp_path / f"run_{mode}")
        loop = DeclutterLoop(executor=executor, state=state, max_iterations=6)
        result = await loop.run("pick up the banana", "banana")

        assert result.status == "success", f"{mode}: {result.reason}"
        assert result.iterations <= 4, f"{mode} took {result.iterations}"

    @pytest.mark.asyncio
    async def test_a_persistent_fault_still_terminates(self, tmp_path):
        """The other half: a fault that never lifts must end the run."""
        executor = MutationExecutor(
            ss.occluded_target_scene(), inject=["drop"], inject_at=None, seed=0
        )
        state = SessionState(tmp_path / "run_persistent")
        loop = DeclutterLoop(executor=executor, state=state, max_iterations=6)
        result = await loop.run("pick up the banana", "banana")

        assert result.status == "failed"
        assert result.iterations <= 3, "it should not run to the cap"
        assert "same stage" in result.reason, result.reason

    @pytest.mark.asyncio
    async def test_the_failed_attempt_is_recorded_before_the_recovery(self, tmp_path):
        """Recovery must not erase the fault from the record."""
        executor = MutationExecutor(ss.occluded_target_scene(), inject=["drop"], seed=0)
        state = SessionState(tmp_path / "run_record")
        loop = DeclutterLoop(executor=executor, state=state, max_iterations=6)
        await loop.run("pick up the banana", "banana")

        first = state.iterations[0]
        assert first.evaluation["action_succeeded"] == "not_moved"
        assert first.evaluation["still_blocking_target"] is True
        assert "slipped" in (first.execution.get("error") or "")
        # And the very next attempt on the same object worked.
        second = state.iterations[1]
        assert second.object_id == first.object_id
        assert second.evaluation["action_succeeded"] == "success"


class TestBlockingSummary:
    def test_renders_each_reason(self, tabletop):
        reg = SceneRegistry()
        reg.update_from_segmentation(
            tabletop["depth"], tabletop["K"], tabletop["seg"], 0, tabletop["label_map"]
        )
        target = reg.resolve_target("banana")
        text = format_blocking(reg.blocking_objects(target.id), target)
        assert "occlusion" in text
        assert "% of the target's outline" in text
        assert "More blockers may appear" in text, "the caveat has to reach the planner"

    def test_says_plainly_when_nothing_blocks(self):
        assert "Nothing blocks" in format_blocking([])


class TestPromptContract:
    """These are `str.format` templates; a stray brace is a runtime KeyError."""

    @pytest.mark.parametrize(
        "template,variables",
        [
            ("TASK_PLAN", {"goal", "target_id", "scene_table", "blocking",
                           "progress", "examples"}),
            ("GRAND_PLAN", {"goal", "target_id", "scene_table", "blocking",
                            "candidates"}),
            ("SELECT_TARGET", {"goal", "scene_table"}),
        ],
    )
    def test_placeholders_are_exactly_as_expected(self, template, variables):
        import string

        text = getattr(task_planner_prompt, template)
        found = {f for _, f, _, _ in string.Formatter().parse(text) if f}
        assert found == variables

    @pytest.mark.parametrize("template",
                             ["TASK_PLAN", "GRAND_PLAN", "SELECT_TARGET"])
    def test_formats_without_raising(self, template):
        text = getattr(task_planner_prompt, template)
        import string

        fields = {f for _, f, _, _ in string.Formatter().parse(text) if f}
        text.format(**{f: "x" for f in fields})

    def test_examples_are_valid_json(self):
        """The in-context examples teach the output format, so they must be it."""
        import re

        blocks = re.findall(
            r"<decision>(.*?)</decision>",
            task_planner_prompt.EXAMPLES_TASK_PLANNER,
            re.S,
        )
        assert len(blocks) >= 4
        for block in blocks:
            payload = json.loads(block)
            assert payload["action"] in (
                "remove", "grasp_target", "retarget", "abort"
            )

    def test_examples_cover_every_action(self):
        text = task_planner_prompt.EXAMPLES_TASK_PLANNER
        for action in ("remove", "grasp_target", "abort"):
            assert f'"action": "{action}"' in text

    def test_examples_name_objects_by_id(self):
        """The rule the whole duplicate-object design depends on."""
        import re

        for block in re.findall(
            r"<decision>(.*?)</decision>", task_planner_prompt.EXAMPLES_TASK_PLANNER, re.S
        ):
            payload = json.loads(block)
            if payload["object_id"] is not None:
                assert re.fullmatch(r"obj_\d+", payload["object_id"])


# ---------------------------------------------------------------------------
# Stage 1: the planner works out what an abstract goal refers to
# ---------------------------------------------------------------------------


class _FakeInstance:
    def __init__(self, oid, label, n_points=500):
        self.id = oid
        self.label = label
        self.n_points = n_points


class _FakeRegistry:
    def __init__(self, instances):
        self.instances = {i.id: i for i in instances}

    def get(self, oid):
        return self.instances[oid]


class _ScriptedLLM:
    """Returns canned replies; records the prompts it was given."""

    system_prompt = "x"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    async def chat(self, system, user, **kw):
        self.prompts.append(user)
        return self.replies.pop(0) if self.replies else ""

    async def chat_with_image(self, system, user, b64, **kw):
        return await self.chat(system, user, **kw)


def _planner(*replies):
    from agents.task_planner import TaskPlanner

    return TaskPlanner(_ScriptedLLM(*replies))


class TestSelectTarget:
    def test_ranked_candidates_are_parsed(self):
        p = _planner("""<target>
        {"interpretation": "a cutting tool", "confidence": "high",
         "candidates": [{"object_id": "obj_001", "label": "knife", "why": "has an edge"},
                        {"object_id": "obj_003", "label": "scissors", "why": "also cuts"}]}
        </target>""")
        choice = asyncio.run(p.select_target("something to cut", "table"))
        assert choice.ids == ["obj_001", "obj_003"]
        assert choice.interpretation == "a cutting tool"
        assert choice.best.label == "knife"

    def test_bare_string_candidates_are_accepted(self):
        """Refusing a correct answer over its packaging would be a bad trade."""
        p = _planner('<target>{"candidates": ["obj_002"]}</target>')
        assert asyncio.run(p.select_target("g", "t")).ids == ["obj_002"]

    def test_more_than_three_candidates_are_truncated(self):
        ids = [f'{{"object_id": "obj_00{i}"}}' for i in range(1, 6)]
        p = _planner('<target>{"candidates": [%s]}</target>' % ",".join(ids))
        assert len(asyncio.run(p.select_target("g", "t")).candidates) == 3

    def test_unparseable_reply_degrades_rather_than_raising(self):
        """A run should fail with "nothing serves that goal", not a parse error."""
        p = _planner("I'm not sure what you mean.")
        assert asyncio.run(p.select_target("g", "t")).candidates == []

    def test_the_goal_and_scene_reach_the_prompt(self):
        p = _planner('<target>{"candidates": [{"object_id": "obj_001"}]}</target>')
        asyncio.run(p.select_target("I am hungry", "| obj_001 | the apple |"))
        prompt = p.llm.prompts[0]
        assert "I am hungry" in prompt and "the apple" in prompt


class TestValidateTarget:
    def _choice(self, *ids):
        from agents.task_planner import TargetCandidate, TargetChoice

        return TargetChoice(candidates=[TargetCandidate(object_id=i) for i in ids])

    def test_unknown_ids_are_dropped_with_a_correction(self):
        from agents.task_planner import TaskPlanner

        reg = _FakeRegistry([_FakeInstance("obj_001", "knife")])
        out = TaskPlanner.validate_target(self._choice("obj_001", "obj_999"), reg)
        assert out.ids == ["obj_001"]
        assert any("obj_999" in c for c in out.corrections)

    def test_too_sparse_an_instance_is_dropped(self):
        """Caught here rather than deep inside the placement search."""
        from agents.task_planner import TaskPlanner

        reg = _FakeRegistry([
            _FakeInstance("obj_001", "knife", n_points=5),
            _FakeInstance("obj_002", "apple", n_points=900),
        ])
        out = TaskPlanner.validate_target(self._choice("obj_001", "obj_002"), reg)
        assert out.ids == ["obj_002"]

    def test_ranking_order_is_preserved(self):
        from agents.task_planner import TaskPlanner

        reg = _FakeRegistry([_FakeInstance(f"obj_00{i}", "x") for i in (1, 2, 3)])
        out = TaskPlanner.validate_target(self._choice("obj_003", "obj_001"), reg)
        assert out.ids == ["obj_003", "obj_001"]

    def test_duplicates_collapse(self):
        from agents.task_planner import TaskPlanner

        reg = _FakeRegistry([_FakeInstance("obj_001", "x")])
        assert TaskPlanner.validate_target(self._choice("obj_001", "obj_001"), reg).ids \
            == ["obj_001"]

    def test_labels_are_filled_in_from_the_registry(self):
        from agents.task_planner import TaskPlanner

        reg = _FakeRegistry([_FakeInstance("obj_001", "knife")])
        assert TaskPlanner.validate_target(self._choice("obj_001"), reg).best.label \
            == "knife"


class _FakeAttempt:
    def __init__(self, succeeded):
        self.succeeded = succeeded


class _FakeState:
    """Just enough SessionState for the retarget gate."""

    def __init__(self, candidates=(), used=0, failures=0, limit=1):
        self.target_candidates = list(candidates)
        self.retargets_used = used
        self._failures = failures
        self._limit = limit

    def can_retarget(self):
        return self.retargets_used < self._limit

    def attempts_on(self, oid):
        return []

    def iterations_without_progress(self):
        return self._failures


class TestRetargetGate:
    """Every condition must hold; failure falls back to abort, never a silent switch.

    Switching what you fetch because the right thing was hard to reach is a
    worse failure than admitting you could not reach it, so the gate is
    deliberately hard to pass.
    """

    def _reg(self):
        return _FakeRegistry([
            _FakeInstance("obj_001", "knife"),
            _FakeInstance("obj_002", "scissors"),
        ])

    def _decide(self, **kw):
        return Decision(action="retarget", object_id=kw.pop("object_id", "obj_002"),
                        rationale="knife is unreachable", **kw)

    def test_a_well_founded_retarget_is_allowed(self):
        state = _FakeState(candidates=["obj_001", "obj_002"], failures=2)
        out = TaskPlanner.validate(self._decide(), self._reg(), "obj_001", state)
        assert out.action == "retarget" and out.object_id == "obj_002"

    def test_a_premature_retarget_is_declined_not_fatal(self):
        """Proposing it early is defensible; dying for it is not.

        Measured live: the model asked to switch after one failed attempt, with
        a sound explanation, and the run ended. Refusals about *timing* now
        defer; only structural impossibility aborts.
        """
        state = _FakeState(candidates=["obj_001", "obj_002"], failures=1)
        out = TaskPlanner.validate(self._decide(), self._reg(), "obj_001", state)
        assert out.action == "defer" and out.is_deferred
        assert not out.is_terminal
        assert any("declined" in c for c in out.corrections)

    def test_structural_refusals_still_abort(self):
        """No alternatives and a spent budget are not matters of timing."""
        for state in (
            _FakeState(candidates=[], failures=5),
            _FakeState(candidates=["obj_001", "obj_002"], failures=5, used=1),
        ):
            out = TaskPlanner.validate(self._decide(), self._reg(), "obj_001", state)
            assert out.action == "abort"

    def test_a_goal_that_named_one_object_cannot_retarget(self):
        """`--target banana` leaves no ranking, so there is nothing to switch to."""
        state = _FakeState(candidates=[], failures=5)
        out = TaskPlanner.validate(self._decide(), self._reg(), "obj_001", state)
        assert out.action == "abort"
        assert any("no alternative" in c for c in out.corrections)

    def test_the_budget_is_enforced(self):
        state = _FakeState(candidates=["obj_001", "obj_002"], failures=5, used=1)
        out = TaskPlanner.validate(self._decide(), self._reg(), "obj_001", state)
        assert out.action == "abort"
        assert any("budget is spent" in c for c in out.corrections)

    def test_an_unranked_id_falls_back_to_the_ranking(self):
        """The ranking is ours; it is what the model was told would be used."""
        state = _FakeState(candidates=["obj_001", "obj_002"], failures=2)
        out = TaskPlanner.validate(
            self._decide(object_id="obj_999"), self._reg(), "obj_001", state
        )
        assert out.action == "retarget" and out.object_id == "obj_002"
        assert any("not a ranked alternative" in c for c in out.corrections)

    def test_a_vanished_alternative_aborts(self):
        state = _FakeState(candidates=["obj_001", "obj_002"], failures=2)
        reg = _FakeRegistry([_FakeInstance("obj_001", "knife")])
        out = TaskPlanner.validate(self._decide(), reg, "obj_001", state)
        assert out.action == "abort"

    def test_without_run_state_it_aborts(self):
        out = TaskPlanner.validate(self._decide(), self._reg(), "obj_001", None)
        assert out.action == "abort"

    def test_retarget_is_a_known_action(self):
        from agents.task_planner import ACTIONS

        assert "retarget" in ACTIONS


class _RetargetingPlanner:
    """Chooses two cutters, then switches to the second once the first stalls.

    Stands in for the model on the one path a model has to volunteer. §7.11 is a
    list of what happens to paths only a model can take when nothing offline has
    ever run them — `find_by_id` was implemented, documented, and broken on its
    very first live call.
    """

    def __init__(self, first, second):
        self.first, self.second = first, second
        self.actions = []

    async def select_target(self, goal, scene_table, image_b64=None):
        from agents.task_planner import TargetCandidate, TargetChoice

        return TargetChoice(
            interpretation="a cutting tool",
            candidates=[
                TargetCandidate(object_id=self.first, label="knife"),
                TargetCandidate(object_id=self.second, label="scissors"),
            ],
        )

    async def make_grand_plan(self, goal, target_id, scene_table, blocking,
                              image_b64=None, candidates=""):
        return {"removal_order": [], "success_criterion": "cut something",
                "reasoning": "clear the knife", "target_id": ""}

    async def __call__(self, goal, target_id, scene_table, blocking, progress,
                       image_b64=None):
        from agents.task_planner import Decision

        if target_id == self.second:
            d = Decision(action="grasp_target", rationale="the scissors are clear")
        elif len(self.actions) >= 2:
            d = Decision(action="retarget", object_id=self.second,
                         rationale="the knife cannot be uncovered")
        else:
            d = Decision(action="remove", object_id=self._blocker,
                         rationale="clear the bottle")
        self.actions.append(d.action)
        return d


class TestRetargetThroughTheLoop:
    """End to end: a stalled run switches target and finishes on the alternative."""

    def _run(self, inject=("drop",)):
        import synth_scene as ss_
        from execution import MutationExecutor
        from session_state import SessionState
        import tempfile
        from pathlib import Path

        ex = MutationExecutor(ss_.SCENARIOS["affordance_choice"](),
                              inject=list(inject), inject_at=None, seed=0)
        state = SessionState(Path(tempfile.mkdtemp()))
        loop = DeclutterLoop(executor=ex, state=state, max_iterations=8)

        obs = loop._perceive(0)
        ids = {i.label: i.id for i in loop.registry.instances.values()}
        planner = _RetargetingPlanner(ids["knife"], ids["scissors"])
        planner._blocker = ids["bottle"]
        loop.planner = planner

        result = asyncio.run(loop.run("i need something to cut"))
        return result, state, planner, ids

    def test_the_run_switches_target_and_succeeds(self):
        result, state, planner, ids = self._run()
        assert "retarget" in planner.actions
        assert result.status == "success"
        assert state.target_id == ids["scissors"]

    def test_the_switch_is_recorded_with_its_reason(self):
        _, state, _, ids = self._run()
        assert state.retargets_used == 1
        rec = state.progress["retargets"][0]
        assert rec["from"]["id"] == ids["knife"]
        assert rec["to"]["id"] == ids["scissors"]
        assert "cannot be uncovered" in rec["reason"]

    def test_the_goal_never_changes(self):
        """The person's words are fixed; only our reading of them may move."""
        _, state, _, _ = self._run()
        assert state.progress["goal"] == "i need something to cut"
        assert state.grand_plan["goal"] == "i need something to cut"

    def test_the_old_plan_is_archived(self):
        _, state, _, ids = self._run()
        assert state.grand_plan["superseded"][0]["target"]["id"] == ids["knife"]

    def test_a_stall_is_deferred_only_once(self):
        """Otherwise a planner that never retargets would run to the cap."""
        _, state, _, _ = self._run()
        assert not state.can_retarget()


class TestTargetOrdering:
    """Stage 2 orders the candidates; it may not invent one.

    An earlier version let the grand plan name any target it liked, which was a
    second retargeting path with no evidence check, no budget and no record — and
    it promptly swapped a priority-1 knife for the scissors because they were
    "fully visible and unobstructed". Ordering a fixed candidate list is a
    different thing, and these pin the difference.
    """

    def _loop(self):
        import tempfile
        from pathlib import Path

        import synth_scene as ss_
        from execution import MutationExecutor
        from session_state import SessionState

        ex = MutationExecutor(ss_.SCENARIOS["affordance_choice"]())
        state = SessionState(Path(tempfile.mkdtemp()))
        loop = DeclutterLoop(executor=ex, state=state, max_iterations=4)
        loop._perceive(0)
        ids = {i.label: i.id for i in loop.registry.instances.values()}
        state.start("i need something to cut", {"id": ids["knife"], "label": "knife"})
        return loop, state, ids

    def _choice(self, ids, priorities=(1, 1)):
        from agents.task_planner import TargetCandidate, TargetChoice

        return TargetChoice(candidates=[
            TargetCandidate(object_id=ids["knife"], label="knife",
                            priority=priorities[0], n_blockers=1),
            TargetCandidate(object_id=ids["scissors"], label="scissors",
                            priority=priorities[1], n_blockers=0),
        ])

    def test_an_ordering_is_applied(self):
        loop, state, ids = self._loop()
        choice = self._choice(ids)
        out = loop._apply_target_order(
            [ids["scissors"], ids["knife"]], choice, ids["knife"]
        )
        assert out == ids["scissors"]
        assert state.target_id == ids["scissors"]
        assert choice.ids == [ids["scissors"], ids["knife"]]

    def test_an_invented_id_is_ignored_and_recorded(self):
        loop, state, ids = self._loop()
        out = loop._apply_target_order(["obj_999"], self._choice(ids), ids["knife"])
        assert out == ids["knife"]
        assert any("not candidates" in n for n in state.progress["run_notes"])

    def test_an_empty_order_changes_nothing(self):
        loop, state, ids = self._loop()
        assert loop._apply_target_order([], self._choice(ids), ids["knife"]) \
            == ids["knife"]

    def test_a_tie_break_on_effort_is_not_flagged(self):
        """Equal priority means either serves; taking the cheaper is free."""
        loop, state, ids = self._loop()
        loop._apply_target_order([ids["scissors"], ids["knife"]],
                                 self._choice(ids, priorities=(1, 1)), ids["knife"])
        assert not state.progress.get("run_notes")

    def test_crossing_a_priority_tier_is_recorded(self):
        """Allowed — the user asked for effort to count — but never silent."""
        loop, state, ids = self._loop()
        loop._apply_target_order([ids["scissors"], ids["knife"]],
                                 self._choice(ids, priorities=(1, 2)), ids["knife"])
        note = " ".join(state.progress["run_notes"])
        assert "traded against suitability" in note
        assert "priority-1" in note

    def test_an_unexpected_plan_key_does_not_crash_the_run(self):
        """`set_grand_plan` takes three fields; a model can emit a fourth.

        It did: `target_id` survived in the returned dict and took the run down
        with a TypeError raised from inside the state writer, naming neither the
        model nor the key.
        """
        import tempfile
        from pathlib import Path

        import synth_scene as ss_
        from execution import MutationExecutor
        from session_state import SessionState

        ex = MutationExecutor(ss_.SCENARIOS["affordance_choice"]())
        state = SessionState(Path(tempfile.mkdtemp()))
        loop = DeclutterLoop(executor=ex, state=state, max_iterations=3)
        obs = loop._perceive(0)
        ids = {i.label: i.id for i in loop.registry.instances.values()}
        state.start("g", {"id": ids["knife"], "label": "knife"})
        loop.planner = _planner(
            '<plan>{"removal_order": [], "target_id": "obj_x", "nonsense": 1}</plan>'
        )
        kept = asyncio.run(loop._draft_grand_plan("g", ids["knife"], obs, None))
        assert kept == ids["knife"]
        assert state.grand_plan is not None

    def test_the_order_becomes_the_retarget_ranking(self):
        """So a later retarget advances to whatever this decided was next."""
        loop, state, ids = self._loop()
        loop._apply_target_order([ids["scissors"], ids["knife"]],
                                 self._choice(ids), ids["knife"])
        assert state.target_candidates == [ids["scissors"], ids["knife"]]


class TestStageOnePriority:
    def test_priority_is_parsed_and_sorted(self):
        p = _planner("""<target>{"candidates": [
            {"object_id": "obj_005", "label": "card", "priority": 3},
            {"object_id": "obj_001", "label": "knife", "priority": 1}]}</target>""")
        choice = asyncio.run(p.select_target("something to cut", "table"))
        assert choice.ids == ["obj_001", "obj_005"]
        assert [c.priority for c in choice.candidates] == [1, 3]

    def test_ties_keep_the_models_order(self):
        """Equal priority is meaningful: it is what lets effort decide."""
        p = _planner("""<target>{"candidates": [
            {"object_id": "obj_001", "priority": 1},
            {"object_id": "obj_003", "priority": 1}]}</target>""")
        choice = asyncio.run(p.select_target("g", "t"))
        assert choice.ids == ["obj_001", "obj_003"]
        assert {c.priority for c in choice.candidates} == {1}

    def test_a_missing_or_junk_priority_defaults_to_one(self):
        p = _planner("""<target>{"candidates": [
            {"object_id": "obj_001"},
            {"object_id": "obj_002", "priority": "high"}]}</target>""")
        choice = asyncio.run(p.select_target("g", "t"))
        assert [c.priority for c in choice.candidates] == [1, 1]

    def test_priority_survives_validation(self):
        from agents.task_planner import TargetCandidate, TargetChoice, TaskPlanner

        reg = _FakeRegistry([_FakeInstance("obj_001", "knife")])
        choice = TargetChoice(candidates=[
            TargetCandidate(object_id="obj_001", priority=2)])
        assert TaskPlanner.validate_target(choice, reg).best.priority == 2

    def test_the_prompt_asks_for_priority_and_allows_ties(self):
        text = task_planner_prompt.SELECT_TARGET
        assert '"priority"' in text
        assert "Ties are allowed" in text

    def test_stage_one_is_not_shown_what_is_in_the_way(self):
        """Suitability and effort are judged separately, then weighed."""
        import string

        fields = {f for _, f, _, _ in
                  string.Formatter().parse(task_planner_prompt.SELECT_TARGET) if f}
        assert fields == {"goal", "scene_table"}
        assert "blocking" not in fields


class TestDeclinedRetargetDoesRealWork:
    """A declined retarget must cost neither the run nor the iteration.

    Two things were wrong with the first version. The iteration did nothing at
    all, so the run paid a whole cycle for a refusal; and because it recorded no
    evaluation, `iterations_without_progress` counted it as evidence — meaning a
    refusal supplied the very evidence justifying the next request, and one real
    failure plus one declined ask cleared a bar meant to need two failures.
    """

    class _AlwaysRetargets:
        system_prompt = "x"

        async def select_target(self, goal, scene_table, image_b64=None):
            from agents.task_planner import TargetCandidate, TargetChoice

            return TargetChoice(candidates=[
                TargetCandidate(object_id=self.first, priority=1),
                TargetCandidate(object_id=self.second, priority=1),
            ])

        async def make_grand_plan(self, *a, **kw):
            return {"removal_order": [], "success_criterion": "", "reasoning": "",
                    "target_order": []}

        async def __call__(self, goal, target_id, scene_table, blocking, progress,
                           image_b64=None):
            from agents.task_planner import Decision

            return Decision(action="retarget", object_id=self.second,
                            rationale="switching immediately")

    def _run(self, inject=("drop",)):
        import tempfile
        from pathlib import Path

        import synth_scene as ss_
        from execution import MutationExecutor
        from session_state import SessionState

        ex = MutationExecutor(ss_.SCENARIOS["affordance_choice"](),
                              inject=list(inject), inject_at=None, seed=0)
        state = SessionState(Path(tempfile.mkdtemp()))
        loop = DeclutterLoop(executor=ex, state=state, max_iterations=5)
        loop._perceive(0)
        ids = {i.label: i.id for i in loop.registry.instances.values()}
        planner = self._AlwaysRetargets()
        planner.first, planner.second = ids["knife"], ids["scissors"]
        loop.planner = planner
        return asyncio.run(loop.run("i need something to cut")), state, ids

    def test_the_iteration_acts_instead_of_idling(self):
        _, state, ids = self._run()
        first = state.iterations[0]
        assert first.action == "remove"
        assert first.object_id == ids["bottle"]
        assert first.evaluation, "an acting iteration must be evaluated"

    def test_the_refusal_is_still_recorded(self):
        _, state, _ = self._run()
        notes = " ".join(state.iterations[0].notes)
        assert "declined" in notes
        assert "falling back to the geometric choice" in notes

    def test_the_run_still_terminates(self):
        result, state, _ = self._run()
        assert result.status in ("failed", "aborted")
        assert len(state.iterations) < 5


class TestDeclineIsNotEvidence:
    """A refusal must not supply the evidence for the next request."""

    def _state(self, tmp_path):
        import session_state as ss_mod

        st = ss_mod.SessionState(tmp_path)
        st.start("g", {"id": "obj_001", "label": "knife"})
        return st

    def _record(self, st, index, action, evaluation=None):
        st.begin_iteration(index, action=action, object_id=None)
        if evaluation is not None:
            st.record_evaluation(evaluation)
        st.end_iteration()

    def test_non_acting_iterations_are_skipped(self, tmp_path):
        st = self._state(tmp_path)
        blocked = {"still_blocking_target": True, "target_blockers": ["obj_002"]}
        self._record(st, 0, "remove", blocked)
        self._record(st, 1, "defer")
        self._record(st, 2, "retarget")
        # One real failure, two decisions that moved nothing.
        assert st.iterations_without_progress() == 1

    def test_two_real_failures_do_count(self, tmp_path):
        st = self._state(tmp_path)
        blocked = {"still_blocking_target": True, "target_blockers": ["obj_002"]}
        self._record(st, 0, "remove", blocked)
        self._record(st, 1, "remove", blocked)
        assert st.iterations_without_progress() == 2

    def test_progress_still_resets_the_count(self, tmp_path):
        st = self._state(tmp_path)
        self._record(st, 0, "remove",
                     {"still_blocking_target": True, "target_blockers": ["a", "b"]})
        self._record(st, 1, "remove",
                     {"still_blocking_target": False, "target_blockers": []})
        assert st.iterations_without_progress() == 0
