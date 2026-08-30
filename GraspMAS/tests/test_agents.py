"""Planner / Coder / Observer, driven by a stub LLM.

No network and no free-tier requests. The point of these tests is the failure
handling that the upstream agents lacked: they parsed model output with bare
`.split()` and `json.loads`, so any model that formatted a reply differently
took down the whole round.
"""

import asyncio
import json

import pytest

from agents.coder import Coder
from agents.observer import Observer
from agents.planner import Planner
from agents.prompt import coder_prompt, observer_prompt, planner_prompt


class StubLLM:
    """Returns scripted replies and records how it was called."""

    system_prompt = "test system prompt"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def _next(self):
        return self.replies.pop(0) if self.replies else ""

    async def chat(self, system, user, agent="?", **params):
        self.calls.append({"agent": agent, "user": user, "vision": False, **params})
        return self._next()

    async def chat_with_image(self, system, user, b64, agent="?", **params):
        self.calls.append({"agent": agent, "user": user, "vision": True, **params})
        return self._next()


class TestPlanner:
    def test_parses_well_formed_reply(self):
        llm = StubLLM(["<thought>reasoning</thought><plan>Step 1: find mug</plan>"])
        p = Planner(planner_prompt, llm)
        thought, plan = asyncio.run(p(query="grasp the mug", previous_plan=None,
                                      observation=None))
        assert thought == "reasoning"
        assert plan == "Step 1: find mug"
        assert len(llm.calls) == 1

    def test_reprompts_once_when_tags_missing(self):
        llm = StubLLM([
            "I think you should find the mug first.",           # untagged
            "<thought>ok</thought><plan>Step 1: find mug</plan>",
        ])
        p = Planner(planner_prompt, llm)
        _, plan = asyncio.run(p(query="grasp the mug", previous_plan=None,
                                observation=None))
        assert plan == "Step 1: find mug"
        assert len(llm.calls) == 2
        assert "IMPORTANT" in llm.calls[1]["user"]

    def test_degrades_to_raw_text_rather_than_crashing(self):
        """Upstream raised IndexError here and lost the round."""
        llm = StubLLM(["no tags", "still no tags"])
        p = Planner(planner_prompt, llm)
        _, plan = asyncio.run(p(query="q", previous_plan=None, observation=None))
        assert plan == "still no tags"

    def test_handles_fenced_reply(self):
        llm = StubLLM(["```\n<thought>t</thought>\n<plan>p</plan>\n```"])
        p = Planner(planner_prompt, llm)
        thought, plan = asyncio.run(p(query="q", previous_plan=None, observation=None))
        assert thought == "t" and plan == "p"

    def test_uses_the_configured_agent_label(self):
        llm = StubLLM(["<thought>a</thought><plan>b</plan>"])
        asyncio.run(Planner(planner_prompt, llm)(query="q", previous_plan=None,
                                                 observation=None))
        # Upstream hardcoded model="gpt-4o" inside the call, ignoring
        # self.model_name; routing now goes through the shared client instead.
        assert llm.calls[0]["agent"] == "planner"


class TestCoder:
    def test_extracts_fenced_code(self):
        llm = StubLLM(["```python\ndef execute_command(image):\n    return None\n```"])
        code = asyncio.run(Coder(coder_prompt, llm)("plan"))
        assert code.startswith("def execute_command")

    def test_strips_preamble(self):
        llm = StubLLM([
            "Sure, here you go:\n```python\ndef execute_command(image):\n    return 1\n```"
        ])
        code = asyncio.run(Coder(coder_prompt, llm)("plan"))
        assert code.startswith("def execute_command")
        ns = {}
        exec(code, ns)
        assert ns["execute_command"](None) == 1

    def test_reprompts_when_no_function_defined(self):
        llm = StubLLM([
            "I cannot write that.",
            "```python\ndef execute_command(image):\n    return 2\n```",
        ])
        code = asyncio.run(Coder(coder_prompt, llm)("plan"))
        assert "def execute_command" in code
        assert len(llm.calls) == 2

    def test_penalties_are_passed_for_the_client_to_strip(self):
        """Coder still sends them; ChatLLM removes them per provider."""
        llm = StubLLM(["```python\ndef execute_command(image):\n    return 1\n```"])
        asyncio.run(Coder(coder_prompt, llm)("plan"))
        assert "presence_penalty" in llm.calls[0]


class TestObserver:
    def _results(self, tmp_path, with_image=True):
        img = None
        if with_image:
            import cv2
            import numpy as np
            img = tmp_path / "overlay.png"
            cv2.imwrite(str(img), np.zeros((32, 32, 3), np.uint8))
        return {
            "grasp": {"pose": [[1, 0, 0, 0]] * 4, "score": 0.8},
            "grasp_summary": {"score": 0.8},
            "image": str(img) if img else None,
            "error_logs": None,
        }

    def test_parses_tagged_json(self, tmp_path):
        payload = json.dumps({"verdict": "VALID", "summary": "looks good"})
        llm = StubLLM([f"<observation>{payload}</observation>"])
        obs = asyncio.run(Observer(observer_prompt, llm)(self._results(tmp_path), "q"))
        assert obs["verdict"] == "VALID"
        assert obs["summary"] == "looks good"
        assert llm.calls[0]["vision"] is True

    def test_parses_fenced_json_without_tags(self, tmp_path):
        llm = StubLLM(['```json\n{"verdict": "INVALID", "summary": "bad"}\n```'])
        obs = asyncio.run(Observer(observer_prompt, llm)(self._results(tmp_path), "q"))
        assert obs["verdict"] == "INVALID"

    def test_reprompts_on_unparseable_reply(self, tmp_path):
        llm = StubLLM([
            "The grasp looks fine to me.",
            '<observation>{"verdict": "VALID", "summary": "ok"}</observation>',
        ])
        obs = asyncio.run(Observer(observer_prompt, llm)(self._results(tmp_path), "q"))
        assert obs["verdict"] == "VALID"
        assert len(llm.calls) == 2

    def test_never_raises_on_persistent_garbage(self, tmp_path):
        """Upstream's bare json.loads raised and killed the loop."""
        llm = StubLLM(["nonsense", "more nonsense"])
        obs = asyncio.run(Observer(observer_prompt, llm)(self._results(tmp_path), "q"))
        assert "summary" in obs
        assert obs.get("parse_error") is True

    def test_falls_back_to_text_only_when_image_missing(self, tmp_path):
        """A missing PNG must degrade, not crash on encode_image(None)."""
        llm = StubLLM(['<observation>{"verdict": "INVALID", "summary": "no image"}</observation>'])
        results = self._results(tmp_path, with_image=False)
        obs = asyncio.run(Observer(observer_prompt, llm)(results, "q"))
        assert obs["verdict"] == "INVALID"
        assert llm.calls[0]["vision"] is False

    def test_summary_key_always_present(self, tmp_path):
        llm = StubLLM(['<observation>{"verdict": "VALID"}</observation>'])
        obs = asyncio.run(Observer(observer_prompt, llm)(self._results(tmp_path), "q"))
        assert "summary" in obs  # graspmas reads this unconditionally


class TestKeepsGrasp:
    """A grasp the Observer rejected must not survive the round.

    The loop used to do `grasp_pose = result["grasp"]` unconditionally on every
    round, with no reference to the verdict. So a query that ran out of rounds
    finished holding the last *rejected* pose and handed it to the executor —
    and a later, worse round silently replaced an earlier, better one.
    """

    def test_a_rejected_grasp_is_not_kept(self):
        from agents.graspmas import GraspMAS

        assert not GraspMAS._keeps_grasp({"verdict": "INVALID"})
        assert not GraspMAS._keeps_grasp({"verdict": "invalid"})

    def test_an_approved_grasp_is_kept(self):
        from agents.graspmas import GraspMAS

        assert GraspMAS._keeps_grasp({"verdict": "VALID"})

    def test_an_unreadable_verdict_does_not_discard_the_grasp(self):
        """A provider hiccup on the critique is no evidence against the pose.

        The Observer retries once and then degrades to raw text; treating that
        as a rejection would throw away sound grasps for a parsing failure.
        """
        from agents.graspmas import GraspMAS

        assert GraspMAS._keeps_grasp({"summary": "...", "parse_error": True})
        assert GraspMAS._keeps_grasp({})
        assert GraspMAS._keeps_grasp(None)


class TestObserverIsTold:
    """Everything `grasp_detection` measured and then logged to a file nobody reads.

    The Observer was asked to judge collisions and approach feasibility from a
    2D projection — where a hand in front of an object and a hand driven
    through it look identical — while the exact 3D answers sat in a
    `logger.warning`. One of those warnings literally reads "keeping them so
    the Observer can see why".
    """

    @staticmethod
    def _grasp(**filters):
        base = {
            "n_candidates": 416, "n_visible": 47, "n_on_target": 29,
            "n_clear_of_scene": 12, "collision_checked": True,
            "every_candidate_collides": False, "no_candidate_inside_mask": False,
            "part_found": None,
        }
        base.update(filters)
        return {"pose": [], "filters": base}

    def test_the_funnel_is_reported(self):
        from agents.graspmas import GraspMAS

        got = GraspMAS._selection_summary(self._grasp())
        assert got["candidates_generated"] == 416
        assert got["approaching_from_the_observed_side"] == 47
        assert got["landing_on_the_requested_region"] == 29
        assert got["clear_of_the_rest_of_the_scene"] == 12

    def test_a_clean_selection_raises_no_warning(self):
        """Only dropped constraints are worth a line in a rate-limited prompt."""
        from agents.graspmas import GraspMAS

        assert GraspMAS._selection_warnings(self._grasp()) is None

    def test_every_candidate_colliding_is_reported(self):
        from agents.graspmas import GraspMAS

        note = GraspMAS._selection_warnings(
            self._grasp(every_candidate_collides=True)
        )
        assert "COLLISION" in note
        assert "least bad" in note

    def test_nothing_inside_the_mask_is_reported(self):
        from agents.graspmas import GraspMAS

        note = GraspMAS._selection_warnings(
            self._grasp(no_candidate_inside_mask=True)
        )
        assert "OFF-TARGET" in note

    def test_a_missing_part_is_reported(self):
        """find_part silently substitutes the whole object; that must be visible."""
        from agents.graspmas import GraspMAS

        note = GraspMAS._selection_warnings(self._grasp(part_found=False))
        assert "WRONG REGION" in note
        assert "NOT on the part" in note

    def test_a_found_part_raises_no_warning(self):
        from agents.graspmas import GraspMAS

        assert GraspMAS._selection_warnings(self._grasp(part_found=True)) is None

    def test_a_grasp_without_provenance_degrades(self):
        from agents.graspmas import GraspMAS

        assert GraspMAS._selection_summary({"pose": []}) is None
        assert GraspMAS._selection_warnings(None) is None


class TestPlannerSeesTheWholeJudgement:
    """The Planner was handed one prose sentence and forbidden the rest.

    `observation.get("summary", "")` threw away the verdict, the failure kind
    and all five checklist fields, because the prompt said "do not directly
    read verdict, checklist, or JSON". The Planner then reconstructed pass/fail
    from prose and behaved as a switch, which is about all a sentence carries.
    """

    def test_every_field_survives(self):
        from agents.graspmas import GraspMAS

        text = GraspMAS._observation_for_planner({
            "verdict": "INVALID",
            "failure": "geometry",
            "checklist": {"target_match": "yes", "collision_risk": "yes"},
            "error_logs": "COLLISION: all 29 grasps hit something",
            "summary": "blocked by the red cylinder",
        })
        assert "INVALID" in text
        assert "geometry" in text
        assert "collision_risk: yes" in text
        assert "COLLISION" in text
        assert "blocked by the red cylinder" in text

    def test_a_clean_observation_stays_short(self):
        """No failure kind and no errors means no noise."""
        from agents.graspmas import GraspMAS

        text = GraspMAS._observation_for_planner({
            "verdict": "VALID", "failure": "none",
            "error_logs": "none", "summary": "looks good",
        })
        assert "FAILURE KIND" not in text
        assert "ERRORS" not in text
        assert "VALID" in text and "looks good" in text

    def test_an_empty_observation_does_not_raise(self):
        from agents.graspmas import GraspMAS

        assert GraspMAS._observation_for_planner({}) == ""
        assert GraspMAS._observation_for_planner(None) == ""
