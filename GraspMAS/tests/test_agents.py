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
