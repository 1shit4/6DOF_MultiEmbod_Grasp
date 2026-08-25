"""agents.llm: config, pacing, retry/failover, and the response extractors.

No network. The OpenAI client is replaced with a stub, so retry policy and rate
limiting are tested as logic rather than by actually hammering a provider — and
the suite costs zero free-tier requests.
"""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
import yaml

import agents.llm as llm_mod
from agents.llm import (
    ChatLLM,
    extract_code,
    extract_json,
    extract_tag,
    reset_shared_llm,
    strip_code_fences,
)


# ---------------------------------------------------------------------------
# Extractors — these exist because the upstream parsers assumed GPT-4o's exact
# formatting and raised IndexError/JSONDecodeError on anything else.
# ---------------------------------------------------------------------------


class TestExtractTag:
    def test_plain_tag(self):
        assert extract_tag("<plan>do the thing</plan>", "plan") == "do the thing"

    def test_multiline(self):
        assert extract_tag("<plan>\nStep 1\nStep 2\n</plan>", "plan") == "Step 1\nStep 2"

    def test_tag_inside_code_fence(self):
        raw = "```\n<plan>fenced plan</plan>\n```"
        assert extract_tag(raw, "plan") == "fenced plan"

    def test_ignores_surrounding_prose(self):
        raw = "Sure! Here you go:\n<plan>the plan</plan>\nHope that helps."
        assert extract_tag(raw, "plan") == "the plan"

    def test_case_insensitive(self):
        assert extract_tag("<PLAN>x</PLAN>", "plan") == "x"

    def test_unclosed_tag_takes_remainder(self):
        # Truncation at max_tokens produces exactly this.
        assert extract_tag("<plan>unfinished plan", "plan") == "unfinished plan"

    def test_missing_tag_returns_none_not_raises(self):
        # The upstream `.split('<thought>')[1]` raised IndexError here.
        assert extract_tag("no tags at all", "plan") is None

    def test_empty_input(self):
        assert extract_tag("", "plan") is None
        assert extract_tag(None, "plan") is None


class TestExtractJson:
    def test_bare_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_prose_around_it(self):
        raw = 'Here is my assessment:\n{"verdict": "VALID"}\nLet me know.'
        assert extract_json(raw) == {"verdict": "VALID"}

    def test_nested_object(self):
        raw = '{"verdict": "VALID", "checklist": {"target_match": "yes"}}'
        assert extract_json(raw)["checklist"]["target_match"] == "yes"

    def test_invalid_returns_none_not_raises(self):
        assert extract_json("definitely not json") is None

    def test_empty_returns_none(self):
        assert extract_json("") is None
        assert extract_json(None) is None


class TestExtractCode:
    def test_fenced_python(self):
        raw = "```python\ndef execute_command(image):\n    return None\n```"
        assert extract_code(raw).startswith("def execute_command")

    def test_strips_preamble_prose(self):
        # Upstream's split('\n')[1:-1] mangled exactly this case.
        raw = "Here is the code:\n```python\ndef execute_command(image):\n    return 1\n```"
        code = extract_code(raw)
        assert code.startswith("def execute_command")
        assert "Here is the code" not in code

    def test_unfenced_code(self):
        raw = "def execute_command(image):\n    return 1"
        assert extract_code(raw).startswith("def execute_command")

    def test_result_is_executable(self):
        raw = "```python\ndef execute_command(image):\n    return 42\n```"
        ns = {}
        exec(extract_code(raw), ns)
        assert ns["execute_command"](None) == 42


def test_strip_code_fences_passthrough():
    assert strip_code_fences("no fences here") == "no fences here"


# ---------------------------------------------------------------------------
# Pacing now lives in the key pool: one sliding window per key rather than one
# for the whole process, so tests for window behaviour are in test_key_pool.py
# (TestRouting.test_no_sleep_while_any_key_has_headroom, test_window_slides,
# test_the_call_past_every_window_waits_exactly_once).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ChatLLM
# ---------------------------------------------------------------------------


@pytest.fixture
def config_file(tmp_path):
    cfg = {
        "provider": "test",
        "base_url": "https://example.invalid/v1/",
        "api_key_env": "TEST_LLM_KEY",
        "api_key_file": str(tmp_path / "missing.key"),
        "model": "test-model",
        "vision_model": "test-vision",
        "rate_limit_rpm": 600,
        "max_retries": 3,
        "retry_base_delay_s": 0.001,
        "unsupported_params": ["frequency_penalty", "presence_penalty"],
    }
    p = tmp_path / "llm_config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


class StubResponse:
    def __init__(self, text):
        msg = type("M", (), {"content": text})
        self.choices = [type("C", (), {"message": msg})]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5,
                                    "total_tokens": 15})


class StubCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0) if self.script else "default"
        if isinstance(item, Exception):
            raise item
        return StubResponse(item)


class StubClient:
    def __init__(self, script):
        self.chat = type("Chat", (), {"completions": StubCompletions(script)})()


@pytest.fixture
def llm(config_file, monkeypatch):
    reset_shared_llm()
    monkeypatch.setenv("TEST_LLM_KEY", "fake-key")
    return ChatLLM(config_path=config_file)


class RateLimitError(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("429 rate limit exceeded")


class TestChatLLMConfig:
    def test_reads_key_from_env(self, llm):
        assert llm.primary.api_key == "fake-key"

    def test_missing_key_raises_with_guidance(self, config_file, monkeypatch):
        monkeypatch.delenv("TEST_LLM_KEY", raising=False)
        with pytest.raises(RuntimeError, match="aistudio.google.com"):
            ChatLLM(config_path=config_file)

    def test_reads_key_from_file(self, config_file, tmp_path, monkeypatch):
        monkeypatch.delenv("TEST_LLM_KEY", raising=False)
        keyfile = tmp_path / "api.key"
        keyfile.write_text("key-from-file\n")
        llm = ChatLLM(config_path=config_file, api_file=str(keyfile))
        assert llm.primary.api_key == "key-from-file"

    def test_prepare_strips_unsupported_params(self, llm):
        out = llm._prepare(
            {"temperature": 0.5, "frequency_penalty": 1.0, "presence_penalty": 2.0},
            llm.primary,
        )
        assert out["temperature"] == 0.5
        assert "frequency_penalty" not in out and "presence_penalty" not in out

    def test_prepare_drops_nones(self, llm):
        assert llm._prepare({"a": 1, "b": None}, llm.primary)["a"] == 1
        assert "b" not in llm._prepare({"a": 1, "b": None}, llm.primary)


class TestThinkingBudget:
    """A thinking model spends max_tokens on reasoning before it writes a word.

    Measured on gemini-3.5-flash: 780-860 thinking tokens per call against an
    agent budget of 900-1000, which left 33-118 tokens for the reply and
    truncated every structured answer mid-JSON.
    """

    def test_max_tokens_is_raised_to_the_floor(self, llm):
        llm.min_max_tokens = 2048
        assert llm._prepare({"max_tokens": 900}, llm.primary)["max_tokens"] == 2048

    def test_a_generous_request_is_left_alone(self, llm):
        llm.min_max_tokens = 2048
        assert llm._prepare({"max_tokens": 4000}, llm.primary)["max_tokens"] == 4000

    def test_floor_applies_when_the_agent_names_no_budget(self, llm):
        llm.min_max_tokens = 2048
        assert llm._prepare({}, llm.primary)["max_tokens"] == 2048

    def test_reasoning_effort_is_applied_but_never_overrides_a_caller(self, llm):
        llm.reasoning_effort = "low"
        assert llm._prepare({}, llm.primary)["reasoning_effort"] == "low"
        assert (
            llm._prepare({"reasoning_effort": "high"}, llm.primary)["reasoning_effort"]
            == "high"
        )

    def test_both_are_off_by_default(self, llm):
        llm.reasoning_effort, llm.min_max_tokens = None, 0
        out = llm._prepare({"max_tokens": 900}, llm.primary)
        assert out["max_tokens"] == 900 and "reasoning_effort" not in out

    def test_reasoning_tokens_derived_from_the_total_shortfall(self):
        """Gemini's compat layer reports thinking only as a gap in the total."""
        from agents.llm import _reasoning_tokens

        usage = SimpleNamespace(
            prompt_tokens=1786, completion_tokens=33, total_tokens=2682
        )
        assert _reasoning_tokens(usage) == 863

    def test_reasoning_tokens_prefers_an_explicit_field(self):
        from agents.llm import _reasoning_tokens

        usage = SimpleNamespace(
            prompt_tokens=100, completion_tokens=10, total_tokens=110,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=42),
        )
        assert _reasoning_tokens(usage) == 42

    def test_reasoning_tokens_absent_when_usage_is_incomplete(self):
        from agents.llm import _reasoning_tokens

        assert _reasoning_tokens(SimpleNamespace(prompt_tokens=1)) is None


class TestChatLLMCalls:
    def test_returns_text(self, llm, monkeypatch):
        stub = StubClient(["hello"])
        monkeypatch.setattr(llm, "_client", lambda p, k=None: stub)
        out = asyncio.run(llm.chat("sys", "user", agent="t"))
        assert out == "hello"

    def test_strips_penalties_on_the_wire(self, llm, monkeypatch):
        stub = StubClient(["ok"])
        monkeypatch.setattr(llm, "_client", lambda p, k=None: stub)
        asyncio.run(llm.chat("s", "u", agent="t", frequency_penalty=1.0,
                             presence_penalty=2.0, temperature=0.3))
        sent = stub.chat.completions.calls[0]
        assert "frequency_penalty" not in sent
        assert "presence_penalty" not in sent
        assert sent["temperature"] == 0.3

    def test_retries_then_succeeds(self, llm, monkeypatch):
        stub = StubClient([RateLimitError(), RateLimitError(), "recovered"])
        monkeypatch.setattr(llm, "_client", lambda p, k=None: stub)
        assert asyncio.run(llm.chat("s", "u", agent="t")) == "recovered"
        assert len(stub.chat.completions.calls) == 3

    def test_gives_up_after_max_retries(self, llm, monkeypatch):
        stub = StubClient([RateLimitError()] * 10)
        monkeypatch.setattr(llm, "_client", lambda p, k=None: stub)
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            asyncio.run(llm.chat("s", "u", agent="t"))
        assert len(stub.chat.completions.calls) == llm.max_retries

    def test_non_retryable_error_fails_fast(self, llm, monkeypatch):
        stub = StubClient([ValueError("bad request: malformed")] * 5)
        monkeypatch.setattr(llm, "_client", lambda p, k=None: stub)
        with pytest.raises(RuntimeError):
            asyncio.run(llm.chat("s", "u", agent="t"))
        assert len(stub.chat.completions.calls) == 1  # no pointless retries

    def test_vision_uses_vision_model_and_image_payload(self, llm, monkeypatch):
        stub = StubClient(["saw it"])
        monkeypatch.setattr(llm, "_client", lambda p, k=None: stub)
        out = asyncio.run(llm.chat_with_image("s", "what?", "QUJD", agent="obs"))
        assert out == "saw it"
        sent = stub.chat.completions.calls[0]
        assert sent["model"] == "test-vision"
        content = sent["messages"][1]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,QUJD")

    def test_records_to_recorder(self, llm, monkeypatch):
        calls = []
        llm.recorder = type("R", (), {"log_llm_call": lambda self, **kw: calls.append(kw)})()
        monkeypatch.setattr(llm, "_client", lambda p, k=None: StubClient(["hi"]))
        asyncio.run(llm.chat("s", "u", agent="planner"))
        assert calls[0]["agent"] == "planner"
        assert calls[0]["usage"]["total_tokens"] == 15

    def test_sync_chat_works_outside_a_loop(self, llm, monkeypatch):
        monkeypatch.setattr(llm, "_client", lambda p, k=None: StubClient(["sync ok"]))
        assert llm.sync_chat("s", "u", agent="t") == "sync ok"

    def test_sync_chat_works_inside_a_running_loop(self, llm, monkeypatch):
        """The generated execute_command is sync but runs under asyncio.run,
        so the sync helpers must not deadlock on the live loop."""
        monkeypatch.setattr(llm, "_client", lambda p, k=None: StubClient(["nested ok"]))

        async def outer():
            return llm.sync_chat("s", "u", agent="t")

        assert asyncio.run(outer()) == "nested ok"


class TestQuotaDetection:
    """`_is_quota_exhausted` now means *daily* exhaustion specifically.

    The distinction is the one the key pool acts on: a per-minute rejection is
    absorbed by rotating to another key, while a per-day one retires the key
    for that model until midnight Pacific. A bare "RESOURCE_EXHAUSTED: quota"
    names neither, and is deliberately read as the *recoverable* one — rotating
    on a misread costs nothing, retiring a key for a day costs a fifth of the
    budget.
    """

    def test_bare_quota_message_is_not_treated_as_daily(self):
        assert not ChatLLM._is_quota_exhausted(Exception("RESOURCE_EXHAUSTED: quota"))
        assert ChatLLM._is_retryable(Exception("RESOURCE_EXHAUSTED: quota"))

    def test_daily_quota_id_is_recognised(self):
        exc = Exception(
            "429 RESOURCE_EXHAUSTED quotaId: "
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )
        assert ChatLLM._is_quota_exhausted(exc)

    def test_plain_rate_limit_is_retryable_not_quota(self):
        assert ChatLLM._is_retryable(RateLimitError())
        assert not ChatLLM._is_quota_exhausted(Exception("please slow down"))


# ---------------------------------------------------------------------------
# Multi-key routing, end to end through ChatLLM
# ---------------------------------------------------------------------------


DAY_429 = "429 quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
MINUTE_429 = "429 quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"


@pytest.fixture
def multikey_llm(config_file, monkeypatch):
    """Three keys on the primary provider, no fallback key configured."""
    reset_shared_llm()
    monkeypatch.setenv("TEST_LLM_KEY", "keyAAA1, keyBBB2, keyCCC3")
    return ChatLLM(config_path=config_file)


class KeyAwareClient:
    """One stub whose behaviour depends on which key it was built for.

    The pool hands `_client` a `KeyState`; recording it is how these tests see
    rotation happen at all.
    """

    def __init__(self, script_by_label, seen):
        self.script_by_label = script_by_label
        self.seen = seen

    def __call__(self, provider, key=None):
        label = key.label if key is not None else "key_1"
        self.seen.append(label)
        item = self.script_by_label.get(label, "ok")
        if isinstance(item, list):
            item = item.pop(0) if item else "ok"
        return StubClient([item])


class TestMultipleKeys:
    def test_all_keys_are_collected(self, multikey_llm):
        assert multikey_llm.primary.api_keys == ["keyAAA1", "keyBBB2", "keyCCC3"]
        assert len(multikey_llm.pool(multikey_llm.primary)) == 3

    def test_api_key_property_still_returns_the_first(self, multikey_llm):
        """Back-compat: single-key call sites read `provider.api_key`."""
        assert multikey_llm.primary.api_key == "keyAAA1"

    def test_consecutive_calls_rotate(self, multikey_llm, monkeypatch):
        seen = []
        monkeypatch.setattr(multikey_llm, "_client", KeyAwareClient({}, seen))
        for _ in range(3):
            asyncio.run(multikey_llm.chat("s", "u", agent="t"))
        assert seen == ["key_1", "key_2", "key_3"]

    def test_minute_429_rotates_without_sleeping(self, multikey_llm, monkeypatch):
        """The headline property. A rate limit must cost a key, not a second."""
        seen = []
        monkeypatch.setattr(
            multikey_llm, "_client",
            KeyAwareClient({"key_1": Exception(MINUTE_429)}, seen),
        )
        slept = []
        monkeypatch.setattr(llm_mod.asyncio, "sleep",
                            lambda s: slept.append(s) or asyncio.sleep(0))

        assert asyncio.run(multikey_llm.chat("s", "u", agent="t")) == "ok"
        assert seen == ["key_1", "key_2"]
        assert slept == []  # no backoff at all

    def test_day_429_retires_only_that_key(self, multikey_llm, monkeypatch):
        seen = []
        monkeypatch.setattr(
            multikey_llm, "_client",
            KeyAwareClient({"key_1": Exception(DAY_429)}, seen),
        )
        asyncio.run(multikey_llm.chat("s", "u", agent="t"))

        pool = multikey_llm.pool(multikey_llm.primary)
        assert pool.keys[0].is_exhausted("test-model", time.time())
        # A different model draws on a different quota, so the key survives there.
        assert not pool.keys[0].is_exhausted("test-vision", time.time())

    def test_exhausted_key_is_skipped_on_later_calls(self, multikey_llm, monkeypatch):
        seen = []
        monkeypatch.setattr(
            multikey_llm, "_client",
            KeyAwareClient({"key_1": Exception(DAY_429)}, seen),
        )
        asyncio.run(multikey_llm.chat("s", "u", agent="t"))
        seen.clear()
        for _ in range(3):
            asyncio.run(multikey_llm.chat("s", "u", agent="t"))
        assert "key_1" not in seen

    def test_every_key_exhausted_raises_after_trying_each(self, multikey_llm, monkeypatch):
        seen = []
        monkeypatch.setattr(
            multikey_llm, "_client",
            KeyAwareClient({lbl: Exception(DAY_429) for lbl in
                            ("key_1", "key_2", "key_3")}, seen),
        )
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            asyncio.run(multikey_llm.chat("s", "u", agent="t"))
        assert sorted(set(seen)) == ["key_1", "key_2", "key_3"]

    def test_trace_names_the_key_but_never_carries_it(self, multikey_llm, monkeypatch):
        calls = []
        multikey_llm.recorder = type(
            "R", (), {"log_llm_call": lambda self, **kw: calls.append(kw)}
        )()
        monkeypatch.setattr(multikey_llm, "_client", KeyAwareClient({}, []))
        asyncio.run(multikey_llm.chat("s", "u", agent="planner"))

        assert calls[0]["key"] == "key_1"
        assert calls[0]["key_fingerprint"] == "...AAA1"
        assert "keyAAA1" not in json.dumps(calls[0])


class TestSingleKeyIsUnchanged:
    """Every recorded run predates the pool; one key must still behave as then."""

    def test_one_key_makes_a_one_key_pool(self, llm):
        assert llm.primary.api_keys == ["fake-key"]
        assert len(llm.pool(llm.primary)) == 1

    def test_backoff_is_still_exponential_with_nothing_to_rotate_to(
        self, llm, monkeypatch
    ):
        """One key, two 429s: wait, double, recover — as the old client did.

        The fake sleep must advance the fake clock. One that does not leaves the
        pool's wait loop re-deciding that the cooldown is still pending and
        spinning, which is the same trap `TestRateLimiter.test_blocks_beyond_limit`
        documents.
        """
        stub = StubClient([RateLimitError(), RateLimitError(), "recovered"])
        monkeypatch.setattr(llm, "_client", lambda p, k=None: stub)

        clock = {"t": 1000.0}
        slept = []
        real_sleep = asyncio.sleep

        async def fake_sleep(seconds, *a, **kw):
            slept.append(seconds)
            clock["t"] += seconds
            return await real_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(llm_mod.kp.time, "monotonic", lambda: clock["t"])

        assert asyncio.run(llm.chat("s", "u", agent="t")) == "recovered"
        # retry_base_delay_s is 0.001 in the fixture, so the per-key cooldown
        # ramps 0.001 then 0.002 — doubling, and nowhere near the flat 60 s that
        # filling the key's minute window instead would have forced.
        assert len(slept) == 2, f"expected two waits, got {slept}"
        assert slept[1] > slept[0]
        assert max(slept) < 1.0
