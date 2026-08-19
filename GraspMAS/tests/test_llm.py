"""agents.llm: config, pacing, retry/failover, and the response extractors.

No network. The OpenAI client is replaced with a stub, so retry policy and rate
limiting are tested as logic rather than by actually hammering a provider — and
the suite costs zero free-tier requests.
"""

import asyncio
import time

import pytest
import yaml

from agents.llm import (
    AsyncRateLimiter,
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
# Rate limiting — the free tier is ~10 RPM and GraspMAS makes 3+ calls a round.
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_burst_up_to_limit(self):
        async def run():
            rl = AsyncRateLimiter(rpm=5)
            t0 = time.monotonic()
            for _ in range(5):
                await rl.acquire()
            return time.monotonic() - t0

        assert asyncio.run(run()) < 0.1  # first N are immediate

    def test_blocks_beyond_limit(self, monkeypatch):
        """The (N+1)th call must wait for the window to roll.

        Both `asyncio.sleep` and `time.monotonic` are stubbed with a shared fake
        clock, and stubbed *before* the loop starts (patching inside the
        coroutine is too late). A fake sleep that does not advance the clock
        would leave the limiter's while-loop spinning for a real minute, which
        is exactly what a naive version of this test did.
        """
        import agents.llm as llm_mod

        clock = {"t": 1000.0}
        slept = []
        real_sleep = asyncio.sleep

        async def fake_sleep(seconds, *a, **kw):
            slept.append(seconds)
            clock["t"] += seconds
            return await real_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(llm_mod.time, "monotonic", lambda: clock["t"])

        async def run():
            rl = AsyncRateLimiter(rpm=2)
            for _ in range(3):
                await rl.acquire()

        asyncio.run(run())
        assert len(slept) == 1, f"expected exactly one wait, got {slept}"
        assert slept[0] > 55  # nearly a full 60 s window

    def test_window_slides_so_old_calls_stop_counting(self, monkeypatch):
        """Once a request ages out of the 60 s window it must not block others."""
        import agents.llm as llm_mod

        clock = {"t": 0.0}
        monkeypatch.setattr(llm_mod.time, "monotonic", lambda: clock["t"])

        async def run():
            rl = AsyncRateLimiter(rpm=2)
            await rl.acquire()
            await rl.acquire()
            clock["t"] += 61.0  # both age out
            waited = await rl.acquire()
            return waited, len(rl._times)

        waited, live = asyncio.run(run())
        assert waited == 0.0
        assert live == 1

    def test_records_timestamps_per_request(self):
        async def run():
            rl = AsyncRateLimiter(rpm=10)
            for _ in range(4):
                await rl.acquire()
            return len(rl._times)

        assert asyncio.run(run()) == 4

    def test_rpm_floor_is_one(self):
        assert AsyncRateLimiter(rpm=0).rpm == 1


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

    def test_sanitize_strips_unsupported_params(self, llm):
        out = ChatLLM._sanitize(
            {"temperature": 0.5, "frequency_penalty": 1.0, "presence_penalty": 2.0},
            llm.primary,
        )
        assert out == {"temperature": 0.5}

    def test_sanitize_drops_nones(self, llm):
        assert ChatLLM._sanitize({"a": 1, "b": None}, llm.primary) == {"a": 1}


class TestChatLLMCalls:
    def test_returns_text(self, llm, monkeypatch):
        stub = StubClient(["hello"])
        monkeypatch.setattr(llm, "_client", lambda p: stub)
        out = asyncio.run(llm.chat("sys", "user", agent="t"))
        assert out == "hello"

    def test_strips_penalties_on_the_wire(self, llm, monkeypatch):
        stub = StubClient(["ok"])
        monkeypatch.setattr(llm, "_client", lambda p: stub)
        asyncio.run(llm.chat("s", "u", agent="t", frequency_penalty=1.0,
                             presence_penalty=2.0, temperature=0.3))
        sent = stub.chat.completions.calls[0]
        assert "frequency_penalty" not in sent
        assert "presence_penalty" not in sent
        assert sent["temperature"] == 0.3

    def test_retries_then_succeeds(self, llm, monkeypatch):
        stub = StubClient([RateLimitError(), RateLimitError(), "recovered"])
        monkeypatch.setattr(llm, "_client", lambda p: stub)
        assert asyncio.run(llm.chat("s", "u", agent="t")) == "recovered"
        assert len(stub.chat.completions.calls) == 3

    def test_gives_up_after_max_retries(self, llm, monkeypatch):
        stub = StubClient([RateLimitError()] * 10)
        monkeypatch.setattr(llm, "_client", lambda p: stub)
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            asyncio.run(llm.chat("s", "u", agent="t"))
        assert len(stub.chat.completions.calls) == llm.max_retries

    def test_non_retryable_error_fails_fast(self, llm, monkeypatch):
        stub = StubClient([ValueError("bad request: malformed")] * 5)
        monkeypatch.setattr(llm, "_client", lambda p: stub)
        with pytest.raises(RuntimeError):
            asyncio.run(llm.chat("s", "u", agent="t"))
        assert len(stub.chat.completions.calls) == 1  # no pointless retries

    def test_vision_uses_vision_model_and_image_payload(self, llm, monkeypatch):
        stub = StubClient(["saw it"])
        monkeypatch.setattr(llm, "_client", lambda p: stub)
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
        monkeypatch.setattr(llm, "_client", lambda p: StubClient(["hi"]))
        asyncio.run(llm.chat("s", "u", agent="planner"))
        assert calls[0]["agent"] == "planner"
        assert calls[0]["usage"]["total_tokens"] == 15

    def test_sync_chat_works_outside_a_loop(self, llm, monkeypatch):
        monkeypatch.setattr(llm, "_client", lambda p: StubClient(["sync ok"]))
        assert llm.sync_chat("s", "u", agent="t") == "sync ok"

    def test_sync_chat_works_inside_a_running_loop(self, llm, monkeypatch):
        """The generated execute_command is sync but runs under asyncio.run,
        so the sync helpers must not deadlock on the live loop."""
        monkeypatch.setattr(llm, "_client", lambda p: StubClient(["nested ok"]))

        async def outer():
            return llm.sync_chat("s", "u", agent="t")

        assert asyncio.run(outer()) == "nested ok"


class TestQuotaDetection:
    def test_quota_message_recognised(self):
        assert ChatLLM._is_quota_exhausted(Exception("RESOURCE_EXHAUSTED: quota"))

    def test_plain_rate_limit_is_retryable_not_quota(self):
        assert ChatLLM._is_retryable(RateLimitError())
        assert not ChatLLM._is_quota_exhausted(Exception("please slow down"))
