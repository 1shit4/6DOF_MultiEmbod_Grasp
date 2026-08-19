"""One configurable, rate-limited LLM client for the whole system.

Replaces the original `OpenAILLM`, which hardcoded paid OpenAI models in five
different places (three agents plus two raw `requests.post` calls inside
`image_patch.py`). Everything now goes through `ChatLLM`, so the provider is a
config choice.

Default provider is **Google Gemini's free tier** via its OpenAI-compatible
endpoint. Two properties made it the right pick: the Observer agent needs image
input, which Gemini's free tier includes; and because the endpoint is
OpenAI-compatible, the existing `AsyncOpenAI` call sites barely change.

Three things the original client lacked and a free tier makes mandatory:

* **Pacing.** ~10 RPM. GraspMAS issues 3+ calls per round and `main_batch.py`
  fires `asyncio.gather` over a batch — unthrottled, the first batch 429s.
* **Backoff and failover.** Retry on 429/5xx, then optionally roll over to a
  second free provider.
* **Tolerant parsing.** The upstream parsers assumed GPT-4o's exact formatting
  (`.split('<thought>')[1]`, bare `json.loads`). A different model breaks them,
  so extraction lives here and is shared.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

BASE_PATH = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = BASE_PATH / "llm_config.yaml"


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*\n?(.*?)```", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Return the content of the first fenced block, or the text unchanged."""
    m = _FENCE_RE.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()


def extract_tag(text: str, tag: str) -> Optional[str]:
    """Pull `<tag>...</tag>` out of a reply, tolerating fences and stray prose.

    The upstream code did `text.split('<thought>')[1].split('</thought>')[0]`,
    which raises IndexError the moment a model omits the tag or wraps the whole
    reply in a code fence. Both happen routinely on non-GPT-4o models.
    """
    if not text:
        return None
    for candidate in (text, strip_code_fences(text)):
        m = re.search(rf"<{tag}\s*>(.*?)</{tag}\s*>", candidate, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # Opening tag but no closing tag — take everything after it.
    m = re.search(rf"<{tag}\s*>(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def extract_json(text: str) -> Optional[dict]:
    """Find and parse the outermost JSON object in a reply.

    Handles: bare JSON, fenced JSON, JSON preceded by prose, and JSON wrapped
    in an XML-ish tag. Returns None rather than raising, so callers can
    re-prompt instead of crashing a round.
    """
    if not text:
        return None
    for candidate in (strip_code_fences(text), text):
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(candidate[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return None


def extract_code(text: str) -> str:
    """Extract a Python function body from a reply.

    Upstream did `'\\n'.join(response.split('\\n')[1:-1])`, which assumes the
    reply is exactly one fenced block with no preamble — off by one line as
    soon as the model adds "Here's the code:".
    """
    if not text:
        return ""
    code = strip_code_fences(text)
    if "def execute_command" in code:
        lines = code.splitlines()
        for i, line in enumerate(lines):
            if line.lstrip().startswith("def execute_command"):
                # Drop any leading prose the model emitted before the def.
                return "\n".join(lines[i:]).strip()
    return code.strip()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class AsyncRateLimiter:
    """Simple async pacer: at most `rpm` requests per rolling 60 s.

    A sliding window rather than a token bucket because free-tier limits are
    literally "N requests per minute" — a bucket would allow a burst of N that
    trips the very limit we're avoiding.
    """

    def __init__(self, rpm: int):
        self.rpm = max(int(rpm), 1)
        self._times: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Block until a slot is free. Returns seconds waited."""
        waited = 0.0
        async with self._lock:
            while True:
                now = time.monotonic()
                self._times = [t for t in self._times if now - t < 60.0]
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return waited
                sleep_for = 60.0 - (now - self._times[0]) + 0.05
                waited += sleep_for
                logger.info(
                    "Rate limit (%d rpm) reached; waiting %.1fs", self.rpm, sleep_for
                )
                await asyncio.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    vision_model: Optional[str] = None
    model_candidates: list = field(default_factory=list)
    vision_model_candidates: list = field(default_factory=list)
    unsupported_params: list = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.api_key)


def _read_key(env_var: Optional[str], key_file: Optional[str]) -> Optional[str]:
    if env_var:
        val = os.environ.get(env_var)
        if val and val.strip():
            return val.strip()
    if key_file:
        path = Path(key_file)
        if not path.is_absolute():
            path = BASE_PATH / path
        if path.exists():
            text = path.read_text().strip().splitlines()
            if text and text[0].strip():
                return text[0].strip()
    return None


def load_config(config_path: Optional[str | os.PathLike] = None) -> dict:
    import yaml

    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"LLM config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class ChatLLM:
    """Async chat client with pacing, retries, failover and shared parsing.

    Not thread-safe across event loops; one instance per process is intended
    (`get_shared_llm()`), which is also what makes the rate limiter correct —
    it can only pace calls it can see.
    """

    def __init__(
        self,
        config_path: Optional[str | os.PathLike] = None,
        api_file: Optional[str] = None,
        recorder: Any = None,
    ):
        cfg = load_config(config_path)
        self.raw_config = cfg

        primary_key = _read_key(
            cfg.get("api_key_env", "LLM_API_KEY"),
            api_file or cfg.get("api_key_file", "api.key"),
        )
        self.primary = ProviderConfig(
            name=cfg.get("provider", "gemini"),
            base_url=cfg["base_url"],
            api_key=primary_key,
            model=cfg.get("model"),
            vision_model=cfg.get("vision_model"),
            model_candidates=list(cfg.get("model_candidates", [])),
            vision_model_candidates=list(cfg.get("vision_model_candidates", [])),
            unsupported_params=list(cfg.get("unsupported_params", [])),
        )

        self.fallback: Optional[ProviderConfig] = None
        fb = cfg.get("fallback")
        if fb:
            fb_key = _read_key(fb.get("api_key_env"), fb.get("api_key_file"))
            self.fallback = ProviderConfig(
                name=fb.get("provider", "fallback"),
                base_url=fb["base_url"],
                api_key=fb_key,
                model=fb.get("model"),
                vision_model=fb.get("vision_model", fb.get("model")),
                unsupported_params=list(fb.get("unsupported_params", [])),
            )

        self.rate_limiter = AsyncRateLimiter(cfg.get("rate_limit_rpm", 10))
        self.max_retries = int(cfg.get("max_retries", 5))
        self.timeout_s = float(cfg.get("timeout_s", 120))
        self.retry_base_delay_s = float(cfg.get("retry_base_delay_s", 2.0))
        self.recorder = recorder

        self._clients: dict[str, Any] = {}
        self._resolved = False

        if not self.primary.usable:
            raise RuntimeError(
                f"No API key found. Set ${cfg.get('api_key_env', 'LLM_API_KEY')} "
                f"or write one to {BASE_PATH / cfg.get('api_key_file', 'api.key')}.\n"
                f"Get a free Gemini key at https://aistudio.google.com/apikey"
            )

    # -- clients -----------------------------------------------------------

    def _client(self, provider: ProviderConfig):
        from openai import AsyncOpenAI

        if provider.name not in self._clients:
            self._clients[provider.name] = AsyncOpenAI(
                api_key=provider.api_key,
                base_url=provider.base_url,
                timeout=self.timeout_s,
                max_retries=0,  # we own retry policy (pacing must come first)
            )
        return self._clients[provider.name]

    async def list_models(self, provider: Optional[ProviderConfig] = None) -> list[str]:
        provider = provider or self.primary
        try:
            resp = await self._client(provider).models.list()
            return sorted(m.id.replace("models/", "") for m in resp.data)
        except Exception as exc:
            logger.warning("Could not list models for %s: %s", provider.name, exc)
            return []

    async def resolve_models(self) -> None:
        """Pin `model` / `vision_model` to something the key can actually reach.

        Free-tier model availability changes often enough that a hard-coded id
        is a liability; probing once at startup costs one request and turns a
        mid-run 404 into a startup-time decision.
        """
        if self._resolved:
            return
        self._resolved = True
        p = self.primary
        if p.model and p.vision_model:
            return

        available = set(await self.list_models(p))
        if not available:
            p.model = p.model or (p.model_candidates[0] if p.model_candidates else None)
            p.vision_model = p.vision_model or p.model
            logger.warning("Model list unavailable; falling back to %s", p.model)
            return

        def pick(candidates: list) -> Optional[str]:
            for c in candidates:
                if c in available:
                    return c
            # Any flash-ish model is a reasonable free-tier default.
            flash = sorted(m for m in available if "flash" in m and "lite" not in m)
            return flash[0] if flash else (sorted(available)[0] if available else None)

        p.model = p.model or pick(p.model_candidates)
        p.vision_model = p.vision_model or pick(p.vision_model_candidates or p.model_candidates)
        logger.info(
            "Resolved models for %s: text=%s vision=%s", p.name, p.model, p.vision_model
        )

    # -- request path ------------------------------------------------------

    @staticmethod
    def _sanitize(params: dict, provider: ProviderConfig) -> dict:
        """Drop parameters the provider does not honour.

        Gemini's OpenAI-compat layer rejects/ignores `frequency_penalty` and
        `presence_penalty`, which `agents/coder.py` passes unconditionally.
        """
        out = dict(params)
        for key in provider.unsupported_params:
            out.pop(key, None)
        return {k: v for k, v in out.items() if v is not None}

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if status in (408, 409, 425, 429, 500, 502, 503, 504):
            return True
        text = str(exc).lower()
        return any(
            s in text
            for s in ("rate limit", "429", "quota", "timeout", "temporarily",
                      "overloaded", "unavailable", "503", "500")
        )

    @staticmethod
    def _is_quota_exhausted(exc: Exception) -> bool:
        text = str(exc).lower()
        return "quota" in text or "resource_exhausted" in text or "daily" in text

    async def _call(
        self,
        provider: ProviderConfig,
        messages: list,
        model: str,
        params: dict,
    ):
        await self.rate_limiter.acquire()
        return await self._client(provider).chat.completions.create(
            model=model, messages=messages, **self._sanitize(params, provider)
        )

    async def complete(
        self,
        messages: list,
        agent: str = "unknown",
        vision: bool = False,
        **params,
    ) -> str:
        """Send a chat completion; return the assistant text.

        Retries with exponential backoff + jitter on transient errors, then
        rolls over to the fallback provider if one is configured and usable.
        """
        await self.resolve_models()

        attempts = 0
        last_exc: Optional[Exception] = None
        providers = [self.primary]
        if self.fallback and self.fallback.usable:
            providers.append(self.fallback)

        for provider in providers:
            model = (provider.vision_model if vision else provider.model) or provider.model
            if model is None:
                continue
            for attempt in range(self.max_retries):
                t0 = time.time()
                try:
                    resp = await self._call(provider, messages, model, params)
                    text = (resp.choices[0].message.content or "").strip()
                    self._record(
                        agent, model, provider.name, messages, text,
                        time.time() - t0, resp, attempts, None, vision,
                    )
                    return text
                except Exception as exc:  # noqa: BLE001 - provider SDKs raise broadly
                    last_exc = exc
                    attempts += 1
                    retryable = self._is_retryable(exc)
                    logger.warning(
                        "%s call failed (%s, attempt %d/%d): %s",
                        agent, provider.name, attempt + 1, self.max_retries, exc,
                    )
                    if not retryable:
                        break
                    if self._is_quota_exhausted(exc) and provider is not providers[-1]:
                        logger.warning("Quota exhausted on %s; failing over", provider.name)
                        break
                    delay = self.retry_base_delay_s * (2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(min(delay, 60.0))

            self._record(
                agent, model, provider.name, messages, "",
                0.0, None, attempts, str(last_exc), vision,
            )

        raise RuntimeError(f"All LLM providers failed for {agent}: {last_exc}") from last_exc

    def _record(self, agent, model, provider, messages, text, latency, resp, retries, error, vision):
        if self.recorder is None:
            return
        try:
            usage = {}
            if resp is not None and getattr(resp, "usage", None):
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }
            self.recorder.log_llm_call(
                agent=agent, model=model or "?", provider=provider,
                prompt=_flatten_prompt(messages), response=text,
                latency_s=latency, usage=usage, retries=retries,
                error=error, has_image=vision,
            )
        except Exception as exc:  # never let telemetry break a run
            logger.debug("Failed to record LLM call: %s", exc)

    # -- convenience -------------------------------------------------------

    async def chat(self, system: str, user: str, agent: str = "unknown", **params) -> str:
        return await self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            agent=agent, **params,
        )

    async def chat_with_image(
        self,
        system: str,
        user: str,
        base64_image: str,
        agent: str = "unknown",
        image_format: str = "png",
        **params,
    ) -> str:
        """Vision call. Payload shape is identical across OpenAI and Gemini."""
        content = [
            {"type": "text", "text": user},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"},
            },
        ]
        return await self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": content}],
            agent=agent, vision=True, **params,
        )

    def sync_chat(self, system: str, user: str, agent: str = "unknown", **params) -> str:
        """Blocking wrapper for the synchronous ImagePatch tool methods.

        The LLM-generated `execute_command` is plain synchronous code, so
        `simple_query` and friends cannot await. When called from inside a
        running loop we hand off to a worker thread rather than deadlock.
        """
        coro = self.chat(system, user, agent=agent, **params)
        return _run_sync(coro)

    def sync_chat_with_image(
        self, system: str, user: str, base64_image: str, agent: str = "unknown", **params
    ) -> str:
        coro = self.chat_with_image(system, user, base64_image, agent=agent, **params)
        return _run_sync(coro)

    # Back-compat: agents constructed as `Planner(prompt, llm)` reach for
    # `llm.system_prompt`.
    system_prompt = "Answer strictly in the format requested."


def _run_sync(coro):
    """Run `coro` to completion from sync code, inside or outside a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _flatten_prompt(messages: list) -> str:
    parts = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            parts.append(f"[{m['role']}] {content}")
        elif isinstance(content, list):
            for c in content:
                if c.get("type") == "text":
                    parts.append(f"[{m['role']}] {c['text']}")
                else:
                    parts.append(f"[{m['role']}] <image>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Process-wide shared instance
# ---------------------------------------------------------------------------

_SHARED: Optional[ChatLLM] = None


def get_shared_llm(
    config_path: Optional[str | os.PathLike] = None,
    api_file: Optional[str] = None,
    recorder: Any = None,
) -> ChatLLM:
    """The one client everything should use.

    Sharing matters: the rate limiter can only pace requests that pass through
    it, so a second instance would silently double the request rate and 429.
    """
    global _SHARED
    if _SHARED is None:
        _SHARED = ChatLLM(config_path=config_path, api_file=api_file, recorder=recorder)
    elif recorder is not None:
        _SHARED.recorder = recorder
    return _SHARED


def reset_shared_llm() -> None:
    """Drop the shared instance (tests, or switching config mid-process)."""
    global _SHARED
    _SHARED = None


# Kept so `from .llm import OpenAILLM` in old code/notebooks still works.
OpenAILLM = ChatLLM


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Probe the configured LLM provider")
    ap.add_argument("--probe", action="store_true", help="List reachable models")
    ap.add_argument("--test", action="store_true", help="Send one text + one vision call")
    args = ap.parse_args()

    async def _main():
        # This is the first command a new user runs, and a missing key is the
        # expected first failure — report it as a message, not a traceback.
        try:
            llm = ChatLLM()
        except RuntimeError as exc:
            print(f"\n{exc}\n", file=sys.stderr)
            raise SystemExit(1)
        if args.probe or not args.test:
            models = await llm.list_models()
            print(f"\n{len(models)} models reachable with this key:")
            for m in models:
                print("  ", m)
            await llm.resolve_models()
            print(f"\nselected text  : {llm.primary.model}")
            print(f"selected vision: {llm.primary.vision_model}")
        if args.test:
            await llm.resolve_models()
            out = await llm.chat("You are terse.", "Reply with exactly: OK", agent="probe")
            print(f"\ntext call  -> {out!r}")

            import base64
            import numpy as np
            import cv2

            img = np.zeros((64, 64, 3), np.uint8)
            cv2.circle(img, (32, 32), 20, (0, 0, 255), -1)
            ok, buf = cv2.imencode(".png", img)
            b64 = base64.b64encode(buf).decode()
            out = await llm.chat_with_image(
                "You are terse.",
                "What colour is the circle? One word.",
                b64,
                agent="probe",
            )
            print(f"vision call -> {out!r}")

    asyncio.run(_main())
