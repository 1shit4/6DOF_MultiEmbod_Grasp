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

* **Pacing.** ~10 RPM per key. GraspMAS issues 3+ calls per round and
  `main_batch.py` fires `asyncio.gather` over a batch — unthrottled, the first
  batch 429s. Pacing lives in `key_pool.KeyPool`, which holds one sliding
  window per key rather than one for the process, so extra keys raise the
  ceiling instead of merely sharing it.
* **Backoff and failover.** A per-minute 429 rotates to another key with no
  delay; a per-day one retires that key for that model; only when nothing is
  left does it back off and roll over to a second free provider.
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

from . import key_pool as kp
from .key_pool import AllKeysExhausted, KeyPool, KeyState

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
# Usage accounting
# ---------------------------------------------------------------------------


def _reasoning_tokens(usage) -> Optional[int]:
    """How many output tokens went to thinking rather than to the reply.

    Providers report this three different ways. OpenAI-style SDKs expose
    `completion_tokens_details.reasoning_tokens`; Gemini's own field is
    `thoughts_token_count`; and Gemini's OpenAI-compat layer exposes neither,
    leaving it only as the gap between `total` and `prompt + completion`.
    """
    details = getattr(usage, "completion_tokens_details", None)
    for obj, attr in ((details, "reasoning_tokens"), (usage, "thoughts_token_count")):
        value = getattr(obj, attr, None) if obj is not None else None
        if isinstance(value, int):
            return value

    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)
    if None in (prompt, completion, total):
        return None
    return max(0, total - prompt - completion)


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_keys: list = field(default_factory=list)
    model: Optional[str] = None
    vision_model: Optional[str] = None
    model_candidates: list = field(default_factory=list)
    vision_model_candidates: list = field(default_factory=list)
    unsupported_params: list = field(default_factory=list)

    @property
    def api_key(self) -> Optional[str]:
        """The first key. Kept so single-key call sites read unchanged."""
        return self.api_keys[0] if self.api_keys else None

    @property
    def usable(self) -> bool:
        return bool(self.api_keys)


def _read_keys(
    env_var: Optional[str],
    key_file: Optional[str],
    env_prefix: Optional[str] = None,
) -> list:
    """Every key configured for one provider, in priority order.

    A superset of the old single-key `_read_key`: the env var still wins over
    the file and still yields the same first key, but each source may now carry
    several. `env_prefix` defaults to the env var plus an underscore, so
    `LLM_API_KEY_1..N` work without being declared.
    """
    if env_prefix is None and env_var:
        env_prefix = f"{env_var}_"
    return kp.collect_keys(
        env_var=env_var,
        env_prefix=env_prefix,
        key_file=key_file,
        base_path=BASE_PATH,
    )


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

        primary_keys = _read_keys(
            cfg.get("api_key_env", "LLM_API_KEY"),
            api_file or cfg.get("api_key_file", "api.key"),
            cfg.get("api_key_env_prefix"),
        )
        self.primary = ProviderConfig(
            name=cfg.get("provider", "gemini"),
            base_url=cfg["base_url"],
            api_keys=primary_keys,
            model=cfg.get("model"),
            vision_model=cfg.get("vision_model"),
            model_candidates=list(cfg.get("model_candidates", [])),
            vision_model_candidates=list(cfg.get("vision_model_candidates", [])),
            unsupported_params=list(cfg.get("unsupported_params", [])),
        )

        self.fallback: Optional[ProviderConfig] = None
        fb = cfg.get("fallback")
        if fb:
            fb_keys = _read_keys(
                fb.get("api_key_env"), fb.get("api_key_file"), fb.get("api_key_env_prefix")
            )
            self.fallback = ProviderConfig(
                name=fb.get("provider", "fallback"),
                base_url=fb["base_url"],
                api_keys=fb_keys,
                model=fb.get("model"),
                vision_model=fb.get("vision_model", fb.get("model")),
                unsupported_params=list(fb.get("unsupported_params", [])),
            )

        self.rpm = int(cfg.get("rate_limit_rpm", 10))
        self.max_retries = int(cfg.get("max_retries", 5))
        self.timeout_s = float(cfg.get("timeout_s", 120))
        self.retry_base_delay_s = float(cfg.get("retry_base_delay_s", 2.0))
        self.reasoning_effort = cfg.get("reasoning_effort") or None
        self.min_max_tokens = int(cfg.get("min_max_tokens", 0))
        self.recorder = recorder

        self._clients: dict[tuple, Any] = {}
        self._resolved = False

        if not self.primary.usable:
            raise RuntimeError(
                f"No API key found. Set ${cfg.get('api_key_env', 'LLM_API_KEY')} "
                f"or write one to {BASE_PATH / cfg.get('api_key_file', 'api.key')}.\n"
                f"Get a free Gemini key at https://aistudio.google.com/apikey"
            )

        # One pool per provider. `rate_limit_rpm` is now **per key**: the limit
        # it names is a property of a project's quota, not of this process, so
        # N keys from N projects genuinely allow N times the rate.
        state_file = cfg.get("quota_state_file")
        state_path = None
        if state_file:
            state_path = Path(state_file)
            if not state_path.is_absolute():
                state_path = BASE_PATH / state_path
        selection = cfg.get("key_selection", "least_used")
        self.pools: dict[str, KeyPool] = {
            self.primary.name: KeyPool(
                self.primary.api_keys, rpm=self.rpm, state_path=state_path,
                selection=selection, base_delay_s=self.retry_base_delay_s,
            )
        }
        if self.fallback and self.fallback.usable:
            self.pools[self.fallback.name] = KeyPool(
                self.fallback.api_keys, rpm=self.rpm, selection=selection,
                base_delay_s=self.retry_base_delay_s,
            )
        if len(self.primary.api_keys) > 1:
            logger.info(
                "%d keys in the %s pool (%d rpm each, %d rpm total)",
                len(self.primary.api_keys), self.primary.name,
                self.rpm, self.rpm * len(self.primary.api_keys),
            )

    # -- clients -----------------------------------------------------------

    def pool(self, provider: ProviderConfig) -> KeyPool:
        return self.pools[provider.name]

    def _client(self, provider: ProviderConfig, key: Optional[KeyState] = None):
        """One cached client per (provider, key).

        `AsyncOpenAI` binds its `api_key` at construction, so the cache has to
        be keyed by both — a single per-provider client would pin the whole pool
        to whichever key happened to build it.
        """
        from openai import AsyncOpenAI

        label = key.label if key is not None else "key_1"
        api_key = key.key if key is not None else provider.api_key
        cache_key = (provider.name, label)
        if cache_key not in self._clients:
            self._clients[cache_key] = AsyncOpenAI(
                api_key=api_key,
                base_url=provider.base_url,
                timeout=self.timeout_s,
                max_retries=0,  # we own retry policy (pacing must come first)
            )
        return self._clients[cache_key]

    async def list_models(
        self, provider: Optional[ProviderConfig] = None, key: Optional[KeyState] = None
    ) -> list[str]:
        """Models this provider can reach, trying each key until one answers.

        Deliberately not routed through `KeyPool.acquire`: listing models is a
        different endpoint from generation and does not draw on the generate
        quota, so spending a pooled slot on it would be pure loss.
        """
        provider = provider or self.primary
        candidates = [key] if key is not None else (self.pool(provider).keys or [None])
        last_exc: Optional[Exception] = None
        for k in candidates:
            try:
                resp = await self._client(provider, k).models.list()
                return sorted(m.id.replace("models/", "") for m in resp.data)
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise broadly
                last_exc = exc
                label = k.label if k is not None else "key_1"
                logger.debug("Model list failed for %s/%s: %s", provider.name, label, exc)
        logger.warning("Could not list models for %s: %s", provider.name, last_exc)
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

    def _prepare(self, params: dict, provider: ProviderConfig) -> dict:
        """Apply provider defaults, then drop parameters it does not honour.

        Two provider realities are handled here rather than in the agents, so
        that every agent gets them and none has to know about them.

        **Thinking tokens come out of `max_tokens`.** Gemini 3.x reasons before
        it answers, and that reasoning is billed and budgeted as output. Measured
        on `gemini-3.5-flash`: 780-860 thinking tokens per call. The agents ask
        for 900-1000, so the reply itself got 33-118 tokens and every structured
        answer was truncated mid-JSON — the task planner failed to parse twice
        running and the run aborted on its first iteration. `min_max_tokens`
        raises every request to a floor that fits reasoning *and* a reply.

        **`reasoning_effort` bounds the thinking.** The OpenAI-compat layer maps
        it to Gemini's `thinking_level`. Every agent here emits structured
        output against an explicit schema, which is not the kind of task extended
        deliberation improves, so capping it saves tokens and latency without
        costing accuracy.
        """
        out = dict(params)
        if self.reasoning_effort and "reasoning_effort" not in out:
            out["reasoning_effort"] = self.reasoning_effort
        if self.min_max_tokens:
            out["max_tokens"] = max(int(out.get("max_tokens") or 0), self.min_max_tokens)
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
        """True for a *daily* exhaustion — the kind waiting does not fix."""
        return kp.classify_quota_error(exc) == "day"

    async def _call(
        self,
        provider: ProviderConfig,
        messages: list,
        model: str,
        params: dict,
        key: Optional[KeyState] = None,
    ):
        return await self._client(provider, key).chat.completions.create(
            model=model, messages=messages, **self._prepare(params, provider)
        )

    async def complete(
        self,
        messages: list,
        agent: str = "unknown",
        vision: bool = False,
        **params,
    ) -> str:
        """Send a chat completion; return the assistant text.

        Rate limits are handled by **rotating keys, not by sleeping**. A
        per-minute 429 puts only that key on cooldown, and the next attempt goes
        out on a different key with no delay at all. A *daily* 429 retires the
        key for that model alone, then falls through to the remaining keys and
        finally to the fallback provider.

        Exponential backoff survives for the errors no other key can help with —
        5xx, timeouts — and, inside the pool, as the cooldown a lone key waits
        out. So a single-key setup behaves as it always did, and extra keys make
        the waiting disappear rather than merely shorten it.
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
            pool = self.pool(provider)
            key: Optional[KeyState] = None
            for attempt in range(self.max_retries):
                try:
                    key = await pool.acquire(model)
                except AllKeysExhausted as exc:
                    last_exc = exc
                    attempts += 1
                    logger.warning("%s: %s; failing over", agent, exc)
                    break

                t0 = time.time()
                try:
                    resp = await self._call(provider, messages, model, params, key)
                    text = (resp.choices[0].message.content or "").strip()
                    pool.note_success(key)
                    self._record(
                        agent, model, provider.name, messages, text,
                        time.time() - t0, resp, attempts, None, vision, key,
                    )
                    return text
                except Exception as exc:  # noqa: BLE001 - provider SDKs raise broadly
                    last_exc = exc
                    attempts += 1
                    kind = pool.note_failure(key, model, exc)
                    retryable = self._is_retryable(exc)
                    logger.warning(
                        "%s call failed (%s/%s, attempt %d/%d): %s",
                        agent, provider.name, key.label, attempt + 1, self.max_retries, exc,
                    )
                    if not retryable:
                        break
                    if kind is not None:
                        # Another key on this provider may serve the request.
                        # Going straight back to `acquire` is the whole point:
                        # it either picks a free key with no delay at all, or
                        # waits precisely until one frees up.
                        if pool.usable_for(model):
                            continue
                        logger.warning(
                            "Every key is out of daily quota on %s; failing over",
                            provider.name,
                        )
                        break
                    delay = self.retry_base_delay_s * (2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(min(delay, 60.0))

            self._record(
                agent, model, provider.name, messages, "",
                0.0, None, attempts, str(last_exc), vision, key,
            )

        raise RuntimeError(f"All LLM providers failed for {agent}: {last_exc}") from last_exc

    def _record(self, agent, model, provider, messages, text, latency, resp, retries,
                error, vision, key=None):
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
                # Thinking tokens are billed as output and drawn from the same
                # max_tokens budget, but Gemini's OpenAI-compat layer reports
                # them only as the shortfall in `total`. Deriving it here makes
                # a starved reply obvious in the trace instead of arithmetic.
                reasoning = _reasoning_tokens(resp.usage)
                if reasoning is not None:
                    usage["reasoning_tokens"] = reasoning
                usage["finish_reason"] = getattr(resp.choices[0], "finish_reason", None)
            # The key's *label* and last four characters, never the key: enough
            # to audit that rotation happened, useless to anyone reading the run.
            self.recorder.log_llm_call(
                agent=agent, model=model or "?", provider=provider,
                prompt=_flatten_prompt(messages), response=text,
                latency_s=latency, usage=usage, retries=retries,
                error=error, has_image=vision,
                key=getattr(key, "label", None),
                key_fingerprint=getattr(key, "fingerprint", None),
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
            pool = llm.pool(llm.primary)
            print(f"\n{len(pool)} key(s) in the {llm.primary.name} pool, "
                  f"{llm.rpm} rpm each ({llm.rpm * len(pool)} rpm total):")
            for key in pool.keys:
                # Probe each key separately. A pool is only as good as its
                # weakest member, and a key that is typo'd or already out of
                # daily quota is invisible until something asks it directly.
                models = await llm.list_models(llm.primary, key)
                dead = ", ".join(sorted(key.exhausted)) or "-"
                print(f"  {key.label:8s} {key.fingerprint:8s} "
                      f"{len(models):3d} models  used today: {key.used_today:3d}  "
                      f"exhausted: {dead}")
                if not models:
                    print(f"           ^ unreachable — bad key, or no quota left")

            models = await llm.list_models()
            print(f"\n{len(models)} models reachable:")
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
