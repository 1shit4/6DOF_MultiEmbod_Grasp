"""A pool of interchangeable API keys, routed by least usage.

Gemini's free tier is metered **per Google Cloud project, per model** — measured
5 requests/minute and 20 requests/day on `gemini-3.5-flash` (CLAUDE.md §6). One
declutter run is 16-30 calls, so a single key buys one run per day and the
verification matrix is simply unreachable. Keys from *separate accounts* have
separate buckets, so N keys multiply both limits by N.

Two properties matter, and they are different:

* **Per-minute limits must never cost wall-clock time.** The old behaviour was a
  process-wide sliding window that *slept* when full (`AsyncRateLimiter`), and a
  429 triggered exponential backoff. On a real arm a 60 s pause mid-manipulation
  is not a rate limit, it is a fault. Here the window is **per key**, and the
  pool only ever sleeps when *every* key is simultaneously at its own limit.
  With 5 keys at 8 rpm that is 40 rpm against a ~20-call run: never.
* **Per-day exhaustion must be remembered.** It survives the process (a run an
  hour later must not re-burn a request rediscovering it) and it is scoped to
  the *model* that returned it, because the quota is. A key out of
  `gemini-3.5-flash` still has its full `gemini-3.1-flash-lite` budget.

Selection is "fewest requests today, then fewest in the current minute" — the
rule requested by the user, and one that naturally round-robins, since every
send increments the count that chose it.

**Keys never leave this module.** Traces and logs carry `label` (`key_1`) and
`fingerprint` (the last four characters), which are enough to audit rotation and
not enough to authenticate.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Google's free-tier daily counters reset at midnight US/Pacific.
_RESET_TZ = "America/Los_Angeles"
_QUOTA_STATE_VERSION = 1


# ---------------------------------------------------------------------------
# Collecting keys
# ---------------------------------------------------------------------------


def _split_keys(blob: str) -> list[str]:
    """Split one env value into keys.

    Accepts commas, whitespace and newlines so a user can paste however they
    like; `#` starts a comment so the same parser serves the key *file*.
    """
    out: list[str] = []
    for line in (blob or "").splitlines():
        line = line.split("#", 1)[0]
        for part in re.split(r"[,\s]+", line):
            part = part.strip()
            if part:
                out.append(part)
    return out


def collect_keys(
    env_var: Optional[str] = "LLM_API_KEY",
    env_prefix: Optional[str] = "LLM_API_KEY_",
    key_file: Optional[str] = None,
    base_path: Optional[Path] = None,
) -> list[str]:
    """Gather every configured key, in priority order, de-duplicated.

    Order is env var, then numbered env vars, then the file — matching the old
    single-key precedence (`_read_key` let the env win over the file), so a
    setup that worked before still resolves to the same *first* key.

    A single `LLM_API_KEY=abc` therefore still yields exactly `["abc"]`: the
    multi-key path is a superset of the old one, not a replacement for it.
    """
    found: list[str] = []

    if env_var:
        found.extend(_split_keys(os.environ.get(env_var, "")))

    if env_prefix:
        numbered: list[tuple[tuple[int, str], str]] = []
        for name, value in os.environ.items():
            if not name.startswith(env_prefix) or name == env_prefix:
                continue
            suffix = name[len(env_prefix):]
            # Numeric suffixes first and in numeric order, so _2 precedes _10;
            # anything else sorts after, by name.
            sort_key = (int(suffix), "") if suffix.isdigit() else (10**9, suffix)
            numbered.append((sort_key, value))
        for _, value in sorted(numbered, key=lambda p: p[0]):
            found.extend(_split_keys(value))

    if key_file:
        path = Path(key_file)
        if not path.is_absolute():
            path = (base_path or Path.cwd()) / path
        if path.exists():
            found.extend(_split_keys(path.read_text()))

    seen: set[str] = set()
    unique: list[str] = []
    for k in found:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def fingerprint(key: str) -> str:
    """A short, human-recognisable handle for logs and traces.

    For *display only*. It is not unique — two keys ending in the same four
    characters share it — so it must never be used to index state. See
    `state_id`, which is what persistence keys on.
    """
    return f"...{key[-4:]}" if len(key) >= 4 else "...?"


def state_id(key: str) -> str:
    """A collision-free, non-reversible identity for the quota state file.

    Keying persistence on `fingerprint` instead looks equivalent and is not:
    two keys sharing a four-character suffix would collapse onto one record, so
    one key's daily exhaustion would retire the other — a healthy key silently
    withdrawn from the pool, which is exactly the failure the pool exists to
    prevent. A hash cannot collide by accident and still reveals nothing.
    """
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return re.sub(r"[\s_-]+", "", text.lower())


def classify_quota_error(exc: BaseException) -> Optional[str]:
    """Return `"minute"`, `"day"`, or None for a non-quota error.

    The numbers themselves appear nowhere in Google's published limits table;
    the only place they surface is the body of the 429, as
    `quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier`. So the
    distinction that decides whether a key is briefly busy or finished for the
    day has to be parsed out of an error string.

    An unrecognised 429 is reported as `"minute"` deliberately: rotating to
    another key is cheap and reversible, while retiring a key for a day on a
    misread throws away a fifth of the budget.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    text = str(exc)
    low = text.lower()
    flat = _normalise(text)

    is_quota = status == 429 or any(
        s in low for s in ("429", "resource_exhausted", "rate limit", "quota")
    )
    if not is_quota:
        return None

    if "perday" in flat or "daily" in flat or "requestsperday" in flat:
        return "day"
    if "perminute" in flat:
        return "minute"
    return "minute"


def retry_delay_s(exc: BaseException) -> Optional[float]:
    """Honour the provider's own `retryDelay` when it gives one."""
    m = re.search(r"retry[_-]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(exc), re.I)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _pacific_now() -> datetime:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(_RESET_TZ))
    except Exception:  # noqa: BLE001 - missing tzdata is not worth failing over
        # Fixed -08:00 rather than UTC: erring toward an *earlier* reset would
        # let a still-exhausted key be retried, which costs a wasted request.
        return datetime.now(timezone(timedelta(hours=-8)))


def next_daily_reset(now: Optional[float] = None) -> float:
    """Epoch seconds of the next midnight US/Pacific."""
    ref = _pacific_now()
    if now is not None:
        ref = datetime.fromtimestamp(now, ref.tzinfo)
    tomorrow = (ref + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return tomorrow.timestamp()


def usage_day(now: Optional[float] = None) -> str:
    """The bucket daily usage is counted into — a Pacific calendar date."""
    ref = _pacific_now()
    if now is not None:
        ref = datetime.fromtimestamp(now, ref.tzinfo)
    return ref.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# One key
# ---------------------------------------------------------------------------


@dataclass
class KeyState:
    """One API key and everything known about its current standing."""

    key: str
    index: int
    rpm: int = 8
    label: str = ""
    used_today: int = 0
    total_used: int = 0
    #: monotonic timestamps of recent sends, pruned to a 60 s window
    window: list = field(default_factory=list)
    #: monotonic time before which this key must not be used again
    cooldown_until: float = 0.0
    #: consecutive per-minute rejections, reset by any success on this key
    consecutive_limits: int = 0
    #: model -> epoch seconds at which its daily quota resets
    exhausted: dict = field(default_factory=dict)

    def __post_init__(self):
        self.label = self.label or f"key_{self.index + 1}"
        self.rpm = max(int(self.rpm), 1)

    @property
    def fingerprint(self) -> str:
        """Display handle. Not unique — never index state with it."""
        return fingerprint(self.key)

    @property
    def state_id(self) -> str:
        """Collision-free identity used by the quota state file."""
        return state_id(self.key)

    def prune(self, now: float) -> None:
        self.window = [t for t in self.window if now - t < 60.0]

    def in_window(self, now: float) -> int:
        self.prune(now)
        return len(self.window)

    def is_exhausted(self, model: Optional[str], wall: float) -> bool:
        """True while this key's daily quota for `model` is spent.

        Entries are dropped once their reset passes, so a key always gets a
        fresh probe after reset rather than being trusted to still be dead.
        """
        if not model:
            return False
        reset = self.exhausted.get(model)
        if reset is None:
            return False
        if wall >= reset:
            del self.exhausted[model]
            return False
        return True

    def available_at(self, now: float) -> float:
        """Monotonic time at which this key could next send."""
        self.prune(now)
        earliest = self.cooldown_until
        if len(self.window) >= self.rpm:
            earliest = max(earliest, self.window[0] + 60.0 + 0.05)
        return earliest

    def register(self, now: float) -> None:
        self.prune(now)
        self.window.append(now)
        self.used_today += 1
        self.total_used += 1


class AllKeysExhausted(RuntimeError):
    """Every key is out of daily quota for the requested model."""


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


class KeyPool:
    """Routes each request to the least-used key that is ready to send."""

    def __init__(
        self,
        keys: list,
        rpm: int = 8,
        state_path: Optional[str | os.PathLike] = None,
        selection: str = "least_used",
        base_delay_s: float = 2.0,
    ):
        if not keys:
            raise ValueError("KeyPool needs at least one key")
        self.rpm = max(int(rpm), 1)
        self.selection = selection
        self.base_delay_s = float(base_delay_s)
        self.state_path = Path(state_path) if state_path else None
        self.keys = [KeyState(key=k, index=i, rpm=self.rpm) for i, k in enumerate(keys)]
        self._by_state_id = {k.state_id: k for k in self.keys}
        self._lock = asyncio.Lock()
        self._rr = 0
        self._load()

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def labels(self) -> list:
        return [k.label for k in self.keys]

    # -- selection ---------------------------------------------------------

    def _eligible(self, model: Optional[str], wall: float) -> list:
        return [k for k in self.keys if not k.is_exhausted(model, wall)]

    def _rank(self, key: KeyState, now: float) -> tuple:
        """Lower sorts first. Fewest requests today is the primary rule.

        Daily usage leads because the daily quota is the scarce one — 20 per
        model against a run's 16-30 calls — and because incrementing it on every
        send makes the rule self-balancing: whichever key is chosen becomes the
        least attractive next time, so `least_used` round-robins for free while
        still recovering correctly when one key has been used outside this run.
        """
        if self.selection == "sticky":
            return (key.index,)
        if self.selection == "round_robin":
            return ((key.index - self._rr) % len(self.keys),)
        return (key.used_today, key.in_window(now), key.index)

    def peek(self, model: Optional[str] = None) -> Optional[KeyState]:
        """The key `acquire` would pick right now, without reserving it."""
        now, wall = time.monotonic(), time.time()
        ready = [k for k in self._eligible(model, wall) if k.available_at(now) <= now]
        return min(ready, key=lambda k: self._rank(k, now)) if ready else None

    async def acquire(self, model: Optional[str] = None) -> KeyState:
        """Reserve a send slot on the best key, sleeping only if none is free.

        Raises `AllKeysExhausted` when every key is out of *daily* quota for
        this model — a condition no amount of waiting fixes, so the caller
        should fail over to another provider rather than retry.
        """
        async with self._lock:
            while True:
                now, wall = time.monotonic(), time.time()
                eligible = self._eligible(model, wall)
                if not eligible:
                    raise AllKeysExhausted(
                        f"all {len(self.keys)} keys are out of daily quota for "
                        f"{model or 'this model'}"
                    )

                ready = [k for k in eligible if k.available_at(now) <= now]
                if ready:
                    chosen = min(ready, key=lambda k: self._rank(k, now))
                    chosen.register(now)
                    self._rr = (chosen.index + 1) % len(self.keys)
                    self._save()
                    return chosen

                sleep_for = max(min(k.available_at(now) for k in eligible) - now, 0.0)
                logger.info(
                    "All %d keys at their per-minute limit; waiting %.1fs",
                    len(eligible), sleep_for,
                )
                await asyncio.sleep(sleep_for + 0.05)

    # -- feedback ----------------------------------------------------------

    def mark_minute_limited(self, key: KeyState, delay_s: Optional[float] = None) -> None:
        """This key is briefly out of per-minute allowance.

        Only *this key* is put on cooldown; nothing global sleeps. With another
        key free the next `acquire` returns instantly, which is the entire point
        of the pool.

        The cooldown is the provider's own `retryDelay` when it sends one, and
        otherwise the same exponential backoff this replaced — **per key** now,
        rather than per process. That matters for the single-key case: filling
        the key's minute window instead (the obvious implementation) would force
        a flat 60 s wait, which is strictly worse than the 2/4/8 s ramp it would
        be replacing. So a lone key behaves as it always did, and extra keys
        make the wait vanish rather than shortening it.
        """
        now = time.monotonic()
        key.prune(now)
        key.consecutive_limits += 1
        wait = float(delay_s) if delay_s else min(
            self.base_delay_s * (2 ** (key.consecutive_limits - 1)), 60.0
        )
        key.cooldown_until = max(key.cooldown_until, now + wait)
        logger.warning(
            "%s hit a per-minute limit; cooling %.1fs (%d other key(s) available)",
            key.label, wait, sum(1 for k in self.keys if k is not key),
        )

    def note_success(self, key: KeyState) -> None:
        """Clear the backoff ramp — the key is demonstrably serving again."""
        key.consecutive_limits = 0
        key.cooldown_until = 0.0

    def mark_day_exhausted(
        self, key: KeyState, model: Optional[str], reset_epoch: Optional[float] = None
    ) -> None:
        """This key's daily quota for `model` is spent until the reset."""
        if not model:
            return
        key.exhausted[model] = float(reset_epoch or next_daily_reset())
        logger.warning(
            "%s is out of daily quota for %s; retired until %s",
            key.label, model,
            datetime.fromtimestamp(key.exhausted[model]).isoformat(timespec="minutes"),
        )
        self._save()

    def note_failure(self, key: KeyState, model: Optional[str], exc: BaseException) -> Optional[str]:
        """Classify a failure and update the key's standing. Returns the kind."""
        kind = classify_quota_error(exc)
        if kind == "day":
            self.mark_day_exhausted(key, model)
        elif kind == "minute":
            self.mark_minute_limited(key, retry_delay_s(exc))
        return kind

    def usable_for(self, model: Optional[str]) -> list:
        """Keys not known to be out of daily quota for `model`."""
        return self._eligible(model, time.time())

    # -- persistence -------------------------------------------------------

    def describe(self) -> list:
        """Per-key standing, safe to print or log."""
        wall, now = time.time(), time.monotonic()
        rows = []
        for k in self.keys:
            rows.append(
                {
                    "label": k.label,
                    "fingerprint": k.fingerprint,
                    "used_today": k.used_today,
                    "in_window": k.in_window(now),
                    "exhausted": {
                        m: datetime.fromtimestamp(r).isoformat(timespec="minutes")
                        for m, r in k.exhausted.items()
                        if r > wall
                    },
                }
            )
        return rows

    def _load(self) -> None:
        """Restore daily usage and exhaustion, keyed by fingerprint.

        Fingerprints only — the state file is ordinary project data and must
        stay useless to anyone who reads it.
        """
        if not self.state_path or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable quota state %s: %s", self.state_path, exc)
            return
        if data.get("version") != _QUOTA_STATE_VERSION:
            return

        wall, today = time.time(), usage_day()
        for sid, entry in (data.get("keys") or {}).items():
            key = self._by_state_id.get(sid)
            if key is None:
                continue
            key.exhausted = {
                m: float(r) for m, r in (entry.get("exhausted") or {}).items()
                if float(r) > wall
            }
            # Usage is only meaningful within its own Pacific day; a stale count
            # would permanently bias routing away from a key that is in fact free.
            if entry.get("day") == today:
                key.used_today = int(entry.get("used_today", 0))

    def _save(self) -> None:
        if not self.state_path:
            return
        today = usage_day()
        payload = {
            "version": _QUOTA_STATE_VERSION,
            "keys": {
                k.state_id: {
                    "fingerprint": k.fingerprint,
                    "day": today,
                    "used_today": k.used_today,
                    "exhausted": k.exhausted,
                }
                for k in self.keys
            },
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, self.state_path)
        except OSError as exc:  # state is an optimisation, never a hard dependency
            logger.debug("Could not write quota state: %s", exc)
