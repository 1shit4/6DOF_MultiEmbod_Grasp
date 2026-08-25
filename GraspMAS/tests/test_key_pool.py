"""The key pool: collection, routing, 429 handling, persistence, secrecy.

Costs zero requests — nothing here touches a network. The properties under test
are the ones that decide whether a run finishes:

* a single key must behave exactly as it did before the pool existed, because
  every previously recorded run was produced that way;
* a per-minute rejection must cost **no wall-clock time** when another key is
  free, which is the whole reason the pool exists;
* a per-day rejection must be remembered, and must not spill onto other models.

Time is faked wherever a test would otherwise sleep. `KeyPool` reads
`time.monotonic` and `time.time` through the `time` module, so patching the
module attributes reaches it.
"""

import asyncio
import json

import pytest

from agents import key_pool as kp
from agents.key_pool import AllKeysExhausted, KeyPool, KeyState, collect_keys


# ---------------------------------------------------------------------------
# Collecting keys
# ---------------------------------------------------------------------------


class TestCollectKeys:
    def test_single_env_key_is_unchanged(self, monkeypatch):
        """The old single-key setup must resolve to exactly the old key."""
        monkeypatch.setenv("K", "abc")
        assert collect_keys(env_var="K", env_prefix=None) == ["abc"]

    def test_comma_separated_env(self, monkeypatch):
        monkeypatch.setenv("K", "a, b ,c")
        assert collect_keys(env_var="K", env_prefix=None) == ["a", "b", "c"]

    def test_whitespace_separated_env(self, monkeypatch):
        monkeypatch.setenv("K", "a b\tc")
        assert collect_keys(env_var="K", env_prefix=None) == ["a", "b", "c"]

    def test_numbered_env_vars_in_numeric_order(self, monkeypatch):
        monkeypatch.delenv("K", raising=False)
        monkeypatch.setenv("K_10", "ten")
        monkeypatch.setenv("K_2", "two")
        monkeypatch.setenv("K_1", "one")
        assert collect_keys(env_var="K", env_prefix="K_") == ["one", "two", "ten"]

    def test_file_is_one_key_per_line(self, tmp_path):
        f = tmp_path / "api.key"
        f.write_text("# account 1\nAAA\n\n  BBB  \n# account 2\nCCC\n")
        assert collect_keys(env_var=None, env_prefix=None, key_file=str(f)) == [
            "AAA", "BBB", "CCC"
        ]

    def test_env_precedes_file_and_duplicates_collapse(self, tmp_path, monkeypatch):
        """Env still wins, so the *first* key matches the old `_read_key`."""
        f = tmp_path / "api.key"
        f.write_text("FILE1\nENVKEY\nFILE2\n")
        monkeypatch.setenv("K", "ENVKEY")
        assert collect_keys(env_var="K", env_prefix=None, key_file=str(f)) == [
            "ENVKEY", "FILE1", "FILE2"
        ]

    def test_missing_file_and_env_yields_nothing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("K", raising=False)
        assert collect_keys(env_var="K", env_prefix="K_",
                            key_file=str(tmp_path / "nope.key")) == []

    def test_relative_file_resolves_against_base_path(self, tmp_path):
        (tmp_path / "api.key").write_text("REL\n")
        assert collect_keys(env_var=None, env_prefix=None,
                            key_file="api.key", base_path=tmp_path) == ["REL"]


class TestFingerprint:
    def test_is_the_last_four_characters(self):
        assert kp.fingerprint("AIzaSyABCDEFGH1234") == "...1234"

    def test_short_key_does_not_leak(self):
        assert kp.fingerprint("ab") == "...?"


# ---------------------------------------------------------------------------
# Classifying 429s
# ---------------------------------------------------------------------------


class TestClassifyQuotaError:
    def test_per_day_quota_id(self):
        exc = Exception(
            "429 RESOURCE_EXHAUSTED quotaId: "
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )
        assert kp.classify_quota_error(exc) == "day"

    def test_per_minute_quota_id(self):
        exc = Exception(
            "429 quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
        )
        assert kp.classify_quota_error(exc) == "minute"

    def test_unknown_429_is_read_as_minute(self):
        """Rotating on a misread is cheap; retiring a key for a day is not."""
        assert kp.classify_quota_error(Exception("429 rate limit exceeded")) == "minute"

    def test_non_quota_error_is_none(self):
        assert kp.classify_quota_error(Exception("500 internal error")) is None
        assert kp.classify_quota_error(ValueError("bad request")) is None

    def test_status_code_alone_is_enough(self):
        exc = Exception("something went wrong")
        exc.status_code = 429
        assert kp.classify_quota_error(exc) == "minute"

    def test_retry_delay_is_parsed(self):
        exc = Exception('429 ... "retryDelay": "27s" ...')
        assert kp.retry_delay_s(exc) == 27.0

    def test_absent_retry_delay_is_none(self):
        assert kp.retry_delay_s(Exception("429")) is None


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@pytest.fixture
def clock(monkeypatch):
    """A monotonic clock that only moves when a test says so.

    `asyncio.sleep` advances it instead of waiting, so a test can assert both
    *that* the pool waited and *how long* it thought it was waiting, in
    milliseconds of real time.
    """

    class Clock:
        def __init__(self):
            self.now = 1000.0
            self.slept = []

        def monotonic(self):
            return self.now

        async def sleep(self, seconds):
            self.slept.append(seconds)
            self.now += seconds

    c = Clock()
    monkeypatch.setattr(kp.time, "monotonic", c.monotonic)
    monkeypatch.setattr(kp.asyncio, "sleep", c.sleep)
    return c


def acquire(pool, model="m"):
    return asyncio.get_event_loop().run_until_complete(pool.acquire(model))


class TestRouting:
    def test_least_used_spreads_across_keys(self, clock):
        pool = KeyPool(["k1", "k2", "k3"], rpm=100)
        picked = [asyncio.run(pool.acquire("m")).label for _ in range(6)]
        assert picked == ["key_1", "key_2", "key_3", "key_1", "key_2", "key_3"]
        assert [k.used_today for k in pool.keys] == [2, 2, 2]

    def test_a_single_key_is_always_chosen(self, clock):
        pool = KeyPool(["only"], rpm=100)
        assert [asyncio.run(pool.acquire("m")).label for _ in range(3)] == ["key_1"] * 3

    def test_a_key_used_outside_this_run_is_deprioritised(self, clock):
        """Persisted usage must actually steer routing, not just be recorded."""
        pool = KeyPool(["k1", "k2"], rpm=100)
        pool.keys[0].used_today = 5
        assert asyncio.run(pool.acquire("m")).label == "key_2"

    def test_no_sleep_while_any_key_has_headroom(self, clock):
        """N keys at rpm=1 serve N immediate calls. This is the point."""
        pool = KeyPool(["k1", "k2", "k3"], rpm=1)
        labels = [asyncio.run(pool.acquire("m")).label for _ in range(3)]
        assert sorted(labels) == ["key_1", "key_2", "key_3"]
        assert clock.slept == []

    def test_the_call_past_every_window_waits_exactly_once(self, clock):
        pool = KeyPool(["k1", "k2"], rpm=1)
        asyncio.run(pool.acquire("m"))
        asyncio.run(pool.acquire("m"))
        assert clock.slept == []
        asyncio.run(pool.acquire("m"))
        assert len(clock.slept) == 1
        assert clock.slept[0] == pytest.approx(60.05, abs=0.2)

    def test_window_slides(self, clock):
        pool = KeyPool(["k1"], rpm=2)
        asyncio.run(pool.acquire("m"))
        asyncio.run(pool.acquire("m"))
        clock.now += 61
        asyncio.run(pool.acquire("m"))
        assert clock.slept == []

    def test_round_robin_selection(self, clock):
        pool = KeyPool(["k1", "k2"], rpm=100, selection="round_robin")
        assert [asyncio.run(pool.acquire("m")).label for _ in range(4)] == [
            "key_1", "key_2", "key_1", "key_2"
        ]

    def test_sticky_selection_stays_put(self, clock):
        pool = KeyPool(["k1", "k2"], rpm=100, selection="sticky")
        assert [asyncio.run(pool.acquire("m")).label for _ in range(3)] == ["key_1"] * 3

    def test_empty_pool_is_rejected(self):
        with pytest.raises(ValueError):
            KeyPool([])


# ---------------------------------------------------------------------------
# Reacting to 429s
# ---------------------------------------------------------------------------


class TestMinuteLimit:
    def test_rotation_costs_no_time(self, clock):
        """The headline property: a per-minute 429 must not stall the arm."""
        pool = KeyPool(["k1", "k2"], rpm=100)
        first = asyncio.run(pool.acquire("m"))
        pool.note_failure(first, "m", Exception("429 quotaId: ...PerMinute..."))
        second = asyncio.run(pool.acquire("m"))
        assert second.label != first.label
        assert clock.slept == []

    def test_a_lone_key_backs_off_exponentially(self, clock):
        """With nothing to rotate to, behave as the old client did: 2, 4, 8 s.

        Filling the key's minute window instead — the obvious implementation —
        would force a flat 60 s wait here, which is worse than what it replaces.
        """
        pool = KeyPool(["only"], rpm=100, base_delay_s=2.0)
        exc = Exception("429 quotaId: ...PerMinute...")
        waits = []
        for _ in range(3):
            key = asyncio.run(pool.acquire("m"))
            pool.note_failure(key, "m", exc)
            waits.append(key.cooldown_until - clock.now)
        assert waits == [2.0, 4.0, 8.0]

    def test_provider_retry_delay_is_honoured(self, clock):
        pool = KeyPool(["only"], rpm=100, base_delay_s=2.0)
        key = asyncio.run(pool.acquire("m"))
        pool.note_failure(key, "m", Exception('429 "retryDelay": "27s"'))
        assert key.cooldown_until - clock.now == pytest.approx(27.0)

    def test_success_clears_the_backoff_ramp(self, clock):
        pool = KeyPool(["only"], rpm=100, base_delay_s=2.0)
        key = asyncio.run(pool.acquire("m"))
        pool.note_failure(key, "m", Exception("429 PerMinute"))
        pool.note_success(key)
        assert key.consecutive_limits == 0 and key.cooldown_until == 0.0

    def test_cooldown_is_waited_out_when_it_is_the_only_key(self, clock):
        pool = KeyPool(["only"], rpm=100, base_delay_s=2.0)
        key = asyncio.run(pool.acquire("m"))
        pool.note_failure(key, "m", Exception("429 PerMinute"))
        asyncio.run(pool.acquire("m"))
        assert clock.slept and clock.slept[0] == pytest.approx(2.05, abs=0.1)


class TestDailyExhaustion:
    DAY = "429 quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"

    def test_key_is_retired_for_that_model_only(self, clock):
        pool = KeyPool(["k1", "k2"], rpm=100)
        key = asyncio.run(pool.acquire("flash"))
        pool.note_failure(key, "flash", Exception(self.DAY))

        assert key not in pool.usable_for("flash")
        assert key in pool.usable_for("lite")  # a different quota entirely

    def test_routing_skips_an_exhausted_key(self, clock):
        pool = KeyPool(["k1", "k2"], rpm=100)
        pool.note_failure(pool.keys[0], "flash", Exception(self.DAY))
        assert {asyncio.run(pool.acquire("flash")).label for _ in range(4)} == {"key_2"}

    def test_all_exhausted_raises_rather_than_waiting(self, clock):
        """No amount of waiting fixes a daily quota, so the caller must fail over."""
        pool = KeyPool(["k1", "k2"], rpm=100)
        for k in pool.keys:
            pool.note_failure(k, "flash", Exception(self.DAY))
        with pytest.raises(AllKeysExhausted):
            asyncio.run(pool.acquire("flash"))
        assert clock.slept == []

    def test_exhaustion_lapses_after_the_reset(self, clock, monkeypatch):
        pool = KeyPool(["k1"], rpm=100)
        key = pool.keys[0]
        pool.mark_day_exhausted(key, "flash", reset_epoch=kp.time.time() + 10)
        assert key.is_exhausted("flash", kp.time.time())
        # Past the reset the entry is dropped, so the key gets a fresh probe
        # rather than being trusted to still be dead.
        assert not key.is_exhausted("flash", kp.time.time() + 20)
        assert "flash" not in key.exhausted

    def test_next_reset_is_in_the_future(self):
        assert kp.next_daily_reset() > kp.time.time()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_exhaustion_survives_a_new_pool(self, tmp_path):
        path = tmp_path / "quota.json"
        pool = KeyPool(["k1", "k2"], rpm=100, state_path=path)
        pool.mark_day_exhausted(pool.keys[0], "flash")

        revived = KeyPool(["k1", "k2"], rpm=100, state_path=path)
        assert revived.keys[0].is_exhausted("flash", kp.time.time())
        assert not revived.keys[1].is_exhausted("flash", kp.time.time())

    def test_daily_usage_survives_and_steers_routing(self, tmp_path, clock):
        path = tmp_path / "quota.json"
        pool = KeyPool(["k1", "k2"], rpm=100, state_path=path)
        for _ in range(3):
            asyncio.run(pool.acquire("m"))

        revived = KeyPool(["k1", "k2"], rpm=100, state_path=path)
        assert [k.used_today for k in revived.keys] == [2, 1]
        assert asyncio.run(revived.acquire("m")).label == "key_2"

    def test_only_fingerprints_are_written(self, tmp_path):
        path = tmp_path / "quota.json"
        pool = KeyPool(["SECRETKEY1234", "SECRETKEY5678"], rpm=100, state_path=path)
        asyncio.run(pool.acquire("m"))

        text = path.read_text()
        assert "SECRETKEY1234" not in text and "SECRETKEY5678" not in text
        assert "...1234" in text

    def test_stale_day_usage_is_ignored(self, tmp_path):
        """A yesterday count must not permanently bias routing away from a key."""
        path = tmp_path / "quota.json"
        pool = KeyPool(["k1"], rpm=100, state_path=path)
        path.write_text(json.dumps({
            "version": 1,
            "keys": {pool.keys[0].state_id: {
                "day": "1999-01-01", "used_today": 99, "exhausted": {}
            }},
        }))
        assert KeyPool(["k1"], rpm=100, state_path=path).keys[0].used_today == 0

    def test_unreadable_state_is_ignored_not_fatal(self, tmp_path):
        path = tmp_path / "quota.json"
        path.write_text("{ not json")
        assert len(KeyPool(["k1"], rpm=100, state_path=path)) == 1

    def test_unknown_state_ids_are_skipped(self, tmp_path):
        path = tmp_path / "quota.json"
        path.write_text(json.dumps({
            "version": 1,
            "keys": {"deadbeefdeadbeef": {"day": kp.usage_day(), "used_today": 7, "exhausted": {}}},
        }))
        assert KeyPool(["k1"], rpm=100, state_path=path).keys[0].used_today == 0

    def test_no_state_path_writes_nothing(self, tmp_path):
        pool = KeyPool(["k1"], rpm=100)
        asyncio.run(pool.acquire("m"))
        assert list(tmp_path.iterdir()) == []


class TestDescribe:
    def test_describe_carries_no_key(self):
        pool = KeyPool(["SECRETKEY1234"], rpm=100)
        rows = pool.describe()
        assert rows[0]["label"] == "key_1"
        assert rows[0]["fingerprint"] == "...1234"
        assert "SECRETKEY1234" not in json.dumps(rows)


class TestKeyState:
    def test_rpm_floor_is_one(self):
        assert KeyState(key="k", index=0, rpm=0).rpm == 1

    def test_label_is_one_based(self):
        assert KeyState(key="k", index=2).label == "key_3"


class TestStateIdentity:
    """Persistence must key on something that cannot collide.

    Two keys ending in the same four characters share a `fingerprint`. Indexing
    the quota file by it collapsed both onto one record, so one key's daily
    exhaustion retired the other — a healthy key silently withdrawn from the
    pool, which is the exact failure the pool exists to prevent.
    """

    def test_suffix_collision_does_not_share_state(self, tmp_path):
        path = tmp_path / "quota.json"
        twins = ["AAAAAAAAsame", "BBBBBBBBsame"]
        assert kp.fingerprint(twins[0]) == kp.fingerprint(twins[1])

        pool = KeyPool(twins, rpm=100, state_path=path)
        pool.mark_day_exhausted(pool.keys[0], "flash")

        revived = KeyPool(twins, rpm=100, state_path=path)
        assert revived.keys[0].is_exhausted("flash", kp.time.time())
        assert not revived.keys[1].is_exhausted("flash", kp.time.time())

    def test_state_id_is_not_the_key(self):
        assert "SECRET" not in kp.state_id("SECRETKEY1234")
        assert kp.state_id("a") != kp.state_id("b")
