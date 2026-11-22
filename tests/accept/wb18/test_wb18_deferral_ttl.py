"""WB18 (RV6 B4): a deferred funded entry never expires while deferred.

§12b.5 LB5 / D35 (a): "A deferral doesn't consume the miner's per-round
submission: the entry stays queued and uncharged until its target isn't the
sentinel", for at most ``sentinel_max_defer_eras`` eras (recommended 4).
WB15's ``rows_held`` deferral is uncharged the same way ("left pending and
queued, uncharged").

The funding queue's ``expire_stale`` (run first in every driver tick)
terminally fails a ``queued`` entry idle past ``entry_ttl_seconds``
(``[round] funded_entry_ttl_hours`` = 36 h), measured from ``active_at``.
Four deferral eras at 12 h are 48 h > 36 h, so without a fix a deferred entry
goes ``failed`` / ``funding_expired`` and the miner pays again to re-enter.
That is a charge, and §12b.5 forbids it.

The rule pinned here: **refresh.** While an entry is deferred
(``sentinel_target`` or ``rows_held``), the trainer re-stamps its liveness
(``last_active``) at each tick that defers it. That is the same proof of life
that lane A gives an admission-held entry (``FundedQueue.touch``,
review 2026-09-02). The TTL clock does not run while the entry is deferred.
When the deferral ends, the clock restarts from the last deferred tick, so
the entry has at least one full normal TTL to start. Why refresh and not
pause: a pause keeps the idle time from before the deferral, so an entry
funded 30 h before its first deferral would have 6 h left afterwards. §12b.5
says the deferral consumes nothing of the miner's, and that includes the
time it has left. Extending the TTL by the deferred span gives the same
lower bound, and these tests accept it too. They pin only the lower bound,
never the mechanism.

Unchanged (invariant): an entry that is never deferred still expires exactly
as today. Its ``active_at`` is not touched, and it fails ``funding_expired``
at the first tick past ``active_at + entry_ttl_seconds``.

Out of scope here: the payer-key vault's own TTL (``PayerKeyVault.ttl_seconds``,
also 36 h). The fixture's ops expose no vault, so it is flagged and not pinned.
"""

from __future__ import annotations

import pytest

from . import helpers as h
from .helpers import GIFT, HK_A, HK_B, HK_C, HK_E, HOUR, ROWS_A, ROWS_B
from .test_wb18_sentinel_deferral import (
    _assert_not_charged,
    _assert_sentinel_row,
    _assert_started,
)

ERA_S = h.ERA_HOURS * HOUR
# two ticks per era: at the era's first hour and 11 h in (the live trainer
# ticks every few minutes; two per era is the sparsest cadence pinned)
OFFSETS_H = (0.0, 11.0)


@pytest.fixture(autouse=True)
def _no_torch():
    with h.torch_blocked():
        yield


def _ttl(sc) -> float:
    ttl = float(sc.fq.entry_ttl_seconds)
    assert ttl == 36 * HOUR, f"fixture TTL is the shipped 36 h; got {ttl / HOUR} h"
    return ttl


def _bound(sc) -> int:
    return int(sc.lb.sentinel_max_defer_eras)


def _not_expired(sc, hotkey: str, when: str) -> None:
    e = sc.fq.get(hotkey)
    assert e is not None and e.status == "queued" and e.last_error_class != "funding_expired", (
        f"{when}: a deferred funded entry must not expire while deferred (§12b.5 uncharged; "
        f"RV6 B4): status={getattr(e, 'status', None)!r} "
        f"class={getattr(e, 'last_error_class', None)!r} "
        f"error={getattr(e, 'last_error', None)!r}")


def _defer_through(sc, hotkey: str, target: str, *, restart: bool = False,
                   trace: list | None = None) -> tuple[float, float]:
    """Open the gate at ``T0`` (the fund time) and tick every era
    ``1 .. bound`` at offsets 0 h and 11 h (12 h per era), each tick
    asserting the entry is still deferred, uncharged and unexpired.
    Returns ``(T0, last deferred tick time)``."""
    t0 = sc.clock.t
    sc.open_gate()
    last = t0
    for era in range(1, _bound(sc) + 1):
        if restart and era > 1:
            sc.restart_all()
        for off in OFFSETS_H:
            sc.clock.t = t0 + (era - 1) * ERA_S + off * HOUR
            sc.tick(era)
            last = sc.clock.t
            idle_h = (last - t0) / HOUR
            when = f"era {era} +{off:g} h ({idle_h:g} h after funding)"
            _not_expired(sc, hotkey, when)
            _assert_not_charged(sc, hotkey)
            _assert_sentinel_row(sc, sc.status(), hotkey, target)
            if trace is not None:
                trace.append(_snapshot(sc, era, off))
    return t0, last


def _snapshot(sc, era: int, off: float) -> tuple:
    return (era, off,
            tuple(sorted((e.hotkey, e.status, e.attempts, e.requeues, e.last_error_class)
                         for e in sc.fq.entries())),
            tuple(sorted(sc.open_legs())),
            tuple(sorted((r.get("hotkey"), r.get("deferred"))
                         for r in h.deferred_rows(sc.status()))))


# ── sentinel_target: never expires across the bound ──────────────────────────


@pytest.mark.parametrize("max_defer", [None, 6], ids=["shipped-4", "configured-6"])
def test_sentinel_deferred_entry_never_expires_across_bound(tmp_path, max_defer):
    """Deferred in every era up to ``sentinel_max_defer_eras`` (4 shipped =
    48 h; 6 configured = 72 h, twice the TTL), ticked twice per era: never
    expired, never re-fund-required, never charged."""
    sc = h.Scenario(tmp_path, max_defer=max_defer)
    for era in range(1, _bound(sc) + 2):
        sc.set_sentinel(era, "gift")
    sc.fund(HK_A, 10, GIFT)
    t0, last = _defer_through(sc, HK_A, GIFT)
    assert last - t0 > _ttl(sc), "the deferral must outlast the TTL for this test to bite"


@pytest.mark.parametrize("release", ["rotation", "past_bound"])
def test_after_deferral_full_ttl_to_start(tmp_path, release):
    """When the deferral ends, the entry has at least one full TTL from its
    last deferred tick. A first tick just short of that still finds it
    ``queued``, and it starts on its own target."""
    sc = h.Scenario(tmp_path, max_defer=None)
    bound = _bound(sc)
    for era in range(1, bound + 1):
        sc.set_sentinel(era, "gift")
    sc.set_sentinel(bound + 1, "boom" if release == "rotation" else "gift")
    sc.fund(HK_A, 10, GIFT)
    _, last = _defer_through(sc, HK_A, GIFT)
    sc.clock.t = last + _ttl(sc) - 60.0
    sc.tick(bound + 1)
    _assert_started(sc, HK_A, GIFT)
    e = sc.fq.get(HK_A)
    assert (e.attempts, e.requeues) == (0, 0), (e.attempts, e.requeues)


def test_paused_clock_survives_restarts(tmp_path):
    """A whole trainer restart (new driver, new queue object, same files) at
    every era boundary changes nothing: still never expired across the
    bound. The trace is identical to the run without restarts."""
    def run(sub, restart):
        sc = h.Scenario(tmp_path / sub, max_defer=None)
        for era in range(1, _bound(sc) + 2):
            sc.set_sentinel(era, "gift")
        sc.fund(HK_A, 10, GIFT)
        trace: list = []
        _, last = _defer_through(sc, HK_A, GIFT, restart=restart, trace=trace)
        if restart:
            sc.restart_all()
        sc.clock.t = last + _ttl(sc) - 60.0
        sc.tick(_bound(sc) + 1)
        _assert_started(sc, HK_A, GIFT)
        return trace

    assert run("plain", False) == run("restarted", True)


def test_deferral_ttl_is_deterministic(tmp_path):
    """The same plan and queue states come out under a permuted funding order
    and a reversed chain order, with two deferred entries and one domain
    entry."""
    def run(sub, order, reverse_chain):
        sc = h.Scenario(tmp_path / sub, max_defer=None)
        for era in range(1, _bound(sc) + 2):
            sc.set_sentinel(era, "gift")
        spec = {HK_A: (10, GIFT), HK_B: (20, GIFT), HK_E: (5, "energy")}
        for hk in order:
            sc.fund(hk, *spec[hk])
        if reverse_chain:
            sc.ops.rows.reverse()
        trace: list = []
        _defer_through(sc, HK_A, GIFT, trace=trace)
        _not_expired(sc, HK_B, "end of the bound")
        return trace

    a = run("a", [HK_A, HK_B, HK_E], False)
    b = run("b", [HK_E, HK_B, HK_A], True)
    assert a == b
    assert a[-1][3] == (HK_E,), a[-1]
    assert a[-1][4] == ((HK_A, h.SENTINEL_TARGET), (HK_B, h.SENTINEL_TARGET)), a[-1]


# ── WB15 rows_held: the same uncharged deferral, the same TTL ────────────────


def test_rows_held_deferred_entry_never_expires(tmp_path):
    """WB15: a declared-rows entry whose rows an open leg holds is
    ``deferred: rows_held``, uncharged. It doesn't expire while held, however
    long the holder runs (here 59 h across five eras, sentinel off)."""
    sc = h.Scenario(tmp_path, sentinel_from_block=0)
    sc.fund(HK_B, 5, "energy", ROWS_B)
    sc.fund(HK_A, 10, "web", ROWS_A)
    t0 = sc.clock.t
    sc.open_gate()
    assert HK_B in sc.open_legs()
    last = t0
    for era in range(1, 6):
        for off in OFFSETS_H:
            sc.clock.t = t0 + (era - 1) * ERA_S + off * HOUR
            sc.tick(era)
            last = sc.clock.t
            _not_expired(sc, HK_A, f"rows_held, era {era} +{off:g} h")
            _assert_not_charged(sc, HK_A)
            row = h.deferred_for(sc.status(), HK_A)
            assert row.get("deferred") == h.ROWS_HELD, row
            assert HK_B in sc.open_legs()
    assert last - t0 > _ttl(sc)


# ── invariant: the TTL of a never-deferred entry is unchanged (paired) ───────


def _idle_entry(sc, hotkey: str) -> float:
    """Fund ``hotkey`` on a domain, then withdraw its chain commitment: the
    driver never sees it, so it is idle and never deferred. Returns its
    ``active_at``."""
    sc.fund(hotkey, 30, "energy")
    sc.ops.rows = [r for r in sc.ops.rows if r.hotkey != hotkey]
    return float(sc.fq.get(hotkey).active_at)


def _tick_until(sc, t0: float, end: float, era0: int = 1) -> int:
    """Tick every 12 h from ``t0`` up to (not past) ``end``; returns the era."""
    era = era0
    t = t0
    while t + ERA_S <= end:
        t += ERA_S
        era += 1
        sc.clock.t = t
        sc.tick(era)
    return era


def test_inv_ttl_unchanged_for_undeferred_entry(tmp_path):
    """Nothing deferred: an idle funded entry is not refreshed by the driver,
    stays ``queued`` up to exactly ``active_at + TTL``, and the first tick
    past that fails it ``funding_expired``, as before RV6."""
    sc = h.Scenario(tmp_path)
    active = _idle_entry(sc, HK_C)
    sc.open_gate()
    ttl = _ttl(sc)
    era = _tick_until(sc, sc.clock.t, active + ttl)
    sc.clock.t = active + ttl
    sc.tick(era)
    e = sc.fq.get(HK_C)
    assert e.status == "queued" and float(e.active_at) == active, (e.status, e.active_at)
    sc.clock.t = active + ttl + 1.0
    sc.tick(era)
    e = sc.fq.get(HK_C)
    assert (e.status, e.last_error_class) == ("failed", "funding_expired"), (
        e.status, e.last_error_class)


def test_inv_ttl_armed_nonempty_lane_d_effect(tmp_path):
    """Same clock, with a sentinel-deferred entry next to the idle one: the
    first tick past the idle entry's TTL expires it. The deferred entry, funded
    at the same moment, stays queued and uncharged."""
    sc = h.Scenario(tmp_path, max_defer=None)
    for era in range(1, 8):
        sc.set_sentinel(era, "gift")
    sc.fund(HK_A, 10, GIFT)
    active = _idle_entry(sc, HK_C)
    sc.open_gate()
    ttl = _ttl(sc)
    era = _tick_until(sc, sc.clock.t, active + ttl)
    sc.clock.t = active + ttl + 1.0
    sc.tick(era)
    e = sc.fq.get(HK_C)
    assert (e.status, e.last_error_class) == ("failed", "funding_expired"), (
        e.status, e.last_error_class)
    _not_expired(sc, HK_A, "tick that expires the idle entry")
    _assert_not_charged(sc, HK_A)
    _assert_sentinel_row(sc, sc.status(), HK_A, GIFT)
