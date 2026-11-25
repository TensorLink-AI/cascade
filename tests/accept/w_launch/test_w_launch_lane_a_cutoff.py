"""W-LAUNCH lane A shutdown (owner 2026-11-25, "lane A should shut down when
the other lanes start"): ``[round] lane_a_until_block`` — mainnet 9586800 on
the operator's trainer + intake.

Contract:

* config: the key parses in ``load_chain_config`` (round trip), defaults to 0
  (off), and is NOT an overlay key nor part of the contract digest;
* intake: at/after the cutoff a new lane A ``/v1/fund`` and a ``/v1/submit``
  carrying a payer key answer a typed ``409 lane_ended`` (``lane`` = ``"a"``)
  BEFORE the key is vaulted; before it (or with no cutoff / no head) the old
  reply; lane B / lane C funds are unaffected; queued, not yet started lane A
  entries are released (nothing charged), in-flight ones never touched;
* trainer: at/after the cutoff no new lane A challenger leg is seated, an
  in-flight leg runs to its verdict and settles, and the king leg plus the
  king-only settlement progress continue.

No network, no torch: in-memory queues / vault, fake legs (tests.unit.test_rolling).
"""

from __future__ import annotations

from dataclasses import replace

MAINNET_CUTOFF = 9586800
REF_A = "org/gen@sha256:" + "a" * 64


def _require():
    """Fail on a base without the cutoff (an assertion, not a crash)."""
    from cascade.funding import lane_ends
    from cascade.shared.config import RoundConfig

    assert hasattr(RoundConfig(), "lane_a_until_block"), "no [round] lane_a_until_block"
    assert hasattr(lane_ends, "lane_a_refusal"), "no lane A refusal in lane_ends"


# ── config ───────────────────────────────────────────────────────────────────


def test_w_launch_lane_a_cutoff_parses_round_trip_and_defaults_off(tmp_path):
    _require()
    from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config

    base = load_chain_config(DEFAULT_CHAIN_TOML)
    assert base.round.lane_a_until_block == 0                  # default: off
    assert base.round.lane_a_ended_at(10**9) is None
    text = DEFAULT_CHAIN_TOML.read_text()
    assert "\n[round]\n" in text
    p = tmp_path / "chain.toml"
    p.write_text(text.replace("\n[round]\n",
                              f"\n[round]\nlane_a_until_block = {MAINNET_CUTOFF}\n", 1))
    cfg = load_chain_config(p)
    assert cfg.round.lane_a_until_block == MAINNET_CUTOFF
    assert cfg.round.lane_a_ended_at(MAINNET_CUTOFF - 1) is None
    assert cfg.round.lane_a_ended_at(MAINNET_CUTOFF) == MAINNET_CUTOFF
    assert cfg.round.lane_a_ended_at(None) is None


def test_w_launch_lane_a_cutoff_is_not_consensus(tmp_path):
    """Not overlay-settable (an unknown overlay key would split external
    validators) and outside the contract digest."""
    _require()
    from cascade.shared import netconfig as nc
    from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config
    from cascade.shared.manifest import contract_digest

    assert "lane_a_until_block" not in nc.ALLOWED["round"]
    assert all(key != "lane_a_until_block" for _, key in nc._UNTIL_KEYS)
    base = load_chain_config(DEFAULT_CHAIN_TOML)
    cut = replace(base, round=replace(base.round, lane_a_until_block=MAINNET_CUTOFF))
    assert contract_digest(cut.training) == contract_digest(base.training)


# ── intake ───────────────────────────────────────────────────────────────────


def _intake(tmp_path, *, until=0, head=None, lb=None, lc=None):
    from cascade.funding.intake import FundingIntake
    from cascade.funding.queue import (
        FundedQueue,
        LaneBQueue,
        LaneCQueue,
        lane_b_queue_path,
        lane_c_queue_path,
    )
    from cascade.funding.store import SubmissionStore
    from cascade.funding.vault import PayerKeyVault
    from tests.accept.wb13.test_wb13_lane_ended import SLOTS, SU, Clock

    clock = Clock()
    qpath = tmp_path / "queue.json"
    kw = {}
    if lb is not None:
        kw.update(lane_b_queue=LaneBQueue(lane_b_queue_path(qpath), clock=clock),
                  resolve_lane_b_reveal=lambda h, t, r: 700, lane_b_targets=SLOTS,
                  lane_b_max_targets=2, lane_b_cfg=lb)
    if lc is not None:
        kw.update(lane_c_queue=LaneCQueue(lane_c_queue_path(qpath), clock=clock),
                  resolve_lane_c_reveal=lambda h, t, r: 700, lane_c_targets=tuple(lc.targets),
                  mkt_suites=SU, lane_c_cfg=lc)
    if until:
        kw["lane_a_until_block"] = until
    return FundingIntake(FundedQueue(qpath, clock=clock), PayerKeyVault(dir=None, clock=clock),
                         resolve_reveal=lambda h, ref: 650, require_signature=False,
                         clock=clock, store=SubmissionStore(tmp_path / "store"),
                         head_block=head, **kw)


def _ha(who: str) -> dict:
    from tests.accept.wb13.test_wb13_lane_ended import hk

    return {"X-Miner-Hotkey": hk(who), "X-Commit-Ref": REF_A, "X-Lium-Api-Key": f"sk-{who}"}


def _hs(who: str, *, key: bool) -> dict:
    from tests.accept.wb13.test_wb13_lane_ended import hk

    h = {"X-Miner-Hotkey": hk(who), "X-Content-Digest": "sha256:" + "0" * 64}
    if key:
        h["X-Lium-Api-Key"] = f"sk-{who}"
    return h


def test_w_launch_lane_a_fund_after_cutoff_is_409_and_keeps_no_key(tmp_path):
    _require()
    from tests.accept.wb13.test_wb13_lane_ended import Head, hk

    head = Head(MAINNET_CUTOFF - 1)
    intake = _intake(tmp_path, until=MAINNET_CUTOFF, head=head)
    assert intake.fund(_ha("A"))[0] == 202                     # before: admitted
    head.block = MAINNET_CUTOFF
    status, body = intake.fund(_ha("B"))
    assert (status, body["code"], body["lane"], body["until_block"], body["block"]) == (
        409, "lane_ended", "a", MAINNET_CUTOFF, MAINNET_CUTOFF)
    assert "lane_a_until_block" in body["message"]
    assert intake.queue.get(hk("B")) is None and intake.vault.get(hk("B")) is None


def test_w_launch_lane_a_funded_submit_after_cutoff_is_409(tmp_path):
    _require()
    from tests.accept.wb13.test_wb13_lane_ended import Head, hk

    head = Head(MAINNET_CUTOFF - 1)
    intake = _intake(tmp_path, until=MAINNET_CUTOFF, head=head)
    assert intake.submit_gate(_hs("A", key=True))[0] is None   # before: passes the gate
    head.block = MAINNET_CUTOFF + 5
    status, body = intake.submit_gate(_hs("B", key=True))
    assert (status, body["code"], body["lane"], body["until_block"]) == (
        409, "lane_ended", "a", MAINNET_CUTOFF)
    status, body = intake.submit(_hs("C", key=True), b"zip")
    assert (status, body["code"], body["lane"]) == (409, "lane_ended", "a")
    assert intake.queue.get(hk("C")) is None and intake.vault.get(hk("C")) is None
    # a plain store (no payer key) stays open: lane B legs may name a vault ref
    assert intake.submit_gate(_hs("D", key=False))[0] is None


def test_w_launch_no_cutoff_or_no_head_is_the_old_reply(tmp_path):
    _require()
    from tests.accept.wb13.test_wb13_lane_ended import Head

    for i, (until, head) in enumerate(((0, Head(MAINNET_CUTOFF + 100)),
                                       (MAINNET_CUTOFF, Head(MAINNET_CUTOFF - 1)),
                                       (MAINNET_CUTOFF, None))):
        plain = _intake(tmp_path / f"p{i}", head=head)
        wired = _intake(tmp_path / str(i), until=until, head=head)
        assert wired.fund(_ha("A")) == plain.fund(_ha("A")) == (202, {"status": "queued"})
        assert wired.submit_gate(_hs("A", key=True))[0] is None
        assert wired.queue_view()[1]["entries"] == plain.queue_view()[1]["entries"]
    assert _intake(tmp_path / "x", head=Head(10**9)).lane_a_until_block == 0


def test_w_launch_lane_b_and_c_funds_unaffected_by_the_lane_a_cutoff(tmp_path):
    _require()
    from tests.accept.wb13.test_wb13_lane_ended import Head, _hb, _hc, _lb, _lc

    intake = _intake(tmp_path, until=MAINNET_CUTOFF, head=Head(MAINNET_CUTOFF + 10),
                     lb=_lb(), lc=_lc())
    assert intake.fund(_ha("A"))[0] == 409
    assert intake.lane_b_fund(_hb("B"))[0] == 202
    assert intake.lane_c_fund(_hc("C"))[0] == 202


def test_w_launch_queued_lane_a_entries_released_in_flight_untouched(tmp_path):
    _require()
    from tests.accept.wb13.test_wb13_lane_ended import Head, hk

    head = Head(MAINNET_CUTOFF - 10)
    intake = _intake(tmp_path, until=MAINNET_CUTOFF, head=head)
    for who in ("A", "F"):
        assert intake.fund(_ha(who))[0] == 202
    q = intake.queue
    assert q.mark_in_flight(hk("F"), REF_A, target_boundary=MAINNET_CUTOFF + 900,
                            era_index=5, started_block=MAINNET_CUTOFF - 5)
    head.block = MAINNET_CUTOFF
    assert intake.queue_view()[0] == 200                       # any request sweeps
    a, f = q.get(hk("A")), q.get(hk("F"))
    assert (a.status, a.last_error_class, a.attempts) == ("failed", "lane_ended", 0)
    assert "nothing charged" in a.last_error
    assert f.status == "in_flight"                             # a running leg is never touched
    assert intake.vault.get(hk("F")) == "sk-F"


# ── trainer ──────────────────────────────────────────────────────────────────


def _trainer(tmp_path, *, cutoff_offset: int | None):
    from tests.accept.w_cmp2.helpers import fixture_cfg
    from tests.unit.test_rolling import Clock, FakeClient, _armed, _sched

    armed = _armed(fixture_cfg(tmp_path / "cfg"))
    era_start = armed.round.rolling_from_block
    if cutoff_offset is not None:
        armed = replace(armed, round=replace(armed.round,
                                             lane_a_until_block=era_start + cutoff_offset))
    clock = Clock()
    sched, ops = _sched(armed, tmp_path / "work", clock)
    return armed, clock, sched, ops, FakeClient(), era_start


def test_w_launch_trainer_seats_no_lane_a_challenger_after_cutoff(tmp_path):
    _require()
    from tests.unit.test_rolling import REF, _commit, _join

    for i, (offset, seated) in enumerate(((5, False), (6, True), (None, True))):
        armed, clock, sched, ops, client, era_start = _trainer(tmp_path / str(i),
                                                               cutoff_offset=offset)
        q = ops.queue()
        ops.commits.append(_commit("ALFA", REF["ALFA"], era_start - 100))
        q.add("ALFA", REF["ALFA"], reveal_block=era_start - 100)
        sched.tick(client, era_start + 5)
        _join(sched)
        assert [c[0] for c in ops.king_calls] == ["KING"]      # the king leg always runs
        if seated:
            assert [c[0] for c in ops.leg_calls] == ["ALFA"], (offset, ops.leg_calls)
        else:
            assert ops.leg_calls == [], ops.leg_calls
            assert q.get("ALFA").status == "queued"            # never failed, never burned
            assert "ALFA" not in ops.burnt


def test_w_launch_in_flight_leg_settles_and_king_only_progress_continues(tmp_path):
    """A leg seated before the cutoff runs to its verdict and settles with the
    king; after the cutoff a new entry is never seated, and a boundary with
    nothing finished still publishes the king-only settlement whenever a lane
    B / C needs progress."""
    _require()
    from tests.unit.test_rolling import EB, REF, _advance, _commit, _join

    armed, clock, sched, ops, client, era_start = _trainer(tmp_path, cutoff_offset=10)
    q = ops.queue()
    ops.commits.append(_commit("ALFA", REF["ALFA"], era_start - 100))
    q.add("ALFA", REF["ALFA"], reveal_block=era_start - 100)
    sched.tick(client, era_start + 5)                          # before the cutoff: seated
    _join(sched)
    assert [c[0] for c in ops.leg_calls] == ["ALFA"]
    # after the cutoff a new lane A entry arrives
    ops.commits.append(_commit("BRAV", REF["BRAV"], era_start + 20))
    q.add("BRAV", REF["BRAV"], reveal_block=era_start + 20)
    sched._needs_settlement_progress = lambda boundary: True   # lane B / C active
    b1 = era_start + EB
    _advance(clock, ops, sched, client, from_block=era_start + 5, to_block=b1 + 1)
    assert [c[0] for c in ops.leg_calls] == ["ALFA"]           # BRAV never seated
    assert len(ops.manifests) == 1
    assert [e.miner_hotkey for e in ops.manifests[0].entries] == ["KING", "ALFA"]
    assert q.get("ALFA").status == "done"
    # the next boundary: nothing finished, lane B / C still need progress ->
    # a king-only settlement, chained to the first
    _advance(clock, ops, sched, client, from_block=b1 + 1, to_block=b1 + EB + 1)
    assert len(ops.manifests) == 2
    m2 = ops.manifests[1]
    assert [e.miner_hotkey for e in m2.entries] == ["KING"]
    assert m2.prev_round_id == ops.manifests[0].round_id
    assert [c[0] for c in ops.leg_calls] == ["ALFA"]
    assert q.get("BRAV").status == "queued"
