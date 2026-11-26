"""#931: the funding intake honours the lane kill switches (``until_block``).

Containment drill, testnet 2026-11-25: overlay seq 9 set ``[lane_c]
until_block`` and ``[lane_b] until_block`` (the #889 kill switches). The
validator and trainer stopped, but the funding intake still ADMITTED lane C
funds (it checked only the competition's own end) and kept the payer's key
in the vault for its 36 h TTL. Contract (``cascade.funding.lane_ends``):

* lane B / lane C: at or after the lane-wide ``until_block`` in force at the
  head, a fund answers ``409 lane_ended`` with the end block, BEFORE the key
  is vaulted (a new payer's key is never kept);
* lane C competitions (a ``[lane_c.competitions.<kind>]`` table or a
  DEC-CA-0075 spec instance): their own end answers ``409
  competition_ended`` with the end block;
* already-queued, not yet started entries of an ended lane / competition are
  released: ``failed`` / ``lane_ended`` (or ``competition_ended``), no
  attempt consumed, and the key forgotten unless something else still needs
  it; an in-flight leg is never touched;
* share 0 (or competition weight 0): accepted, with a ``pay_warning``;
* nothing ended (or no head wired): byte-for-byte the old replies.

No network, no torch: in-memory queues and vault, a fake head.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from cascade.funding.intake import LANE_B_TARGETS_HEADER, LANE_C_TARGET_HEADER, FundingIntake
from cascade.funding.queue import (
    FundedQueue,
    LaneBQueue,
    LaneCQueue,
    lane_b_queue_path,
    lane_c_entry_ref,
    lane_c_queue_path,
)
from cascade.funding.vault import PayerKeyVault
from cascade.shared.config import LaneBConfig
from cascade.shared.config.lane_c import LaneCCompetition, parse_lane_c

ERA = 7200
FROM = 2 * ERA
UNTIL = 6 * ERA
SU = ("returns", "vol")
MKT = "mkt:returns"
SLOTS = ("energy", "web")
REF_B = "org/gen@sha256:" + "b" * 64
REF_C = "org/recipe@sha256:" + "c" * 64


def hk(name: str) -> str:
    return ("5" + name * 47)[:48]


def _require():
    """Fail on main at the missing module (its PATHS entry)."""
    from cascade.funding.lane_ends import LANE_ENDED

    assert LANE_ENDED == "lane_ended"


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class Head:
    def __init__(self, block: int) -> None:
        self.block = block

    def __call__(self) -> int:
        return self.block


def _lc(*, until: int = 0, share: float = 0.25, comps=None, targets=(MKT,)):
    lc = replace(parse_lane_c({}), from_block=FROM, until_block=until, share=share,
                 targets=tuple(targets))
    return replace(lc, competitions=tuple(comps)) if comps is not None else lc


def _lb(*, until: int = 0, share: float = 0.5):
    return replace(LaneBConfig(), from_block=FROM, until_block=until, share=share,
                   domains=SLOTS)


def _intake(tmp_path, *, lc=None, lb=None, head=None, wired=True):
    clock = Clock()
    qpath = tmp_path / "queue.json"
    lane_a = FundedQueue(qpath, clock=clock)
    kw = {}
    if lc is not None:
        kw.update(lane_c_queue=LaneCQueue(lane_c_queue_path(qpath), clock=clock),
                  resolve_lane_c_reveal=lambda h, t, r: 700, lane_c_targets=tuple(lc.targets),
                  mkt_suites=SU)
    if lb is not None:
        kw.update(lane_b_queue=LaneBQueue(lane_b_queue_path(qpath), clock=clock),
                  resolve_lane_b_reveal=lambda h, t, r: 700, lane_b_targets=SLOTS,
                  lane_b_max_targets=2, lane_b_cfg=lb)
    if wired:
        kw.update(lane_c_cfg=lc, head_block=head)
    return FundingIntake(lane_a, PayerKeyVault(dir=None, clock=clock),
                         resolve_reveal=lambda h, ref: None, require_signature=False,
                         clock=clock, **kw)


def _hc(who: str, target: str = MKT) -> dict:
    return {"X-Miner-Hotkey": hk(who), "X-Commit-Ref": REF_C, LANE_C_TARGET_HEADER: target,
            "X-Lium-Api-Key": f"sk-{who}"}


def _hb(who: str) -> dict:
    return {"X-Miner-Hotkey": hk(who), "X-Commit-Ref": REF_B, LANE_B_TARGETS_HEADER: "energy",
            "X-Lium-Api-Key": f"sk-{who}"}


# ── lane C: the lane-wide kill switch ────────────────────────────────────────


def test_lane_c_fund_after_lane_until_is_refused_and_no_key_kept(tmp_path):
    _require()
    head = Head(UNTIL - 1)
    intake = _intake(tmp_path, lc=_lc(until=UNTIL), head=head)
    assert intake.lane_c_fund(_hc("A"))[0] == 202
    head.block = UNTIL
    status, body = intake.lane_c_fund(_hc("B"))
    assert (status, body["code"], body["until_block"], body["lane"]) == (
        409, "lane_ended", UNTIL, "c")
    assert intake.lane_c_queue.get(hk("B")) is None
    assert intake.vault.get(hk("B")) is None             # a refused fund keeps no key
    # the competition's own end (none here) is not what refused it
    assert "competition" not in body


def test_lane_c_queued_entries_are_released_at_the_end(tmp_path):
    _require()
    head = Head(UNTIL - 10)
    intake = _intake(tmp_path, lc=_lc(until=UNTIL), head=head)
    q = intake.lane_c_queue
    for who in ("A", "F"):
        assert intake.lane_c_fund(_hc(who))[0] == 202
    assert q.mark_in_flight(hk("F"), lane_c_entry_ref(MKT, REF_C), target_boundary=UNTIL,
                            era_index=5, started_block=UNTIL - 5)
    head.block = UNTIL + 3
    assert intake.lane_c_queue_view()[0] == 200          # any lane request sweeps
    a, f = q.get(hk("A")), q.get(hk("F"))
    assert (a.status, a.last_error_class, a.attempts) == ("failed", "lane_ended", 0)
    assert "nothing charged" in a.last_error
    assert intake.vault.get(hk("A")) is None             # released: the key is forgotten
    assert f.status == "in_flight"                       # a running leg is never touched
    assert intake.vault.get(hk("F")) == "sk-F"
    rows = {r["hotkey"]: r for r in intake.lane_c_queue_view()[1]["entries"]}
    assert rows[hk("A")]["last_error_class"] == "lane_ended"


def test_lane_c_competition_end_is_typed_with_its_block(tmp_path):
    _require()
    lc = _lc(comps=(LaneCCompetition("mkt", until_block=UNTIL),))
    intake = _intake(tmp_path, lc=lc, head=Head(UNTIL))
    status, body = intake.lane_c_fund(_hc("A"))
    assert (status, body["code"], body["until_block"], body["competition"]) == (
        409, "competition_ended", UNTIL, "mkt")
    assert intake.vault.get(hk("A")) is None


@pytest.fixture
def spec_kind():
    from tests.accept.w_cmp2 import helpers as H

    H.register_kind()
    yield H
    H.unregister_kind()


def test_spec_competition_end_refuses_and_releases_only_its_entries(tmp_path, spec_kind):
    _require()
    H = spec_kind
    spec_target = f"{H.KIND}:returns"
    base = H.lane_c(with_instance=True)
    lc = replace(base, until_block=0, competitions=(
        LaneCCompetition(H.KIND, from_block=H.EB, until_block=UNTIL), LaneCCompetition("mkt")))
    head = Head(UNTIL - 1)
    intake = _intake(tmp_path, lc=lc, head=head)
    assert intake.lane_c_fund(_hc("A", spec_target))[0] == 202
    assert intake.lane_c_fund(_hc("M", MKT))[0] == 202
    head.block = UNTIL
    status, body = intake.lane_c_fund(_hc("B", spec_target))
    assert (status, body["code"], body["until_block"], body["competition"]) == (
        409, "competition_ended", UNTIL, H.KIND)
    assert intake.lane_c_fund(_hc("C", MKT))[0] == 202  # mkt runs on
    q = intake.lane_c_queue
    a = q.get(hk("A"))
    assert (a.status, a.last_error_class) == ("failed", "competition_ended")
    assert q.get(hk("M")).status == "queued"
    assert intake.vault.get(hk("A")) is None and intake.vault.get(hk("M")) == "sk-M"


def test_lane_c_share_or_weight_zero_is_accepted_with_a_warning(tmp_path):
    _require()
    intake = _intake(tmp_path, lc=_lc(share=0.0), head=Head(FROM + 10))
    status, body = intake.lane_c_fund(_hc("A"))
    assert status == 202 and "share is 0" in body["pay_warning"]
    lc = _lc(comps=(LaneCCompetition("mkt", weight=0.0),))
    intake = _intake(tmp_path / "w", lc=lc, head=Head(FROM + 10))
    status, body = intake.lane_c_fund(_hc("A"))
    assert status == 202 and "weight 0" in body["pay_warning"]


def test_lane_c_nothing_ended_is_the_old_reply(tmp_path):
    _require()
    for i, (lc, head) in enumerate(((_lc(), Head(UNTIL)), (_lc(until=UNTIL), Head(UNTIL - 1)),
                                    (_lc(until=UNTIL), None))):
        plain = _intake(tmp_path / f"p{i}", lc=_lc(), wired=False)
        wired = _intake(tmp_path / str(i), lc=lc, head=head)
        assert wired.lane_c_fund(_hc("A")) == plain.lane_c_fund(_hc("A"))
        assert wired.lane_c_queue.get(hk("A")).status == "queued"
        assert wired.lane_c_queue_view()[1]["entries"] == plain.lane_c_queue_view()[1]["entries"]


def test_head_read_failure_admits(tmp_path):
    _require()

    def down():
        raise RuntimeError("substrate down")

    intake = _intake(tmp_path, lc=_lc(until=UNTIL), head=down)
    assert intake.lane_c_fund(_hc("A"))[0] == 202


# ── lane B ───────────────────────────────────────────────────────────────────


def test_lane_b_fund_after_until_is_refused_and_no_key_kept(tmp_path):
    _require()
    head = Head(UNTIL - 1)
    intake = _intake(tmp_path, lb=_lb(until=UNTIL), head=head)
    assert intake.lane_b_fund(_hb("A"))[0] == 202
    head.block = UNTIL
    status, body = intake.lane_b_fund(_hb("B"))
    assert (status, body["code"], body["until_block"], body["lane"]) == (
        409, "lane_ended", UNTIL, "b")
    assert intake.lane_b_queue.get(hk("B")) is None and intake.vault.get(hk("B")) is None


def test_lane_b_release_keeps_a_key_another_lane_still_needs(tmp_path):
    _require()
    head = Head(UNTIL - 1)
    intake = _intake(tmp_path, lb=_lb(until=UNTIL), head=head)
    assert intake.lane_b_fund(_hb("A"))[0] == 202
    assert intake.lane_b_fund(_hb("K"))[0] == 202
    intake.queue.add(hk("K"), "org/lane-a@sha256:" + "d" * 64, 650)   # a live lane A entry
    head.block = UNTIL
    assert intake.lane_b_queue_view()[0] == 200
    qb = intake.lane_b_queue
    assert (qb.get(hk("A")).status, qb.get(hk("A")).last_error_class) == ("failed",
                                                                          "lane_ended")
    assert qb.get(hk("K")).status == "failed"
    assert intake.vault.get(hk("A")) is None
    assert intake.vault.get(hk("K")) == "sk-K"           # lane A still needs it


def test_lane_b_share_zero_warns_and_nothing_ended_is_unchanged(tmp_path):
    _require()
    intake = _intake(tmp_path, lb=_lb(share=0.0), head=Head(UNTIL))
    status, body = intake.lane_b_fund(_hb("A"))
    assert status == 202 and "share is 0" in body["pay_warning"]
    plain = _intake(tmp_path / "p", lb=_lb(), wired=False)
    wired = _intake(tmp_path / "w", lb=_lb(until=UNTIL), head=Head(UNTIL - 1))
    assert wired.lane_b_fund(_hb("A")) == plain.lane_b_fund(_hb("A"))


# ── production wiring ────────────────────────────────────────────────────────


def test_effective_lanes_points_the_kill_switches_at_the_view(tmp_path):
    _require()
    from cascade.funding.main import CachedHead, EffectiveLanes

    cfg = replace(parse_lane_c({}), from_block=FROM, until_block=UNTIL)
    view = type("View", (), {"lane_c": cfg, "lane_b": _lb(until=UNTIL), "netconfig": None})()
    qpath = tmp_path / "queue.json"
    lanes = EffectiveLanes(
        None, lane_b_mode="auto", lane_c_mode="auto",
        open_lane_b_queue=lambda: LaneBQueue(lane_b_queue_path(qpath)),
        open_lane_c_queue=lambda: LaneCQueue(lane_c_queue_path(qpath)))
    intake = FundingIntake(FundedQueue(qpath), PayerKeyVault(dir=None),
                           resolve_reveal=lambda h, ref: None, require_signature=False,
                           head_block=Head(UNTIL), lane_refresh=lanes)
    lanes.apply(intake, view)
    assert intake.lane_c_cfg is cfg and intake.lane_b_cfg is view.lane_b
    assert intake.lane_c_fund({})[1]["code"] == "lane_ended"
    assert intake.lane_b_fund({})[1]["code"] == "lane_ended"

    reads, t = [], {"now": 0.0}

    def read():
        reads.append(1)
        return 100 + len(reads)

    head = CachedHead(read, ttl_seconds=6.0, clock=lambda: t["now"])
    assert (head(), head()) == (101, 101)
    t["now"] = 6.0
    assert head() == 102 and len(reads) == 2
