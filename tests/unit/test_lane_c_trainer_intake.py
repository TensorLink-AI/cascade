"""DEC-CA-0056 lane C — intake + queue: ``POST /v1/lane-c/fund`` queues into
the LANE-C queue (never the challenger queue) behind the /v1/fund identity
gate, with the signature binding the key digest, the target and the ref.
MarketCast-only lane C (owner 2026-10-20): targets are ``mkt:<suite>``."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

from cascade.funding.intake import (
    LANE_C_TARGET_HEADER,
    FundingIntake,
    canonical_fund_message,
    lane_c_signed_ref,
)
from cascade.funding.main import ChainRevealResolver
from cascade.funding.queue import (
    FundedQueue,
    LaneCQueue,
    lane_c_entry_ref,
    lane_c_queue_path,
    split_lane_c_ref,
)
from cascade.funding.vault import PayerKeyVault
from cascade.shared.lane_c_manifest import format_finetune_commit

HK = "5FakeHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
REF = "org/recipe@sha256:" + "a" * 64
TGT = "mkt:returns"
SU = ("returns", "vol", "range")      # admitted suites; "range" is no [lane_c] target


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _intake(tmp_path, *, resolve=lambda hk, t, r: 700, verify=None, sig=False,
            targets=(TGT, "mkt:vol")):
    clock = Clock()
    lane_a = FundedQueue(tmp_path / "queue.json", clock=clock)
    lane_c = LaneCQueue(lane_c_queue_path(tmp_path / "queue.json"), clock=clock)
    intake = FundingIntake(
        lane_a, PayerKeyVault(dir=None, clock=clock),
        resolve_reveal=lambda hk, ref: None, require_signature=sig,
        verify=verify or (lambda hk, msg, s: False), clock=clock,
        lane_c_queue=lane_c, resolve_lane_c_reveal=resolve, lane_c_targets=targets,
        mkt_suites=SU)
    return intake, lane_a, lane_c, clock


def _h(**kw):
    h = {"X-Miner-Hotkey": HK, "X-Commit-Ref": REF, LANE_C_TARGET_HEADER: TGT,
         "X-Lium-Api-Key": "sk-live"}
    h.update(kw)
    return {k: v for k, v in h.items() if v is not None}


def test_lane_c_queue_path_is_derived_beside_the_challenger_queue(tmp_path):
    assert lane_c_queue_path(tmp_path / "q" / "funded.json") == tmp_path / "q" / "lane_c_queue.json"
    assert split_lane_c_ref(lane_c_entry_ref(TGT, REF)) == (TGT, REF)


def test_fund_queues_into_lane_c_only_and_vaults_the_key(tmp_path):
    intake, lane_a, lane_c, _ = _intake(tmp_path)
    status, body = intake.lane_c_fund(_h())
    assert (status, body["status"], body["lane"], body["target"]) == (202, "queued", "c", TGT)
    e = lane_c.get(HK)
    assert e.reveal_block == 700 and split_lane_c_ref(e.ref) == (TGT, REF)
    assert lane_a.entries() == []                # the challenger queue is untouched
    assert intake.vault.get(HK) == "sk-live"
    # idempotent
    assert intake.lane_c_fund(_h())[0] == 200
    # a re-fund for another target replaces the entry (one live entry per hotkey)
    intake.resolve_lane_c_reveal = lambda hk, t, r: 800
    status, body = intake.lane_c_fund(_h(**{LANE_C_TARGET_HEADER: "mkt:vol"}))
    assert (status, body["status"]) == (202, "replaced")
    assert split_lane_c_ref(lane_c.get(HK).ref)[0] == "mkt:vol"


def test_fund_gate_rejections(tmp_path):
    intake, _, lane_c, _ = _intake(tmp_path, resolve=lambda hk, t, r: None)
    assert intake.lane_c_fund(_h())[1]["code"] == "not_revealed"
    assert intake.lane_c_fund(_h(**{LANE_C_TARGET_HEADER: None}))[1]["code"] == "missing_target"
    assert intake.lane_c_fund(_h(**{LANE_C_TARGET_HEADER: "mkt:Bad!"}))[1]["code"] == "bad_target"
    assert intake.lane_c_fund(_h(**{LANE_C_TARGET_HEADER: "mkt:range"}))[1]["code"] \
        == "unknown_target"
    # a generic DEC-CA-0056 target has no forward-only eval: refused
    status, body = intake.lane_c_fund(_h(**{LANE_C_TARGET_HEADER: "domain:energy"}))
    assert (status, body["code"]) == (400, "bad_target") and "forward-only" in body["message"]
    intake.mkt_suites = ()                        # [marketcast] off
    assert intake.lane_c_fund(_h())[1]["code"] == "marketcast_inactive"
    intake, _, lane_c, _ = _intake(tmp_path)
    assert intake.lane_c_fund(_h(**{"X-Lium-Api-Key": None}))[1]["code"] == "missing_lium_api_key"
    assert lane_c.entries() == []


def test_reveal_oracle_gets_target_and_ref(tmp_path):
    seen = []
    intake, *_ = _intake(tmp_path, resolve=lambda hk, t, r: (seen.append((hk, t, r)), 5)[1])
    assert intake.lane_c_fund(_h())[0] == 202
    assert seen == [(HK, TGT, REF)]


def test_disabled_without_a_lane_c_queue(tmp_path):
    clock = Clock()
    intake = FundingIntake(FundedQueue(tmp_path / "q.json", clock=clock),
                           PayerKeyVault(dir=None, clock=clock),
                           resolve_reveal=lambda hk, ref: 1, require_signature=False, clock=clock)
    status, body = intake.lane_c_fund(_h())
    assert (status, body["code"]) == (503, "lane_c_disabled")
    assert intake.lane_c_queue_view()[0] == 503


def test_signature_binds_target_ref_and_key_digest(tmp_path):
    seen = {}

    def verify(hk, msg, sig):
        seen["msg"] = msg
        return sig == "ok"

    intake, _, lane_c, clock = _intake(tmp_path, verify=verify, sig=True)
    ts = str(int(clock()))
    status, _ = intake.lane_c_fund(_h(**{"X-Timestamp": ts, "X-Signature": "ok"}))
    assert status == 202
    expect = canonical_fund_message("lane-c-fund", HK, lane_c_signed_ref(TGT, REF), ts,
                                    hashlib.sha256(b"sk-live").hexdigest())
    assert seen["msg"] == expect
    assert b"mkt:returns|org/recipe" in expect
    # a lane-A fund signature can never be replayed as a lane-C fund
    assert canonical_fund_message("fund", HK, REF, ts, "-") != expect


def test_withdraw_and_public_view(tmp_path):
    intake, _, lane_c, _ = _intake(tmp_path)
    intake.lane_c_fund(_h())
    status, view = intake.lane_c_queue_view()
    assert status == 200 and view["lane"] == "c"
    (row,) = view["entries"]
    assert (row["target"], row["ref"], row["status"]) == (TGT, REF, "queued")
    assert "sk-live" not in str(view)
    assert intake.lane_c_withdraw(_h(**{"X-Lium-Api-Key": None}))[1]["status"] == "withdrawn"
    assert lane_c.get(HK).status == "withdrawn"
    assert intake.lane_c_withdraw(_h(**{"X-Lium-Api-Key": None}))[0] == 409
    # the shared payer key is kept (other legs of the hotkey may rent on it)
    assert intake.vault.get(HK) == "sk-live"


def test_queue_semantics_mirror_the_challenger_queue(tmp_path):
    clock = Clock()
    q = LaneCQueue(tmp_path / "lane_c_queue.json", clock=clock)
    q.add_lane_c(HK, TGT, REF, 10)
    er = lane_c_entry_ref(TGT, REF)
    assert [e.hotkey for e in q.queued()] == [HK]
    assert q.mark_in_flight(HK, er, target_boundary=0, era_index=3, started_block=50)
    # operator fault: no burn, backoff — not startable until not_before
    assert q.requeue(HK, error="cpu tier down", error_class="operator", burn_attempt=False)
    assert q.get(HK).attempts == 0 and q.queued() == []
    clock.t += 10_000
    assert [e.hotkey for e in q.queued()] == [HK]
    # infra faults burn one attempt each, then the entry is terminal
    for _ in range(3):
        assert q.mark_in_flight(HK, er, target_boundary=0, era_index=3, started_block=50)
        assert q.requeue(HK, error="pod died", error_class="infra", burn_attempt=True)
    assert q.mark_in_flight(HK, er, target_boundary=0, era_index=3, started_block=50)
    assert not q.requeue(HK, error="pod died", error_class="infra", burn_attempt=True)
    assert q.get(HK).status == "failed"


def test_chain_resolver_matches_finetune_commitments_only():
    rows = [SimpleNamespace(hotkey=HK, payload=format_finetune_commit(TGT, REF, mkt_suites=SU),
                            commit_block=40),
            SimpleNamespace(hotkey=HK, payload=f"metro-v1:gen:hippius:{REF}", commit_block=50)]
    client = SimpleNamespace(poll_commitments=lambda include_history=True: rows)
    r = ChainRevealResolver(client)
    assert r.lane_c(HK, TGT, REF) is None        # MarketCast off: no mkt: commit parses
    r.mkt_suites = SU
    assert r.lane_c(HK, TGT, REF) == 40
    assert r.lane_c(HK, "mkt:vol", REF) is None
    assert r(HK, REF) == 50                      # the lane-A oracle is unchanged
