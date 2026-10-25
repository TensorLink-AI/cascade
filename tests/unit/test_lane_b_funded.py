"""DEC-CA-0059 item 8 — lane B is miner-funded: ``POST /v1/lane-b/fund``
queues into the lane-B FUNDING queue behind the /v1/fund identity gate (the
signature binding key digest, targets and ref; DEC-CA-0060 GPU headers);
the trainer dispatches a leg only for a funded, latest commitment of a free
hotkey, on the miner's key (``_run_funded_leg``, role ``expert``), with no
operator fallback; the miner CLI builds the signed request."""

from __future__ import annotations

import dataclasses
import hashlib
import threading
from types import SimpleNamespace

import pytest

from cascade.funding.intake import (
    LANE_B_TARGETS_HEADER,
    FundingIntake,
    canonical_fund_message,
    lane_b_signed_ref,
)
from cascade.funding.main import ChainRevealResolver
from cascade.funding.queue import (
    FundedQueue,
    LaneBQueue,
    lane_b_entry_ref,
    lane_b_queue_path,
    split_lane_b_ref,
)
from cascade.funding.vault import PayerKeyVault
from cascade.shared.expert_manifest import format_expert_commit, format_expert_targets

HK = "5FakeHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
REF = "org/gen@sha256:" + "a" * 64
SLOTS = ("energy", "web", "horizon:64")


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _intake(tmp_path, *, resolve=lambda hk, t, r: 700, verify=None, sig=False, catalog=None):
    clock = Clock()
    lane_a = FundedQueue(tmp_path / "queue.json", clock=clock)
    lane_b = LaneBQueue(lane_b_queue_path(tmp_path / "queue.json"), clock=clock)
    intake = FundingIntake(
        lane_a, PayerKeyVault(dir=None, clock=clock),
        resolve_reveal=lambda hk, ref: None, require_signature=sig,
        verify=verify or (lambda hk, msg, s: False), clock=clock, gpu_catalog=catalog,
        lane_b_queue=lane_b, resolve_lane_b_reveal=resolve, lane_b_targets=SLOTS,
        lane_b_max_targets=2)
    return intake, lane_a, lane_b, clock


def _h(**kw):
    h = {"X-Miner-Hotkey": HK, "X-Commit-Ref": REF, LANE_B_TARGETS_HEADER: "energy",
         "X-Lium-Api-Key": "sk-live"}
    h.update(kw)
    return {k: v for k, v in h.items() if v is not None}


# ── queue ────────────────────────────────────────────────────────────────────


def test_entry_ref_round_trip_and_derived_path(tmp_path):
    assert lane_b_queue_path(tmp_path / "q" / "f.json") == tmp_path / "q" / "lane_b_queue.json"
    r = lane_b_entry_ref(("energy", "horizon:64"), REF)
    assert r == f"energy,horizon:64|{REF}"
    assert split_lane_b_ref(r) == (("energy", "horizon:64"), REF)
    for bad in ((), ("energy", "energy"), ("a|b",), ("a,b",)):
        with pytest.raises(ValueError):
            lane_b_entry_ref(bad, REF)
    with pytest.raises(ValueError):
        split_lane_b_ref("no-separator")


def test_public_view_splits_targets(tmp_path):
    q = LaneBQueue(tmp_path / "lane_b_queue.json", clock=Clock())
    assert q.add_lane_b(HK, ("energy", "web"), REF, 5) == "queued"
    (row,) = q.public_view()["entries"]
    assert row["targets"] == ["energy", "web"] and row["ref"] == REF
    assert q.public_view()["lane"] == "b"


# ── intake ───────────────────────────────────────────────────────────────────


def test_fund_queues_into_lane_b_only_and_vaults_the_key(tmp_path):
    intake, lane_a, lane_b, _ = _intake(tmp_path)
    status, body = intake.lane_b_fund(_h())
    assert (status, body["status"], body["lane"], body["targets"]) == (
        202, "queued", "b", ["energy"])
    e = lane_b.get(HK)
    assert e.reveal_block == 700 and split_lane_b_ref(e.ref) == (("energy",), REF)
    assert lane_a.entries() == [] and intake.vault.get(HK) == "sk-live"
    assert intake.lane_b_fund(_h())[0] == 200                       # idempotent
    # stored in the slot order, whatever order was sent
    status, body = intake.lane_b_fund(_h(**{LANE_B_TARGETS_HEADER: "horizon:64, energy"}))
    assert (status, body["status"], body["targets"]) == (202, "replaced",
                                                         ["energy", "horizon:64"])
    assert intake.lane_b_queue_view()[1]["entries"][0]["targets"] == ["energy", "horizon:64"]


@pytest.mark.parametrize("value,code", [
    (None, "missing_targets"), ("energy,energy", "bad_targets"),
    ("solar", "unknown_target"), ("energy,web,horizon:64", "too_many_targets")])
def test_fund_refuses_bad_target_lists(tmp_path, value, code):
    intake, _, lane_b, _ = _intake(tmp_path)
    status, body = intake.lane_b_fund(_h(**{LANE_B_TARGETS_HEADER: value}))
    assert status == 400 and body["code"] == code and lane_b.entries() == []


def test_fund_needs_a_matching_reveal_and_lane_b_on(tmp_path):
    intake, _, lane_b, _ = _intake(tmp_path, resolve=lambda hk, t, r: None)
    assert intake.lane_b_fund(_h())[1]["code"] == "not_revealed"
    seen = []
    intake.resolve_lane_b_reveal = lambda hk, t, r: seen.append((hk, t, r)) or 9
    intake.lane_b_fund(_h(**{LANE_B_TARGETS_HEADER: "web,energy"}))
    assert seen == [(HK, ("energy", "web"), REF)]
    intake.lane_b_queue = None
    assert intake.lane_b_fund(_h()) == (503, {"code": "lane_b_disabled", "message": intake.
                                              lane_b_fund(_h())[1]["message"]})
    assert intake.lane_b_queue_view()[0] == 503


def test_signature_binds_targets_ref_and_key(tmp_path):
    seen = []

    def verify(hk, msg, sig):
        seen.append(msg)
        return sig == "good"

    intake, _, lane_b, clock = _intake(tmp_path, verify=verify, sig=True)
    ts = str(int(clock()))
    digest = hashlib.sha256(b"sk-live").hexdigest()
    h = _h(**{LANE_B_TARGETS_HEADER: "web,energy", "X-Timestamp": ts, "X-Signature": "good"})
    assert intake.lane_b_fund(h)[0] == 202
    assert seen[-1] == canonical_fund_message(
        "lane-b-fund", HK, lane_b_signed_ref(("web", "energy"), REF), ts, digest)
    assert seen[-1].startswith(b"cascade-lane-b-fund:v2:")
    # withdraw signs its own action, no key
    clock.t += 1
    h = _h(**{"X-Timestamp": str(int(clock())), "X-Signature": "good",
              "X-Lium-Api-Key": None})
    status, body = intake.lane_b_withdraw(h)
    assert (status, body["status"]) == (200, "withdrawn")
    assert seen[-1].startswith(b"cascade-lane-b-withdraw:v2:") and seen[-1].endswith(b":-")
    assert intake.lane_b_withdraw(dict(h, **{"X-Timestamp": str(int(clock()) + 1)}))[0] == 409


def test_gpu_headers_select_the_v3_message_and_ride_the_entry(tmp_path):
    from cascade.funding.gpu_choice import GpuCatalog

    seen = []
    intake, _, lane_b, clock = _intake(
        tmp_path, sig=True, verify=lambda hk, m, s: seen.append(m) or True,
        catalog=GpuCatalog(skus=("L40S", "H100")))
    h = _h(**{"X-Timestamp": str(int(clock())), "X-Signature": "x", "X-Gpu-Type": "h100",
              "X-Gpu-Allow-Fallback": "1", "X-Gpu-Max-Price": "3.5"})
    status, body = intake.lane_b_fund(h)
    assert status == 202 and body["gpu"] == "H100" and body["allow_fallback"] is True
    assert b":v3:" in seen[-1] and seen[-1].endswith(b"gpu=h100;fallback=1;max_price=3.5")
    e = lane_b.get(HK)
    assert (e.gpu, e.gpu_allow_fallback, e.gpu_max_price_per_hour) == ("H100", True, 3.5)
    assert intake.lane_b_fund(_h(**{"X-Gpu-Type": "TPU"}))[1]["code"] == "unknown_gpu"


def test_reveal_resolver_matches_single_and_multi_target_commitments():
    rows = [SimpleNamespace(hotkey=HK, commit_block=10, payload=format_expert_commit("energy",
                                                                                    REF)),
            SimpleNamespace(hotkey=HK, commit_block=20, payload=format_expert_targets(
                ("energy", "horizon:64"), REF, SLOTS)),
            SimpleNamespace(hotkey=HK, commit_block=30, payload="metro-v1:gen:hippius:" + REF)]
    client = SimpleNamespace(poll_commitments=lambda include_history=True: rows)
    r = ChainRevealResolver(client, clock=lambda: 0.0)
    assert r.lane_b(HK, ("energy",), REF, slots=SLOTS) == 10
    assert r.lane_b(HK, ("horizon:64", "energy"), REF, slots=SLOTS) == 20
    assert r.lane_b(HK, ("web",), REF, slots=SLOTS) is None
    assert r.lane_b(HK, ("energy", "horizon:64"), REF, slots=SLOTS, max_targets=1) is None


def test_http_routes(tmp_path):
    import json
    import urllib.request

    intake, *_ = _intake(tmp_path)
    srv = intake.make_server("127.0.0.1", 0, per_ip_rpm=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        req = urllib.request.Request(f"{base}/v1/lane-b/fund", method="POST", headers=_h())
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 202
        with urllib.request.urlopen(f"{base}/v1/lane-b/queue", timeout=10) as resp:
            assert json.loads(resp.read())["entries"][0]["targets"] == ["energy"]
    finally:
        srv.shutdown()
        srv.server_close()


# ── miner CLI ────────────────────────────────────────────────────────────────


def test_cli_targets_and_signed_headers():
    from cascade.miner.lane_b import build_lane_b_headers, lane_b_fund_targets
    from cascade.shared.config import parse_lane_b

    cfg = SimpleNamespace(lane_b=parse_lane_b({"domains": ["energy", "web"],
                                                "horizons": [64]}))
    assert lane_b_fund_targets(cfg, "h64,energy") == (("energy", "horizon:64"), None)
    assert lane_b_fund_targets(cfg, "solar")[1]
    signed = []
    h = build_lane_b_headers("lane-b-fund", HK, ("energy", "horizon:64"), REF, "sk",
                             lambda m: signed.append(m) or b"\x01", now=lambda: 5.0,
                             gpu="L40S")
    assert h[LANE_B_TARGETS_HEADER] == "energy,horizon:64" and h["X-Lium-Api-Key"] == "sk"
    assert signed[0] == canonical_fund_message(
        "lane-b-fund", HK, lane_b_signed_ref(("energy", "horizon:64"), REF), "5",
        hashlib.sha256(b"sk").hexdigest(), "gpu=L40S;fallback=0;max_price=")
    w = build_lane_b_headers("lane-b-withdraw", HK, ("energy",), REF, "sk",
                             lambda m: b"\x01", now=lambda: 5.0, gpu="L40S")
    assert "X-Lium-Api-Key" not in w and "X-Gpu-Type" not in w


def test_cli_parser_has_lane_b():
    from cascade.miner.cli import build_parser

    a = build_parser().parse_args(["lane-b", "fund", "https://x", "--targets", "energy",
                                   "--ref", REF, "--wallet-name", "w", "--wallet-hotkey", "h",
                                   "--gpu", "L40S"])
    assert a.command == "lane-b fund" and a.gpu == "L40S"
    assert build_parser().parse_args(["lane-b", "status", "https://x"]).command == \
        "lane-b status"


# ── trainer driver (tiny real MoE checkpoints) ──────────────────────────────

torch_needed = pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None, reason="needs torch")


def _funded_driver(tmp_path, *, dispatch=None, busy=()):
    from cascade.trainer.lane_b_driver import LaneBDriver

    from .test_lane_b_driver import _driver, _kings
    from .test_moe_trainer import _C, _train

    _train(tmp_path / "king", dataclasses.replace(_C(), copula_loss_cap=0.0), budget=600)
    _, ops = _driver(tmp_path, {"energy": 0.9})
    fq = LaneBQueue(tmp_path / "lane_b_queue.json")
    ops.miner_funded = True
    ops.funded_queue = lambda: fq
    ops.hotkey_busy = lambda hk: hk in busy
    if dispatch is not None:
        ops.dispatch_leg = dispatch
    drv = LaneBDriver(ops.cfg, ops, tmp_path / "lane_b2.json")
    prev, king = _kings(tmp_path, ops)
    drv.add_lineage_tokens(1000)
    drv.tick(1, 0, prev)
    return drv, ops, fq, king


GEN = "alice/metro-gen@sha256:" + "a" * 64


@torch_needed
def test_unfunded_commitment_never_dispatches_then_funding_starts_it(tmp_path):
    drv, ops, fq, king = _funded_driver(tmp_path)
    assert drv.tick(2, 1, king) == [] and drv.st.gate_opened_block == 2
    assert [s.hotkey for s in drv.queue.pending] == ["alice"]
    fq.add_lane_b("alice", ("energy",), GEN, 5, gpu_choice=SimpleNamespace(
        gpu="L40S", allow_fallback=False, max_price_per_hour=2.0))
    (leg,) = drv.tick(3, 1, king)
    assert leg.funding["entry_ref"] == lane_b_entry_ref(("energy",), GEN)
    assert leg.funding["gpu"] == "L40S" and leg.funding["gpu_max_price_per_hour"] == 2.0
    # Ops.dispatch_leg trains synchronously: finished, entry still in flight
    assert drv.st.finished and fq.get("alice").status == "in_flight"
    drv.settle(round_id="r1", block=20, era_index=1, king_pointer=king,
               work_dir=tmp_path / "w")
    assert fq.get("alice").status == "done"


@torch_needed
def test_busy_hotkey_waits_and_no_funded_queue_means_no_dispatch(tmp_path):
    drv, ops, fq, king = _funded_driver(tmp_path, busy=("alice",))
    fq.add_lane_b("alice", ("energy",), GEN, 5)
    assert drv.tick(2, 1, king) == [] and fq.get("alice").status == "queued"
    ops.hotkey_busy = lambda hk: False
    ops.funded_queue = lambda: None                 # funding off: nothing, no fallback
    assert drv.tick(3, 1, king) == []


@torch_needed
def test_capacity_miss_requeues_unburned_and_miner_fault_ends_the_entry(tmp_path):
    from cascade.trainer.lane_c_driver import LegFailure

    fails = [LegFailure("sold out", miner_fault=False, error_class="no_capacity", burn=False),
             LegFailure("bad gen", miner_fault=True, error_class="generator", burn=True)]
    drv, ops, fq, king = _funded_driver(
        tmp_path, dispatch=lambda leg, era, block, done: done(leg, None, failure=fails.pop(0)))
    fq.add_lane_b("alice", ("energy",), GEN, 5)
    (leg,) = drv.tick(2, 1, king)
    e = fq.get("alice")
    assert (e.status, e.attempts, e.last_error_class) == ("queued", 0, "no_capacity")
    assert [s.hotkey for s in drv.queue.pending] == ["alice"] and not drv.queue.attempts
    (leg,) = drv.tick(3, 1, king)
    e = fq.get("alice")
    assert (e.status, e.last_error_class) == ("failed", "generator")
    assert [o["outcome"] for o in drv.st.outcomes] == ["requeued", "miner_fault"]


@torch_needed
def test_superseded_and_spent_entries_end(tmp_path):
    drv, ops, fq, king = _funded_driver(tmp_path)
    fq.add_lane_b("alice", ("web",), GEN, 5)         # the chain says energy
    drv.tick(2, 1, king)
    assert fq.get("alice").last_error_class == "superseded"
    fq.add_lane_b("alice", ("energy",), GEN, 5)
    drv.tick(3, 1, king)
    drv.settle(round_id="r1", block=20, era_index=1, king_pointer=king,
               work_dir=tmp_path / "w")
    fq.add_lane_b("alice", ("energy",), GEN, 5)      # re-fund of the SAME commitment
    assert drv.tick(4, 1, king) == []
    assert fq.get("alice").last_error_class == "spent"


@torch_needed
def test_unjudged_drop_requeues_and_readmits(tmp_path):
    drv, ops, fq, king = _funded_driver(tmp_path)
    fq.add_lane_b("alice", ("energy",), GEN, 5)
    drv.tick(2, 1, king)
    drv.settle(round_id="r9", block=20, era_index=9, king_pointer=king,   # base aged out
               work_dir=tmp_path / "w")
    e = fq.get("alice")
    assert (e.status, e.attempts, e.last_error_class) == ("queued", 0, "no_capacity")
    (leg,) = drv.tick(5, 9, king)                    # admitted again, retrains
    assert leg.sub.hotkey == "alice"


@torch_needed
def test_restart_requeues_orphaned_funded_legs(tmp_path):
    from cascade.trainer.lane_b_driver import LaneBDriver

    drv, ops, fq, king = _funded_driver(
        tmp_path, dispatch=lambda leg, era, block, done: None)
    fq.add_lane_b("alice", ("energy",), GEN, 5)
    drv.tick(2, 1, king)
    assert fq.get("alice").status == "in_flight" and drv.live_hotkeys() == {"alice"}
    drv2 = LaneBDriver(ops.cfg, ops, tmp_path / "lane_b2.json")
    assert drv2.live_hotkeys() == set()
    drv2.tick(3, 1, king)                             # recovers, then re-dispatches
    assert [leg.sub.hotkey for leg in drv2.queue.open] == ["alice"]
    assert fq.get("alice").status == "in_flight"
    assert fq.get("alice").operator_billed  # a lost allocated leg must not rent on the miner again


@torch_needed
def test_expired_leg_keeps_hotkey_busy_until_its_worker_returns(tmp_path):
    callbacks = []
    drv, ops, fq, king = _funded_driver(
        tmp_path, dispatch=lambda leg, era, block, done: callbacks.append((leg, done)))
    fq.add_lane_b("alice", ("energy",), GEN, 5)
    (old,) = drv.tick(2, 1, king)
    assert len(callbacks) == 1
    # The old worker is still alive when its base becomes too old to settle.
    assert drv.tick(3, 9, king) == []
    assert len(callbacks) == 1 and drv.live_hotkeys() == {"alice"}
    callbacks[0][1](old, None)
    assert drv.live_hotkeys() == set()
    assert len(drv.tick(4, 9, king)) == 1
    assert len(callbacks) == 2


# ── production ops: the funded path, never the operator pool ────────────────


def test_runner_ops_dispatch_funded_leg_on_the_miners_key(cfg, tmp_path, monkeypatch):
    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg
    from cascade.trainer.faults import _FundedOperatorFallback
    from cascade.trainer.lane_b_driver import RunnerLaneBOps
    from cascade.trainer.loop import TrainerRunner

    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    era = SimpleNamespace(base_seed=12345, start_block=7200, index=3)
    r.cfg = dataclasses.replace(cfg, lane_b=dataclasses.replace(cfg.lane_b, leg_hours=0.5),
                                round=dataclasses.replace(cfg.round, funded_mode="required"))
    r.__dict__["_rolling_sched"] = SimpleNamespace(state=SimpleNamespace(current=era))
    monkeypatch.setattr(TrainerRunner, "_dispatcher", lambda self: object())
    monkeypatch.setattr(TrainerRunner, "_rolling_lane_pool",
                        lambda self: pytest.fail("operator pool used"))
    calls, results = [], []

    def run(self, disp, gen, seeds, block, contract, suffix, **kw):
        assert contract.target_train_hours == 0.5
        assert contract.max_train_seconds == 1800
        assert kw["funded_queue"].path.name == "lane_b_queue.json"
        calls.append((gen, suffix, kw))
        if len(calls) == 2:
            raise _FundedOperatorFallback(gen.hotkey)
        return SimpleNamespace(trained_pointer="ptr", corpus_digest="cd", train_block=block,
                               gpu_name="L40S")

    monkeypatch.setattr(TrainerRunner, "_run_funded_leg", run)
    ops = RunnerLaneBOps(r, client=None, window_source=None)
    assert ops.miner_funded
    leg = OpenLeg(sub=LaneBSubmission("alice", 7, GEN, "energy", 5,
                                      targets=["energy", "web"]),
                  selection={0: [1, 2]}, base_king_ref="kingref", base_era=3,
                  funding={"entry_ref": "x", "gpu": "H100", "gpu_allow_fallback": True})
    for _ in range(2):
        ev = threading.Event()
        ops.dispatch_leg(leg, 3, 7300, lambda leg, ptr, *a, failure=None, ev=ev:
                         (results.append((ptr, failure)), ev.set()))
        assert ev.wait(5)
    gen, suffix, kw = calls[0]
    assert gen.hotkey == "alice" and suffix == "-xenergy.web-u7"
    assert kw["role"] == "expert" and kw["bench"] is False and kw["domain"] == "energy.web"
    assert kw["expert_selection"] == {0: [1, 2]} and kw["warm_start_ref"] == "kingref"
    assert kw["gpu_choice"].gpu == "H100" and kw["gpu_choice"].allow_fallback is True
    assert results[0] == ("ptr", None)
    ptr, f = results[1]
    assert ptr is None and (f.error_class, f.burn, f.miner_fault) == ("no_capacity", False,
                                                                       False)
    # a leg without a funded entry never runs under leg_funding = "miner"
    ev = threading.Event()
    got = []
    ops.dispatch_leg(dataclasses.replace(leg, funding={}), 3, 7300,
                     lambda leg, ptr, *a, **k: (got.append(ptr), ev.set()))
    assert ev.wait(5) and got == [None] and len(calls) == 2


def test_funded_leg_passes_expert_kwargs_to_the_dispatcher(cfg, tmp_path, monkeypatch):
    """``_run_funded_leg(role="expert")`` hands the worker its domain + rows."""
    import cascade.trainer.runner_funded_legs as rfl
    from cascade.trainer.loop import ResolvedGenerator, TrainerRunner

    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    from cascade.trainer.admission import remember_owners

    remember_owners(r, [SimpleNamespace(hotkey="h", coldkey="fixture-owner", commit_block=1)])
    seen = {}

    class Stop(Exception):
        pass

    class Disp:
        def dispatch(self, host, **kw):
            seen.update(kw)
            raise Stop

    monkeypatch.setattr(rfl, "leg_corpus", lambda *a, **k: {})
    monkeypatch.setattr(rfl, "_reupload_pending", lambda *a, **k: None)
    monkeypatch.setattr(TrainerRunner, "_rent_funded_host",
                        lambda self, rid, gen, **k: (SimpleNamespace(), SimpleNamespace()))
    monkeypatch.setattr(TrainerRunner, "_funded_pod_identity_mismatch", lambda self, p: None)
    monkeypatch.setattr(TrainerRunner, "_funded_harvest", lambda self: True)
    monkeypatch.setattr(TrainerRunner, "_teardown_funded_pod", lambda self, p: None)
    monkeypatch.setattr(TrainerRunner, "_record_funded_failure", lambda self, *a, **k: None)
    monkeypatch.setattr(TrainerRunner, "_funded_queue",
                        lambda self: SimpleNamespace(get=lambda hk: None,
                                                     set_operator_billed=lambda *a: None))
    monkeypatch.setattr("cascade.trainer.remote.probe_worker_runtime", lambda host: "")
    monkeypatch.setattr(rfl, "classify_funded_worker_failure",
                        lambda *a, **k: (False, "infra", True))
    contract = cfg.training.primary_size
    with pytest.raises(Stop):
        r._run_funded_leg(Disp(), ResolvedGenerator(hotkey="h", uid=1, ref=GEN),
                          SimpleNamespace(base_seed=1), 5, contract, "-xenergy-u1",
                          warm_start_ref="k", role="expert", bench=False, domain="energy",
                          expert_selection={0: [3]})
    assert seen["role"] == "expert" and seen["domain"] == "energy"
    assert seen["expert_selection"] == {0: [3]} and seen["warm_start_ref"] == "k"
