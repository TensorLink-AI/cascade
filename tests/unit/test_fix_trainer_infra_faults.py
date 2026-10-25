"""T6 + leg-verdict hygiene: operator-side faults never burn a miner's attempt
and back off; a failed upload keeps the harvested checkpoint and the requeued
leg re-uploads it instead of retraining; a new leg never reads back the
previous leg's verdict."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cascade.trainer.loop as loop_module
from cascade.funding.queue import (
    INFRA_BACKOFF_BASE_SECONDS,
    INFRA_BACKOFF_MAX_SECONDS,
    OPERATOR_FAULT_CLASS,
    FundedQueue,
)
from cascade.generation.stage import GenerationFault, GenerationInfraError
from cascade.trainer.faults import _OperatorFault
from cascade.trainer.runner_common import read_pending_upload
from cascade.trainer.runner_funded_legs import _record_generation_failure
from tests.unit.test_funded_pod_wiring import (
    _challenger,
    _FakeLocalDisp,
    _harvest_runner,
    _leg_runner,
)


class _FakeDisp:
    def __init__(self):
        self.calls = []

    def dispatch(self, host, **kw):
        self.calls.append((host, kw))
        return SimpleNamespace(hotkey=kw["hotkey"])


# ── the queue: operator faults back off, never burn ─────────────────────────

def _queue(tmp_path, t):
    q = FundedQueue(tmp_path / "q.json", clock=lambda: t[0])
    q.add("hk", "r@1", 10)
    assert q.mark_in_flight("hk", "r@1", target_boundary=1, era_index=1, started_block=1)
    return q


def test_operator_requeue_backs_off_and_never_burns(tmp_path):
    t = [1000.0]
    q = _queue(tmp_path, t)
    assert q.requeue("hk", error="upload failed", error_class=OPERATOR_FAULT_CLASS,
                     burn_attempt=False)
    e = q.get("hk")
    assert e.attempts == 0 and e.infra_streak == 1
    assert e.not_before == 1000.0 + INFRA_BACKOFF_BASE_SECONDS
    assert not q.mark_in_flight("hk", "r@1", target_boundary=1, era_index=1, started_block=2)
    t[0] = e.not_before + 1
    assert q.mark_in_flight("hk", "r@1", target_boundary=1, era_index=1, started_block=2)
    # consecutive operator faults double, capped
    for _ in range(10):
        q.requeue("hk", error="x", error_class=OPERATOR_FAULT_CLASS, burn_attempt=False)
        t[0] = q.get("hk").not_before
        q.mark_in_flight("hk", "r@1", target_boundary=1, era_index=1, started_block=3)
    q.requeue("hk", error="x", error_class=OPERATOR_FAULT_CLASS, burn_attempt=False)
    assert q.get("hk").not_before - t[0] == INFRA_BACKOFF_MAX_SECONDS
    assert q.get("hk").attempts == 0


def test_other_requeues_reset_the_backoff(tmp_path):
    t = [1000.0]
    q = _queue(tmp_path, t)
    q.requeue("hk", error="x", error_class=OPERATOR_FAULT_CLASS, burn_attempt=False)
    q.requeue("hk", error="pod died", error_class="infra", burn_attempt=True)
    e = q.get("hk")
    assert e.not_before == 0.0 and e.infra_streak == 0 and e.attempts == 1
    assert q.mark_in_flight("hk", "r@1", target_boundary=1, era_index=1, started_block=2)


def test_backoff_fields_survive_a_reload(tmp_path):
    t = [1000.0]
    q = _queue(tmp_path, t)
    q.requeue("hk", error="x", error_class=OPERATOR_FAULT_CLASS, burn_attempt=False)
    again = FundedQueue(tmp_path / "q.json", clock=lambda: t[0]).get("hk")
    assert again.not_before == 1000.0 + INFRA_BACKOFF_BASE_SECONDS and again.infra_streak == 1


# ── generation-tier verdicts ────────────────────────────────────────────────

def _rec_runner():
    rec = {}
    return SimpleNamespace(
        _funded_queue=lambda: None,
        _record_funded_failure=lambda hk, msg, **kw: rec.__setitem__(hk, (msg, kw))), rec


@pytest.mark.parametrize("exc", [GenerationInfraError("no CPU capacity: 32 vCPU"),
                                 RuntimeError("generation tier: no public key")])
def test_generation_tier_infra_is_the_operators_unburned(exc):
    r, rec = _rec_runner()
    _record_generation_failure(r, _challenger("hkA"), exc)
    _msg, kw = rec["hkA"]
    assert kw == {"miner_fault": False, "error_class": OPERATOR_FAULT_CLASS, "burn": False}


def test_generator_fault_on_the_cpu_tier_stays_the_miners():
    r, rec = _rec_runner()
    _record_generation_failure(r, _challenger("hkA"), GenerationFault("generator fault"))
    _msg, kw = rec["hkA"]
    assert kw["miner_fault"] is True and kw["error_class"] == "generator"


# ── a new leg never inherits the last leg's verdict ─────────────────────────

def test_a_new_leg_clears_the_hotkeys_previous_verdict(tmp_path, monkeypatch):
    disp = _FakeDisp()
    runner, _torn, seeds, contract = _leg_runner(tmp_path, monkeypatch, disp=disp)
    runner._funded_leg_failures["hkA"] = ("old leg: tamper", True, "tamper", False)
    runner._run_funded_leg(disp, _challenger("hkA"), seeds, 100, contract, "",
                           warm_start_ref=None)
    assert "hkA" not in runner._funded_leg_failures


def test_stale_runtime_image_is_an_operator_fault_unburned(tmp_path, monkeypatch):
    disp = _FakeDisp()
    runner, torn, seeds, contract = _leg_runner(tmp_path, monkeypatch, disp=disp)
    monkeypatch.setattr("cascade.trainer.remote.probe_worker_runtime",
                        lambda host, **kw: "pod booted a STALE worker image")
    with pytest.raises(_OperatorFault):
        runner._run_funded_leg(disp, _challenger("hkA"), seeds, 100, contract, "",
                               warm_start_ref=None)
    _msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, OPERATOR_FAULT_CLASS, False)
    assert torn and disp.calls == []


# ── upload failure: retried, kept, re-uploaded by the requeued leg ──────────

def _harvest_ok(monkeypatch):
    def _harvest(host, rdir, dest, **kw):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "weights.safetensors").write_bytes(b"W")
        return dest
    monkeypatch.setattr("cascade.trainer.remote.harvest_remote_dir", _harvest)
    monkeypatch.setattr("cascade.eval.checkpoint_guard.verify_checkpoint",
                        lambda d, contract=None, **kw: None)
    monkeypatch.setattr("cascade.eval.checkpoint_guard.verify_weights_finite", lambda d: None)


def _uploader(monkeypatch, fail_first: int):
    calls = []
    up = SimpleNamespace(ref=SimpleNamespace(
        immutable_ref="cascade/ckpt-r777-challenger-toto2-4m@sha256:" + "d" * 64))

    def _up(d, repo, hub, hf_repo=None):
        calls.append(repo)
        if len(calls) <= fail_first:
            raise RuntimeError("hippius 503")
        return up
    monkeypatch.setattr(loop_module, "upload_dir_to_hub_or_hf", _up)
    return calls


def test_upload_blip_is_retried_and_the_local_copy_deleted(tmp_path, monkeypatch):
    disp = _FakeLocalDisp()
    runner, _torn, seeds, contract = _harvest_runner(tmp_path, monkeypatch, disp=disp)
    runner._hf_ckpt_repo = lambda repo: None
    sleeps = []
    runner._upload_retry_sleep = sleeps.append
    _harvest_ok(monkeypatch)
    calls = _uploader(monkeypatch, fail_first=2)
    entry = runner._run_funded_leg(disp, _challenger("hkA"), seeds, 100, contract, "",
                                   warm_start_ref=None)
    assert len(calls) == 3 and len(sleeps) == 2 and sleeps[1] == 2 * sleeps[0]
    assert entry.miner_hotkey == "hkA"
    assert not (tmp_path / "_funded_harvest" / "777" / "hkA").exists()   # verified ⇒ deleted


def test_upload_outage_keeps_the_checkpoint_and_the_requeue_reuploads(tmp_path, monkeypatch):
    disp = _FakeLocalDisp()
    runner, torn, seeds, contract = _harvest_runner(tmp_path, monkeypatch, disp=disp)
    runner._hf_ckpt_repo = lambda repo: None
    runner._upload_retry_sleep = lambda s: None
    _harvest_ok(monkeypatch)
    calls = _uploader(monkeypatch, fail_first=loop_module.HARVEST_UPLOAD_ATTEMPTS)
    with pytest.raises(_OperatorFault):
        runner._run_funded_leg(disp, _challenger("hkA"), seeds, 100, contract, "",
                               warm_start_ref=None)
    _msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, OPERATOR_FAULT_CLASS, False)
    rec = read_pending_upload(tmp_path, 777, "hkA")
    assert rec is not None and rec["warm_start_ref"] == "" and rec["gen_ref"]
    assert len(disp.calls) == 1 and len(torn) == 1
    # The requeued leg: no rent, no dispatch — the kept copy is uploaded.
    runner._rent_funded_host = lambda *a, **k: pytest.fail("must not rent again")
    entry = runner._run_funded_leg(disp, _challenger("hkA"), seeds, 100, contract, "",
                                   warm_start_ref=None)
    assert entry.miner_hotkey == "hkA" and entry.corpus_digest == "cd"
    assert len(disp.calls) == 1
    assert len(calls) == loop_module.HARVEST_UPLOAD_ATTEMPTS + 1
    assert read_pending_upload(tmp_path, 777, "hkA") is None
    assert not (tmp_path / "_funded_harvest" / "777" / "hkA").exists()


def test_a_pending_upload_of_another_init_is_not_reused(tmp_path, monkeypatch):
    disp = _FakeLocalDisp()
    runner, _torn, seeds, contract = _harvest_runner(tmp_path, monkeypatch, disp=disp)
    runner._hf_ckpt_repo = lambda repo: None
    runner._upload_retry_sleep = lambda s: None
    _harvest_ok(monkeypatch)
    _uploader(monkeypatch, fail_first=loop_module.HARVEST_UPLOAD_ATTEMPTS)
    with pytest.raises(_OperatorFault):
        runner._run_funded_leg(disp, _challenger("hkA"), seeds, 100, contract, "",
                               warm_start_ref=None)
    _uploader(monkeypatch, fail_first=0)
    runner._run_funded_leg(disp, _challenger("hkA"), seeds, 100, contract, "",
                           warm_start_ref="cascade/init@sha256:" + "e" * 64)
    assert len(disp.calls) == 2                  # retrained under the other init


# ── operator-rented legs: our pod, our fault ────────────────────────────────

def test_operator_rented_leg_failure_never_burns(tmp_path, monkeypatch):
    from cascade.trainer.loop import TrainerRunner
    from cascade.trainer.remote import RemoteDispatchError

    rec = {}
    torn = []
    r = SimpleNamespace(
        _rent_operator_leg_host=lambda rid, gen: (object(), "op-pod-0"),
        _teardown_operator_pod=lambda pod: torn.append(pod.instance_id),
        _funded_queue=lambda: None,
        _record_funded_failure=lambda hk, msg, **kw: rec.__setitem__(hk, kw),
        _leg_corpus=lambda *a, **k: None,
    )

    class _Boom:
        def dispatch(self, host, **kw):
            raise RemoteDispatchError("worker exited rc=1", returncode=1)

    with pytest.raises(RemoteDispatchError):
        TrainerRunner._run_operator_rented_leg(
            r, _Boom(), _challenger("hkA"), SimpleNamespace(base_seed=7), 1,
            SimpleNamespace(arch_preset="toto2-4m", corpus_mode="stream_cpu"), "",
            warm_start_ref=None)
    assert rec["hkA"] == {"miner_fault": False, "error_class": OPERATOR_FAULT_CLASS,
                          "burn": False}
    assert torn == ["op-pod-0"]
