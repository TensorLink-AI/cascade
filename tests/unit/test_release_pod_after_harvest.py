"""Owner 2026-11-01: "stop pods as soon as we're sure the checkpoint is saved".

A funded / operator-rented leg's pod is released the moment its harvested
copy on the box matches the pod's own listing (every file, size + sha256)
and the ingest guards passed — BEFORE the box → Hippius upload, which runs
from the box copy. Never on a mismatch; never for a pod the trainer still
benches on."""

from __future__ import annotations

import hashlib
import logging
from types import SimpleNamespace

import pytest

import cascade.trainer.loop as loop_module
from cascade.shared.config import RoundConfig
from cascade.trainer.faults import _OperatorFault
from cascade.trainer.harvest_release import (
    RELEASE_LOG,
    VerifiedHarvest,
    manifest_mismatches,
    release_if_verified,
)
from cascade.trainer.remote import RemoteDispatchError, RemoteHost, remote_file_manifest
from cascade.trainer.runner_common import read_pending_upload
from tests.unit.test_funded_pod_wiring import _challenger, _FakeLocalDisp, _harvest_runner

FILES = {"weights.safetensors": b"W" * 7, "weights_stable.safetensors": b"S" * 5,
         "optimizer.pt": b"O" * 11}
POD = "cascade-n91-777-funded-hka-0"


def _listing(files: dict) -> dict:
    return {name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in files.items()}


def _scripted(tmp_path, monkeypatch, *, pod_files=None, harvests=None, upload_fails=0,
              **round_kw):
    """A payer leg on scripted ssh: ``harvests`` is the box copy each harvest
    writes (default: the pod's files), ``pod_files`` what the pod lists."""
    disp = _FakeLocalDisp()
    runner, _torn, seeds, contract = _harvest_runner(tmp_path, monkeypatch, disp=disp, **round_kw)
    runner._hf_ckpt_repo = lambda repo: None
    runner._upload_retry_sleep = lambda s: None
    runner._harvest_retry_sleep = lambda s: None
    events: list[str] = []
    runner._teardown_funded_pod = lambda pod: events.append(f"teardown:{pod.instance_id}")
    pod_files = dict(FILES if pod_files is None else pod_files)
    copies = list(harvests or [pod_files])

    def harvest(host, rdir, dest, **kw):
        files = copies.pop(0) if len(copies) > 1 else copies[0]
        dest.mkdir(parents=True, exist_ok=True)
        for p in dest.iterdir():
            p.unlink()
        for name, data in files.items():
            (dest / name).write_bytes(data)
        events.append("harvest")
        return dest

    def listing(host, rdir, **kw):
        events.append("pod-listing")
        return _listing(pod_files)

    monkeypatch.setattr("cascade.trainer.remote.harvest_remote_dir", harvest)
    monkeypatch.setattr("cascade.trainer.remote.remote_file_manifest", listing)
    monkeypatch.setattr("cascade.eval.checkpoint_guard.verify_checkpoint",
                        lambda d, contract=None, **kw: events.append("verify"))
    monkeypatch.setattr("cascade.eval.checkpoint_guard.verify_weights_finite", lambda d: None)
    up = SimpleNamespace(ref=SimpleNamespace(
        immutable_ref="cascade/ckpt-r777-challenger-toto2-4m@sha256:" + "d" * 64))
    calls: list[str] = []

    def upload(d, repo, hub, hf_repo=None):
        calls.append(repo)
        # the upload reads the BOX copy, complete
        assert {p.name for p in d.iterdir()} == set(pod_files)
        events.append("upload")
        if len(calls) <= upload_fails:
            raise RuntimeError("hippius 503")
        return up

    monkeypatch.setattr(loop_module, "upload_dir_to_hub_or_hf", upload)
    return runner, disp, seeds, contract, events, calls


def _leg(runner, disp, seeds, contract):
    return runner._run_funded_leg(disp, _challenger("hkA"), seeds, 100, contract, "",
                                  warm_start_ref=None)


def test_pod_released_after_verified_harvest_and_before_upload(tmp_path, monkeypatch, caplog):
    runner, disp, seeds, contract, events, _ = _scripted(tmp_path, monkeypatch)
    with caplog.at_level(logging.INFO, logger="cascade.trainer"):
        entry = _leg(runner, disp, seeds, contract)
    assert entry.miner_hotkey == "hkA"
    assert events == ["harvest", "pod-listing", "verify", f"teardown:{POD}", "upload"]
    nbytes = sum(len(v) for v in FILES.values())
    assert RELEASE_LOG % (3, nbytes / 1e9) in caplog.text
    assert runner._funded_bench_pods == {} and runner._final_role_hosts == {}


def test_hash_mismatch_reharvests_once_then_releases(tmp_path, monkeypatch):
    bad = dict(FILES, **{"optimizer.pt": b"O" * 10 + b"X"})   # same size, other bytes
    runner, disp, seeds, contract, events, _ = _scripted(
        tmp_path, monkeypatch, harvests=[bad, FILES])
    _leg(runner, disp, seeds, contract)
    assert events == ["harvest", "pod-listing", "harvest", "pod-listing", "verify",
                      f"teardown:{POD}", "upload"]


@pytest.mark.parametrize("copy", [
    dict(FILES, **{"optimizer.pt": b"O" * 10 + b"X"}),              # sha256 differs
    {k: v for k, v in FILES.items() if k != "optimizer.pt"},         # file missing
    dict(FILES, **{"optimizer.pt": b"O" * 3}),                       # truncated
])
def test_persistent_mismatch_is_never_a_success_and_keeps_the_pod(tmp_path, monkeypatch, copy):
    """2026-11-02 (saving robustness): a copy that still differs from the
    pod's own listing after its re-harvest is NEVER uploaded; the harvest is
    retried (bounded) and then the pod is KEPT for the re-attach — an
    operator fault, nothing burned."""
    from cascade.trainer.harvest_guard import HarvestKeepPod

    runner, disp, seeds, contract, events, _ = _scripted(
        tmp_path, monkeypatch, harvests=[copy], harvest_disk_retries=1)
    monkeypatch.setattr(loop_module, "upload_dir_to_hub_or_hf",
                        lambda d, repo, hub, hf_repo=None: events.append("upload") or SimpleNamespace(
                            ref=SimpleNamespace(immutable_ref="cascade/c@sha256:" + "d" * 64)))
    with pytest.raises(HarvestKeepPod):
        _leg(runner, disp, seeds, contract)
    # two tries x (harvest + one re-harvest), never verified, uploaded or torn down
    assert events == ["harvest", "pod-listing", "harvest", "pod-listing"] * 2
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "operator", False)


def test_unlistable_pod_is_kept_until_the_leg_ends(tmp_path, monkeypatch):
    runner, disp, seeds, contract, events, _ = _scripted(tmp_path, monkeypatch)

    def dead(host, rdir, **kw):
        raise RemoteDispatchError("ssh rc=255")
    monkeypatch.setattr("cascade.trainer.remote.remote_file_manifest", dead)
    _leg(runner, disp, seeds, contract)
    assert events == ["harvest", "verify", "upload", f"teardown:{POD}"]


def test_guard_failure_after_a_complete_harvest_never_releases_early(tmp_path, monkeypatch):
    from cascade.eval.checkpoint_guard import CheckpointTampered
    from cascade.trainer.loop import _FundedTamper

    runner, disp, seeds, contract, events, _ = _scripted(tmp_path, monkeypatch)

    def tampered(d, contract=None, **kw):
        raise CheckpointTampered("model.py differs")
    monkeypatch.setattr("cascade.eval.checkpoint_guard.verify_checkpoint", tampered)
    with pytest.raises(_FundedTamper):
        _leg(runner, disp, seeds, contract)
    assert events == ["harvest", "pod-listing", f"teardown:{POD}"]   # leg-end teardown only


def test_upload_failure_after_release_reuploads_from_the_box_copy(tmp_path, monkeypatch):
    runner, disp, seeds, contract, events, calls = _scripted(
        tmp_path, monkeypatch, upload_fails=loop_module.HARVEST_UPLOAD_ATTEMPTS)
    with pytest.raises(_OperatorFault):
        _leg(runner, disp, seeds, contract)
    # released once, before the first upload attempt; never torn down twice
    assert events[:4] == ["harvest", "pod-listing", "verify", f"teardown:{POD}"]
    assert events.count(f"teardown:{POD}") == 1
    assert events[4:] == ["upload"] * loop_module.HARVEST_UPLOAD_ATTEMPTS
    rec = read_pending_upload(tmp_path, 777, "hkA")
    assert rec is not None                       # the box copy is kept
    # The requeued leg: no rent, no pod, no dispatch — the box copy uploads.
    runner._rent_funded_host = lambda *a, **k: pytest.fail("must not rent again")
    entry = _leg(runner, disp, seeds, contract)
    assert entry.miner_hotkey == "hkA" and len(disp.calls) == 1
    assert len(calls) == loop_module.HARVEST_UPLOAD_ATTEMPTS + 1
    assert read_pending_upload(tmp_path, 777, "hkA") is None


def test_trainer_bench_source_keeps_the_pod(tmp_path, monkeypatch):
    runner, disp, seeds, contract, events, _ = _scripted(tmp_path, monkeypatch)
    runner.cascade_bench_plan = object()          # the trainer benches on this pod
    _leg(runner, disp, seeds, contract)
    assert "pod-listing" not in events and not any(e.startswith("teardown") for e in events)
    assert runner._funded_bench_pods["hkA"].instance_id == POD
    assert ("challenger", "toto2-4m", "hkA") in runner._final_role_hosts


def test_validator_bench_source_releases_even_with_the_bench_armed(tmp_path, monkeypatch):
    import threading

    runner, disp, seeds, contract, events, _ = _scripted(tmp_path, monkeypatch)
    runner.cascade_bench_plan = object()
    local = runner.__dict__.setdefault("_leg_local_obj", threading.local())
    local.validator_bench = True                  # DEC-CA-0063 era: no trainer bench
    _leg(runner, disp, seeds, contract)
    assert events == ["harvest", "pod-listing", "verify", f"teardown:{POD}", "upload"]
    assert runner._funded_bench_pods == {} and runner._final_role_hosts == {}


def test_knob_off_keeps_todays_teardown(tmp_path, monkeypatch):
    runner, disp, seeds, contract, events, _ = _scripted(
        tmp_path, monkeypatch, funded_release_after_harvest=False)
    _leg(runner, disp, seeds, contract)
    assert events == ["harvest", "verify", "upload", f"teardown:{POD}"]


# ── operator-rented legs ─────────────────────────────────────────────────────

def _operator_runner(events):
    def harvest(host, receipt, contract, seeds, suffix):
        events.append("harvest")
        release_if_verified(r, VerifiedHarvest(files=3, nbytes=23))
        events.append("upload")
        return SimpleNamespace(miner_hotkey="hkA")

    r = SimpleNamespace(
        _rent_operator_leg_host=lambda rid, gen: (object(), "op-pod-0"),
        _teardown_operator_pod=lambda pod: events.append(f"teardown:{pod.instance_id}"),
        _funded_queue=lambda: None,
        _record_funded_failure=lambda hk, msg, **kw: events.append("failure"),
        _harvest_funded_checkpoint=harvest,
    )
    return r


class _Disp:
    def __init__(self, events, harvest_in_dispatch=None):
        self.events = events
        self.harvest_in_dispatch = harvest_in_dispatch

    def dispatch(self, host, **kw):
        self.events.append("dispatch")
        if self.harvest_in_dispatch is not None:          # the isolated-host hook
            return self.harvest_in_dispatch(host, None, None, None, "")
        return SimpleNamespace(miner_hotkey="hkA")


@pytest.mark.parametrize("role", ["challenger", "expert"])
def test_operator_rented_leg_releases_before_upload(monkeypatch, role):
    from cascade.trainer.loop import TrainerRunner

    monkeypatch.setattr("cascade.trainer.runner_operator_pods.leg_corpus",
                        lambda *a, **k: {})
    events: list[str] = []
    r = _operator_runner(events)
    disp = _Disp(events, harvest_in_dispatch=(r._harvest_funded_checkpoint
                                              if role == "challenger" else None))
    TrainerRunner._run_operator_rented_leg(
        r, disp, _challenger("hkA"), SimpleNamespace(base_seed=7), 1,
        SimpleNamespace(arch_preset="toto2-4m", corpus_mode="stream_cpu"), "",
        warm_start_ref=None, role=role)
    assert events == ["dispatch", "harvest", "teardown:op-pod-0", "upload"]


def test_operator_rented_leg_without_a_verified_harvest_tears_down_at_leg_end(monkeypatch):
    from cascade.trainer.loop import TrainerRunner

    monkeypatch.setattr("cascade.trainer.runner_operator_pods.leg_corpus",
                        lambda *a, **k: {})
    events: list[str] = []
    r = _operator_runner(events)

    def unverified(host, receipt, contract, seeds, suffix):
        events.append("harvest")
        release_if_verified(r, None)
        events.append("upload")
        return SimpleNamespace(miner_hotkey="hkA")
    r._harvest_funded_checkpoint = unverified
    TrainerRunner._run_operator_rented_leg(
        r, _Disp(events), _challenger("hkA"), SimpleNamespace(base_seed=7), 1,
        SimpleNamespace(arch_preset="toto2-4m", corpus_mode="stream_cpu"), "",
        warm_start_ref=None, role="expert")
    assert events == ["dispatch", "harvest", "upload", "teardown:op-pod-0"]


def test_release_outside_an_armed_leg_is_a_no_op():
    r = SimpleNamespace()
    assert release_if_verified(r, VerifiedHarvest(files=1, nbytes=1)) is False


# ── the pod-side listing ─────────────────────────────────────────────────────

def _host():
    return RemoteHost(name="p", host="h", workdir="/root/cascade")


def test_remote_file_manifest_parses_sizes_and_hashes():
    seen = {}
    a, b = "a" * 64, "b" * 64
    out = (f"S 7 ./weights.safetensors\nS 11 ./sub/optimizer.pt\n"
           f"{a}  ./weights.safetensors\n{b}  ./sub/optimizer.pt\n")

    def run(argv, timeout):
        seen["cmd"] = argv[-1]
        return SimpleNamespace(returncode=0, stdout=out, stderr="")
    got = remote_file_manifest(_host(), "_train_work/ck", runner=run)
    assert got == {"weights.safetensors": (7, a), "sub/optimizer.pt": (11, b)}
    assert seen["cmd"].startswith("cd /root/cascade && cd _train_work/ck && ")


@pytest.mark.parametrize("out", [
    "S 7 ./w\n",                                     # size without hash
    "S x ./w\n" + "a" * 64 + "  ./w\n",              # bad size
    "S 7 ./w\n\\" + "a" * 64 + "  ./w\\nx\n",        # escaped (odd) filename
])
def test_remote_file_manifest_refuses_what_it_cannot_parse(out):
    def run(argv, timeout):
        return SimpleNamespace(returncode=0, stdout=out, stderr="")
    with pytest.raises(RemoteDispatchError):
        remote_file_manifest(_host(), "ck", runner=run)


def test_remote_file_manifest_ssh_failure_raises():
    def run(argv, timeout):
        return SimpleNamespace(returncode=255, stdout="", stderr="no route")
    with pytest.raises(RemoteDispatchError, match="rc=255"):
        remote_file_manifest(_host(), "ck", runner=run)


def test_manifest_mismatches_names_every_kind():
    pod = {"a": (1, "x"), "b": (2, "y"), "c": (3, "z")}
    box = {"a": (1, "x"), "b": (2, "q"), "d": (4, "w")}
    assert manifest_mismatches(pod, box) == [
        "b: sha256 differs from the pod's", "c: missing from the box copy",
        "d: on the box but not on the pod"]
    assert manifest_mismatches(pod, dict(pod)) == []


# ── the knob ─────────────────────────────────────────────────────────────────

def test_release_knob_defaults_on_and_round_trips_through_the_loader(tmp_path):
    """Config knobs need loader parsing: the dataclass field alone would make
    TOML disarming a silent no-op."""
    from pathlib import Path

    from cascade.shared.config import load_chain_config

    assert RoundConfig().funded_release_after_harvest is True
    repo_toml = Path(__file__).resolve().parents[2] / "chain.toml"
    assert load_chain_config(repo_toml).round.funded_release_after_harvest is True
    src = repo_toml.read_text(encoding="utf-8")
    edited = src.replace("\n[round]\n", "\n[round]\nfunded_release_after_harvest = false\n", 1)
    assert edited != src
    (tmp_path / "chain.toml").write_text(edited, encoding="utf-8")
    assert load_chain_config(tmp_path / "chain.toml").round.funded_release_after_harvest is False


# The owner-approval gate (2026-11-02) is tested in test_money_safety.py;
# these tests drive the operator-lane / operator-rent machinery behind it.
pytestmark = pytest.mark.usefixtures("operator_legs_preapproved")
