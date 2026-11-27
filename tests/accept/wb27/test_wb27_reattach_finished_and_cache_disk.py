"""WB27 incident fixes (2026-11-27), two lane B robustness bugs hit live.

1. **A FINISHED expert run "not found" by the re-attach.** At 11:42 the
   trainer restarted while the 5GMzsP lane B expert leg's worker had finished
   on its operator pod (8xH200; step ~23,800; exit 0; receipt + checkpoint on
   the pod). The re-attach logged "ReattachMissing: re-attach refused: no
   live or finished run of this expert leg …", requeued, and ``lium rm`` the
   pod. Cause: the detached run tag is ``<role>-<hk12>-<seed>-<identity
   hash>`` and the identity (continuation key) folds in ``--contract-block``,
   which the re-attach re-derived from the CURRENT era (2666, block 9597600)
   instead of the leg's dispatch era (2665, block 9594000) — another hash, a
   glob miss. Pinned: the re-attach finds the run by its stable name and
   harvests it; the lane B re-attach re-derives the dispatch era; a pod
   holding a finished, unharvested run is never removed on a failure.
   The fixture dispatch dir is the live one (postmortem copy; stderr tail).

2. **Lane B eval-pod cache on the small root disk** ("6.8 GB free on the
   lane B cache filesystem < 26.0 GB needed" on a 43 GB ``/`` overlay while
   ``/workspace`` had 159 GB): the offload / eval cache dirs are placed on
   the largest writable data mount; the disk-short error names the
   alternative mount.

Torch-free (every module imported with torch blocked); pods are local
``bash`` (``build_ssh_argv`` patched) over a tmp workdir."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.accept.wb27.helpers import torch_blocked

FIXTURE = Path(__file__).parent / "fixtures" / "reattach_5gmzsp"
HK = "5GMzsPiUhy6gtqU5GxeUgzJ4yyQnJ5iyx8EUdSgZD2riSDgn"
SEED = 824229856147703689
RUN = "expert-5GMzsPiUhy6g-824229856147703689-cca9bc324c19-e35ab556"
CKPT = "_train_work/824229856147703689/cascade-moe-v2-120m/challenger-xenergy-u171/checkpoint"
SEL = {"0": [23, 31, 57, 66, 72, 77, 78, 93], "1": [6, 23, 71, 163, 173, 188, 211, 244],
       "2": [2, 55, 103, 107, 112, 156, 202, 244], "3": [15, 42, 197, 215, 400, 444, 478, 503],
       "4": [192, 233, 248, 251, 359, 367, 385, 462], "5": [55, 66, 80, 144, 169, 200, 474, 489],
       "6": [61, 82, 84, 94, 111, 218, 293, 314, 347, 349, 416, 499]}
INIT = ("metro-v1:trained:hippius:cascade/ckpt-r824229856147703689-challenger-cascade-moe-v2-"
        "120m-u164@sha256:0d7e107f50e5ba38aef64f24454a0eff422b9e4409967a134654083d19aff1be")
GEN = ("thomas05101990/gen-574ca46dc640@sha256:"
       "72e839e5b15f22eb2949ff5c81655045f35ca3204c7c5152fc3cb031c96fa8f3")


def _outcome(fn):
    """``("ok", result)`` or ``("raised", exc)`` — a base without the fix
    fails by ASSERTION, not by an unexpected exception."""
    try:
        return "ok", fn()
    except Exception as e:  # noqa: BLE001
        return "raised", e


def _pod(tmp_path: Path, *, finished: bool = True) -> Path:
    """A pod workdir holding the live dispatch dir (+ the checkpoint)."""
    work = tmp_path / "pod"
    run = work / "_train_work" / "_dispatch" / RUN
    run.mkdir(parents=True)
    if finished:
        for name in ("pid", "progress.json", "stdout", "stderr", "exit_code"):
            shutil.copyfile(FIXTURE / name, run / name)
        ck = work / CKPT
        ck.mkdir(parents=True)
        (ck / "weights.safetensors").write_bytes(b"w" * 64)
    return work


def _local_pod(monkeypatch):
    """Every ssh to a pod runs the command in local bash."""
    with torch_blocked():
        from cascade.trainer import remote
    monkeypatch.setattr(remote, "build_ssh_argv", lambda host, cmd: ["bash", "-c", cmd])
    monkeypatch.setattr(remote, "_SLEEP", lambda s: None)
    calls: list[str] = []

    def run(argv, timeout, stdin=None):
        calls.append(argv[-1])
        if "setsid nohup" in argv[-1]:
            # a LAUNCH is never executed here (no worker, no torch on the box)
            return subprocess.CompletedProcess(argv, 1, "", "launch refused by the test pod")
        return subprocess.run(argv, input=stdin or "", capture_output=True, text=True,
                              timeout=min(float(timeout), 60))
    return run, calls


def _host(work: Path, remote):
    return remote.RemoteHost(name="operator-leg-5GMzsPiUhy6g", host="203.98.89.164",
                             port=30301, workdir=str(work), isolated=True)


def _dispatch(disp, host, *, contract_block: int):
    """The live re-attach call (lane B expert leg, operator-billed pod)."""
    return disp.dispatch(
        host, gen_ref=GEN, uid=171, hotkey=HK, role="expert", base_seed=SEED, block=9596663,
        arch_preset="cascade-moe-v2-120m", repo_suffix="-xenergy-u171", warm_start_ref=INIT,
        lane_count=1, local_checkpoint=True, contract_block=contract_block, domain="energy",
        expert_selection=SEL)


# ── 1. the finished run is found and harvested ───────────────────────────────


def test_live_identity_hash_drifts_with_the_era_contract_block(tmp_path):
    """The root cause, pinned: the run dir's hash is the continuation key of
    the dispatch-era identity (contract block 9594000); the current era's
    9597600 hashes elsewhere."""
    import hashlib

    with torch_blocked():
        from cascade.trainer.continuation import canonical

    def key(cb):
        return hashlib.sha256(canonical({
            "gen_ref": GEN, "hotkey": HK, "role": "expert", "seed": SEED,
            "arch": "cascade-moe-v2-120m", "init": INIT, "contract_block": cb,
            "experts": SEL, "domain": "energy", "recipe": None, "suffix": "-xenergy-u171",
            "train_hours": None, "router_slow": False, "anneal": False})).hexdigest()[:12]

    assert RUN.split("-")[3] == key(9594000) == "cca9bc324c19"
    assert key(9597600) != key(9594000)


def test_reattach_harvests_the_finished_expert_run_on_the_live_dispatch_dir(tmp_path,
                                                                             monkeypatch):
    with torch_blocked():
        from cascade.trainer import leg_reattach, remote
    work = _pod(tmp_path)
    run, calls = _local_pod(monkeypatch)
    monkeypatch.setattr(remote, "run_ssh", run)
    disp = remote.RemoteDispatcher(trainer_spec="cascade.trainer.toto2_trainer:Toto2Trainer",
                                   detached=True, poll_seconds=1,
                                   harvest=lambda h, e, **k: e,
                                   continuation_root=str(tmp_path / "continuations"),
                                   _runner=run)
    # the restart re-derived the CURRENT era's contract block (the live bug)
    go, torn, harvested = _operator_runner(monkeypatch, tmp_path, work, None, disp=disp,
                                           contract_block=9597600)
    with leg_reattach.reattach_only():
        # the restart re-derived the CURRENT era's contract block (the live bug)
        kind, out = _outcome(go)
    assert kind == "ok", f"finished run not re-attached: {out!r}"
    assert [(r.checkpoint_dir, r.miner_hotkey, sfx) for r, sfx in harvested] == \
        [(CKPT, HK, "-xenergy-u171")]                           # harvested, not requeued
    assert not any("setsid nohup" in c for c in calls)          # nothing relaunched
    assert (work / "_train_work" / "_dispatch" / RUN / "exit_code").exists()
    assert (work / CKPT / "weights.safetensors").exists()       # the pod was not touched


def test_reattach_still_refuses_when_the_pod_holds_no_run_of_the_leg(tmp_path, monkeypatch):
    with torch_blocked():
        from cascade.trainer import leg_reattach, remote
    work = _pod(tmp_path, finished=False)
    run, calls = _local_pod(monkeypatch)
    disp = remote.RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=1,
                                   harvest=lambda h, e, **k: e,
                                   continuation_root=str(tmp_path / "c"), _runner=run)
    with leg_reattach.reattach_only():
        kind, e = _outcome(lambda: _dispatch(disp, _host(work, remote),
                                             contract_block=9597600))
    assert kind == "raised" and isinstance(e, remote.ReattachMissing), repr(e)
    assert not any("setsid nohup" in c for c in calls)


def test_reattach_refuses_another_legs_finished_run(tmp_path, monkeypatch):
    """A run found by its stable name whose receipt is another leg's (another
    domain suffix of the same miner) is refused, never harvested as this one."""
    with torch_blocked():
        from cascade.trainer import leg_reattach, remote
    work = _pod(tmp_path)
    run, _ = _local_pod(monkeypatch)
    harvested = []
    disp = remote.RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=1,
                                   harvest=lambda h, e, **k: harvested.append(e) or e,
                                   continuation_root=str(tmp_path / "c"), _runner=run)
    with leg_reattach.reattach_only():
        kind, e = _outcome(lambda: disp.dispatch(
            _host(work, remote), gen_ref=GEN, uid=171, hotkey=HK, role="expert",
            base_seed=SEED, block=9596663, repo_suffix="-xsolar-u171", local_checkpoint=True,
            contract_block=9597600, domain="solar", expert_selection=SEL))
    assert kind == "raised" and getattr(remote, "ReattachMissing", None) is not None
    assert isinstance(e, remote.ReattachMissing) and harvested == []


def test_lane_b_reattach_rederives_the_dispatch_era(tmp_path):
    """``_dispatch_era``: a leg dispatched at block 9596663 (era 2665, start
    9594000) re-attached in era 2666 keeps era 2665's start block / index."""
    with torch_blocked():
        from cascade.shared.config.rounds import RoundConfig
        from cascade.trainer import lane_b_driver
    fn = getattr(lane_b_driver, "_dispatch_era", None)
    assert fn is not None, "the re-attach re-derives the leg in the CURRENT era"
    rnd = RoundConfig(epoch_blocks=900, era_settlements=4)
    cfg = SimpleNamespace(round=rnd)
    current = SimpleNamespace(index=2666, start_block=9597600, seed_block=9507600)
    era = fn(cfg, current, 9596663, {"era_index": 2665})
    assert (era.index, era.start_block, era.seed_block) == (2665, 9594000, 9507600)
    assert fn(cfg, current, 9598000, {}) is current            # same era: unchanged
    assert fn(cfg, current, 0, {}) is current                  # unknown block: unchanged


# ── 1b. a pod holding a finished run is never removed without a harvest ──────


def _operator_runner(monkeypatch, tmp_path, work, failure, *, disp=None,
                     contract_block: int = 9594000):
    """A runner for ``_run_operator_rented_leg`` whose dispatch fails with
    ``failure`` on an adopted pod at ``work``."""
    with torch_blocked():
        from cascade.trainer import (
            harvest_guard,
            harvest_release,
            leg_pins,
            money_guard,
            progress,
            remote,
            runner_funded_legs,
            runner_operator_pods,
        )
    import contextlib

    monkeypatch.setattr(money_guard, "operator_leg_allocated", lambda *a, **k: True)
    monkeypatch.setattr(leg_pins, "leg_contract_scope", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(runner_operator_pods, "leg_corpus", lambda *a, **k: {})
    monkeypatch.setattr(harvest_release, "arm_release", lambda *a, **k: None)
    monkeypatch.setattr(harvest_release, "disarm_release", lambda *a, **k: None)
    monkeypatch.setattr(progress, "report_progress", lambda *a, **k: None)
    monkeypatch.setattr(runner_funded_legs, "_with_leg_watchdog", lambda r, d, *a, **k: d)
    monkeypatch.setattr(harvest_guard, "handle_our_side_fault", lambda *a, **k: False)
    monkeypatch.setattr(runner_operator_pods, "classify_leg_exception",
                        lambda *a, **k: (False, "no_capacity", False))
    host = _host(work, remote)
    torn: list[str] = []
    failures: list[str] = []
    harvested: list = []

    class Disp:
        def dispatch(self, host, **kw):
            raise failure

    r = SimpleNamespace(
        _leg_local_obj=SimpleNamespace(),
        _rent_operator_leg_host=lambda rid, gen: (host, "cascade-n91-x-funded-5gmzsp-op-0"),
        _stage_vault_zip_on=lambda h, d: h,
        _teardown_operator_pod=lambda pod: torn.append(pod.instance_id),
        _load_funded_ledger=lambda: [SimpleNamespace(instance_id="cascade-n91-x-funded-5gmzsp-op-0",
                                                     provider="lium")],
        _funded_queue=lambda: None,
        _record_funded_failure=lambda hk, msg, **k: failures.append(msg),
        _harvest_funded_checkpoint=lambda h, receipt, c, s, sfx: harvested.append(
            (receipt, sfx)) or "entry",
    )
    r.__dict__["_leg_local_obj"] = SimpleNamespace()
    gen = SimpleNamespace(hotkey=HK, uid=171, ref=GEN)
    seeds = SimpleNamespace(base_seed=SEED)
    contract = SimpleNamespace(arch_preset="cascade-moe-v2-120m")

    def go():
        return runner_operator_pods._OperatorPodMixin._run_operator_rented_leg(
            r, disp or Disp(), gen, seeds, 9596663, contract, "-xenergy-u171",
            warm_start_ref=INIT, contract_block=contract_block, role="expert", domain="energy",
            expert_selection=SEL)
    return go, torn, harvested


def test_a_failed_leg_keeps_a_pod_holding_a_finished_unharvested_run(tmp_path, monkeypatch):
    with torch_blocked():
        from cascade.trainer import remote
    run, _ = _local_pod(monkeypatch)
    from cascade.trainer import runner_operator_pods

    monkeypatch.setattr(remote, "run_ssh", run)
    work = _pod(tmp_path)
    go, torn, _ = _operator_runner(monkeypatch, tmp_path, work, remote.ReattachMissing(
        "re-attach refused: no live or finished run of this expert leg"))
    kind, e = _outcome(go)
    assert kind == "raised" and isinstance(e, remote.ReattachMissing)
    assert torn == [], "the pod holding the finished run was removed"
    held = getattr(runner_operator_pods, "unharvested_finished_run", None)
    assert held is not None and RUN in held(SimpleNamespace(), _host(work, remote))


def test_a_failed_leg_on_a_pod_without_a_finished_run_is_still_torn_down(tmp_path, monkeypatch):
    with torch_blocked():
        from cascade.trainer import remote
    run, _ = _local_pod(monkeypatch)
    monkeypatch.setattr(remote, "run_ssh", run)
    work = _pod(tmp_path, finished=False)
    go, torn, _ = _operator_runner(monkeypatch, tmp_path, work,
                                   remote.RemoteDispatchError("remote expert failed (rc=1)"))
    kind, _ = _outcome(go)
    assert kind == "raised" and torn == ["cascade-n91-x-funded-5gmzsp-op-0"]


def test_a_requeued_fresh_dispatch_harvests_the_finished_run_instead_of_retraining(
        tmp_path, monkeypatch):
    """The requeued leg adopts the kept pod and dispatches afresh (not attach
    only): a FINISHED run of this leg (its receipt names this leg's
    checkpoint dir) is harvested, never trained over."""
    with torch_blocked():
        from cascade.trainer import remote
    work = _pod(tmp_path)
    run, calls = _local_pod(monkeypatch)
    disp = remote.RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=1,
                                   harvest=lambda h, e, **k: e,
                                   continuation_root=str(tmp_path / "c"), _runner=run)
    kind, out = _outcome(lambda: _dispatch(disp, _host(work, remote), contract_block=9597600))
    assert kind == "ok" and getattr(out, "checkpoint_dir", "") == CKPT, repr(out)
    assert not any("setsid nohup" in c for c in calls)


# ── 2. cache dirs on the large data mount ────────────────────────────────────


@pytest.fixture
def two_filesystems(tmp_path):
    """``small``: a dir on /dev/shm (tmpfs); ``big``: a dir on tmp_path's
    filesystem — two different filesystems, ``big`` with more free space."""
    if not os.path.isdir("/dev/shm"):
        pytest.skip("no /dev/shm")
    small = Path(tempfile.mkdtemp(prefix="wb27-small-", dir="/dev/shm"))
    big = tmp_path / "workspace"
    big.mkdir()
    if os.stat(small).st_dev == os.stat(big).st_dev or \
            shutil.disk_usage(big).free <= shutil.disk_usage(small).free:
        shutil.rmtree(small, ignore_errors=True)
        pytest.skip("no second, larger filesystem here")
    try:
        yield small, big
    finally:
        shutil.rmtree(small, ignore_errors=True)


def _pdr():
    import importlib

    try:
        return importlib.import_module("cascade.shared.pod_data_root")
    except ImportError:
        raise AssertionError("no data-mount placement for the pod cache dirs") from None


def _sh(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30)


def test_offload_dir_is_linked_onto_the_larger_data_mount(two_filesystems):
    small, big = two_filesystems
    with torch_blocked():
        pdr = _pdr()
    link = small / "_lane_b_offload"
    assert _sh(pdr.place_on_data_mount_command(str(link), candidates=(str(big),))).returncode == 0
    assert link.is_symlink() and Path(os.readlink(link)) == big / "_lane_b_offload"
    # idempotent, and a non-empty real dir is never moved
    assert _sh(pdr.place_on_data_mount_command(str(link), candidates=(str(big),))).returncode == 0
    other = small / "_eval_offload"
    (other / "cache").mkdir(parents=True)
    _sh(pdr.place_on_data_mount_command(str(other), candidates=(str(big),)))
    assert other.is_dir() and not other.is_symlink() and (other / "cache").is_dir()
    # an EMPTY real dir is replaced; a same-filesystem candidate never links
    empty = small / "_private_eval_offload"
    empty.mkdir()
    _sh(pdr.place_on_data_mount_command(str(empty), candidates=(str(big),)))
    assert empty.is_symlink()
    same = big / "w" / "_lane_b_offload"
    _sh(pdr.place_on_data_mount_command(str(same), candidates=(str(big),)))
    assert not same.exists()


def test_lane_b_runner_places_its_root_before_any_use(monkeypatch):
    """The runner's first pod command puts ``<workdir>/_lane_b_offload`` on
    the data mount (no explicit ``cache_root``); with ``cache_root`` set the
    root is that mount and nothing is linked."""
    with torch_blocked():
        from cascade.trainer import lane_b_offload as off
        from cascade.trainer.remote import RemoteHost
    seen: list[str] = []

    def run(argv, timeout, stdin=""):
        seen.append(argv[-1])
        raise RuntimeError("stop after the first command")

    for host, linked in ((RemoteHost(name="valeval-blackwell4", host="h"), True),
                         (RemoteHost(name="v", host="h", cache_root="/workspace"), False)):
        seen.clear()
        r = off.LaneBPodRunner(lambda h=host: h, run=run, bundle=lambda: ("sha", b""))
        with pytest.raises(off.LaneBOffloadUnavailable):
            r._call("probe", "ptr", {}, None)
        first = seen[0]
        assert ("ln -sfn" in first and "/workspace" in first) is linked, first
        assert off.LaneBPodRunner.root_of(host) in first


def test_disk_short_names_the_alternative_mount(two_filesystems):
    small, big = two_filesystems
    with torch_blocked():
        pdr = _pdr()
        from cascade.trainer import lane_b_offload as off
    ckpt = small / "_lane_b_offload" / "ckpt"
    import unittest.mock as um

    with um.patch.object(pdr, "DATA_MOUNT_CANDIDATES", (str(big),)), \
            um.patch.object(off, "local_alternatives", create=True, new=
                            lambda p, **k: pdr.local_alternatives(p, candidates=(str(big),))):
        kind, e = _outcome(lambda: off._ensure_room(ckpt, 1 << 60, set(), what="the rebase"))
    assert kind == "raised" and isinstance(e, off.PodDiskShort)
    assert str(big) in str(e) and f'cache_root = "{big}"' in str(e), str(e)


def test_eval_offload_names_the_alternative_mount_and_places_its_dirs(two_filesystems, caplog):
    small, big = two_filesystems
    with torch_blocked():
        pdr = _pdr()
        from cascade.trainer.remote import RemoteHost
        from cascade.validator import eval_offload as eo
    cache = small / "_eval_offload" / "cache"
    cache.mkdir(parents=True)
    cmd = eo.bench_make_room_command(str(cache), [], 1 << 60).replace(
        pdr.alternative_mounts_command("."),
        pdr.alternative_mounts_command(".", candidates=(str(big),)))
    proc = _sh(cmd)
    assert eo.bench_free_bytes(proc.stdout) is not None
    host = RemoteHost(name="valeval-blackwell4", host="h")
    with caplog.at_level("ERROR"):
        assert eo._room_ok(host, proc, 1 << 60, "the benchmark") is False
    assert f'cache_root = "{big}"' in caplog.text, caplog.text
    # the bench prep links <workdir>/_eval_offload; an explicit cache_root roots it
    assert "ln -sfn" in eo._placement(host, "/root/cascade/_eval_offload")
    assert eo._placement(RemoteHost(name="v", host="h", cache_root="/workspace"),
                         "/workspace/_eval_offload") == ""


def test_bench_prep_runs_the_placement_and_finds_through_the_link(monkeypatch):
    with torch_blocked():
        from cascade.trainer.remote import RemoteHost
        from cascade.validator import eval_offload as eo
    seen: list[str] = []

    def run(argv, timeout, stdin=None):
        seen.append(argv[-1])
        return SimpleNamespace(returncode=1, stdout="", stderr="stop")

    eo.run_bench_via_host_many(RemoteHost(name="v", host="h", workdir="/root/cascade"),
                               ["/tmp/x/ckpt"], suites="gift-eval", runner=run)
    assert seen and "ln -sfn" in seen[0] and "/root/cascade/_eval_offload/ " in seen[0]
    host = RemoteHost(name="v", host="h", workdir="/root/cascade", cache_root="/workspace")
    seen.clear()
    eo.run_bench_via_host_many(host, ["/tmp/x/ckpt"], suites="gift-eval", runner=run)
    assert "/workspace/_eval_offload/ckpt-" in seen[0] and "ln -sfn" not in seen[0]
