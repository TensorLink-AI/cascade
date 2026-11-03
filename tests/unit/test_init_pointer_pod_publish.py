"""The DEC-CA-0055 init entry on the trainer (2026-11-01 incident: an operator
8xH100 rented at era start purely to bench the init, which the main validator
benches itself under DEC-CA-0063).

* validator-sourced era: no init bench, no rent — only the random-init
  POINTER, built on the era's already-mapped operator king pod and harvested
  as plain files; no pod ⇒ deferred, never rented;
* trainer-sourced (legacy) era: benched as before, but the random init is
  built ON the pod, never on the trainer box;
* the retry cap is per seed epoch and logs once when it gives up."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.shared.bench_report import SIX_SCORE_KEYS
from cascade.trainer import runner_rolling as RR
from cascade.trainer.rolling import EraState, RollingScheduler

from .test_rolling import EB, Clock, FakeOps, _armed, _sched

INIT_PTR = "metro-v1:trained:hippius:cascade/ckpt-4242-moe-random-init@sha256:" + "e" * 64
WARM_PTR = "metro-v1:trained:hippius:cascade/member@sha256:" + "f" * 64
KING_PTR = "metro-v1:trained:hippius:cascade/king@sha256:" + "b" * 64
MOE = SimpleNamespace(arch_preset="cascade-moe-v2-120m", base_arch="cascade_moe")
SIX = {k: 0.5 for k in SIX_SCORE_KEYS}


def _vsrc(cfg):
    """Every era validator-sourced (DEC-CA-0063)."""
    return replace(cfg, v2=replace(cfg.v2, bench_source="validator", bench_source_from_block=1))


def _gated(cfg):
    armed = _armed(cfg)
    return replace(armed, scoring=replace(armed.scoring, bench_gate_mode="shadow"))


def _era(**kw):
    king = {"miner_hotkey": "KING", "miner_uid": 1, "role": "king", "gen_ref": "g",
            "trained_pointer": KING_PTR, "corpus_digest": "d", "train_block": 1}
    return EraState(index=12, start_block=12 * 4 * EB, seed_block=11 * 4 * EB, base_seed=4242,
                    epoch_start=12, king_entry=king, king_hotkey="KING", **kw)


class _Runner(RR._RollingMixin):
    def __init__(self, cfg, tmp_path):
        self.cfg = cfg
        self.work_root = tmp_path
        self.cascade_bench_plan = object()
        self._final_role_hosts = {}
        self._funded_king_lock = threading.Lock()
        self._funded_king_host = None
        self._rolling_king_host_era = None
        self._funded_wait_abort = threading.Event()
        self.rents: list[str] = []
        self.builds: list = []
        self.published: list = []
        self.ledger: list = []

    def _load_funded_ledger(self):
        return list(self.ledger)

    def _rent_king_host(self, round_id):
        self.rents.append(round_id)
        host = SimpleNamespace(name="rented-king")
        self._funded_king_host = host
        return host

    def _fetch_checkpoint_dir(self, pointer):
        return f"/local/{pointer}"

    def _rolling_build_init_on_pod(self, host, era, contract):
        self.builds.append(host)
        return "/pod/_train_work/4242/cascade-moe-v2-120m/init/checkpoint"

    def _rolling_harvest_publish_init(self, host, remote_dir, era, contract):
        self.published.append((host, remote_dir))
        return INIT_PTR

    def _stash_bench_rows(self, pointer, report):
        pass


@pytest.fixture
def wired(monkeypatch):
    """No sweep may run and nothing may be built on the box unless a test
    re-wires it; the era's contract is the MoE."""
    from cascade.trainer import bench_hook, moe_trainer

    monkeypatch.setattr(RR, "era_contracts", lambda cfg, era: [MOE])
    monkeypatch.setattr(RR, "host_cannot_bench", lambda runner, host: "")
    monkeypatch.setattr(moe_trainer, "build_random_init",
                        lambda *a, **k: pytest.fail("random init built on the trainer box"))
    monkeypatch.setattr(bench_hook, "run_post_round_benchmark",
                        lambda *a, **k: pytest.fail("init benched by the trainer"))
    monkeypatch.setattr(bench_hook, "push_checkpoint",
                        lambda *a, **k: pytest.fail("checkpoint pushed for an init bench"))
    return monkeypatch


# ── validator-sourced era ────────────────────────────────────────────────────

def test_validator_era_without_a_king_pod_rents_nothing_and_defers(cfg, tmp_path, wired,
                                                                   caplog):
    r = _Runner(_vsrc(_gated(cfg)), tmp_path)
    with caplog.at_level(logging.WARNING, logger="cascade.trainer"):
        out = r._rolling_bench_init(_era())
        again = r._rolling_bench_init(_era())
    assert "_deferred" in out and "_deferred" in again
    assert r.rents == [] and r.builds == [] and r.published == []
    assert caplog.text.count("NOT renting") == 1          # logged once per era
    assert "4242" in r._init_pointer_wanted                # parked for an operator pod


def test_validator_era_publishes_the_pointer_from_the_mapped_king_pod(cfg, tmp_path, wired):
    r = _Runner(_vsrc(_gated(cfg)), tmp_path)
    pod = SimpleNamespace(name="era-king-pod")
    r._rolling_king_hosts = {12: pod}
    out = r._rolling_bench_init(_era())
    assert r.rents == []
    assert r.builds == [pod]                                 # built on the pod
    assert r.published == [(pod, "/pod/_train_work/4242/cascade-moe-v2-120m/init/checkpoint")]
    assert out == {**RR.pointer_only_init_scores(), "_pointer": INIT_PTR}
    # pointer-only: sentinel scores no CRPS / MASE can take
    assert all(v < 0 for k, v in out.items() if k != "_pointer")


def test_validator_era_never_builds_on_a_payer_pod(cfg, tmp_path, wired):
    r = _Runner(_vsrc(_gated(cfg)), tmp_path)
    r._final_role_hosts[("king", MOE.arch_preset, "KING")] = SimpleNamespace(
        name="payer", isolated=True, allocation_id="payer-pod-0")
    r.ledger = [SimpleNamespace(instance_id="payer-pod-0", payer_hotkey="KING")]
    assert "_deferred" in r._rolling_bench_init(_era())
    assert r.rents == [] and r.builds == []


def test_validator_era_warm_start_needs_nothing(cfg, tmp_path, wired):
    r = _Runner(_vsrc(_gated(cfg)), tmp_path)
    r._rolling_king_hosts = {12: SimpleNamespace(name="era-king-pod")}
    assert r._rolling_bench_init(_era(warm_start_ckpt=WARM_PTR)) is None
    assert r.rents == [] and r.builds == []


def test_validator_era_reuses_the_epochs_published_pointer(cfg, tmp_path, wired):
    r = _Runner(_vsrc(_gated(cfg)), tmp_path)
    out = r._rolling_bench_init(_era(init_pointer=INIT_PTR))
    assert out["_pointer"] == INIT_PTR
    assert r.rents == [] and r.builds == []


def test_scheduler_owes_nothing_for_a_warm_started_validator_era(cfg):
    v = _vsrc(cfg)
    sched = SimpleNamespace(cfg=v, _ladder_era=lambda era: True,
                            INIT_BENCH_MAX_TRIES=RollingScheduler.INIT_BENCH_MAX_TRIES)
    assert RollingScheduler._init_bench_owed(sched, _era(warm_start_ckpt=WARM_PTR)) is False
    assert RollingScheduler._init_bench_owed(sched, _era()) is True       # random init
    legacy = SimpleNamespace(cfg=cfg, _ladder_era=lambda era: True,
                             INIT_BENCH_MAX_TRIES=RollingScheduler.INIT_BENCH_MAX_TRIES)
    assert RollingScheduler._init_bench_owed(legacy, _era(warm_start_ckpt=WARM_PTR)) is True


# ── trainer-sourced (legacy) era ────────────────────────────────────────────

def test_legacy_era_still_benches_but_builds_the_init_on_the_pod(cfg, tmp_path, wired):
    from cascade.eval import benchmarks
    from cascade.trainer import bench_hook

    swept = []
    # the legacy bench-host rent pushes the king checkpoint first (unchanged)
    wired.setattr(bench_hook, "push_checkpoint", lambda *a, **k: None)
    wired.setattr(bench_hook, "run_post_round_benchmark",
                  lambda host, rid, preset, plan, **kw: swept.append((host, kw["role"])) or {})
    wired.setattr(benchmarks, "extract_bench_scores", lambda report, **kw: dict(SIX))
    r = _Runner(_gated(cfg), tmp_path)
    out = r._rolling_bench_init(_era())
    assert r.rents == ["4242"]                       # the legacy bench-host rent, unchanged
    host = r._rolling_king_hosts[12]
    assert r.builds == [host] and swept == [(host, "init")]
    assert out["_pointer"] == INIT_PTR and {k: out[k] for k in SIX} == SIX


def test_legacy_warm_start_pushes_the_member_and_builds_nothing(cfg, tmp_path, wired):
    from cascade.eval import benchmarks
    from cascade.trainer import bench_hook

    pushed = []
    wired.setattr(bench_hook, "push_checkpoint",
                  lambda host, local, rid, preset, role, **kw: pushed.append((local, role)))
    wired.setattr(bench_hook, "run_post_round_benchmark", lambda *a, **k: {})
    wired.setattr(benchmarks, "extract_bench_scores", lambda report, **kw: dict(SIX))
    r = _Runner(_gated(cfg), tmp_path)
    r._rolling_king_hosts = {12: SimpleNamespace(name="era-king-pod")}
    out = r._rolling_bench_init(_era(warm_start_ckpt=WARM_PTR))
    assert "_pointer" not in out and {k: out[k] for k in SIX} == SIX
    assert pushed == [(f"/local/{WARM_PTR}", "init")]
    assert r.builds == [] and r.published == [] and r.rents == []


# ── the pod-side build and the harvest ──────────────────────────────────────

def test_init_build_command_runs_on_the_pod_cpu_only():
    host = SimpleNamespace(remote_python="/opt/venv/bin/python", workdir="/root/cascade",
                           chain_toml="/root/cascade/chain.deployed.toml")
    cmd = RR.build_init_remote_command(host, "/root/cascade/_train_work/4242/p/init/checkpoint",
                                       contract_block=480, training_seed=99,
                                       arch_preset="p", digest="abc123")
    assert cmd.startswith("cd /root/cascade && rm -rf /root/cascade/_train_work/4242/p/init/"
                          "checkpoint && CUDA_VISIBLE_DEVICES= /opt/venv/bin/python -c ")
    for arg in ("/root/cascade/chain.deployed.toml", " 480 ", " 99 ", " abc123 "):
        assert arg in cmd
    assert "build_random_init" in cmd and "init build refused" in cmd


def test_build_on_pod_ssh_and_failure(cfg, tmp_path, monkeypatch):
    from cascade.trainer import remote

    monkeypatch.setattr(RR, "era_contracts", lambda cfg, era: [MOE])
    from cascade.shared import manifest

    monkeypatch.setattr(manifest, "contract_digest", lambda c: "dig")
    seen = []
    rc = {"v": 0}
    monkeypatch.setattr(remote, "build_ssh_argv", lambda host, cmd: ["ssh", cmd])
    monkeypatch.setattr(remote, "run_ssh", lambda argv, timeout: seen.append(argv) or
                        SimpleNamespace(returncode=rc["v"], stderr="boom", stdout=""))
    host = SimpleNamespace(name="pod", remote_python="python", workdir="/w", chain_toml=None)
    r = RR._RollingMixin.__new__(RR._RollingMixin)
    r.cfg = cfg
    out = RR._RollingMixin._rolling_build_init_on_pod(r, host, _era(), MOE)
    assert out == "/w/_train_work/4242/cascade-moe-v2-120m/init/checkpoint"
    assert " dig " in seen[0][1]
    rc["v"] = 1
    with pytest.raises(RuntimeError, match="random-init build on pod failed"):
        RR._RollingMixin._rolling_build_init_on_pod(r, host, _era(), MOE)


def test_harvest_publish_verifies_the_copy_and_drops_it(tmp_path, monkeypatch):
    from cascade.trainer import remote

    body = b"weights"
    pod_listing = {"weights.safetensors": (len(body), hashlib.sha256(body).hexdigest())}

    def harvest(host, remote_dir, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "weights.safetensors").write_bytes(body)

    monkeypatch.setattr(remote, "remote_file_manifest", lambda host, d: dict(pod_listing))
    monkeypatch.setattr(remote, "harvest_remote_dir", harvest)
    seen = []
    r = SimpleNamespace(work_root=tmp_path,
                        _publish_init_checkpoint=lambda local, era, c: seen.append(
                            sorted(p.name for p in Path(local).iterdir())) or INIT_PTR)
    ptr = RR._RollingMixin._rolling_harvest_publish_init(r, "host", "/pod/init", _era(), MOE)
    assert ptr == INIT_PTR and seen == [["weights.safetensors"]]
    assert not (tmp_path / "init_publish" / "4242").exists()     # no box copy kept
    pod_listing["weights.safetensors"] = (len(body), "0" * 64)   # a corrupted copy
    with pytest.raises(RuntimeError, match="differs"):
        RR._RollingMixin._rolling_harvest_publish_init(r, "host", "/pod/init", _era(), MOE)
    assert len(seen) == 1                                        # never uploaded


# ── the retry cap ───────────────────────────────────────────────────────────

class _EmptyInitOps(FakeOps):
    def __init__(self, tmp_path, clock):
        super().__init__(tmp_path, clock)
        self.init_calls = 0

    def bench_init(self, era):
        self.init_calls += 1
        return None


def test_init_retries_are_capped_and_give_up_once(cfg, tmp_path, caplog):
    clock = Clock()
    sched, ops = _sched(_armed(cfg), tmp_path, clock, ops=_EmptyInitOps(tmp_path, clock))
    sched._ladder_era = lambda era: True
    era = _era()
    with caplog.at_level(logging.WARNING, logger="cascade.trainer.rolling"):
        for _ in range(6):
            clock.t += sched.INIT_BENCH_SECONDS + 1
            sched._maybe_bench_init(era, clock.t)
            end = time.time() + 10
            while sched._init_threads and time.time() < end:
                time.sleep(0.01)
    assert ops.init_calls == sched.INIT_BENCH_MAX_TRIES
    assert era.init_bench_tries == sched.INIT_BENCH_MAX_TRIES
    assert caplog.text.count("GAVE UP") == 1


def test_the_tries_cap_carries_across_the_seed_epoch():
    src, dst = _era(init_bench_tries=3), _era()
    dst.index = 13
    fake = SimpleNamespace(_same_epoch=lambda a, b: True)
    RollingScheduler._carry_epoch_king(fake, src, dst)
    assert dst.init_bench_tries == 3 and dst.init_bench is None
    # a landed pointer still carries as before
    src2 = _era(init_bench=dict(SIX), init_bench_tries=1, init_pointer=INIT_PTR)
    dst2 = _era()
    RollingScheduler._carry_epoch_king(fake, src2, dst2)
    assert dst2.init_pointer == INIT_PTR and dst2.init_bench == SIX


# ── operator-billed pods only (owner follow-up) ─────────────────────────────

def _op_pod(name="op-king", aid="op-pod-0"):
    """A credential-free (isolated) pod — operator king / leg pods are too."""
    return SimpleNamespace(name=name, isolated=True, allocation_id=aid)


def test_isolated_operator_king_pod_is_used(cfg, tmp_path, wired):
    r = _Runner(_vsrc(_gated(cfg)), tmp_path)
    pod = _op_pod()
    r.ledger = [SimpleNamespace(instance_id="op-pod-0", payer_hotkey="")]
    r._rolling_king_hosts = {12: pod}
    assert r._rolling_bench_init(_era())["_pointer"] == INIT_PTR
    assert r.builds == [pod] and r.rents == []


def _lanes(*hosts):
    import queue

    q = queue.Queue()
    for h in hosts:
        q.put(h)
    return q


def test_an_idle_operator_lane_builds_it_and_is_handed_back(cfg, tmp_path, wired):
    r = _Runner(_vsrc(_gated(cfg)), tmp_path)
    a, b = SimpleNamespace(name="lane-a"), SimpleNamespace(name="lane-b")
    r._rolling_lane_pool_obj = pool = _lanes(a, b)
    assert r._rolling_bench_init(_era())["_pointer"] == INIT_PTR
    assert r.builds == [a] and r.rents == []
    assert pool.qsize() == 2                                # handed back


def test_busy_last_or_payer_lanes_are_never_taken(cfg, tmp_path, wired):
    for hosts in ((),                                       # every lane busy
                  (SimpleNamespace(name="lane-a"),),        # the last idle lane
                  (_op_pod("payer-1", "p1"), _op_pod("payer-2", "p2"))):   # payer pods
        r = _Runner(_vsrc(_gated(cfg)), tmp_path)
        r.ledger = [SimpleNamespace(instance_id="p1", payer_hotkey="M1"),
                    SimpleNamespace(instance_id="p2", payer_hotkey="M2")]
        r._rolling_lane_pool_obj = pool = _lanes(*hosts)
        assert "_deferred" in r._rolling_bench_init(_era())
        assert r.builds == [] and r.rents == [] and pool.qsize() == len(hosts)


def test_a_parked_init_is_built_on_the_next_operator_leg_pod_after_its_leg(cfg, tmp_path,
                                                                          wired):
    r = _Runner(_vsrc(_gated(cfg)), tmp_path)
    assert "_deferred" in r._rolling_bench_init(_era())
    torn = []
    payer = _op_pod("payer-leg", "pl")
    r.ledger = [SimpleNamespace(instance_id="pl", payer_hotkey="MINER"),
                SimpleNamespace(instance_id="ol", payer_hotkey="")]
    # a payer pod or another seed epoch's pod never claims it
    assert r._rolling_init_on_leg_pod(payer, 4242, lambda: torn.append("p")) is False
    assert r._rolling_init_on_leg_pod(_op_pod("other", "ol"), 999,
                                      lambda: torn.append("x")) is False
    leg = _op_pod("op-leg", "ol")
    assert r._rolling_init_on_leg_pod(leg, 4242, lambda: torn.append("leg")) is True
    end = time.time() + 10
    while torn != ["leg"] and time.time() < end:
        time.sleep(0.01)
    assert torn == ["leg"]                       # torn down by the build thread, once
    assert r.builds == [leg] and r.rents == []
    out = r._rolling_bench_init(_era())          # the next scheduler poll picks it up
    assert out["_pointer"] == INIT_PTR and r.builds == [leg]


def test_a_deferral_spends_no_try(cfg, tmp_path):
    clock = Clock()

    class _Parked(FakeOps):
        calls = 0

        def bench_init(self, era):
            type(self).calls += 1
            return {"_deferred": "no operator pod"}

    sched, ops = _sched(_armed(cfg), tmp_path, clock, ops=_Parked(tmp_path, clock))
    sched._ladder_era = lambda era: True
    era = _era()
    for _ in range(5):
        clock.t += sched.INIT_BENCH_SECONDS + 1
        sched._maybe_bench_init(era, clock.t)
        end = time.time() + 10
        while sched._init_threads and time.time() < end:
            time.sleep(0.01)
    assert type(ops).calls == 5 and era.init_bench_tries == 0 and era.init_bench is None


# The owner-approval gate (2026-11-02) is tested in test_money_safety.py;
# these tests drive the operator-lane / operator-rent machinery behind it.
pytestmark = pytest.mark.usefixtures("operator_legs_preapproved")
