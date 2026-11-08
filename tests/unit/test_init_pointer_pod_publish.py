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
import json
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
    def fake(c):
        ns = SimpleNamespace(cfg=c, _ladder_era=lambda era: True,
                             INIT_BENCH_MAX_TRIES=RollingScheduler.INIT_BENCH_MAX_TRIES)
        ns._init_bench_wanted = lambda era: RollingScheduler._init_bench_wanted(ns, era)
        return ns

    sched = fake(_vsrc(cfg))
    assert RollingScheduler._init_bench_owed(sched, _era(warm_start_ckpt=WARM_PTR)) is False
    assert RollingScheduler._init_bench_owed(sched, _era()) is True       # random init
    # past the cap: still wanted (slow retries), no longer owed (no pod held)
    assert RollingScheduler._init_bench_owed(sched, _era(init_bench_tries=3)) is False
    assert RollingScheduler._init_bench_wanted(sched, _era(init_bench_tries=3)) is True
    legacy = fake(cfg)
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

    def harvest(host, remote_dir, dest, **kw):
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


def _tick_init(sched, era, clock, step):
    clock.t += step + 1
    sched._maybe_bench_init(era, clock.t)
    end = time.time() + 10
    while sched._init_threads and time.time() < end:
        time.sleep(0.01)


def test_init_retries_slow_down_after_the_cap_but_never_stop(cfg, tmp_path, caplog):
    """2026-10-31: three ssh timeouts GAVE UP the pointer for the whole seed
    epoch and every king check deferred for days. The cap now only ends the
    fast cadence (and the pod hold); the pointer is retried every
    INIT_BENCH_RETRY_SECONDS until it lands."""
    clock = Clock()
    sched, ops = _sched(_armed(cfg), tmp_path, clock, ops=_EmptyInitOps(tmp_path, clock))
    sched._ladder_era = lambda era: True
    era = _era()
    with caplog.at_level(logging.WARNING, logger="cascade.trainer.rolling"):
        for _ in range(6):
            _tick_init(sched, era, clock, sched.INIT_BENCH_SECONDS)
    # the fast cadence stops at the cap …
    assert ops.init_calls == sched.INIT_BENCH_MAX_TRIES
    assert era.init_bench_tries == sched.INIT_BENCH_MAX_TRIES
    assert caplog.text.count("GAVE UP") == 1
    assert "rebuild-init-pointer" in caplog.text
    # … the pointer is no longer OWED (no pod is held for it) but still WANTED
    assert sched._init_bench_owed(era) is False
    assert sched._init_bench_wanted(era) is True
    # … and the slow cadence keeps trying
    with caplog.at_level(logging.WARNING, logger="cascade.trainer.rolling"):
        _tick_init(sched, era, clock, sched.INIT_BENCH_RETRY_SECONDS)
    assert ops.init_calls == sched.INIT_BENCH_MAX_TRIES + 1
    assert era.init_bench_tries == sched.INIT_BENCH_MAX_TRIES + 1
    assert "slow retry 4 failed" in caplog.text
    assert caplog.text.count("GAVE UP") == 1                 # logged once, at the cap


def test_a_landed_pointer_ends_the_slow_retries(cfg, tmp_path):
    clock = Clock()

    class _LandsLater(_EmptyInitOps):
        def bench_init(self, era):
            self.init_calls += 1
            if self.init_calls <= RollingScheduler.INIT_BENCH_MAX_TRIES:
                return None
            return {**RR.pointer_only_init_scores(), "_pointer": INIT_PTR}

    sched, ops = _sched(_armed(cfg), tmp_path, clock, ops=_LandsLater(tmp_path, clock))
    sched._ladder_era = lambda era: True
    era = _era()
    for _ in range(3):
        _tick_init(sched, era, clock, sched.INIT_BENCH_SECONDS)
    _tick_init(sched, era, clock, sched.INIT_BENCH_RETRY_SECONDS)
    assert era.init_pointer == INIT_PTR and era.init_bench is not None
    assert sched._init_bench_wanted(era) is False
    _tick_init(sched, era, clock, sched.INIT_BENCH_RETRY_SECONDS)
    assert ops.init_calls == 4                              # nothing more once landed


def test_rebuild_request_resets_the_tries_and_retries_now(cfg, tmp_path, caplog):
    from cascade.trainer.operator_legs_cli import INIT_POINTER_REBUILD_FILE

    clock = Clock()
    sched, ops = _sched(_armed(cfg), tmp_path, clock, ops=_EmptyInitOps(tmp_path, clock))
    sched._ladder_era = lambda era: True
    era = _era()
    for _ in range(4):
        _tick_init(sched, era, clock, sched.INIT_BENCH_SECONDS)
    assert ops.init_calls == 3 and era.init_bench_tries == 3
    # a fast tick does nothing while capped …
    _tick_init(sched, era, clock, sched.INIT_BENCH_SECONDS)
    assert ops.init_calls == 3
    # … the owner's request resets the counter and the next fast tick retries
    req = Path(sched.state_path).parent / INIT_POINTER_REBUILD_FILE
    req.write_text(json.dumps({"requested_at": "now", "reason": "ssh fixed"}), encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="cascade.trainer.rolling"):
        _tick_init(sched, era, clock, sched.INIT_BENCH_SECONDS)
    assert not req.exists()                                  # consumed
    assert "operator requested an init pointer rebuild (ssh fixed)" in caplog.text
    assert ops.init_calls == 4 and era.init_bench_tries == 1  # reset to 0, then one failure
    # a malformed request is consumed too, and still resets
    era.init_bench_tries = 3
    req.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="cascade.trainer.rolling"):
        _tick_init(sched, era, clock, sched.INIT_BENCH_SECONDS)
    assert not req.exists() and "unreadable" in caplog.text
    assert ops.init_calls == 5 and era.init_bench_tries == 1


def test_rebuild_request_resets_every_era_of_the_state_that_wants_its_init(cfg, tmp_path):
    from cascade.trainer.operator_legs_cli import INIT_POINTER_REBUILD_FILE

    clock = Clock()
    sched, ops = _sched(_armed(cfg), tmp_path, clock, ops=_EmptyInitOps(tmp_path, clock))
    sched._ladder_era = lambda era: True
    cur, nxt, done = _era(init_bench_tries=3), _era(init_bench_tries=3), _era(
        init_bench=dict(SIX), init_bench_tries=2, init_pointer=INIT_PTR)
    nxt.index, done.index = 13, 14
    sched.state.current, sched.state.next = cur, nxt
    (Path(sched.state_path).parent / INIT_POINTER_REBUILD_FILE).write_text("{}", encoding="utf-8")
    sched._consume_init_pointer_rebuild(done)
    assert cur.init_bench_tries == 0 and nxt.init_bench_tries == 0
    assert done.init_bench_tries == 2                        # landed: untouched
    assert not (Path(sched.state_path).parent / INIT_POINTER_REBUILD_FILE).exists()


def test_rebuild_init_pointer_command_writes_the_request(tmp_path, capsys):
    from cascade.trainer.operator_legs_cli import INIT_POINTER_REBUILD_FILE, main

    assert main(["rebuild-init-pointer", "--work-root", str(tmp_path),
                 "--reason", "ssh timeouts fixed"]) == 0
    body = json.loads((tmp_path / INIT_POINTER_REBUILD_FILE).read_text(encoding="utf-8"))
    assert body["reason"] == "ssh timeouts fixed" and body["requested_at"].endswith("Z")
    assert "rebuild requested" in capsys.readouterr().out
    # no queue file is needed (the trainer's work root may hold none yet)
    assert not (tmp_path / "funded_queue.json").exists()


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


# ── init-only lanes (hosts.toml stage = "init") ──────────────────────────────

def _hosts_file(tmp_path, *entries):
    """hosts.toml with one ``[[host]]`` per ``(name, ip, stage)``."""
    text = "".join(
        f'[[host]]\nname = "{n}"\nhost = "{ip}"\nkey_path = "/k"\n'
        f'remote_python = "/root/cascade/.venv/bin/python"\nworkdir = "/root/cascade"\n'
        f'stage = "{stage}"\n\n' for n, ip, stage in entries)
    p = tmp_path / "hosts.toml"
    p.write_text(text)
    return p


def test_hosts_file_accepts_the_init_stage(tmp_path):
    from cascade.trainer.remote import load_hosts

    (h,) = load_hosts(_hosts_file(tmp_path, ("init-lane", "10.0.0.9", "init")))
    assert h.stage == "init"


def test_an_init_lane_on_file_builds_the_pointer_without_a_pool(cfg, tmp_path, wired):
    r = _Runner(_vsrc(_gated(cfg)), tmp_path)
    r.remote_hosts_path = _hosts_file(tmp_path, ("profile", "127.0.0.1", "final"),
                                      ("init-lane", "10.0.0.9", "init"))
    assert r._rolling_bench_init(_era())["_pointer"] == INIT_PTR
    assert [h.name for h in r.builds] == ["init-lane"] and r.rents == []
    assert "4242" not in r.__dict__.get("_init_pointer_wanted", {})   # no longer parked
    assert r._rolling_final_hosts() == []                             # never a leg lane
    # the second ask reuses the published pointer (no second build)
    assert r._rolling_bench_init(_era())["_pointer"] == INIT_PTR and len(r.builds) == 1


def test_init_lanes_are_read_live_and_blocked_ones_are_skipped(cfg, tmp_path, wired,
                                                               monkeypatch):
    r = _Runner(_vsrc(_gated(cfg)), tmp_path)
    r.remote_hosts_path = _hosts_file(tmp_path, ("profile", "127.0.0.1", "final"))
    assert "_deferred" in r._rolling_bench_init(_era())                # nothing on file: parked
    _hosts_file(tmp_path, ("profile", "127.0.0.1", "final"), ("bad", "10.0.0.1", "init"),
                ("good", "10.0.0.2", "init"))                             # hand-rented later
    monkeypatch.setattr("cascade.trainer.host_preflight.host_blocked_reason",
                        lambda runner, ip: "quarantined" if ip == "10.0.0.1" else "")
    assert r._rolling_bench_init(_era())["_pointer"] == INIT_PTR
    assert [h.name for h in r.builds] == ["good"]


def test_final_lanes_and_profiles_are_not_init_lanes(cfg, tmp_path, wired):
    r = _Runner(_vsrc(_gated(cfg)), tmp_path)
    r.remote_hosts_path = _hosts_file(tmp_path, ("lane-a", "10.0.0.5", "final"),
                                      ("loop", "127.0.0.1", "init"))
    assert "_deferred" in r._rolling_bench_init(_era())
    assert r.builds == [] and r.rents == []


def test_init_lane_gets_the_deployed_chain_toml(cfg, tmp_path, wired):
    r = _Runner(_vsrc(_gated(cfg)), tmp_path)
    r.remote_hosts_path = _hosts_file(tmp_path, ("init-lane", "10.0.0.9", "init"))
    pushed = []
    r._push_deployed_chain_toml = lambda host: (pushed.append(host.name)
                                                or replace(host, chain_toml="/root/cascade/chain.deployed.toml"))
    assert r._rolling_bench_init(_era())["_pointer"] == INIT_PTR
    assert pushed == ["init-lane"] and r.builds[0].chain_toml == "/root/cascade/chain.deployed.toml"


# The owner-approval gate (2026-11-02) is tested in test_money_safety.py;
# these tests drive the operator-lane / operator-rent machinery behind it.
pytestmark = pytest.mark.usefixtures("operator_legs_preapproved")
