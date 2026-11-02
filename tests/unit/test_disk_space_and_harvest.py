"""2026-11-02 disk safety: the ONE space manager (eval cache + trainer caches
+ harvest leftovers), the atomic harvest save, and "a full disk never costs a
finished run" (ENOSPC = operator fault, bounded retry, the pod kept).

The incident: 29 x 14 GB eval checkpoints never evicted filled the volume;
every finished leg then failed its harvest with ENOSPC, was classed
``[infra]`` (a burned miner attempt) and lost its pod before anything was saved.
"""
from __future__ import annotations

import errno
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.shared import disk_space
from cascade.shared.config import RoundConfig, ValidatorConfig
from cascade.trainer import harvest_guard
from cascade.trainer.faults import _OperatorFault, classify_leg_exception

D = ["sha256-" + c * 64 for c in "abcde"]


def _mk(root: Path, name: str, nbytes: int, age: float) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "weights.safetensors").write_bytes(b"w" * nbytes)
    t = time.time() - age
    os.utime(d / "weights.safetensors", (t, t))
    os.utime(d, (t, t))
    return d


@pytest.fixture
def fs(monkeypatch, tmp_path):
    """A fake filesystem of ``cap`` bytes: free = cap - bytes under tmp_path."""
    state = {"cap": 10_000}

    def usage(_p):
        used = disk_space.tree_bytes(tmp_path)
        return SimpleNamespace(total=state["cap"], used=used, free=state["cap"] - used)

    monkeypatch.setattr(disk_space, "DISK_USAGE", usage)
    return state


def _cfg(tmp_path, **v):
    vc = ValidatorConfig(weight_set_interval_blocks=1, poll_seconds=1,
                         state_db_path=str(tmp_path / "validator_state.db"),
                         cascade_state_db_path=str(tmp_path / "cascade_state.json"),
                         warm_start_init_path=str(tmp_path / "warm_start_init.json"),
                         eval_ckpt_cache_dir=str(tmp_path / "_eval_ckpts"), **v)
    return SimpleNamespace(validator=vc, round=RoundConfig())


# ── the space manager ───────────────────────────────────────────────────────


def test_lru_eviction_never_touches_the_king_state_refs_or_pins(tmp_path, fs):
    cache = tmp_path / "_eval_ckpts"
    for i, name in enumerate(D):
        _mk(cache, name, 1000, age=10_000 - i * 1000)        # D[0] oldest
    # the king (champion state), the warm-start init, the unsettled manifest
    (tmp_path / "validator_state.db").write_text(
        '{"king": "cascade/ckpt@' + D[0].replace("-", ":", 1) + '"}')
    (tmp_path / "warm_start_init.json").write_text('{"ref": "x@' + D[1].replace("-", ":", 1) + '"}')
    disk_space.write_pins(cache, [D[2]])
    fs["cap"] = 5_000 + 3_000                                  # 3000 free; floor 4500
    cfg = _cfg(tmp_path, eval_ckpt_cache_max_gb=0.0, eval_ckpt_cache_grace_seconds=60)
    rep = disk_space.reclaim(cfg, cache, floor_bytes=4_500)
    assert [c.path.name for c in rep.evicted] == [D[3], D[4]][:len(rep.evicted)]
    assert (cache / D[0]).is_dir() and (cache / D[1]).is_dir() and (cache / D[2]).is_dir()
    assert rep.short_bytes == 0 and not (cache / D[3]).exists()


def test_byte_budget_evicts_oldest_first(tmp_path, fs):
    cache = tmp_path / "_eval_ckpts"
    for i, name in enumerate(D[:4]):
        _mk(cache, name, 1000, age=10_000 - i * 1000)
    src = disk_space.digest_cache_source("eval_cache", cache, set(), max_bytes=2_500)
    rep = disk_space.ensure_space(cache, [src], floor_bytes=0, grace_seconds=60)
    assert sorted(c.path.name for c in rep.evicted) == sorted(D[:2])
    assert sorted(p.name for p in cache.iterdir()) == sorted(D[2:4])


def test_recent_entries_are_inside_the_grace(tmp_path, fs):
    cache = tmp_path / "_eval_ckpts"
    _mk(cache, D[0], 1000, age=5)
    fs["cap"] = 1_000
    src = disk_space.digest_cache_source("eval_cache", cache, set())
    rep = disk_space.ensure_space(cache, [src], floor_bytes=10_000, grace_seconds=3600)
    assert rep.evicted == [] and rep.short_bytes > 0 and (cache / D[0]).is_dir()


def test_unreadable_protection_evicts_nothing(tmp_path, fs):
    cache = tmp_path / "_eval_ckpts"
    _mk(cache, D[0], 1000, age=10_000)
    (cache / disk_space.PINS_FILE).write_text("{torn")
    fs["cap"] = 1_000
    rep = disk_space.reclaim(_cfg(tmp_path, eval_ckpt_cache_grace_seconds=0), cache,
                             floor_bytes=10_000)
    assert rep.blocked and rep.evicted == [] and (cache / D[0]).is_dir()


def test_harvest_with_a_pending_upload_or_in_progress_is_never_reclaimed(tmp_path, fs):
    wr = tmp_path / "wr"
    seed = wr / "_funded_harvest" / "777"
    settled = _mk(seed, "hkA--j1", 1000, age=10_000)
    pending = _mk(seed, "hkB--j2", 1000, age=10_000)
    (seed / "hkB--j2.pending.json").write_text("{}")
    writing = _mk(seed, "hkC--j3.partial-1-2", 1000, age=10_000)
    leftover = _mk(wr / "_funded_harvest.box-20261031" / "776", "hkD--j4", 1000, age=10_000)
    boxed_pending = _mk(wr / "_funded_harvest.box-20261031" / "776", "hkE--j5", 1000,
                        age=10_000)
    (boxed_pending.parent / "hkE--j5.pending.json").write_text("{}")
    fs["cap"] = 5_100
    cfg = _cfg(tmp_path, eval_ckpt_cache_grace_seconds=60)
    rep = disk_space.reclaim(cfg, wr, floor_bytes=10_000, work_root=wr,
                             active_harvests=lambda: {str((seed / "hkC--j3").resolve())})
    gone = {c.path for c in rep.evicted}
    assert settled in gone and leftover in gone
    assert pending.is_dir() and writing.is_dir() and boxed_pending.is_dir()


def test_validator_trims_its_cache_and_pins_the_manifest_in_progress(tmp_path, fs):
    from cascade.validator.evaluation import EvaluationMixin

    cache = tmp_path / "_eval_ckpts"
    for i, name in enumerate(D[:3]):
        _mk(cache, name, 1000, age=10_000 - i * 1000)
    (tmp_path / "validator_state.db").write_text("{}")         # a validator lives here
    v = SimpleNamespace(cfg=_cfg(tmp_path, eval_ckpt_cache_max_gb=0.0,
                                 eval_ckpt_cache_min_free_gb=2_500 / disk_space.GIB,
                                 eval_ckpt_cache_grace_seconds=60),
                        cache_dir=None, state=None)
    EvaluationMixin._pin_eval_checkpoint(v, cache, D[0])       # the manifest in progress
    assert disk_space.read_pins(cache) == {D[0]}
    fs["cap"] = 3_000 + 1_500                                  # ~1.4 kB free < 2.5 kB floor
    EvaluationMixin._trim_eval_cache(v, cache)
    assert (cache / D[0]).is_dir() and not (cache / D[1]).exists()
    EvaluationMixin._release_eval_pins(v)                      # settled: _mark_handled
    assert disk_space.read_pins(cache) == set()


# ── ENOSPC is the operator's fault; the pod is kept ─────────────────────────


def _runner(tmp_path, **round_kw):
    rnd = RoundConfig(min_free_disk_gb=0.0, expected_checkpoint_gb=0.0, **round_kw)
    r = SimpleNamespace(cfg=SimpleNamespace(round=rnd), work_root=tmp_path)
    r.sleeps = []
    r._harvest_retry_sleep = r.sleeps.append
    return r


def _enospc():
    return OSError(errno.ENOSPC, "No space left on device")


def test_local_disk_errors_never_burn_a_miner_attempt(tmp_path):
    gen = SimpleNamespace(hotkey="hkA", ref="r@1")
    seeds = SimpleNamespace(base_seed=7)
    wrapped = RuntimeError("tar extraction failed")
    wrapped.__cause__ = _enospc()
    for exc in (_enospc(), wrapped, OSError(errno.EIO, "I/O error")):
        assert classify_leg_exception(_runner(tmp_path), gen, seeds, exc, rc=None, text=str(exc),
                                      stalled_before=False, shards_leg=False) \
            == (False, "operator", False)
    assert harvest_guard.local_disk_error(OSError(errno.ENOENT, "x")) is None


def test_enospc_mid_harvest_drops_the_partial_frees_space_and_retries(tmp_path):
    r = _runner(tmp_path, harvest_disk_retries=2, harvest_disk_retry_seconds=10)
    dest = tmp_path / "_funded_harvest" / "7" / "hkA--j"
    calls = []

    def attempt():
        calls.append(1)
        if len(calls) == 1:
            dest.mkdir(parents=True)
            (dest / "half.safetensors").write_bytes(b"x")
            raise _enospc()
        return "entry"

    assert harvest_guard.guarded_harvest(r, dest, attempt, label="hkA") == "entry"
    assert len(calls) == 2 and r.sleeps == [10.0] and not dest.exists()
    assert r.__dict__["_active_harvest_dirs"] == set()


def test_exhausted_budget_keeps_the_pod_or_goes_pod_to_r2(tmp_path):
    r = _runner(tmp_path, harvest_disk_retries=1, harvest_disk_retry_seconds=5)

    def attempt():
        raise _enospc()

    with pytest.raises(harvest_guard.HarvestDiskFault) as ei:
        harvest_guard.guarded_harvest(r, tmp_path / "d", attempt, label="hkA")
    assert harvest_guard.keeps_pod(ei.value) and isinstance(ei.value, _OperatorFault)
    assert r.sleeps == [5.0]
    assert harvest_guard.guarded_harvest(r, tmp_path / "d", attempt, label="hkA",
                                         r2_fallback=lambda: "via-r2") == "via-r2"


def test_after_release_the_box_copy_is_never_deleted(tmp_path):
    r = _runner(tmp_path)
    dest = tmp_path / "d"

    def attempt():
        dest.mkdir()
        (dest / "w").write_bytes(b"verified")
        r.__dict__["_leg_local_obj"].harvest_released = True
        raise _enospc()

    with pytest.raises(_OperatorFault) as ei:
        harvest_guard.guarded_harvest(r, dest, attempt, label="hkA")
    assert not harvest_guard.keeps_pod(ei.value) and (dest / "w").read_bytes() == b"verified"


def test_no_room_means_no_harvest_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(disk_space, "DISK_USAGE",
                        lambda p: SimpleNamespace(total=1, used=1, free=0))
    r = _runner(tmp_path, harvest_disk_retries=0)
    r.cfg.round = RoundConfig(min_free_disk_gb=1.0, expected_checkpoint_gb=1.0,
                              harvest_disk_retries=0)
    r.cfg.validator = ValidatorConfig(weight_set_interval_blocks=1, poll_seconds=1,
                                      state_db_path=str(tmp_path / "s.db"))
    tried = []
    with pytest.raises(harvest_guard.HarvestDiskFault):
        harvest_guard.guarded_harvest(r, tmp_path / "d", lambda: tried.append(1),
                                      label="hkA")
    assert tried == []


# ── the atomic save ─────────────────────────────────────────────────────────


def test_failed_harvest_stream_leaves_no_copy_and_no_partial(tmp_path, monkeypatch):
    import io
    import tarfile

    from cascade.trainer import remote as remote_mod
    from cascade.trainer.remote import RemoteDispatchError, RemoteHost

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo("./weights.safetensors")
        info.size = 4
        tf.addfile(info, io.BytesIO(b"WWWW"))
    tar = tmp_path / "ok.tar"
    tar.write_bytes(buf.getvalue())
    out = tmp_path / "h" / "hkA--j"
    monkeypatch.setattr(remote_mod, "build_ssh_argv",
                        lambda host, cmd: ["sh", "-c", f"cat {tar}; exit 1"])
    with pytest.raises(RemoteDispatchError):
        remote_mod.harvest_remote_dir(RemoteHost(name="p", host="h"), "/w/ck", out)
    assert not out.exists() and list(out.parent.iterdir()) == []
    monkeypatch.setattr(remote_mod, "build_ssh_argv", lambda host, cmd: ["cat", str(tar)])
    remote_mod.harvest_remote_dir(RemoteHost(name="p", host="h"), "/w/ck", out)
    assert (out / "weights.safetensors").read_bytes() == b"WWWW"
    assert [p.name for p in out.parent.iterdir()] == ["hkA--j"]


# ── pre-rent disk reservation ───────────────────────────────────────────────


def test_reservation_counts_only_live_legs_holding_a_pod(tmp_path, monkeypatch):
    from cascade.funding.queue import FundedQueue
    from cascade.trainer import money_guard, runner_common

    gib = 1 << 30
    monkeypatch.setattr(runner_common, "DISK_USAGE",
                        lambda p: SimpleNamespace(total=1, used=0, free=70 * gib))
    rnd = RoundConfig(min_free_disk_gb=30.0, expected_checkpoint_gb=15.0,
                      funded_mode="required")
    r = SimpleNamespace(cfg=SimpleNamespace(round=rnd), work_root=tmp_path)
    q = FundedQueue(tmp_path / "funded_queue.json")
    for i, hk in enumerate(("hkA", "hkB", "hkC")):
        q.add(hk, f"r@{i}", 10 + i)
        q.mark_in_flight(hk, f"r@{i}", target_boundary=1, era_index=1, started_block=1)
    assert runner_common.disk_guard_refusal(r) == ""          # none holds a pod yet
    for hk, i in (("hkA", 0), ("hkB", 1)):
        e = q.get(hk)
        assert q.set_phase(hk, "training", expect_ref=f"r@{i}", expect_started_block=1,
                           expect_funded_at=e.funded_at, expect_attempt_id=e.attempt_id)
    msg = runner_common.disk_guard_refusal(r)                  # 30 + 15 * 3 = 75 > 70
    assert msg.startswith("operator disk") and "3 checkpoint(s)" in msg
    assert any(c["code"] == "operator_disk" for c in money_guard.health_conditions(r, {}))


def test_the_disk_manager_evicts_nothing_when_cwd_differs_or_state_is_missing(
        tmp_path, fs, monkeypatch):
    cache = tmp_path / "deploy" / "_eval_ckpts"
    _mk(cache, D[0], 1000, age=10_000)
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")                 # relative state paths miss
    vc = ValidatorConfig(weight_set_interval_blocks=1, poll_seconds=1,
                         state_db_path="validator_state.db", eval_ckpt_cache_dir=str(cache),
                         eval_ckpt_cache_grace_seconds=0)
    cfg = SimpleNamespace(validator=vc, round=RoundConfig())
    fs["cap"] = 1_000
    rep = disk_space.reclaim(cfg, cache, floor_bytes=10_000)
    assert rep.blocked and rep.evicted == [] and (cache / D[0]).is_dir()


# ── box health: today's sequence (owner, 2026-11-02) ────────────────────────


def test_a_full_volume_halts_every_rent_keeps_the_pod_and_reharvests_it(tmp_path, monkeypatch):
    """The volume is full, a leg finishes, its harvest hits ENOSPC: no lium rm
    before the harvest, no new rent for ANY leg, the halt set and alerted.
    Space comes back: the canary passes, the halt clears, and the SAME pod is
    re-harvested — adopted, never retrained."""
    from dataclasses import replace as dc_replace

    from cascade.provision import funded as funded_mod
    from cascade.trainer import money_guard, runner_common
    from cascade.trainer.box_health import BoxHealth
    from cascade.trainer.faults import _FundedLegSkip
    from tests.unit.test_funded_pod_wiring import _challenger, _rent_ok, _vault
    from tests.unit.test_release_pod_after_harvest import POD, _leg, _scripted

    runner, disp, seeds, contract, events, _ = _scripted(tmp_path, monkeypatch)
    rents = []

    def rent(**kw):
        rents.append(kw["hotkey"])
        # the second rent of hkA finds its kept pod (the ledgered name) live
        return dc_replace(_rent_ok(kw["hotkey"]), adopted=rents.count(kw["hotkey"]) > 1)

    monkeypatch.setattr(funded_mod, "rent_funded_pod", rent)
    from cascade.trainer import remote as remote_mod

    good_harvest = remote_mod.harvest_remote_dir            # the scripted (working) harvest

    def full_harvest(host, rdir, dest, **kw):
        events.append("harvest-enospc")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("cascade.trainer.remote.harvest_remote_dir", full_harvest)
    gib = 1 << 30
    free = {"b": 0}
    monkeypatch.setattr(runner_common, "DISK_USAGE",
                        lambda p: SimpleNamespace(total=1, used=0, free=free["b"]))
    monkeypatch.setattr(disk_space, "DISK_USAGE",
                        lambda p: SimpleNamespace(total=1, used=0, free=free["b"]))
    free["b"] = 500 * gib                                   # room when the leg starts
    with pytest.raises(harvest_guard.HarvestKeepPod):
        _leg(runner, disp, seeds, contract)
    assert f"teardown:{POD}" not in events                  # no lium rm, pod KEPT
    assert "harvest-enospc" in events and rents == ["hkA"]
    halt = BoxHealth.for_runner(runner).halt()
    assert halt is not None and halt["cls"] == "local_disk"
    conds = {c["code"]: c for c in money_guard.health_conditions(runner, {})}
    assert conds["rents_halted"]["severity"] == "critical"
    assert conds["our_side_fault"]["severity"] == "critical"
    assert runner._funded_leg_failures["hkA"][1:] == (False, "operator_kept_pod", False)

    free["b"] = 0                                            # the volume is full
    runner._payer_pod_liveness = lambda pod_id: (object(), True)   # the kept pod is up
    _vault(tmp_path, "hkB")
    for hk in ("hkB", "hkA"):                                # a new leg, and the retry
        runner.__dict__.pop("_box_health_checked", None)
        with pytest.raises(_FundedLegSkip):
            runner._rent_funded_host("777", _challenger(hk))
        assert runner._funded_leg_failures[hk][2] == "operator_disk"
    assert rents == ["hkA"]                                  # nothing rented, nothing retrained

    free["b"] = 500 * gib                                    # space is back
    runner.__dict__.pop("_box_health_checked", None)
    from cascade.trainer.box_health import rent_halt_refusal

    assert rent_halt_refusal(runner) == ""                   # canary + space pass: cleared
    assert BoxHealth.for_runner(runner).halt() is None
    monkeypatch.setattr("cascade.trainer.remote.harvest_remote_dir", good_harvest)
    entry = _leg(runner, disp, seeds, contract)
    assert entry.miner_hotkey == "hkA"
    assert rents == ["hkA", "hkA"]                           # the second was an ADOPTION
    assert BoxHealth.for_runner(runner).leg("hkA") is None
    assert "upload" in events
