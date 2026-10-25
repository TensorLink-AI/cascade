"""Regressions from the pre-launch review (2026-10-25): the dated v2 launch
must be honoured by every replay path, miner tooling must judge admission
the way the trainer does, a pod that loses isolation mid-leg is ours, and
an enforced bench gate refuses a capped sweep however it is configured."""
from __future__ import annotations

import io
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.miner.training_context import UnresolvedLaunch, configured_context
from cascade.miner.verify import VerifyReport
from cascade.shared.config import load_chain_config, parse_chain_config
from cascade.shared.manifest import TrainedEntry, contract_digest
from cascade.shared.netconfig import effective_raw
from cascade.trainer import sandbox as sandbox_mod
from cascade.trainer.corpus import CorpusError
from cascade.trainer.worker import _pod_fault

ROOT = Path(__file__).resolve().parents[2]


def _shipped_with_main_validator():
    """chain.toml as shipped (dated launch, from_block 0) plus the one key the
    operator must type before launch, so the v2 config loads."""
    cfg = load_chain_config(ROOT / "chain.toml")
    raw = effective_raw(cfg)
    raw["netconfig"]["owner_hotkey"] = cfg.manifest.validator_hotkey
    return parse_chain_config(raw)


# ── miner tooling judges admission the way the trainer does ──────────────────

def test_at_block_cannot_pin_an_unresolved_dated_launch_offline():
    shipped = load_chain_config(ROOT / "chain.toml")
    with pytest.raises(UnresolvedLaunch, match="--at-block"):
        configured_context(shipped, 9_999_999)
    assert configured_context(shipped, None).training.base_arch == shipped.training.base_arch
    typed = replace(shipped, v2=replace(shipped.v2, from_block=43200))
    assert configured_context(typed, 43200 + 1).training is not None


def test_verify_at_block_is_a_usage_error_while_the_launch_is_dated(example_generator_dir, capsys):
    from cascade.miner.cli import main

    rc = main(["verify", str(example_generator_dir), "--quick", "--at-block", "9999999",
               "--json", "--chain-toml", str(ROOT / "chain.toml")])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 2 and doc["exit_code"] == 2
    assert doc["error"]["code"] == "launch_unresolved"


def _recording_verify(calls):
    def fake(repo_dir, cfg, **kw):
        calls.append(kw)
        return VerifyReport(ok=False, failures=[], scope="quick")
    return fake


def test_iterate_and_score_check_the_launch_intersection_like_verify(example_generator_dir, tmp_path,
                                                                     monkeypatch, capsys):
    from cascade.miner import cli
    from cascade.miner import verify as verify_mod

    calls: list = []
    monkeypatch.setattr(verify_mod, "verify_repo", _recording_verify(calls))   # iterate imports lazily
    monkeypatch.setattr(cli, "verify_repo", _recording_verify(calls))
    toml = str(ROOT / "chain.toml")
    rc = cli.main(["iterate", str(example_generator_dir), "--work-dir", str(tmp_path / "w"),
                   "--baseline", "none", "--json", "--chain-toml", toml])
    assert rc == 1 and json.loads(capsys.readouterr().out)["error"]["code"] == "verify_failed"
    rc = cli.main(["score", str(example_generator_dir), "--json", "--chain-toml", toml])
    assert rc == 1 and json.loads(capsys.readouterr().out)["error"]["code"] == "verify_failed"
    assert [c.get("across_launch") for c in calls] == [True, True]


# ── a pod that loses isolation mid-leg is an infra fault, never a rejection ───

def _dead_child(stderr: bytes):
    r, w = os.pipe()
    os.close(w)                       # stdout: immediate EOF
    return SimpleNamespace(stdout=os.fdopen(r, "rb", buffering=0), stderr=io.BytesIO(stderr),
                           poll=lambda: 1, wait=lambda timeout=None: 1, pid=1)


def test_isolation_lost_at_exec_time_is_an_infra_fault_and_reprobed(monkeypatch):
    monkeypatch.setattr(sandbox_mod, "_NETNS_PROBE", ("unshare", "--user", "--map-root-user", "--net"))
    monkeypatch.setattr(sandbox_mod, "_wrapper_works", lambda prefix: False)
    with pytest.raises(CorpusError) as ei:
        list(sandbox_mod._frame_iter(_dead_child(b"unshare: unshare failed: Operation not permitted"), 5))
    assert "sandbox_isolation_unavailable" in str(ei.value)
    assert _pod_fault(ei.value)                      # worker rc 1, not the miner's rc 3
    assert sandbox_mod._NETNS_PROBE is None          # the next leg re-probes the host


def test_a_generator_printing_unshare_errors_cannot_relabel_its_own_crash(monkeypatch):
    monkeypatch.setattr(sandbox_mod, "_NETNS_PROBE", ("unshare", "--net"))
    monkeypatch.setattr(sandbox_mod, "_wrapper_works", lambda prefix: True)
    with pytest.raises(CorpusError) as ei:
        list(sandbox_mod._frame_iter(_dead_child(b"unshare: unshare failed: Operation not permitted"), 5))
    assert str(ei.value).startswith("sandbox_stream_failed")
    assert not _pod_fault(ei.value)
    assert sandbox_mod._NETNS_PROBE == ("unshare", "--net")


# ── an enforced gate refuses a capped sweep however it is configured ─────────

def test_capped_sweep_is_refused_while_the_launch_is_still_dated():
    cfg = load_chain_config(ROOT / "chain.toml")
    raw = effective_raw(cfg)
    raw["netconfig"]["owner_hotkey"] = cfg.manifest.validator_hotkey
    raw["eval"]["cascade_bench_max_series"] = 3
    assert raw["v2"]["from_block"] == 0 and raw["v2"]["bench_gate_mode"] == "enforce"
    with pytest.raises(ValueError, match="complete benchmark suites"):
        parse_chain_config(raw)


def test_an_overlay_cannot_cap_the_sweep_or_dodge_the_check_by_reverting_the_rule():
    from cascade.shared.netconfig import NetConfigError, build_config

    base = _shipped_with_main_validator()
    network = {"netuid": base.subnet.netuid, "chain": base.netconfig.network_id}
    era = base.round.epoch_blocks * base.round.era_settlements

    def body(overrides):
        return {"protocol": "cascade-netconfig/2", "network": network, "seq": 1, "prev": "",
                "effective_block": 0, "overrides": overrides}

    with pytest.raises(NetConfigError, match="complete benchmark suites"):
        build_config(base, body({"eval": {"cascade_bench_max_series": 5}}))
    with pytest.raises(NetConfigError, match="complete benchmark suites"):
        build_config(base, body({"v2": {"bench_gate_v2_from_block": 0},
                                 "eval": {"cascade_bench_max_series": 5}}))
    # an overlay that loads only while v2 is inert (the validator-sourced bench
    # needs the king ladder once the launch resolves) is refused up front
    with pytest.raises(NetConfigError, match="dated launch"):
        build_config(base, body({"v2": {"king_ladder": False}}))
    ok = build_config(base, body({"scoring": {"bench_gate_min_items": 11}}), applied_block=era)
    assert ok.scoring.bench_gate_min_items == 11 and ok.eval.cascade_bench_max_series == 0


# ── a stale worker image cannot publish an unbound v2 entry ─────────────────

def test_trainer_refuses_v2_entries_without_their_init_and_contract_binding(cfg, tmp_path):
    from cascade.shared.hippius import StorageError
    from cascade.shared.manifest import format_trained_pointer
    from cascade.trainer import rolling as R
    from tests.unit.test_rolling import Clock, _sched
    from tests.unit.test_v2_golive_validator import _v2

    v2cfg, n = _v2(cfg)
    sched, ops = _sched(v2cfg, tmp_path, Clock())
    era_len = v2cfg.round.epoch_blocks * v2cfg.round.era_settlements
    era = R.EraState(index=n // era_len, start_block=n, seed_block=n - era_len, base_seed=7,
                     warm_start_ckpt="")
    v2c = v2cfg.training.v2_contract
    bound = TrainedEntry("king_hk", 0, "king", "ns/k@sha256:" + "11" * 32,
                         format_trained_pointer("ns/ck@sha256:" + "22" * 32), "d", n + 10,
                         size=v2c.arch_preset, init_pointer="",
                         training_contract_digest=contract_digest(v2c))
    settle = n + v2cfg.round.epoch_blocks
    m = sched._build_manifest(era, bound, [], "round-x", settle, pin_block=settle)
    assert m.entries[0].training_contract_digest == contract_digest(v2c)
    # the pinned worker-v0.9.0 image emits neither field: the manifest is refused
    # (the settlement holds until the image is rebuilt), never published unbound
    stale = replace(bound, init_pointer=None, training_contract_digest="")
    with pytest.raises(StorageError, match="init/contract identity"):
        sched._build_manifest(era, stale, [], "round-y", settle, pin_block=settle)
