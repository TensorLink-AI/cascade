"""``[v2] init_epoch_bind_from_block``: the king check accepts a random-init
epoch's init pointer only when it binds THIS seed epoch (release-then-
activate: off by default, old behaviour byte-identical until the block)."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.audit import checks
from cascade.shared.bench_report import (
    BenchEntry,
    BenchReport,
    init_pointer_bound_to_epoch,
    random_init_repo_name,
)
from cascade.shared.config import load_chain_config
from cascade.shared.era import init_epoch_bind_at
from cascade.shared.manifest import BenchScores, format_trained_pointer
from cascade.validator.king_check import KingCheckMixin

ROOT = Path(__file__).resolve().parents[2]
SB = 5000
KING_PTR = format_trained_pointer("cascade/king@sha256:" + "b" * 64)
CUR = format_trained_pointer(f"cascade/{random_init_repo_name(42, 'p', SB)}@sha256:" + "e" * 64)
STALE = format_trained_pointer(
    f"cascade/{random_init_repo_name(41, 'p', SB - 3600)}@sha256:" + "d" * 64)
LEGACY = format_trained_pointer("cascade/ckpt-41-p-random-init@sha256:" + "c" * 64)
KING = BenchScores(*([0.5] * 6))
INIT = BenchScores(*([0.9] * 6))


def _on(cfg, block=100):
    return replace(cfg, v2=replace(cfg.v2, init_epoch_bind_from_block=block))


# ── the knob (loader-parsed, shipped OFF) ───────────────────────────────────

def test_knob_ships_off_on_mainnet_and_testnet():
    assert load_chain_config(ROOT / "chain.toml").v2.init_epoch_bind_from_block == 0
    assert load_chain_config(ROOT / "chain.testnet.toml").v2.init_epoch_bind_from_block == 0


def _with(tmp_path, value):
    text = (ROOT / "chain.toml").read_text()
    text = re.sub(r"(?m)^init_epoch_bind_from_block\s*=.*$", "", text)
    # the [v2] key line itself (the phrase also appears in comments above it)
    text, n = re.subn(r"(?m)^king_ladder = true$",
                      f"king_ladder = true\ninit_epoch_bind_from_block = {value}", text, count=1)
    assert n == 1
    p = tmp_path / "chain.toml"
    p.write_text(text)
    return p


def test_knob_round_trips_through_the_loader(tmp_path):
    cfg = load_chain_config(_with(tmp_path, 123456))
    assert cfg.v2.init_epoch_bind_from_block == 123456
    assert init_epoch_bind_at(cfg, 123456) and not init_epoch_bind_at(cfg, 123455)


@pytest.mark.parametrize("value", ["-1", "1.5", '"v2"', "true"])
def test_knob_refuses_bad_values(tmp_path, value):
    with pytest.raises(ValueError, match="init_epoch_bind_from_block"):
        load_chain_config(_with(tmp_path, value))


# ── the binding ─────────────────────────────────────────────────────────────

def test_binding_is_the_repo_basename():
    assert random_init_repo_name(42, "p", SB) == f"ckpt-42-p-random-init-sb{SB}"
    assert init_pointer_bound_to_epoch(CUR, SB)
    assert not init_pointer_bound_to_epoch(STALE, SB)
    assert not init_pointer_bound_to_epoch(LEGACY, SB)          # pre-binding name
    assert not init_pointer_bound_to_epoch(CUR, SB + 1)         # no suffix collision
    assert not init_pointer_bound_to_epoch(CUR, None)
    assert not init_pointer_bound_to_epoch("garbage", SB)
    hf = format_trained_pointer(f"hfns/{random_init_repo_name(42, 'p', SB)}@sha256:" + "f" * 64)
    assert init_pointer_bound_to_epoch(hf, SB)


def test_trainer_publishes_a_bound_name(monkeypatch):
    from cascade.shared import hippius
    from cascade.trainer.runner_rolling import _RollingMixin

    seen = []

    def upload(local, full, hub, *, hf_repo=None):
        seen.append(full)
        return SimpleNamespace(ref=SimpleNamespace(immutable_ref=f"{full}@sha256:" + "a" * 64))

    monkeypatch.setattr(hippius, "upload_dir_to_hub_or_hf", upload)
    fake = SimpleNamespace(hub=lambda: SimpleNamespace(namespace="cascade"),
                           _hf_ckpt_repo=lambda full: None)
    era = SimpleNamespace(base_seed=42, seed_block=SB)
    ptr = _RollingMixin._publish_init_checkpoint(fake, "/tmp/x", era,
                                                 SimpleNamespace(arch_preset="p"))
    assert seen == [f"cascade/ckpt-42-p-random-init-sb{SB}"]
    assert init_pointer_bound_to_epoch(ptr, SB)


# ── the king check: trainer-report path ─────────────────────────────────────

def _report(init_ptr):
    return BenchReport(round_id="100", created_block=7, entries=(
        BenchEntry("king", "p", "KING", 1, KING_PTR, KING),
        BenchEntry("init", "p", "", -1, init_ptr, INIT)))


def _manifest(warm=""):
    return SimpleNamespace(prev_round_id="100", warm_start_ckpt=warm, round_id="101",
                           era={"seed_block": SB, "index": 3})


def _trainer(cfg, init_ptr, boundary=200):
    fake = SimpleNamespace(cfg=cfg, _epoch_start_block=lambda m: boundary,
                           _fetch_bench_report=lambda rid, require=None: _report(init_ptr),
                           _king_check_need_rows=False)
    rec = {"reasons": []}
    fake._kc_init_unbound = lambda m, p, r: KingCheckMixin._kc_init_unbound(fake, m, p, r)
    return KingCheckMixin._v2_king_check_trainer_bench(fake, _manifest(), KING_PTR, rec), rec


def test_active_stale_pointer_defers(cfg):
    got, rec = _trainer(_on(cfg), STALE)
    assert got is None
    assert rec["reasons"][0].startswith("no init pointer for this seed epoch")


def test_active_current_epoch_pointer_is_judged(cfg):
    got, rec = _trainer(_on(cfg), CUR)
    assert rec["reasons"] == [] and got == (KING, INIT, CUR)


def test_inactive_is_the_old_behaviour(cfg):
    for c, boundary in ((cfg, 200), (_on(cfg, 10_000), 200)):   # off / not yet in force
        got, rec = _trainer(c, STALE, boundary)
        assert rec["reasons"] == [] and got == (KING, INIT, STALE)


# ── the king check: validator-bench path ────────────────────────────────────

def _vbench(cfg, init_ptr):
    swept = []

    def side(e, role, need_rows=False, pair_with=None):
        swept.append(role)
        return {}, dict(zip(BenchScores.__dataclass_fields__, [0.5] * 6, strict=False))

    fake = SimpleNamespace(
        cfg=cfg, _epoch_start_block=lambda m: 200, _king_check_need_rows=False,
        _vbench={"mode": "sweep", "round_id": "101"},
        _fetch_bench_report=lambda rid, raise_unavailable=False: _report(init_ptr),
        _vbench_side=side, _vbench_infra=lambda e: None)
    fake._kc_init_unbound = lambda m, p, r: KingCheckMixin._kc_init_unbound(fake, m, p, r)
    king = SimpleNamespace(miner_hotkey="KING", miner_uid=1, gen_ref="g", size="p")
    rec = {"reasons": []}
    got = KingCheckMixin._v2_king_check_validator_bench(fake, _manifest(), king, KING_PTR, rec)
    return got, rec, swept


def test_vbench_active_stale_pointer_defers_before_any_sweep(cfg):
    got, rec, swept = _vbench(_on(cfg), STALE)
    assert got is None and swept == []
    assert rec["reasons"][0].startswith("no init pointer for this seed epoch")


def test_vbench_current_pointer_and_inactive_are_swept(cfg):
    for c, ptr in ((_on(cfg), CUR), (cfg, STALE)):
        got, rec, swept = _vbench(c, ptr)
        assert rec["reasons"] == [] and got[2] == ptr and swept == ["king", "init"]


def test_warm_started_settlements_are_untouched(cfg):
    fake = SimpleNamespace(cfg=_on(cfg), _epoch_start_block=lambda m: 200)
    rec = {"reasons": []}
    assert KingCheckMixin._kc_init_unbound(fake, _manifest(warm=KING_PTR), STALE, rec) is False
    assert rec["reasons"] == []


# ── the audit ───────────────────────────────────────────────────────────────

def test_audit_allows_the_reason_only_under_the_binding(cfg):
    why = f"no init pointer for this seed epoch (the published init {STALE!r} ...)"
    kc = {"reasons": [why], "king_pointer": KING_PTR, "bench_round": "100"}
    manifest = SimpleNamespace(entry_for_role=lambda role: None, warm_start_ckpt="",
                               prev_round_id="100")
    receipt = SimpleNamespace(epoch_start_block=200, bench_report=None)
    assert checks._kc_deferred(receipt, _on(cfg), kc, manifest, None).status == checks.PASS
    assert checks._kc_deferred(receipt, cfg, kc, manifest, None).status == checks.FAIL
