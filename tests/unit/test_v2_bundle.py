"""DEC-CA-0053/0054 ride the v2 go-live (DEC-CA-0050): one typed block."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cascade.shared.config import load_chain_config
from cascade.shared.era import bench_gate_configured, bench_gate_mode_at

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mainnet():
    return load_chain_config(ROOT / "chain.toml")


def test_mainnet_bundle_is_inert_until_v2(mainnet):
    assert mainnet.v2.from_block == 0
    assert mainnet.training.corpus_mode == "stream_cpu"          # the live contract
    assert mainnet.training.at_block(10**9).corpus_mode == "stream_cpu"
    assert bench_gate_mode_at(mainnet, 10**9) == "off"
    assert not bench_gate_configured(mainnet)


def test_v2_contract_carries_the_bundle(mainnet):
    v2 = mainnet.training.v2_contract
    assert (v2.corpus_mode, v2.long_series) == ("shards", "crop")
    assert mainnet.v2.bench_gate_mode == "enforce"
    assert mainnet.generation.enabled


def test_typing_the_block_switches_everything_at_once(mainnet):
    era = mainnet.round.epoch_blocks * mainnet.round.era_settlements
    gate = 10_000 * era
    cfg = replace(mainnet, v2=replace(mainnet.v2, from_block=gate),
                  training=replace(mainnet.training, v2_from_block=gate))
    before, after = gate - era, gate + era
    assert cfg.training.at_block(before).corpus_mode == "stream_cpu"
    assert cfg.training.at_block(gate).corpus_mode == "shards"
    assert cfg.training.at_block(gate).long_series == "crop"
    # the settlement AT the v2 block still judges the last pre-v2 era
    assert bench_gate_mode_at(cfg, gate) == "off"
    assert bench_gate_mode_at(cfg, after) == "enforce"
    assert bench_gate_configured(cfg)


def test_shards_contract_needs_the_tier(tmp_path):
    text = (ROOT / "chain.toml").read_text(encoding="utf-8")
    text = text.replace("enabled            = true          # allowed;",
                        "enabled            = false         # allowed;", 1)
    p = tmp_path / "chain.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="needs \\[generation\\] enabled"):
        load_chain_config(p)


def test_generation_lead_applies_once_v2_is_typed(mainnet):
    from types import SimpleNamespace

    from cascade.trainer.runner_generation import _GenerationMixin

    lead = _GenerationMixin._generation_lead_seconds
    assert lead(SimpleNamespace(cfg=mainnet)) == 0.0          # v2 not typed: no lead
    era = mainnet.round.epoch_blocks * mainnet.round.era_settlements
    typed = replace(mainnet, v2=replace(mainnet.v2, from_block=10_000 * era))
    want = typed.generation.window_seconds + _GenerationMixin.GENERATION_OVERHEAD_SECONDS
    assert lead(SimpleNamespace(cfg=typed, GENERATION_OVERHEAD_SECONDS=7200.0)) == want


def _typed(mainnet):
    era = mainnet.round.epoch_blocks * mainnet.round.era_settlements
    gate = 10_000 * era
    return replace(mainnet, v2=replace(mainnet.v2, from_block=gate),
                   training=replace(mainnet.training, v2_from_block=gate)), gate, era


def test_settlement_waits_for_the_kings_bench(mainnet):
    from types import SimpleNamespace

    from cascade.trainer.rolling import RollingScheduler

    cfg, gate, era = _typed(mainnet)
    me = SimpleNamespace(cfg=cfg)
    ready = [SimpleNamespace(hotkey="A", bench={"x": 1}), SimpleNamespace(hotkey="B", bench=None)]
    mid = gate + era + mainnet.round.epoch_blocks          # not an era's last settlement
    hold = RollingScheduler._hold_unbenched
    assert [f.hotkey for f in hold(me, ready, mid)] == ["A"]
    no_king_bench = SimpleNamespace(king_entry={"k": 1}, king_bench=None)
    assert hold(me, ready, mid, no_king_bench) == []
    benched = SimpleNamespace(king_entry={"k": 1}, king_bench={"x": 1})
    assert [f.hotkey for f in hold(me, ready, mid, benched)] == ["A"]


def test_a_late_republish_never_adds_gate_rows():
    from types import SimpleNamespace

    from cascade.trainer.rolling import RollingScheduler

    me = SimpleNamespace(state=SimpleNamespace(finished=[], bench_reports={
        "77": {"created_block": 5, "pairs": [[{"trained_pointer": "K"}, {}]],
               "rows_pointers": ["K"], "warm_start_ckpt": ""}}))
    out = RollingScheduler._join_bench_locked(me, "LATE", {"trained_pointer": "LATE"},
                                              {"s": 1}, "77")
    assert out[0] == "77" and out[-1] == frozenset({"K"})


def test_publish_bench_filters_rows_for_a_republish(monkeypatch):
    from types import SimpleNamespace

    from cascade.shared import bench_report as BR
    from cascade.shared.manifest import BenchScores
    from cascade.trainer.runner_rolling import _RollingMixin

    seen = {}
    monkeypatch.setattr(BR, "publish_bench_report",
                        lambda store, text, rid: seen.setdefault("t", text) and "k")
    rows = {"K": {"key": "rk", "sha256": "a" * 64}, "L": {"key": "rl", "sha256": "b" * 64}}
    me = SimpleNamespace(cfg=load_chain_config(ROOT / "chain.toml"), wallet=None,
                         _bench_rows_registry=lambda: rows, manifest_store=lambda: None,
                         _log_bench_pair_wandb=lambda r: None)
    six = BenchScores(**{f: 1.0 for f in BenchScores.__dataclass_fields__})
    ent = [(SimpleNamespace(role=r, size="", miner_hotkey=r, miner_uid=1,
                            trained_pointer=p), six) for r, p in (("king", "K"), ("challenger", "L"))]
    rep = _RollingMixin._rolling_publish_bench(me, "1", 1, ent, rows_for=frozenset({"K"}))
    assert rep.entry_for_pointer("K").rows_key == "rk"
    assert rep.entry_for_pointer("L").rows_key == ""
