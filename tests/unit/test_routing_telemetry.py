"""MoE routing telemetry (cascade.trainer.routing_telemetry): the stats are
right on tiny count vectors, the cadence knob round-trips through the loader,
nothing in the inputs is mutated, and a failure can never escape into
training. CPU-only, tiny tensors."""

from __future__ import annotations

import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.shared.config import load_chain_config
from cascade.trainer.routing_telemetry import (
    ROUTING_EVERY_DEFAULT,
    RoutingTelemetry,
    layer_routing_stats,
    routing_record,
)

ROOT = Path(__file__).resolve().parents[2]


# ── per-layer stats ──────────────────────────────────────────────────────────


def test_balanced_layer_is_max_entropy_no_dead():
    s = layer_routing_stats([5, 5, 5, 5])
    assert s["dead_experts"] == 0 and s["starved_experts"] == 0
    assert s["entropy"] == pytest.approx(1.0)
    assert s["entropy_nats"] == pytest.approx(math.log(4))
    assert s["load_max_over_mean"] == pytest.approx(1.0)
    assert s["load_min_over_mean"] == pytest.approx(1.0)
    assert s["load_max_over_min"] == pytest.approx(1.0)
    for q in ("p10", "p50", "p90"):
        assert s[f"load_{q}_over_mean"] == pytest.approx(1.0)


def test_collapsed_layer_counts_dead_and_omits_inf_ratio():
    s = layer_routing_stats([12, 0, 0, 0])
    assert s["dead_experts"] == 3 and s["starved_experts"] == 3
    assert s["entropy"] == pytest.approx(0.0)
    assert s["load_max_over_mean"] == pytest.approx(4.0)
    assert s["load_min_over_mean"] == 0.0
    assert "load_max_over_min" not in s          # never inf/NaN on the log channel


def test_skewed_layer_values():
    s = layer_routing_stats([6, 2, 1, 1])        # mean 2.5
    f = [0.6, 0.2, 0.1, 0.1]
    h = -sum(p * math.log(p) for p in f)
    assert s["entropy_nats"] == pytest.approx(h)
    assert s["entropy"] == pytest.approx(h / math.log(4))
    assert s["load_max_over_mean"] == pytest.approx(2.4)
    assert s["load_max_over_min"] == pytest.approx(6.0)
    assert s["load_p50_over_mean"] == pytest.approx(0.6)   # median(0.4,0.4,0.8,2.4)


def test_empty_and_zero_layers():
    assert layer_routing_stats([]) == {}
    s = layer_routing_stats([0, 0])
    assert s["dead_experts"] == 2 and "entropy" not in s


def test_accepts_tensors_and_leaves_them_untouched():
    torch = pytest.importorskip("torch")
    load = torch.tensor([3, 1, 0, 4], dtype=torch.int64)
    before = load.clone()
    s = layer_routing_stats(load)
    assert s["dead_experts"] == 1
    assert torch.equal(load, before) and load.dtype == torch.int64


# ── the flat record ─────────────────────────────────────────────────────────


def test_record_keys_and_summary():
    rec = routing_record([[4, 4], [8, 0, 0, 0]], block_ids=[1, 5],
                         aux=[0.25, None], bias=[[0.1, -0.3], None])
    assert rec["routing/L1/entropy"] == pytest.approx(1.0)
    assert rec["routing/L5/dead_experts"] == 3
    assert rec["routing/L1/aux_loss"] == pytest.approx(0.25)
    assert "routing/L5/aux_loss" not in rec
    assert rec["routing/L1/bias_absmax"] == pytest.approx(0.3)
    assert rec["routing/dead_experts"] == 3
    assert rec["routing/dead_frac"] == pytest.approx(3 / 6)
    assert rec["routing/entropy_min"] == pytest.approx(0.0)
    assert rec["routing/entropy_mean"] == pytest.approx(0.5)
    assert rec["routing/load_max_over_mean"] == pytest.approx(4.0)
    assert rec["routing/aux_loss_mean"] == pytest.approx(0.25)
    # every value is a JSON scalar (the S3 LogSink / wandb contract)
    assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in rec.values())
    assert all(re.fullmatch(r"routing/(L\d+/)?[a-z0-9_]+", k) for k in rec)


def test_record_block_ids_default_and_mismatch():
    assert "routing/L0/entropy" in routing_record([[1, 1]])
    with pytest.raises(ValueError):
        routing_record([[1, 1]], block_ids=[1, 2])


# ── cadence + failure guard ─────────────────────────────────────────────────


class _Layer:
    def __init__(self, bias):
        self.route_bias = bias
        self.last_aux = None


def _model(torch, n=2):
    return SimpleNamespace(cfg=SimpleNamespace(moe_blocks=tuple(range(1, n + 1))),
                           moe_layers=lambda: [_Layer(torch.zeros(4)) for _ in range(n)])


def test_emit_cadence_and_record_shape():
    torch = pytest.importorskip("torch")
    got = []
    t = RoutingTelemetry(ROUTING_EVERY_DEFAULT)
    loads = [torch.tensor([1, 2, 3, 4]), torch.tensor([0, 0, 5, 5])]
    for step in (1, 2, 49, 50, 100):
        t.emit(got.append, step=step, tokens=step * 10, model=_model(torch), loads=loads)
    assert [r["step"] for r in got] == [1, 50, 100]
    r = got[0]
    assert r["event"] == "routing" and r["tokens"] == 10
    assert r["routing/L2/dead_experts"] == 2 and "routing/L1/entropy" in r
    assert "loss" not in r and "lr" not in r                # no duplicated step keys


def test_disabled_and_no_logger_are_noops():
    torch = pytest.importorskip("torch")
    got = []
    RoutingTelemetry(0).emit(got.append, step=50, tokens=0, model=_model(torch),
                             loads=[torch.ones(4), torch.ones(4)])
    RoutingTelemetry(50).emit(None, step=50, tokens=0, model=_model(torch),
                              loads=[torch.ones(4), torch.ones(4)])
    assert got == []
    assert RoutingTelemetry("junk").every == ROUTING_EVERY_DEFAULT


def test_failure_warns_once_and_never_raises(caplog):
    class Boom:
        def moe_layers(self):
            raise RuntimeError("boom")

    got = []
    t = RoutingTelemetry(1)
    with caplog.at_level("WARNING", logger="cascade.trainer.routing"):
        t.emit(got.append, step=1, tokens=0, model=Boom(), loads=[])
        t.emit(got.append, step=2, tokens=0, model=Boom(), loads=[])
    assert got == []
    assert sum("routing telemetry disabled" in r.message for r in caplog.records) == 1
    assert not t.due(3)


# ── [telemetry] routing_every_steps (loader-parsed) ─────────────────────────


def test_knob_defaults_when_absent():
    for f in ("chain.toml", "chain.testnet.toml"):
        assert load_chain_config(ROOT / f).telemetry.routing_every_steps == ROUTING_EVERY_DEFAULT


@pytest.mark.parametrize("value,expected", [(10, 10), (0, 0), (-5, 0)])
def test_knob_round_trips_through_the_loader(tmp_path, value, expected):
    text = (ROOT / "chain.toml").read_text()
    text, n = re.subn(r"(?m)^\[telemetry\]$",
                      f"[telemetry]\nrouting_every_steps = {value}", text, count=1)
    assert n == 1
    p = tmp_path / "chain.toml"
    p.write_text(text)
    assert load_chain_config(p).telemetry.routing_every_steps == expected
