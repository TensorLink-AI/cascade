"""[training] budget_denomination (DEC-CA-0042): a width-proportional budget.

Under the legacy "points" rule a (C, L) series costs C×L budget points, so a
wide corpus buys C× fewer optimizer steps per budget unless the batch fill
shrinks with C ("sequences", DEC-CA-0041). "series_points" bills a (C, L)
series L points — per time-step positions — so, paired with the "series"
batch fill (64 series per step whatever C), every width earns the SAME step
count and a wide corpus trains C× the channel tokens per step (univariate
1×, C=32 → 32×). The load-bearing properties pinned here:

* the stream's stop and the trainer's counter apply ONE rule (the corpus
  digest covers the consumed prefix, so the stop is consensus-relevant);
* C = 1 is bit-identical under both denominations (every historical round);
* the field is digest-inert at its default and a deliberate digest bump when
  armed (the drop-when-default convention);
* the shipped config pairs "series" fill with "series_points" budget.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cascade.shared.config import validate_budget_denomination
from cascade.trainer.stream import element_points
from cascade.trainer.toto2_trainer import batch_points, iter_training_batches

PS = 8
MAXP = 16
L = PS * MAXP


# ── the point rule itself ────────────────────────────────────────────────────

@pytest.mark.parametrize("c", [1, 2, 4, 32])
def test_element_points_rules(c):
    arr = np.zeros((c, L))
    assert element_points(arr, "points") == c * L
    assert element_points(arr, "series_points") == L
    # record carrier: only `values` are points, the mask adds none
    rec = {"values": arr, "mask": np.ones((c, L), dtype=np.uint8), "roles": None}
    assert element_points(rec, "points") == c * L
    assert element_points(rec, "series_points") == L


def test_element_points_1d_is_denomination_blind():
    arr = np.zeros(L)
    assert element_points(arr, "points") == element_points(arr, "series_points") == L
    assert element_points(arr) == L      # default = legacy "points"


@pytest.mark.parametrize("b,c", [(64, 1), (64, 4), (2, 32), (16, 4)])
def test_batch_points_matches_element_points(b, c):
    """The loop-side counter is the sum of the stream-side billing of the
    same rows — the two stops must agree or the trainer under/over-runs the
    prefix the audit re-derives."""
    batch = np.zeros((b, c, L))
    for denom in ("points", "series_points"):
        assert batch_points(batch, denom) == sum(
            element_points(batch[i], denom) for i in range(b))


# ── the property the pairing buys: equal steps, C× tokens ────────────────────

@pytest.mark.parametrize("c", [1, 4, 32])
def test_series_fill_plus_series_points_gives_width_independent_steps(c):
    rng = np.random.default_rng(0)
    budget = 64 * L * 5                              # 5 full steps of 64 series
    stream = [rng.normal(size=(c, L)) for _ in range(64 * 5)]
    steps, points, channel_tokens = 0, 0, 0
    for batch in iter_training_batches(iter(stream), patch_size=PS, max_ctx_patches=MAXP,
                                       batch_size=64, batch_denomination="series"):
        steps += 1
        points += batch_points(batch, "series_points")
        channel_tokens += batch.size
        if points >= budget:
            break
    assert steps == 5                                # same step count at every C
    assert channel_tokens == c * budget              # C× the trained tokens


def test_legacy_points_budget_collapses_steps_with_width():
    """The regression the pairing exists to avoid, kept as the contrast."""
    rng = np.random.default_rng(0)
    budget = 64 * L * 5
    stream = [rng.normal(size=(4, L)) for _ in range(64 * 5)]
    steps, points = 0, 0
    for batch in iter_training_batches(iter(stream), patch_size=PS, max_ctx_patches=MAXP,
                                       batch_size=64, batch_denomination="series"):
        steps += 1
        points += batch_points(batch, "points")
        if points >= budget:
            break
    assert steps == 2                                # ceil(5 / 4)


# ── the stream stops on the same rule ───────────────────────────────────────

class _FakeCorpus:
    def __init__(self, series):
        self.series = series
        self.digest = "d" * 64
        self.n_series = len(series)
        self.total_points = sum(int(np.size(s)) for s in series)


def _cache_stream(series, budget, denom, monkeypatch):
    from cascade.trainer import stream as st

    monkeypatch.setattr(st, "build_round_corpus", lambda *a, **k: _FakeCorpus(series))
    return st._CacheReuseStream("repo", 1, None, budget, use_sandbox=False, blocked=(),
                                budget_denomination=denom)


def test_cache_reuse_stream_stops_on_denomination(monkeypatch):
    series = [np.full((4, L), float(i)) for i in range(10)]
    seen_points = list(_cache_stream(series, 3 * L, "series_points", monkeypatch).series())
    seen_legacy = list(_cache_stream(series, 3 * L, "points", monkeypatch).series())
    assert len(seen_points) == 3              # L per series ⇒ 3 series fill 3L
    assert len(seen_legacy) == 1              # 4L per series ⇒ the first overshoots


def test_stream_default_is_legacy_points(monkeypatch):
    from cascade.trainer import stream as st

    series = [np.full((2, L), 1.0) for _ in range(10)]
    monkeypatch.setattr(st, "build_round_corpus", lambda *a, **k: _FakeCorpus(series))
    rs = st._CacheReuseStream("repo", 1, None, 4 * L, use_sandbox=False, blocked=())
    assert len(list(rs.series())) == 2        # 2L each under "points"
    assert rs.total_points == 4 * L


# ── the trainer counts on the same rule ──────────────────────────────────────

@pytest.mark.parametrize("denom,expect", [("points", 4 * 2 * 16 + 2 * 16),
                                          ("series_points", 4 * 16 + 2 * 16)])
def test_trainer_tokens_seen_follows_denomination(tmp_path, denom, expect):
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from cascade.trainer.toto2_trainer import Toto2Trainer

    contract = SimpleNamespace(
        context_length=16, horizon=8, patch_size=4, d_model=16, num_layers=1,
        num_heads=1, head_dim=16, mlp_expansion=2, num_quantiles=9,
        batch_size=2, max_train_seconds=30, base_lr=1e-3, weight_decay=0.0,
        optimizer="adamw", warmup_tokens=0, input_transform="arcsinh_causal",
        batch_denomination="series", budget_denomination=denom,
    )
    rng = np.random.default_rng(0)
    series = ([rng.normal(size=(2, 32)).cumsum(axis=-1) for _ in range(4)]
              + [rng.normal(size=32).cumsum() for _ in range(2)])
    result = Toto2Trainer(device="cpu", deterministic=False).train(
        iter(series), contract, training_seed=1, token_budget=10**6,
        out_dir=tmp_path / "ckpt",
    )
    assert result.metrics["steps"] == 3       # 2 series per step at either C
    assert result.metrics["tokens_seen"] == expect


def test_trainer_stops_at_series_points_budget(tmp_path):
    """A budget of 2 series-points-steps stops after 2 steps even though the
    channel tokens are 2× that — the budget is width-blind by design."""
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from cascade.trainer.toto2_trainer import Toto2Trainer

    contract = SimpleNamespace(
        context_length=16, horizon=8, patch_size=4, d_model=16, num_layers=1,
        num_heads=1, head_dim=16, mlp_expansion=2, num_quantiles=9,
        batch_size=2, max_train_seconds=30, base_lr=1e-3, weight_decay=0.0,
        optimizer="adamw", warmup_tokens=0, input_transform="arcsinh_causal",
        batch_denomination="series", budget_denomination="series_points",
    )
    rng = np.random.default_rng(0)
    series = [rng.normal(size=(2, 32)).cumsum(axis=-1) for _ in range(20)]
    result = Toto2Trainer(device="cpu", deterministic=False).train(
        iter(series), contract, training_seed=1, token_budget=2 * 2 * 16,
        out_dir=tmp_path / "ckpt",
    )
    assert result.metrics["steps"] == 2
    assert result.metrics["tokens_seen"] == 2 * 2 * 16
    assert result.metrics["deadline_hit"] is False


# ── config + digest ──────────────────────────────────────────────────────────

def test_validate_budget_denomination():
    assert validate_budget_denomination("points") == "points"
    assert validate_budget_denomination("series_points") == "series_points"
    with pytest.raises(ValueError, match="budget_denomination"):
        validate_budget_denomination("tokens")


def test_budget_denomination_is_digest_inert_at_default_and_a_bump_when_armed(cfg):
    from cascade.shared.manifest import contract_digest, contract_payload

    base = replace(cfg.training, budget_denomination="points")
    armed = replace(cfg.training, budget_denomination="series_points")
    assert "budget_denomination" not in contract_payload(base), (
        "the inert default must stay out of the hash (drop-when-default)")
    assert contract_payload(armed)["budget_denomination"] == "series_points"
    assert contract_digest(base) != contract_digest(armed)


def test_loader_rejects_unknown_budget_denomination(tmp_path):
    from pathlib import Path

    from cascade.shared.config import load_chain_config

    root = Path(__file__).resolve().parents[2]
    text = (root / "chain.toml").read_text().replace(
        'budget_denomination         = "series_points"',
        'budget_denomination         = "tokens"')
    assert 'budget_denomination         = "tokens"' in text
    bad = tmp_path / "chain.toml"
    bad.write_text(text)
    with pytest.raises(ValueError, match="budget_denomination"):
        load_chain_config(bad)


def test_chain_toml_arms_five_hour_series_points_legs():
    from pathlib import Path

    from cascade.shared.config import load_chain_config

    root = Path(__file__).resolve().parents[2]
    t = load_chain_config(root / "chain.toml").training
    assert t.target_train_hours == 5.0
    assert t.max_train_seconds == 5 * 3600
    assert t.batch_denomination == "series"
    assert t.budget_denomination == "series_points"
    # audit replay reads the field back off a declared body
    from cascade.shared.manifest import contract_payload
    assert contract_payload(t)["budget_denomination"] == "series_points"
