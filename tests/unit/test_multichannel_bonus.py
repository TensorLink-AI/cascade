"""[training] budget_denomination = "points+mv<PCT>" (DEC-CA-0047): token billing
with a width bonus.

Under "points" every values entry is a budget point, so a (C, L) series costs
C×L and every width trains the SAME model tokens — the GPU drawn never decides
the token count, because the budget (not the wall) stops every leg. The bonus
bills a C > 1 series at 100/(100+PCT) of its points, so an all-multichannel
corpus trains PCT % more tokens than a univariate one; a corpus that stacks a
share s of its points earns 1/(1 - s·PCT/(100+PCT)) — proportional, not flat.
Pinned here:

* exact integer arithmetic (ceil) — the stream's stop and the trainer's counter
  agree row for row, so the audit replays the same prefix;
* univariate series are bit-identical to "points" (every historical round);
* the grammar is validated loud (PCT 1..100) and the field stays digest-inert
  at its default "points";
* the bonus is proportional to the multichannel share of the corpus.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cascade.shared.config import (
    budget_denomination_parts,
    validate_budget_denomination,
)
from cascade.shared.manifest import contract_digest
from cascade.trainer.stream import element_points
from cascade.trainer.toto2_trainer import batch_points

L = 4096


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# ── grammar ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,parts", [
    ("points", ("points", 0)),
    ("series_points", ("series_points", 0)),
    ("points+mv20", ("points", 20)),
    ("points+mv1", ("points", 1)),
    ("points+mv100", ("points", 100)),
])
def test_parts_and_validation(mode, parts):
    assert budget_denomination_parts(mode) == parts
    assert validate_budget_denomination(mode) == mode


@pytest.mark.parametrize("bad", ["points+mv0", "points+mv101", "points+mv", "points+mv020",
                                 "pointsmv20", "series_points+mv20", "POINTS+mv20", "mv20"])
def test_bad_grammar_fails_loud(bad):
    with pytest.raises(ValueError):
        validate_budget_denomination(bad)


# ── the point rule ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("c", [2, 3, 4, 8, 32])
@pytest.mark.parametrize("pct", [1, 20, 50, 100])
def test_multichannel_series_billed_at_discount(c, pct):
    arr = np.zeros((c, L))
    want = _ceil_div(c * L * 100, 100 + pct)
    assert element_points(arr, f"points+mv{pct}") == want
    assert want < c * L                       # a real bonus …
    assert want >= c * L * 100 // (100 + pct)  # … never rounded below the exact ratio
    rec = {"values": arr, "mask": np.ones((c, L), dtype=np.uint8), "roles": None}
    assert element_points(rec, f"points+mv{pct}") == want   # mask adds no points


@pytest.mark.parametrize("pct", [1, 20, 100])
def test_univariate_is_bit_identical_to_points(pct):
    for arr in (np.zeros(L), np.zeros((1, L)), np.zeros(37)):
        assert element_points(arr, f"points+mv{pct}") == element_points(arr, "points") == arr.size


def test_full_stacking_earns_exactly_the_bonus():
    """An all-2-channel corpus under points+mv20 reaches the budget after
    1.2× the model tokens a univariate corpus does (to within one series)."""
    budget = 39_960_000_000
    cost = element_points(np.zeros((2, L)), "points+mv20")
    n_series = budget // cost
    model_tokens = n_series * 2 * L
    assert abs(model_tokens / budget - 1.2) < 1e-3


def test_partial_stacking_is_proportional():
    """22 % of series two-channel (the observed king shape) is ≈ 36 % of the
    POINTS (a 2-channel series carries twice the points) and earns ≈ 6.4 %:
    1 / (1 − 0.36 × 20/120). Not the flat 20 %, and a single stacked series
    in a thousand earns nothing measurable — no token-gesture bonus."""
    uni = element_points(np.zeros(L), "points+mv20")
    two = element_points(np.zeros((2, L)), "points+mv20")
    n = 1000
    n2 = 220
    billed = (n - n2) * uni + n2 * two
    tokens = (n - n2) * L + n2 * 2 * L
    ratio = tokens / billed
    assert 1.06 < ratio < 1.07
    one_in_a_thousand = ((n - 1) * uni + two) / ((n - 1) * L + 2 * L)
    assert abs(1 / one_in_a_thousand - 1.0) < 1e-3


@pytest.mark.parametrize("b,c", [(64, 1), (64, 2), (16, 4), (2, 32)])
@pytest.mark.parametrize("pct", [20, 100])
def test_batch_points_matches_element_points(b, c, pct):
    """The loop-side counter is the sum of the stream-side billing of the
    same rows — ceil applied per row, so both stops agree exactly."""
    batch = np.zeros((b, c, L))
    denom = f"points+mv{pct}"
    assert batch_points(batch, denom) == sum(element_points(batch[i], denom) for i in range(b))
    if c == 1:
        assert batch_points(batch, denom) == batch_points(batch, "points")


def test_copied_channel_spends_real_budget():
    """Under token billing a second channel that carries no information still
    costs (almost) its full points — the bonus never pays for junk."""
    two = element_points(np.zeros((2, L)), "points+mv20")
    one = element_points(np.zeros(L), "points+mv20")
    assert two > 1.6 * one            # the copy costs ≥ 60 % extra budget …
    assert two < 2 * one              # … minus only the 1/6 bonus


# ── contract digest ───────────────────────────────────────────────────────────

def test_default_is_digest_inert_and_arming_bumps(cfg):
    from cascade.shared.manifest import contract_payload

    base = replace(cfg.training, budget_denomination="points")
    d0 = contract_digest(base)
    assert "budget_denomination" not in contract_payload(base)
    d_mv = contract_digest(replace(cfg.training, budget_denomination="points+mv20"))
    d_sp = contract_digest(replace(cfg.training, budget_denomination="series_points"))
    assert len({d0, d_mv, d_sp}) == 3
    assert contract_digest(replace(cfg.training, budget_denomination="points+mv25")) != d_mv


def test_loader_round_trips_the_bonus_denomination(tmp_path):
    """The TOML string reaches the dataclass unchanged (config knobs need loader
    parsing — a silently defaulted knob would bill the legacy rule)."""
    from pathlib import Path as _P

    from cascade.shared.config import load_chain_config

    root = _P(__file__).resolve().parents[2]
    text = (root / "chain.toml").read_text().replace(
        'budget_denomination         = "series_points"',
        'budget_denomination         = "points+mv20"')
    assert 'budget_denomination         = "points+mv20"' in text
    good = tmp_path / "chain.toml"
    good.write_text(text)
    t = load_chain_config(good).training
    assert t.budget_denomination == "points+mv20"
    assert budget_denomination_parts(t.budget_denomination) == ("points", 20)
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace("points+mv20", "points+mv0"))
    with pytest.raises(ValueError, match="budget_denomination"):
        load_chain_config(bad)
