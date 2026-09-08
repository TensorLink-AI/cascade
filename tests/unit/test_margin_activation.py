"""Scheduled fresh-king margin change (``[scoring] win_margin_start_prev`` /
``margin_activation_block``).

Rounds before the block are judged at the previous value, rounds from it on
at ``win_margin_start``; the validator resolves it from the round's epoch
boundary and ``cascade-audit`` replays each receipt under its own value.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from cascade.shared.config import ScoringConfig, effective_win_margin_start


def _scoring(**kw) -> ScoringConfig:
    base = dict(win_margin_start=0.01, win_margin_end=0.005, margin_warmup_rounds=8,
                min_windows=200, bootstrap_B=100, bootstrap_alpha=0.05, dethrone_cp=1)
    base.update(kw)
    return ScoringConfig(**base)


def test_margin_knobs_parse_and_default_off(tmp_path):
    from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config

    bare = re.sub(r"^(win_margin_start_prev|margin_activation_block)\s*=.*$", "",
                  DEFAULT_CHAIN_TOML.read_text(), flags=re.M)
    p = tmp_path / "chain.toml"
    p.write_text(bare)
    cfg = load_chain_config(p)
    assert cfg.scoring.win_margin_start_prev == 0.0
    assert cfg.scoring.margin_activation_block == 0
    assert effective_win_margin_start(cfg.scoring, 1) == cfg.scoring.win_margin_start

    p.write_text(bare.replace(
        "\n[scoring]\n",
        "\n[scoring]\nwin_margin_start_prev = 0.02\nmargin_activation_block = 5000\n", 1))
    cfg = load_chain_config(p)
    assert cfg.scoring.win_margin_start_prev == 0.02
    assert cfg.scoring.margin_activation_block == 5000

    # An armed shipped toml flips on an epoch boundary, in step with the
    # ladder and the duel-only rounds (one announced block).
    shipped = load_chain_config(DEFAULT_CHAIN_TOML)
    if shipped.scoring.margin_activation_block:
        assert shipped.scoring.win_margin_start_prev > 0.0
        assert shipped.scoring.margin_activation_block % shipped.round.epoch_blocks == 0
        assert shipped.scoring.margin_activation_block == shipped.eval.scored_from_block


def test_effective_margin_resolves_per_block():
    armed = _scoring(win_margin_start_prev=0.02, margin_activation_block=5000)
    assert effective_win_margin_start(armed, 4999) == 0.02
    assert effective_win_margin_start(armed, 5000) == 0.01
    assert effective_win_margin_start(armed, 9000) == 0.01
    assert effective_win_margin_start(armed, None) == 0.01     # steady state
    steady = _scoring()
    assert effective_win_margin_start(steady, 1) == 0.01
    # a block without a previous value is no schedule
    assert effective_win_margin_start(_scoring(margin_activation_block=5000), 1) == 0.01


def test_effective_margin_two_step_schedule():
    """A 2% -> 1% -> 0.5% history resolves at each era, so old rounds stay
    audit-resolvable — a pre-flip block re-derives the value it recorded, and
    check_koth_params does not fail on the earlier era after a second flip."""
    sched = _scoring(win_margin_start=0.005,
                     win_margin_start_prev=0.01, margin_activation_block=9043200,
                     win_margin_start_prev2=0.02, margin_activation_block2=8992800)
    assert effective_win_margin_start(sched, 8992799) == 0.02    # launch..flip2 (2%)
    assert effective_win_margin_start(sched, 8992800) == 0.01    # flip2..flip1 start
    assert effective_win_margin_start(sched, 9043199) == 0.01    # flip2..flip1 end
    assert effective_win_margin_start(sched, 9043200) == 0.005   # flip1 on (0.5%)
    assert effective_win_margin_start(sched, None) == 0.005      # steady / head
    # *2 unset must stay bit-identical to the one-step resolver
    one = _scoring(win_margin_start_prev=0.02, margin_activation_block=5000)
    for b in (None, 1, 4999, 5000, 9000):
        assert effective_win_margin_start(one, b) == effective_win_margin_start(
            replace(one, win_margin_start_prev2=0.02, margin_activation_block2=0), b)


def test_koth_params_carry_the_round_value(cfg):
    from cascade.eval.koth import margin_for_tenure

    armed = replace(cfg, scoring=replace(cfg.scoring, win_margin_start=0.01,
                                         win_margin_start_prev=0.02,
                                         margin_activation_block=5000))
    before, after = armed.koth_params(block=4999), armed.koth_params(block=5000)
    assert before.win_margin_start == 0.02 and after.win_margin_start == 0.01
    assert margin_for_tenure(before, 0) == 0.02
    assert margin_for_tenure(after, 0) == 0.01
    # the decayed floor is untouched on both sides
    assert margin_for_tenure(before, 8) == pytest.approx(cfg.scoring.win_margin_end)
    assert margin_for_tenure(after, 8) == pytest.approx(cfg.scoring.win_margin_end)
    assert armed.koth_params().win_margin_start == 0.01


def test_audit_replays_each_receipt_under_its_own_margin(cfg):
    from cascade.audit import checks as C
    from tests.unit.receipt_fixture import make_scored_receipt

    armed = replace(cfg, scoring=replace(cfg.scoring, win_margin_start=0.01,
                                         win_margin_start_prev=0.02,
                                         margin_activation_block=5000))
    receipt, _, _ = make_scored_receipt(armed)          # params at the steady value
    post = replace(receipt, epoch_start_block=5000)
    assert C.check_koth_params(post, armed).status == C.PASS
    pre = replace(receipt, epoch_start_block=4999)      # a pre-flip round recorded 0.01?
    r = C.check_koth_params(pre, armed)
    assert r.status == C.FAIL and "win_margin_start" in r.detail
    old_params = dict(receipt.verdict.params, win_margin_start=0.02)
    pre_ok = replace(pre, verdict=replace(receipt.verdict, params=old_params))
    assert C.check_koth_params(pre_ok, armed).status == C.PASS


def test_dashboard_margin_line_uses_the_round_value():
    from cascade.miner.dashboard import margin_for_tenure_cfg

    armed = _scoring(win_margin_start_prev=0.02, margin_activation_block=5000)
    assert margin_for_tenure_cfg(armed, 0, block=4999) == 0.02
    assert margin_for_tenure_cfg(armed, 0, block=5000) == 0.01
    assert margin_for_tenure_cfg(armed, 0) == 0.01
    assert margin_for_tenure_cfg(armed, 8, block=4999) == pytest.approx(0.005)


# ── DEC-CA-0039: block-gated increment-margin activation ─────────────────────

def test_increment_from_block_resolves_per_block():
    from cascade.shared.config import effective_margin_mode

    armed = _scoring(margin_mode="level", increment_from_block=5000)
    assert effective_margin_mode(armed, 4999) == "level"       # pre-gate: base
    assert effective_margin_mode(armed, 5000) == "increment"   # from the gate on
    assert effective_margin_mode(armed, 9000) == "increment"
    assert effective_margin_mode(armed, None) == "level"       # unknown block ⇒ base
    # gate off ⇒ the base mode for every block
    assert effective_margin_mode(_scoring(margin_mode="level"), 9000) == "level"


def test_koth_params_flip_margin_mode_at_the_block(cfg):
    armed = replace(cfg, scoring=replace(cfg.scoring, margin_mode="level",
                                         increment_from_block=5000))
    assert armed.koth_params(block=4999).margin_mode == "level"
    assert armed.koth_params(block=5000).margin_mode == "increment"
    # And the audit resolves the SAME rule off the receipt's block, so
    # check_koth_params expects increment for a post-gate round and level before.
    assert armed.koth_params(block=None).margin_mode == "level"
