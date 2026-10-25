"""The v2-era lane-A dethrone bar ([v2] win_margin_*, owner 2026-10-19)."""

from dataclasses import replace

import pytest

from cascade.shared.config import load_chain_config
from cascade.shared.config.v2 import parse_v2_economics

V2_BLOCK = 9_000_000


def _armed(cfg, **v2):
    return replace(cfg, v2=replace(cfg.v2, from_block=V2_BLOCK, **v2))


def test_mainnet_ships_ten_to_one_percent():
    cfg = load_chain_config("chain.toml")
    assert (cfg.v2.win_margin_start, cfg.v2.win_margin_end,
            cfg.v2.margin_warmup_blocks) == (0.10, 0.01, 18000)


def test_v2_settlement_uses_the_v2_bar():
    cfg = _armed(load_chain_config("chain.toml"))
    p = cfg.koth_params(block=V2_BLOCK + 2 * 3600)
    assert (p.win_margin_start, p.win_margin_end) == (0.10, 0.01)


def test_settlement_at_the_v2_block_keeps_the_old_bar():
    # The settlement AT from_block belongs to the last toto2 era.
    cfg = _armed(load_chain_config("chain.toml"))
    old = replace(cfg, v2=replace(cfg.v2, win_margin_start=0.0, win_margin_end=0.0))
    assert cfg.koth_params(block=V2_BLOCK) == old.koth_params(block=V2_BLOCK)


def test_unset_inherits_scoring_bit_identically():
    cfg = _armed(load_chain_config("chain.toml"))
    off = replace(cfg, v2=replace(cfg.v2, win_margin_start=0.0, win_margin_end=0.0,
                                  margin_warmup_blocks=0))
    b = V2_BLOCK + 2 * 3600
    assert off.koth_params(block=b).win_margin_start != 0.10
    assert off.koth_params(block=b) == replace(cfg, v2=replace(
        cfg.v2, win_margin_start=0.0, win_margin_end=0.0)).koth_params(block=b)


def test_level_fallback_keeps_the_level_bar():
    cfg = _armed(load_chain_config("chain.toml"))
    b = V2_BLOCK + 2 * 3600
    lvl = cfg.judged_level_params(cfg.koth_params(block=b), b)
    assert lvl.margin_mode == "level"
    assert lvl.win_margin_end == cfg.scoring.win_margin_end


def test_testnet_leaves_it_unset():
    assert not load_chain_config("chain.testnet.toml").v2.margin_set()


def test_king_check_bar_is_fixed_and_separate():
    cfg = _armed(load_chain_config("chain.toml"))
    b = V2_BLOCK + 2 * 3600
    kc = cfg.king_check_params(b)
    assert (kc.win_margin_start, kc.win_margin_end) == (0.2, 0.2)   # no decay
    assert kc.margin_mode == "increment" and kc.init_gate_mode == "off"
    assert cfg.koth_params(block=b).win_margin_start == 0.10        # dethrone bar untouched


def test_king_check_unset_follows_the_dethrone_bar():
    cfg = _armed(load_chain_config("chain.toml"), king_check_margin=0.0)
    b = V2_BLOCK + 2 * 3600
    kc = cfg.king_check_params(b)
    assert (kc.win_margin_start, kc.win_margin_end) == (0.10, 0.01)


@pytest.mark.parametrize("bad", [-0.1, 2.0, "0.2", True])
def test_king_check_margin_refused(bad):
    with pytest.raises(ValueError):
        parse_v2_economics({"king_check_margin": bad})


@pytest.mark.parametrize("raw", [
    {"win_margin_start": 0.01, "win_margin_end": 0.1, "margin_warmup_blocks": 18000},
    {"win_margin_start": 0.1, "win_margin_end": 0.0, "margin_warmup_blocks": 18000},
    {"win_margin_start": 0.1, "win_margin_end": 0.01, "margin_warmup_blocks": 0},
    {"win_margin_start": 1.0, "win_margin_end": 0.01, "margin_warmup_blocks": 18000},
    {"win_margin_start": "0.1", "win_margin_end": 0.01, "margin_warmup_blocks": 18000},
])
def test_parse_refuses_bad_bars(raw):
    with pytest.raises(ValueError):
        parse_v2_economics(raw)
