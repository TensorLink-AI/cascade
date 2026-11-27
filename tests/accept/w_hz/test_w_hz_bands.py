"""W-HZ: the GIFT-Eval horizon bands and the ``[lane_d]`` keys (owner
2026-11-27). Contract: README.md."""

from __future__ import annotations

import pytest

from .helpers import GIFT_TABLE, REPO, band_cfg, need

MOD = "cascade.shared.lane_d_horizons"


def test_gift_eval_base_horizons_and_term_multipliers():
    hz = need(MOD, "GIFT_BASE_HORIZONS", "GIFT_TERM_MULTIPLIERS")
    # gift-eval src/gift_eval/data.py: PRED_LENGTH_MAP + M4's Q / A, Term.multiplier
    assert hz.GIFT_BASE_HORIZONS == {"S": 60, "T": 48, "H": 48, "D": 30, "W": 8, "M": 12,
                                     "Q": 8, "A": 6}
    assert hz.GIFT_TERM_MULTIPLIERS == {"short": 1, "medium": 10, "long": 15}


def test_band_horizons_are_patch_aligned_and_capped():
    hz = need(MOD, "band_table", "band_horizon")
    assert hz.band_table(("short", "medium", "long")) == GIFT_TABLE
    for row in GIFT_TABLE.values():
        assert all(h % 32 == 0 and 32 <= h <= 2048 for h in row.values())
    assert hz.band_horizon("S", "long", max_horizon=512) == 512


@pytest.mark.parametrize("freq,base", [("H", "H"), ("h", "H"), ("5min", "T"), ("min", "T"),
                                        ("15min", "T"), ("10S", "S"), ("150S", "S"),
                                        ("240S", "T"), ("21600S", "H"), ("10800S", "H"),
                                        ("D", "D"), ("B", "D"), ("W-SUN", "W"), ("W", "W"),
                                        ("ME", "M"), ("MS", "M"), ("Q-DEC", "Q"),
                                        ("A-DEC", "A"), ("YE", "A"), ("10T", "T")])
def test_frequencies_normalise_like_gift_eval(freq, base):
    hz = need(MOD, "gift_base_freq")
    assert hz.gift_base_freq(freq) == base


def test_unknown_and_sub_second_frequencies_have_no_band():
    hz = need(MOD, "bands_for")
    for f in ("", "ms", "100L", "xyz", None):
        assert hz.bands_for(f, ("short", "medium", "long")) == {}


def test_config_defaults_to_the_band_rule_and_round_trips():
    need("cascade.shared.config.lane_d", "LANE_D_HORIZON_BANDS")
    from cascade.shared.config.lane_d import LaneDConfig, parse_lane_d

    d = parse_lane_d({})
    assert d.horizon_bands == ("short", "medium", "long") == LaneDConfig().horizon_bands
    assert (d.max_horizon, d.band_slots, d.min_cell_windows) == (2048, 1, 50)
    assert d.horizon == 64
    fixed = parse_lane_d({"horizon_bands": []})
    assert fixed.horizon_bands == ()
    got = parse_lane_d({"horizon_bands": ["long", "short"], "max_horizon": 1024,
                        "band_slots": 3, "min_cell_windows": 7})
    assert (got.horizon_bands, got.max_horizon, got.band_slots, got.min_cell_windows) == (
        ("short", "long"), 1024, 3, 7)
    for bad in ({"horizon_bands": ["huge"]}, {"horizon_bands": ["short", "short"]},
                {"horizon_bands": "short"}, {"max_horizon": 100}, {"max_horizon": 4096},
                {"band_slots": 0}, {"min_cell_windows": 0}):
        with pytest.raises(ValueError):
            parse_lane_d(bad)


def test_shipped_chain_tomls_carry_the_band_rule():
    import tomllib

    from cascade.shared.config.lane_d import parse_lane_d

    for name in ("chain.toml", "chain.testnet.toml"):
        raw = tomllib.loads((REPO / name).read_text())["lane_d"]
        assert raw.get("horizon_bands") == ["short", "medium", "long"], name
        assert raw.get("max_horizon") == 2048 and raw.get("horizon") == 64, name
        cfg = parse_lane_d(raw)
        assert cfg.horizon_bands == ("short", "medium", "long")


def test_band_keys_are_release_only():
    from cascade.shared.netconfig_bounds import RELEASE_ONLY_KEYS, key_refusals

    for k in ("horizon_bands", "max_horizon", "band_slots", "min_cell_windows"):
        assert ("lane_d", k) in RELEASE_ONLY_KEYS, k
    assert key_refusals({"lane_d": {"horizon_bands": []}})


def test_the_fixed_rule_is_the_band_list_empty():
    cfg = band_cfg(horizon_bands=[])
    hz = need(MOD, "bands_on")
    assert not hz.bands_on(cfg) and hz.bands_on(band_cfg())
