"""W-OV (plan rev6.24): the overlay envelope — per-key bounds, release-only
keys inside allowed tables, content-addressed data references, no silent
share clamp — and the lane C eval-data keys made overlay-settable."""

from __future__ import annotations

import pytest

from .helpers import NETWORK, SHIPPED, refusal, shipped_cfg, shipped_raw

# Values the loader accepts but the envelope refuses: one per danger class
# and lane.
DANGEROUS = [
    ({"lane_c": {"alpha": 0.3}}, "lane_c] alpha"),                   # free win
    ({"lane_b": {"gain_margin": 0.0005}}, "lane_b] gain_margin"),    # a noise-level bar
    ({"lane_d": {"bootstrap_B": 10_000_000}}, "lane_d] bootstrap_B"),  # satellite DoS
    ({"lane_d": {"d_margin": 0.0005, "backtest_weight_cap": 0.0}}, "lane_d] d_margin"),
    ({"lane_d": {"max_windows_per_settlement": 1_000_000}}, "max_windows_per_settlement"),
    ({"lane_c": {"n_windows": 1_000_000}}, "lane_c] n_windows"),
    ({"marketcast": {"alpha": 0.4}}, "marketcast] alpha"),
    ({"marketcast": {"windows_per_cell": 5000}}, "windows_per_cell"),
    ({"scoring": {"bootstrap_alpha": 0.45}}, "scoring] bootstrap_alpha"),
    ({"scoring": {"bootstrap_B": 5_000_000}}, "scoring] bootstrap_B"),
    ({"lane_b": {"share": 0.8}, "lane_c": {"share": 0.1}}, "lane_b] share"),
    ({"lane_d": {"court_weights": [0.2, 0.4, 0.4]}}, "lane_d] court_weights"),  # KING_D < half
    ({"lane_c": {"full_pay_max_days": 1000.0}}, "lane_c] full_pay_max_days"),
]


@pytest.mark.parametrize("overrides,where", DANGEROUS)
def test_w_ov_envelope_refuses_loader_legal_dangerous_values(overrides, where):
    from cascade.shared import netconfig_bounds as nb

    assert nb.value_refusals(overrides), f"{overrides} is inside the envelope"
    why = refusal(shipped_cfg(), overrides)
    assert why is not None and "out of bounds" in why and where in why, why


@pytest.mark.parametrize("name", SHIPPED)
def test_w_ov_envelope_contains_every_shipped_value(name):
    """A restating overlay (S0c) always passes: every bounded key's shipped
    value is inside its envelope."""
    from cascade.shared import netconfig_bounds as nb

    raw = shipped_raw(name)
    restated = {}
    for (table, key) in [*nb.BOUNDS, *nb.SEQ_BOUNDS]:
        if key in (raw.get(table) or {}):
            restated.setdefault(table, {})[key] = raw[table][key]
    assert restated
    assert nb.value_refusals(restated) == []


def test_w_ov_every_bound_names_a_parser_key():
    """No typo'd bound: each key is typed in a shipped file or is a field of
    its parsed section."""
    from dataclasses import fields

    from cascade.shared import netconfig_bounds as nb

    cfg = shipped_cfg()
    raws = [shipped_raw(n) for n in SHIPPED]
    bad = []
    for (table, key) in [*nb.BOUNDS, *nb.SEQ_BOUNDS]:
        typed = any(key in (r.get(table) or {}) for r in raws)
        section = getattr(cfg, table, None)
        parsed = section is not None and key in {f.name for f in fields(section)}
        if not (typed or parsed):
            bad.append(f"{table}.{key}")
    assert bad == []


@pytest.mark.parametrize("table,key,value", [
    ("scoring", "prior_contract_digest", "ab" * 32),
    ("eval", "shared_ladder_feature", "lad-00000001"),
    ("eval", "gift_gate_data_dir", "/elsewhere"),
    ("lane_d", "teacher_api", "other"),
])
def test_w_ov_release_only_keys_inside_allowed_tables(table, key, value):
    from cascade.shared import netconfig as nc
    from cascade.shared import netconfig_bounds as nb

    assert (table, key) in nb.RELEASE_ONLY_KEYS
    try:
        nc.check_overrides({table: {key: value}})
    except nc.NetConfigError as e:
        assert f"[{table}] {key}" in str(e)
    else:
        raise AssertionError(f"[{table}] {key} accepted by check_overrides")


def test_w_ov_data_references_must_be_content_addressed():
    from cascade.shared import netconfig as nc
    from cascade.shared import netconfig_bounds as nb

    assert nb.pin_refusals({"eval": {"window_pool": "org/pool@3f2a"}}) == []
    assert nb.pin_refusals({"eval": {"window_pool": ""}}) == []           # off
    assert nb.pin_refusals({"lane_d": {"real_corpus_ref": "org/corpus@rev1"}}) == []
    for ov in ({"eval": {"window_pool": "org/pool"}},
               {"lane_d": {"real_corpus_ref": "org/corpus@"}}):
        try:
            nc.check_overrides(ov)
        except nc.NetConfigError as e:
            assert "content-addressed" in str(e)
        else:
            raise AssertionError(f"unpinned data reference accepted: {ov}")


def test_w_ov_lane_share_the_carve_would_clamp_is_refused():
    """main: B 0.25, C 0.25 from lane A ⇒ A's pool is 0.5. Lane D 0.5 from
    lane A fits; lane D 0.45 from lane C (which holds 0.25) would be
    clamped to 0.25 silently — refused."""
    from cascade.shared import netconfig_bounds as nb

    base = shipped_cfg()
    assert nb.share_refusals(base) == []
    why = refusal(base, {"lane_d": {"share": 0.45, "takes_from": "lane_c"}})
    assert why is not None and "clamp" in why, why
    assert refusal(base, {"lane_d": {"share": 0.25, "takes_from": "lane_a"}}) is None


def test_w_ov_lane_c_eval_data_is_overlay_settable():
    """Owner 2026-11-24: lane C's data (assets, timeframes, suite weights,
    grid density) changes by overlay; the model pins, the suite list, data
    sources, pool and embargo stay release-only."""
    from cascade.shared import netconfig as nc
    from cascade.shared import netconfig_bounds as nb

    assert nb.value_refusals({"marketcast": {"windows_per_cell": 96}}) == []
    base = shipped_cfg()
    assets = {k: list(v) for k, v in base.raw["marketcast"]["assets"].items()}
    assets["crypto"] = [a for a in assets["crypto"] if a != "JUP"]
    ov = {"marketcast": {"assets": assets, "windows_per_cell": 96}}
    body = nc.next_overlay(ov, 10**9, None, network=NETWORK)
    out = nc.build_config(base, body)
    assert out.marketcast.windows_per_cell == 96
    assert "JUP" not in out.marketcast.assets_map["crypto"]
    assert "JUP" in base.marketcast.assets_map["crypto"]
    for key in ("embargo_days", "suites", "data_sources", "pool_prefix", "base_ref"):
        try:
            nc.check_overrides({"marketcast": {key: base.raw["marketcast"][key]}})
        except nc.NetConfigError:
            continue
        raise AssertionError(f"[marketcast] {key} is overlay-settable")


def test_w_ov_policy_publishes_the_envelope():
    from cascade.shared import netconfig as nc
    from cascade.shared import netconfig_bounds as nb

    env = nc.overlay_policy()["envelope"]
    assert env == nb.policy()
    assert "lane_c.n_windows" in env["bounds"] and "lane_d.d_margin" in env["bounds"]
    assert "[netconfig] owner_hotkey" in env["never_overlay"]
