"""Installing a release, the burn vote and the dated lane launch are distinct."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.shared.activation import prelaunch_burn_feature
from cascade.shared.calendar_launch import (
    check_calendar_launch,
    launch_timestamp,
    resolve_calendar_launch,
)
from cascade.shared.config import load_chain_config, parse_chain_config
from cascade.shared.netconfig import effective_raw


def _chain(target, head):
    return SimpleNamespace(finalized_block=lambda: head,
                           block_timestamp=lambda block: target + (block - 30001) * 12,
                           current_block=lambda: pytest.fail("must use finalized timestamps"))


def test_release_has_the_noon_utc_launch_and_separate_update_vote():
    cfg = load_chain_config()
    assert cfg.v2.from_block == 0
    assert cfg.v2.launch_at_utc == "2026-10-30T12:00:00Z"
    assert cfg.v2.prelaunch_burn == 0.95
    assert cfg.v2.prelaunch_burn_from_block == 0
    assert cfg.activation.threshold == 0.51
    assert prelaunch_burn_feature(cfg).name == "prelaunch-95-r2"


def test_old_release_votes_cannot_activate_the_new_release_burn():
    from cascade.shared.activation import ValidatorStake, format_signal, tally

    cfg = load_chain_config()
    feature = prelaunch_burn_feature(cfg).name
    validators = [ValidatorStake("updated", 51, True, 0), ValidatorStake("old", 49, True, 0)]
    signals = {"updated": format_signal("prelaunch-95"), "old": format_signal("prelaunch-95")}
    assert not tally(feature, validators, signals, threshold=0.51, block=14400).locked
    signals["updated"] = format_signal(feature)
    current = tally(feature, validators, signals, threshold=0.51, block=14400)
    assert current.locked and current.signed == ("updated",)


def test_calendar_uses_the_same_finalized_era_for_live_and_late_nodes():
    cfg = load_chain_config()
    raw = effective_raw(cfg)
    raw["netconfig"]["owner_hotkey"] = cfg.manifest.validator_hotkey
    cfg = parse_chain_config(raw)
    target = launch_timestamp(cfg.v2.launch_at_utc)
    opening = resolve_calendar_launch(cfg, _chain(target, 30000))
    assert opening.v2.from_block == 0
    assert opening.v2.commit_from_block == 29101
    first = resolve_calendar_launch(cfg, _chain(target, 30001))
    late = resolve_calendar_launch(cfg, _chain(target, 100000))
    assert first.v2.from_block == late.v2.from_block == 43200
    assert first.training.v2_from_block == 43200
    assert first.v2.bench_gate_v2_from_block == 43200
    assert first.v2.prelaunch_burn_from_block == 0  # time alone never activates burn
    assert first.lane_b.from_block == first.lane_c.from_block == 0
    assert resolve_calendar_launch(first, object()) is first


def test_a_node_observing_launch_immediately_needs_only_recent_timestamps():
    cfg = load_chain_config()
    raw = effective_raw(cfg)
    raw["netconfig"]["owner_hotkey"] = cfg.manifest.validator_hotkey
    cfg = parse_chain_config(raw)
    target = launch_timestamp(cfg.v2.launch_at_utc)
    # A running node first resolves the opening while its timestamp is recent.
    cfg = resolve_calendar_launch(cfg, _chain(target, 29101))
    chain = _chain(target, 30001)
    recent = chain.block_timestamp
    reads = []

    def pruned(block):
        reads.append(block)
        if block < 29900:
            raise RuntimeError("not an archive RPC")
        return recent(block)

    chain.block_timestamp = pruned
    assert resolve_calendar_launch(cfg, chain).v2.from_block == 43200
    assert min(reads) >= 29900


@pytest.mark.parametrize("bad", ["2026-10-30T12:00:00", "2026-10-30T12:00:00+10:00", "noon"])
def test_launch_requires_an_unambiguous_utc_timestamp(bad):
    with pytest.raises(ValueError):
        launch_timestamp(bad)


def test_failed_or_nonfinite_chain_time_does_not_guess_a_launch():
    cfg = load_chain_config()
    chain = _chain(launch_timestamp(cfg.v2.launch_at_utc), 30001)
    chain.block_timestamp = lambda _: float("nan")
    with pytest.raises(ValueError, match="unresolved"):
        resolve_calendar_launch(cfg, chain)
    # A testnet's separately typed historical activation remains authoritative.
    typed = replace(cfg, v2=replace(cfg.v2, from_block=43200,
                                   commit_open_at_utc="", commit_from_block=0))
    assert resolve_calendar_launch(typed, object()) is typed


def test_service_start_refuses_a_dated_launch_whose_v2_config_cannot_load():
    # A deployment missing the main-validator pin is refused before launch.
    cfg = load_chain_config()
    head = SimpleNamespace(finalized_block=lambda: 30000)
    check_calendar_launch(cfg, head)
    assert cfg.netconfig.owner_hotkey == cfg.manifest.validator_hotkey
    raw = effective_raw(cfg)
    raw["netconfig"]["owner_hotkey"] = ""
    with pytest.raises(ValueError, match="launch_at_utc"):
        check_calendar_launch(parse_chain_config(raw), head)
    check_calendar_launch(replace(cfg, v2=replace(cfg.v2, launch_at_utc="")), object())


def test_a_pruned_late_joiner_gets_an_actionable_error_not_a_guess():
    cfg = load_chain_config()
    raw = effective_raw(cfg)
    raw["netconfig"]["owner_hotkey"] = cfg.manifest.validator_hotkey
    cfg = parse_chain_config(raw)
    chain = _chain(launch_timestamp(cfg.v2.launch_at_utc), 100000)
    recent = chain.block_timestamp

    def pruned(block):
        if block < 100000 - 256:
            raise RuntimeError("State already discarded")
        return recent(block)

    chain.block_timestamp = pruned
    with pytest.raises(RuntimeError, match="archive endpoint or pin the resolved.*from_block and commit_from_block"):
        resolve_calendar_launch(cfg, chain)


def test_an_enforced_v2_gate_cannot_be_configured_with_a_capped_sweep():
    # Launch benchmarks must be complete: a capped sweep is partial by design.
    cfg = load_chain_config()
    raw = effective_raw(cfg)
    raw["netconfig"]["owner_hotkey"] = cfg.manifest.validator_hotkey
    raw["v2"]["from_block"] = 43200
    assert parse_chain_config(raw).eval.cascade_bench_max_series == 0
    raw["eval"]["cascade_bench_max_series"] = 3
    with pytest.raises(ValueError, match="complete benchmark suites"):
        parse_chain_config(raw)
