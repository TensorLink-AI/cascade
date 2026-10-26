"""DEC-CA-0057 owner-signed network config: the overlay, its chain, the
allow-list, the merged validation and the block-indexed history."""

from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path

import pytest

import cascade.shared.activation as A
from cascade.shared import netconfig as nc
from cascade.shared.config import load_chain_config, parse_chain_config
from cascade.shared.hippius import ObjectNotFound

REPO = Path(__file__).resolve().parents[2]
with (REPO / "chain.toml").open("rb") as _fh:
    SHIPPED_RAW = tomllib.load(_fh)

# A typed DEC-CA-0043 rollover (3600 -> 900 grid at R, eras of 4 x 900 after
# it), so only the overlay's own cfg- segment is ever voted on in these tests.
R = 9050400
GRID = 900
ERA = 3600
E1, E2, E3 = R + 2 * ERA, R + 3 * ERA, R + 4 * ERA


def owner_keypair(uri: str = "//Owner"):
    from bittensor.sp_core import Keypair

    return Keypair.create_from_uri(uri)


OWNER = owner_keypair()


def base_raw(owner: str | None = None) -> dict:
    raw = copy.deepcopy(SHIPPED_RAW)
    raw["round"].update(rolling_from_block=R, epoch_blocks=900, epoch_blocks_prev=3600,
                        epoch_activation_block=R)
    raw["scoring"].update(era_king_from_block=R, tenure_blocks_from_block=R,
                          cohort_maxt_increment_from_block=R, forfeit_hotkeys=[],
                          forfeit_from_block=0, forfeit_successor_hotkey="",
                          win_margin_start_v2=0.0, win_margin_end_v2=0.0,
                          margin_warmup_blocks_v2=0, margin_v2_from_block=0)
    raw["v2"]["prelaunch_burn"] = 0.0
    raw["v2"]["launch_at_utc"] = ""
    raw["v2"]["commit_open_at_utc"] = ""
    raw["v2"]["commit_from_block"] = 0
    raw["netconfig"] = {"owner_hotkey": OWNER.ss58_address if owner is None else owner}
    # DEC-CA-0063: once the validator bench source arms, [manifest]
    # validator_hotkey (the shipped local override) must name the same main
    # validator as owner_hotkey — the fixture's owner is that validator
    raw["manifest"]["validator_hotkey"] = raw["netconfig"]["owner_hotkey"]
    return raw


def base_cfg(owner: str | None = None):
    return parse_chain_config(base_raw(owner))


class MemStore:
    """Duck-typed manifest store (get_text / put_text)."""

    def __init__(self) -> None:
        self.objects: dict[str, str] = {}
        self.gets: list[str] = []

    def get_text(self, key: str) -> str:
        self.gets.append(key)
        if key not in self.objects:
            raise ObjectNotFound(key)
        return self.objects[key]

    def put_text(self, key, text, *, content_type="text/plain", acl=None) -> None:
        self.objects[key] = text


def publish(store: MemStore, overrides: dict, effective_block: int, previous=None,
            keypair=OWNER) -> dict:
    body = nc.next_overlay(overrides, effective_block, previous, network={"netuid": 91, "chain": "bittensor:finney"})
    nc.publish_overlay(store, nc.sign_overlay(body, keypair))
    return body


# ── the loader refactor ───────────────────────────────────────────────────────


def test_parse_chain_config_is_the_loader_and_does_not_mutate_its_input():
    raw = copy.deepcopy(SHIPPED_RAW)
    before = json.dumps(raw, sort_keys=True, default=str)
    assert parse_chain_config(raw) == load_chain_config(REPO / "chain.toml")
    assert json.dumps(raw, sort_keys=True, default=str) == before


def test_netconfig_table_defaults_inert_and_refuses_typos():
    for path in ("chain.toml", "chain.testnet.toml"):
        c = load_chain_config(REPO / path)
        if path == "chain.toml":
            assert c.netconfig.owner_hotkey == c.manifest.validator_hotkey
            assert c.netconfig.enabled
        else:
            assert c.netconfig.owner_hotkey == "" and not c.netconfig.enabled
    raw = copy.deepcopy(SHIPPED_RAW)
    raw.pop("netconfig", None)
    missing = parse_chain_config(raw).netconfig
    assert missing.owner_hotkey == "" and not missing.enabled
    raw["netconfig"] = {"owner_hotkeys": "x"}
    with pytest.raises(ValueError, match="unknown keys"):
        parse_chain_config(raw)
    raw = base_raw()
    raw["activation"]["feature"] = ""
    with pytest.raises(ValueError, match="activation"):
        parse_chain_config(raw)


# ── inert ─────────────────────────────────────────────────────────────────────


def test_empty_owner_is_completely_inert(tmp_path):
    cfg = base_cfg(owner="")
    store = MemStore()
    publish(store, {"scoring": {"min_windows": 7}}, E1)
    res = nc.NetConfigResolver(cfg, state_path=tmp_path / "s.json", store=store, signal=True)
    assert not res.enabled
    assert res.refresh(cfg, force=True) is False and store.gets == []
    assert res.tick(cfg, object(), now_block=E2) is False
    assert res.config_at(cfg, E3) is cfg and res.segment() is None
    assert not (tmp_path / "s.json").exists()
    # the note is byte-identical with or without the (absent) segment
    rec = A.ActivationRecord()
    assert A.own_signal_payload(cfg, rec) == A.own_signal_payload(cfg, rec, None, None, None,
                                                                 None, None)


# ── signature, seq and prev ──────────────────────────────────────────────────


def test_chain_verifies_signature_seq_and_prev():
    store = MemStore()
    b1 = publish(store, {"scoring": {"min_windows": 7}}, E1)
    b2 = publish(store, {"scoring": {"min_windows": 8}}, E2, b1)
    envs = nc.fetch_overlays(store, owner=OWNER.ss58_address)
    assert [e["body"] for e in envs] == [b1, b2]
    assert nc.fetch_overlays(store, owner=OWNER.ss58_address, last=b2) == []
    assert [e["body"] for e in nc.fetch_overlays(store, owner=OWNER.ss58_address, last=b1)] == [b2]
    # another key's signature
    with pytest.raises(nc.NetConfigError):
        nc.fetch_overlays(store, owner=owner_keypair("//Mallory").ss58_address)
    # a tampered body
    bad = MemStore()
    bad.objects = dict(store.objects)
    env = json.loads(bad.objects[nc.LATEST_KEY])
    env["body"]["overrides"]["scoring"]["min_windows"] = 1
    bad.objects[nc.LATEST_KEY] = json.dumps(env)
    with pytest.raises(nc.NetConfigError, match="signature"):
        nc.fetch_overlays(bad, owner=OWNER.ss58_address)


def test_rollback_fork_and_broken_chain_are_refused():
    store = MemStore()
    b1 = publish(store, {"scoring": {"min_windows": 7}}, E1)
    b2 = publish(store, {"scoring": {"min_windows": 8}}, E2, b1)
    # latest rolled back to seq 1 after seq 2 was accepted
    old = MemStore()
    publish(old, {"scoring": {"min_windows": 7}}, E1)
    with pytest.raises(nc.NetConfigError, match="rolled back"):
        nc.fetch_overlays(old, owner=OWNER.ss58_address, last=b2)
    # a second, different body under seq 2
    fork = MemStore()
    fork.objects = dict(store.objects)
    publish(fork, {"scoring": {"min_windows": 9}}, E2, b1)
    with pytest.raises(nc.NetConfigError, match="share seq"):
        nc.fetch_overlays(fork, owner=OWNER.ss58_address, last=b2)
    # seq 2 that does not chain from the seq 1 this node holds
    other_b1 = nc.next_overlay({"scoring": {"min_windows": 6}}, E1, None, network={"netuid": 91, "chain": "bittensor:finney"})
    with pytest.raises(nc.NetConfigError, match="does not chain"):
        nc.fetch_overlays(store, owner=OWNER.ss58_address, last=other_b1)
    # a gap in the published chain
    gap = MemStore()
    gap.objects = dict(store.objects)
    del gap.objects[nc.seq_key(1)]
    with pytest.raises(nc.NetConfigError, match="missing"):
        nc.fetch_overlays(gap, owner=OWNER.ss58_address)
    with pytest.raises(nc.NetConfigError, match="prev"):
        nc.check_body({**b2, "prev": None})


# ── the allow-list ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("overrides", [
    {"training": {"train_seed_salt": "x"}},
    {"v2": {"training": {"arch_preset": "x"}}},
    {"manifest": {"trainer_hotkey": "5x"}},
    {"storage": {"manifest_bucket": "evil"}},
    {"storage": {"pool_bucket": "evil"}},
    {"chain": {"bittensor": "12"}},
    {"netconfig": {"owner_hotkey": "5x"}},
    {"activation": {"threshold": 0.1}},
    {"validator": {"force_burn": True}},
    {"round": {"funded_pods": "off"}},
    {"generator": {"max_length": 1}},
])
def test_allow_list_refuses_release_only_keys(overrides):
    with pytest.raises(nc.NetConfigError, match="release-only"):
        nc.check_overrides(overrides)
    with pytest.raises(nc.NetConfigError):
        nc.next_overlay(overrides, E1, None, network={"netuid": 91, "chain": "bittensor:finney"})


def test_allow_list_accepts_verdict_and_payout_tables():
    cfg = base_cfg()
    body = nc.next_overlay({"scoring": {"min_windows": 7}, "eval": {"n_windows": 1100},
                            "lane_b": {"share": 0.4}, "v2": {"carryover_king_share": 0.04},
                            "round": {"blocked_hotkeys": ["5Blocked"]}}, E1, None, network={"netuid": 91, "chain": "bittensor:finney"})
    out = nc.validate_overlay(cfg, body)
    assert out.scoring.min_windows == 7 and out.eval.n_windows == 1100
    assert out.lane_b.share == 0.4 and out.v2.carryover_king_share == 0.04
    assert out.round.blocked_hotkeys == ("5Blocked",)
    assert out.netconfig.seq == 1 and out.netconfig.digest == nc.body_digest(body)
    # untouched sections are the base's
    assert out.training == cfg.training and out.storage == cfg.storage


def test_a_key_no_parser_reads_is_refused():
    with pytest.raises(nc.NetConfigError, match="no effect"):
        nc.validate_overlay(base_cfg(), nc.next_overlay({"scoring": {"min_windowz": 1}}, E1, None, network={"netuid": 91, "chain": "bittensor:finney"}))


def test_runtime_tuning_keeps_old_settlements_and_pins_unchanged():
    cfg = base_cfg()
    overrides = {
        "round": {"funded_field_cap": 2},
        "v2": {"bench_report_timeout_settlements": 3, "bench_budget_hours": 3.0,
               "bench_sweep_hours": 1.0},
        "marketcast": {"gain_margin": 0.02, "coverage_tolerance": 0.04,
                       "min_windows": 220, "report_timeout_settlements": 4,
                       "bootstrap_B": 500},
    }
    body = nc.next_overlay(overrides, E2, None, network={"netuid": 91, "chain": "bittensor:finney"})
    out = nc.validate_overlay(cfg, body)
    assert out.v2.bench_report_timeout_settlements == 3
    assert out.round.funded_field_cap == 2
    assert out.v2.bench_budget_hours == 3.0 and out.v2.bench_sweep_hours == 1.0
    assert out.marketcast.gain_margin == 0.02 and out.marketcast.min_windows == 220
    assert out.marketcast.bootstrap_B == 500
    assert out.marketcast.base_ref == cfg.marketcast.base_ref
    assert out.marketcast.data_sources == cfg.marketcast.data_sources
    assert out.manifest == cfg.manifest and out.storage == cfg.storage
    history = [nc.Adopted(apply_block=E2, seq=body["seq"], digest=nc.body_digest(body),
                          envelope=nc.sign_overlay(body, OWNER))]
    old = nc.config_from_history(cfg, history, E2 - 1)
    new = nc.config_from_history(cfg, history, E2)
    assert old.marketcast == cfg.marketcast and old.v2 == cfg.v2
    assert new.marketcast == out.marketcast and new.v2 == out.v2


def test_bench_policy_switch_checks_identity_alignment_and_retroactivity():
    raw = base_raw()
    raw["v2"].update(from_block=R, bench_source="trainer", bench_source_from_block=0,
                     bench_gate_v2_from_block=0)
    cfg = parse_chain_config(raw)
    ov = {"v2": {"bench_source": "validator", "bench_source_from_block": E2,
                 "bench_gate_v2_from_block": E2}}
    out = nc.validate_overlay(cfg, nc.next_overlay(ov, E1, None, network={"netuid": 91, "chain": "bittensor:finney"}))
    assert out.v2.bench_source == "validator"
    assert nc.retroactive_changes(cfg, out, E2) == []
    assert any("bench_source_from_block" in why
               for why in nc.retroactive_changes(cfg, out, E3))
    with pytest.raises(nc.NetConfigError, match="era boundary"):
        nc.validate_overlay(cfg, nc.next_overlay(
            {"v2": {"bench_source_from_block": E2 + GRID,
                    "bench_source": "validator"}}, E1, None, network={"netuid": 91, "chain": "bittensor:finney"}))
    raw["netconfig"]["owner_hotkey"] = ""
    raw["manifest"]["validator_hotkey"] = ""
    unpinned = parse_chain_config(raw)
    with pytest.raises(nc.NetConfigError, match="owner_hotkey"):
        nc.validate_overlay(unpinned, nc.next_overlay(ov, E1, None, network={"netuid": 91, "chain": "bittensor:finney"}))


@pytest.mark.parametrize("overrides", [
    {"v2": {"bench_budget_hours": 1.0, "bench_sweep_hours": 2.0}},
    {"v2": {"bench_report_timeout_settlements": 0}},
    {"marketcast": {"gain_margin": 0.0}},
    {"marketcast": {"alpha": 2.0}},
    {"marketcast": {"bootstrap_B": 50}},
    {"marketcast": {"full_pay_tiers": [[0.01, 100000]]}},
])
def test_new_runtime_controls_still_pass_the_loader(overrides):
    with pytest.raises(nc.NetConfigError, match="loader"):
        nc.validate_overlay(base_cfg(), nc.next_overlay(overrides, E1, None, network={"netuid": 91, "chain": "bittensor:finney"}))


# ── merged validation ────────────────────────────────────────────────────────


def test_merged_table_runs_the_loader_checks():
    cfg = base_cfg()
    # [v2] from_block that is not an era boundary
    with pytest.raises(nc.NetConfigError, match="era boundary"):
        nc.validate_overlay(cfg, nc.next_overlay({"v2": {"from_block": E1 + GRID}}, E1, None, network={"netuid": 91, "chain": "bittensor:finney"}))
    # a malformed economics value
    with pytest.raises(nc.NetConfigError, match="loader"):
        nc.validate_overlay(cfg, nc.next_overlay({"v2": {"carryover_king_share": 1.5}}, E1,
                                                 None, network={"netuid": 91, "chain": "bittensor:finney"}))
    # effective_block itself must be an era start
    with pytest.raises(nc.NetConfigError, match="era start"):
        nc.validate_overlay(cfg, nc.next_overlay({"scoring": {"min_windows": 7}}, E1 + GRID,
                                                 None, network={"netuid": 91, "chain": "bittensor:finney"}))
    # an era-aligned v2 go-live passes the same checks
    ok = nc.validate_overlay(cfg, nc.next_overlay({"v2": {"from_block": E2}}, E1, None, network={"netuid": 91, "chain": "bittensor:finney"}))
    assert ok.v2.from_block == E2 and ok.training.v2_from_block == E2


# ── retroactivity and the apply block ────────────────────────────────────────


def test_apply_block_is_effective_or_first_era_start_after_a_late_lock():
    cfg = base_cfg()
    body = nc.next_overlay({"scoring": {"min_windows": 7}}, E2, None, network={"netuid": 91, "chain": "bittensor:finney"})
    early = A.ActivationRecord(lock_block=E1 + GRID, activation_block=E2)
    assert nc.apply_block_for(body, early, cfg.round) == E2
    late = A.ActivationRecord(lock_block=E2 + GRID, activation_block=E3)
    assert nc.apply_block_for(body, late, cfg.round) == E3
    on_it = A.ActivationRecord(lock_block=E2, activation_block=E3)
    assert nc.apply_block_for(body, on_it, cfg.round) == E3


def test_lane_b_launch_overlay_is_independent_and_cannot_be_retroactive():
    raw = base_raw()
    raw["v2"]["from_block"] = R
    cfg = parse_chain_config(raw)
    assert cfg.lane_b.from_block == 0
    body = nc.next_overlay({"lane_b": {"from_block": E2}}, E1, None, network={"netuid": 91, "chain": "bittensor:finney"})
    launched = nc.validate_overlay(cfg, body)
    assert launched.v2.from_block == R and launched.lane_b.from_block == E2 + 1
    assert not launched.lane_b.active(E2) and launched.lane_b.active(E2 + GRID)
    assert nc.retroactive_changes(cfg, launched, E1) == []
    assert any("lane_b" in why for why in nc.retroactive_changes(cfg, launched, E3))


def test_retroactive_switch_blocks_and_grid_moves_are_refused():
    cfg = base_cfg()
    new = nc.build_config(cfg, nc.next_overlay({"scoring": {"entry_fault_from_block": E2}},
                                               E2, None, network={"netuid": 91, "chain": "bittensor:finney"}))
    assert nc.retroactive_changes(cfg, new, E2) == []
    why = nc.retroactive_changes(cfg, new, E3)
    assert why and "entry_fault_from_block" in why[0]
    # an unchanged past block is fine (snapshots restate keys)
    same = nc.build_config(cfg, nc.next_overlay({"scoring": {"mv_score_from_block": 9068400}},
                                                E2, None, network={"netuid": 91, "chain": "bittensor:finney"}))
    assert nc.retroactive_changes(cfg, same, E3) == []
    # the grid / era length before the apply block never moves
    grid = nc.build_config(cfg, nc.next_overlay({"round": {"era_settlements": 2}}, E2, None, network={"netuid": 91, "chain": "bittensor:finney"}))
    assert any("grid" in w for w in nc.retroactive_changes(cfg, grid, E2))


# ── history and config_at ────────────────────────────────────────────────────


def _adopted(cfg, body, apply_block):
    return nc.Adopted(apply_block=apply_block, seq=body["seq"], digest=nc.body_digest(body),
                      envelope=nc.sign_overlay(body, OWNER))


def test_config_at_walks_the_history_by_block():
    cfg = base_cfg()
    b1 = nc.next_overlay({"scoring": {"min_windows": 7}}, E1, None, network={"netuid": 91, "chain": "bittensor:finney"})
    b2 = nc.next_overlay({"scoring": {"min_windows": 8}, "eval": {"n_windows": 1100}}, E2, b1, network={"netuid": 91, "chain": "bittensor:finney"})
    hist = [_adopted(cfg, b1, E1), _adopted(cfg, b2, E2)]
    cache: dict = {}
    assert nc.config_from_history(cfg, hist, E1 - 1, cache) is cfg
    at1 = nc.config_from_history(cfg, hist, E1, cache)
    assert at1.scoring.min_windows == 7 and at1.eval == cfg.eval and at1.netconfig.seq == 1
    assert nc.config_from_history(cfg, hist, E2 - 1, cache) is at1          # cached
    at2 = nc.config_from_history(cfg, hist, E3, cache)
    # snapshots, not deltas: seq 2 restates what it keeps
    assert at2.scoring.min_windows == 8 and at2.eval.n_windows == 1100
    assert at2.netconfig.applied_block == E2
    assert nc.config_from_history(cfg, hist, None) is cfg


def test_resolved_activation_blocks_survive_an_overlay():
    raw = copy.deepcopy(SHIPPED_RAW)
    raw["netconfig"] = {"owner_hotkey": OWNER.ss58_address}
    raw["scoring"].update(forfeit_hotkeys=[], win_margin_start_v2=0.0, win_margin_end_v2=0.0,
                          margin_warmup_blocks_v2=0)
    raw["v2"]["prelaunch_burn"] = 0.0
    shipped = parse_chain_config(raw)
    armed = A.apply_activation(shipped, R)              # the fleet's DEC-CA-0045 decision
    body = nc.next_overlay({"scoring": {"min_windows": 7}}, R + 3 * ERA, None, network={"netuid": 91, "chain": "bittensor:finney"})
    out = nc.build_config(armed, body)
    assert out.round.rolling_from_block == R and out.round.epoch_blocks == 900
    assert A.resolved_rollover(out) == R and A.configured_rollover(out) == 0
    # an overlay that TYPES the decided block makes it typed (its segment drops)
    typed = nc.build_config(armed, nc.next_overlay({"round": {"rolling_from_block": R}},
                                                   R + 3 * ERA, None, network={"netuid": 91, "chain": "bittensor:finney"}))
    assert A.resolved_rollover(typed) == 0 and A.configured_rollover(typed) == R


def test_state_round_trips_atomically(tmp_path):
    cfg = base_cfg()
    b1 = nc.next_overlay({"scoring": {"min_windows": 7}}, E1, None, network={"netuid": 91, "chain": "bittensor:finney"})
    st = nc.NetConfigState(last=b1, candidate=nc.sign_overlay(b1, OWNER), refused=["ab"],
                           history=[_adopted(cfg, b1, E1)],
                           record=A.ActivationRecord(feature="cfg-x", lock_block=R,
                                                     activation_block=R + ERA, source="tally"))
    p = tmp_path / "netconfig_state.json"
    from cascade.shared.atomic import write_text_atomic

    write_text_atomic(p, st.to_json(), backup=True)
    back = nc.load_state(p)
    assert back.history == st.history and back.record == st.record and back.last == b1
    p.write_text("{torn")
    with pytest.raises(nc.NetConfigStateUnreadable):
        nc.load_state(p)
    write_text_atomic(p, st.to_json(), backup=True)
    write_text_atomic(p, st.to_json(), backup=True)
    p.write_text("{torn")
    assert nc.load_state(p).history == st.history          # the .bak


# ── the note ─────────────────────────────────────────────────────────────────


def test_note_trimming_drops_older_segments_before_the_cfg_segment():
    shipped = load_chain_config(REPO / "chain.toml")      # forfeit + margin v2 + pre-launch armed
    assert A.forfeit_feature(shipped) and A.margin_v2_feature(shipped)
    assert A.prelaunch_burn_feature(shipped)

    def locked(spec):
        return A.ActivationRecord(feature=spec.name, lock_block=R, activation_block=R + 3600,
                                  source="tally")

    primary = locked(A.primary_feature(shipped))
    recs = (locked(A.forfeit_feature(shipped)), locked(A.margin_v2_feature(shipped)),
            A.ActivationRecord())
    seg = A.ReadySignal("cfg-0123abcd", lock_block=R, activation_block=R + 3600)
    note = A.own_signal_payload(shipped, primary, *recs, seg)
    assert len(note.encode()) <= A.NOTE_MAX_BYTES
    assert A.signal_for(note, "cfg-0123abcd") == seg
    assert A.signal_for(note, "prelaunch-95") is None        # the newest extra went first
    # without the cfg segment the note is exactly the pre-DEC-CA-0057 one
    assert A.own_signal_payload(shipped, primary, *recs) == A.own_signal_payload(
        shipped, primary, *recs, None, None)
    # an in-force overlay that types a decided block drops that segment
    typed = parse_chain_config({**copy.deepcopy(shipped.raw),
                                "scoring": {**shipped.raw["scoring"],
                                            "forfeit_from_block": R + 3600}})
    trimmed = A.own_signal_payload(shipped, primary, *recs, seg, typed)
    assert A.signal_for(trimmed, A.forfeit_feature(shipped).name) is None
    assert A.signal_for(trimmed, "cfg-0123abcd") == seg


def test_cfg_segment_alone_keeps_a_typed_rollover_note_alive():
    cfg = base_cfg()
    rec = A.ActivationRecord()
    assert A.own_signal_payload(cfg, rec) is None             # typed rollover: inert
    note = A.own_signal_payload(cfg, rec, netconfig_segment=A.ReadySignal("cfg-0123abcd"))
    assert note == "cascade-ready:1:rolling-era-king:0:0:cfg-0123abcd:0:0"
    assert A.parse_signals(note) is not None


# ── the publish plan ─────────────────────────────────────────────────────────


def test_plan_overlay_prints_the_diff_and_every_check():
    cfg = base_cfg()
    plan = nc.plan_overlay(cfg, {"scoring": {"min_windows": 7}}, E1, None, head_block=R)
    assert plan.ok and any("min_windows" in d for d in plan.diff)
    out = plan.render()
    assert "VALID" in out and nc.cfg_feature_name(plan.body) in out
    bad = nc.plan_overlay(cfg, {"training": {"x": 1}}, E1, None)
    assert not bad.ok and "INVALID" in bad.render()
    retro = nc.plan_overlay(cfg, {"scoring": {"entry_fault_from_block": E1}}, E2, None)
    assert not retro.ok and "entry_fault_from_block" in retro.render()
