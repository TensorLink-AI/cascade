"""DEC-CA-0055 config keys, the pre-launch burn vote (DEC-CA-0045 secondary
feature), and the drop-when-default wire fields (manifest ``funded_by``, the
bench report's ``init`` role, receipt stamps, champion-state ladder)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cascade.shared import activation as A
from cascade.shared.bench_report import BenchEntry, BenchReport, load_bench_report
from cascade.shared.config import load_chain_config
from cascade.shared.manifest import (
    BenchScores,
    TrainedEntry,
    TrainingManifest,
    format_trained_pointer,
    load_manifest,
)
from cascade.validator import state as state_mod
from cascade.validator.state import ChampionState

ROOT = Path(__file__).resolve().parents[2]
CID = "alice/gen@sha256:" + "a" * 64
CKPT = format_trained_pointer("cascade/ckpt@sha256:" + "b" * 64)


def test_mainnet_and_testnet_values():
    v2 = load_chain_config(ROOT / "chain.toml").v2
    assert (v2.random_init_eras, v2.reign_warm_start_blocks, v2.prelaunch_burn,
            v2.carryover_king_share, v2.king_ladder) == (14, 21600, 0.95, 0.05, True)
    assert v2.prelaunch_burn_from_block == 0                  # decided by the vote
    t = load_chain_config(ROOT / "chain.testnet.toml").v2
    assert (t.random_init_eras, t.reign_warm_start_blocks, t.prelaunch_burn,
            t.carryover_king_share, t.king_ladder) == (0, 0, 0.0, 0.0, False)


@pytest.mark.parametrize("key,value", [
    ("prelaunch_burn", "1.0"), ("prelaunch_burn", "-0.1"), ("carryover_king_share", "1.5"),
    ("reign_warm_start_blocks", "-1"), ("prelaunch_burn_from_block", "1.5"),
])
def test_loader_refuses_bad_values(tmp_path, key, value):
    text = (ROOT / "chain.toml").read_text()
    import re

    text = re.sub(rf"(?m)^{key}\s*=.*$", "", text)
    text = text.replace("king_ladder = true", f"king_ladder = true\n{key} = {value}", 1)
    p = tmp_path / "chain.toml"
    p.write_text(text)
    with pytest.raises(ValueError):
        load_chain_config(p)


def test_loader_refuses_non_bool_ladder(tmp_path):
    text = (ROOT / "chain.toml").read_text().replace("king_ladder = true", "king_ladder = 1")
    p = tmp_path / "chain.toml"
    p.write_text(text)
    with pytest.raises(ValueError, match="king_ladder"):
        load_chain_config(p)


# ── the pre-launch burn vote ─────────────────────────────────────────────────


def test_prelaunch_feature_rides_the_note():
    cfg = load_chain_config(ROOT / "chain.toml")
    spec = A.prelaunch_burn_feature(cfg)
    assert spec is not None and spec.name == "prelaunch-95-r2" and spec.decided_on_chain
    assert A.prelaunch_burn_feature_name(0.955) == "prelaunch-95.5"
    assert A.prelaunch_burn_feature_name(0.0) == ""
    note = A.own_signal_payload(cfg, A.ActivationRecord())
    assert note.endswith(":prelaunch-95-r2:0:0")
    # typed ⇒ owner override, nothing signalled for it
    typed = replace(cfg, v2=replace(cfg.v2, prelaunch_burn_from_block=7200))
    assert not A.prelaunch_burn_feature(typed).decided_on_chain
    assert "prelaunch" not in (A.own_signal_payload(typed, A.ActivationRecord()) or "")


def test_note_never_exceeds_the_commitment_field():
    cfg = load_chain_config(ROOT / "chain.toml")
    lp = A.ActivationRecord(feature=A.primary_feature(cfg).name, lock_block=91830000,
                            activation_block=91833600)
    lf = A.ActivationRecord(feature=A.forfeit_feature(cfg).name, lock_block=91854000,
                            activation_block=91854900)
    lm = A.ActivationRecord(feature=A.margin_v2_feature(cfg).name, lock_block=91863000,
                            activation_block=91863900)
    note = A.own_signal_payload(cfg, lp, lf, lm, A.ActivationRecord())
    assert len(note.encode()) <= A.NOTE_MAX_BYTES


def test_apply_prelaunch_activation_is_one_way_and_boundary_aligned():
    cfg = load_chain_config(ROOT / "chain.toml")
    eb = cfg.round.epoch_blocks
    b = 2600 * eb
    armed = A.apply_prelaunch_burn_activation(cfg, b)
    assert armed.v2.prelaunch_burn_from_block == b
    assert A.resolved_prelaunch_burn_block(armed) == b
    assert A.configured_prelaunch_burn_block(armed) == 0     # resolved, not typed
    assert A.apply_prelaunch_burn_activation(armed, b) is armed
    with pytest.raises(ValueError, match="one-way"):
        A.apply_prelaunch_burn_activation(armed, b + eb)
    with pytest.raises(ValueError, match="boundary"):
        A.apply_prelaunch_burn_activation(cfg, b + 1)
    # the audit replays a receipt under the block it stamps
    from types import SimpleNamespace

    rc = A.apply_receipt_activation(cfg, SimpleNamespace(prelaunch_burn_block=b))
    assert rc.v2.prelaunch_burn_from_block == b


# ── wire fields ──────────────────────────────────────────────────────────────


def _manifest(funded_by=""):
    return TrainingManifest(round_id="1", created_block=10, contract_digest="c",
                            base_arch_digest="d", eval_dataset="e",
                            entries=[TrainedEntry("k", 0, "king", CID, CKPT, "d", 10,
                                                  funded_by=funded_by)])


def test_funded_by_is_dropped_when_empty_and_signed_when_set():
    old, new = _manifest(), _manifest("operator")
    assert b"funded_by" not in old.canonical_body()
    assert b'"funded_by":"operator"' in new.canonical_body()
    from cascade.shared.manifest import dump_manifest

    assert load_manifest(dump_manifest(new)).entries[0].funded_by == "operator"
    with pytest.raises(ValueError, match="funded_by"):
        TrainedEntry("k", 0, "king", CID, CKPT, "d", 10, funded_by="someone")


def test_bench_report_accepts_one_init_entry():
    s = BenchScores(*[1.0] * 6)
    rep = BenchReport(round_id="1", created_block=1, entries=(
        BenchEntry(role="king", size="m", miner_hotkey="k", miner_uid=0,
                   trained_pointer=CKPT, scores=s),
        BenchEntry(role="init", size="m", miner_hotkey="", miner_uid=-1,
                   trained_pointer=CKPT.replace("b" * 64, "c" * 64), scores=s)))
    from cascade.shared.bench_report import dump_bench_report

    back = load_bench_report(dump_bench_report(rep))
    assert back.init_entry() is not None and back.init_entry().role == "init"
    with pytest.raises(ValueError):
        BenchEntry(role="baseline", size="m", miner_hotkey="", miner_uid=-1,
                   trained_pointer=CKPT, scores=s)


def test_receipt_stamps_drop_when_default():
    from cascade.shared.receipt import RoundReceipt, load_receipt

    base = RoundReceipt(round_id="1", status="rejected", epoch_start_block=0,
                        epoch_block_hash="0x", base_seed=1, generation_seed=1,
                        training_seed=1, manifest={}, reject_reason="x")
    assert b"emission" not in base.canonical_body() and b"king_check" not in base.canonical_body()
    stamped = replace(base, prelaunch_burn_block=7200, emission={"boundary": 0},
                      king_check={"result": "deferred"})
    from cascade.shared.receipt import dump_receipt

    back = load_receipt(dump_receipt(stamped))
    assert (back.prelaunch_burn_block, back.emission, back.king_check) == (
        7200, {"boundary": 0}, {"result": "deferred"})


def test_champion_state_ladder_fields_roundtrip_and_drop_when_default():
    st = ChampionState(king_hotkey="k")
    assert "king_share_bp" not in state_mod.dumps(st)
    full = replace(st, king_share_bp=7000, court_hotkey="old", court_until_block=99,
                   king_slashed=True, king_check_pass={"block": 5, "king_pointer": "p"},
                   epoch_init={"epoch": 3, "init": ""})
    assert state_mod.loads(state_mod.dumps(full)) == full


def test_dethrone_resets_the_ladder_but_keeps_the_epoch_init():
    from cascade.eval.koth import RoundResult

    st = ChampionState(king_hotkey="k", king_share_bp=9000, king_check_pass={"block": 1},
                       epoch_init={"epoch": 3, "init": "x"}, court_hotkey="older",
                       court_until_block=10)
    win = RoundResult(challenger_wins_round=True, lcb=0.1, margin=0.01, n_windows=10,
                      king_geomean=1.0, chal_geomean=0.9, inconclusive=False)
    t = state_mod.apply_round(st, challenger_hotkey="c", challenger_uid=2, result=win,
                              dethrone_cp=1)
    assert t.dethroned and t.state.king_share_bp is None and t.state.king_check_pass is None
    assert t.state.court_hotkey is None and t.state.epoch_init == {"epoch": 3, "init": "x"}


def test_forfeit_clears_a_forfeited_court():
    st = ChampionState(king_hotkey="k", court_hotkey="bad", court_until_block=50)
    t = state_mod.apply_forfeit(st, forfeited={"bad"})
    assert t is not None and t.state.court_hotkey is None and t.state.court_until_block == 0
