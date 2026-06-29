"""End-to-end validator round with an injected evaluator (no HF/torch).

Exercises the manifest gate → eval → KOTH → state path, plus the trainer's
pairing logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from metronome.eval.scoring import WindowScore
from metronome.shared.chain import Commitment
from metronome.shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    format_trained_pointer,
)
from metronome.trainer.loop import plan_round, resolve_commitments
from metronome.validator.loop import ValidatorRunner
from metronome.validator.state import genesis

CID = "alice/gen@sha256:" + "a" * 64
CID2 = "metronome/ckpt@sha256:" + "b" * 64


def _scores(scale, seed, n=300):
    rng = np.random.default_rng(seed)
    return [
        WindowScore(
            series_id=str(i),
            mase=float(rng.uniform(0.5, 1.5) * scale),
            qloss_per_q=rng.uniform(0.1, 1.0, size=9) * scale,
            abs_target=float(rng.uniform(5.0, 10.0)),
        )
        for i in range(n)
    ]


def _manifest(cfg):
    entries = [
        TrainedEntry("king_hk", 0, "king", CID, format_trained_pointer(CID2), "d", 10),
        TrainedEntry("chal_hk", 1, "challenger", CID, format_trained_pointer(CID2), "d", 10),
    ]
    return TrainingManifest(
        round_id="1",
        created_block=10,
        contract_digest=contract_digest(cfg.training),
        base_arch_digest=cfg.training.base_arch_digest,
        eval_dataset=cfg.eval.eval_dataset,
        entries=entries,
    )


def test_process_round_strong_challenger_wins(cfg):
    # Challenger scores share the king's windows (paired abs_target) at 0.6x.
    king_scores = _scores(1.0, 0)
    chal_scores = [WindowScore(s.series_id, s.mase * 0.6, s.qloss_per_q * 0.6, s.abs_target) for s in king_scores]

    def fake_eval(entry, windows):
        return king_scores if entry.role == "king" else chal_scores

    runner = ValidatorRunner(cfg=cfg, state=genesis("king_hk", 0), evaluate_fn=fake_eval, verify_signatures=False)
    outcome = runner.process_round(_manifest(cfg), windows=[], base_seed=7)
    assert outcome is not None
    assert outcome.result.challenger_wins_round
    # chain.toml ships dethrone_cp = 1, so one winning round takes the throne.
    assert outcome.transition.dethroned
    assert runner.state.king_hotkey == "chal_hk"


def test_win_below_cp_only_increments_streak(cfg):
    # With a sticky (dethrone_cp > 1) config, a single win records the streak but
    # does NOT dethrone — the multi-round path is still intact when configured.
    from dataclasses import replace

    cfg = replace(cfg, scoring=replace(cfg.scoring, dethrone_cp=2))
    king_scores = _scores(1.0, 0)
    chal_scores = [WindowScore(s.series_id, s.mase * 0.6, s.qloss_per_q * 0.6, s.abs_target) for s in king_scores]

    def fake_eval(entry, windows):
        return king_scores if entry.role == "king" else chal_scores

    runner = ValidatorRunner(cfg=cfg, state=genesis("king_hk", 0), evaluate_fn=fake_eval, verify_signatures=False)
    outcome = runner.process_round(_manifest(cfg), windows=[], base_seed=7)
    assert outcome.result.challenger_wins_round
    assert not outcome.transition.dethroned
    assert runner.state.streaks.get("chal_hk") == 1


def test_process_round_rejects_contract_mismatch(cfg):
    m = _manifest(cfg)
    bad = TrainingManifest(
        round_id=m.round_id,
        created_block=m.created_block,
        contract_digest="0" * 64,  # wrong
        base_arch_digest=m.base_arch_digest,
        eval_dataset=m.eval_dataset,
        entries=m.entries,
    )
    runner = ValidatorRunner(cfg=cfg, state=genesis("king_hk", 0), evaluate_fn=lambda e, w: [], verify_signatures=False)
    assert runner.process_round(bad, windows=[], base_seed=1) is None


def test_dethrone_after_consecutive_wins(cfg):
    king_scores = _scores(1.0, 0)
    chal_scores = [WindowScore(s.series_id, s.mase * 0.5, s.qloss_per_q * 0.5, s.abs_target) for s in king_scores]

    def fake_eval(entry, windows):
        return king_scores if entry.role == "king" else chal_scores

    runner = ValidatorRunner(cfg=cfg, state=genesis("king_hk", 0), evaluate_fn=fake_eval, verify_signatures=False)
    dethroned = False
    for r in range(cfg.scoring.dethrone_cp):
        outcome = runner.process_round(_manifest(cfg), windows=[], base_seed=r)
        dethroned = outcome.transition.dethroned
    assert dethroned
    assert runner.state.king_hotkey == "chal_hk"


def test_process_round_is_atomic_on_eval_failure(cfg):
    # The live loop marks a round consumed only after process_round returns, which
    # is safe only because a transient eval/fetch error leaves champion state
    # UNCHANGED (so the retry can't double-count the streak/tenure).
    def boom(entry, windows):
        raise RuntimeError("registry fetch failed")

    runner = ValidatorRunner(cfg=cfg, state=genesis("king_hk", 0), evaluate_fn=boom,
                             verify_signatures=False)
    before = runner.state
    with pytest.raises(RuntimeError):
        runner.process_round(_manifest(cfg), windows=[], base_seed=1)
    assert runner.state is before  # no mutation ⇒ clean retry


def test_reward_uids_include_registered_former_kings(cfg):
    import types

    from metronome.validator.state import ChampionState

    # Current king is uid 0 (manifest king); former court = ["fk1", "fk2"], but
    # only fk1 is still registered (fk2 deregistered ⇒ dropped).
    state = ChampionState(king_hotkey="king_hk", king_uid=0, former_kings=("fk1", "fk2"))
    runner = ValidatorRunner(cfg=cfg, state=state, evaluate_fn=lambda e, w: [], verify_signatures=False)
    client = types.SimpleNamespace(uid_for_hotkey=lambda hk: {"fk1": 5}.get(hk))
    uids = runner._reward_uids(_manifest(cfg), None, client)
    assert uids == [0, 5]


def test_reward_uids_empty_when_no_king(cfg):
    import types

    # No manifest king and an empty throne ⇒ empty reward set; the loop then
    # burns to burn_uid (teutonic-style) rather than skipping the weight-set.
    runner = ValidatorRunner(cfg=cfg, evaluate_fn=lambda e, w: [], verify_signatures=False)
    empty = TrainingManifest(
        round_id="1", created_block=10,
        contract_digest=contract_digest(cfg.training),
        base_arch_digest=cfg.training.base_arch_digest,
        eval_dataset=cfg.eval.eval_dataset, entries=[],
    )
    client = types.SimpleNamespace(uid_for_hotkey=lambda hk: None)
    assert runner._reward_uids(empty, None, client) == []


def test_vote_prefers_manifest_king_without_dethrone(cfg):
    runner = ValidatorRunner(cfg=cfg, evaluate_fn=lambda e, w: [], verify_signatures=False)
    # No outcome (e.g. king-only round) ⇒ keep voting the manifest's king (uid 0).
    assert runner._king_uid_to_vote(_manifest(cfg), None) == 0


def test_vote_switches_to_new_king_on_dethrone(cfg):
    import types

    runner = ValidatorRunner(cfg=cfg, state=genesis("chal_hk", 1), evaluate_fn=lambda e, w: [],
                             verify_signatures=False)
    dethroned = types.SimpleNamespace(transition=types.SimpleNamespace(dethroned=True))
    assert runner._king_uid_to_vote(_manifest(cfg), dethroned) == 1  # state's (new) king


def test_trainer_pairing_logic():
    commits = [
        Commitment(uid=0, hotkey="a", coldkey=None, payload=f"metro-v1:gen:hippius:{CID}", commit_block=5),
        Commitment(uid=1, hotkey="b", coldkey=None, payload=f"metro-v1:gen:hippius:{CID2}", commit_block=6),
        Commitment(uid=2, hotkey="c", coldkey=None, payload="garbage", commit_block=7),
    ]
    resolved = resolve_commitments(commits)
    assert len(resolved) == 2  # garbage dropped
    plan = plan_round(resolved, king_hotkey="a")
    assert plan.king.hotkey == "a"
    assert [c.hotkey for c in plan.challengers] == ["b"]


def test_trainer_pairing_promotes_interim_king_when_absent():
    commits = [
        Commitment(uid=3, hotkey="x", coldkey=None, payload=f"metro-v1:gen:hippius:{CID}", commit_block=5),
    ]
    plan = plan_round(resolve_commitments(commits), king_hotkey=None)
    assert plan.king.hotkey == "x"
    assert plan.challengers == []
