"""`cascade score --replay-round` — rebuild a past round's verdict windows from a
revealed snapshot, verify them against the receipt, and judge a new challenger
against the king's signed receipt scores. Heavy train/eval steps are mocked."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cascade.eval.koth import evaluate_round
from cascade.eval.scoring import WindowScore
from cascade.miner import replay as replay_mod
from cascade.miner.replay import ReplayError
from cascade.shared.receipt import (
    EntryScores,
    EvalContext,
    VerdictRecord,
    WindowScoreRecord,
    build_receipt,
)
from cascade.trainer.contract import RoundSeeds
from cascade.validator import state as state_mod

from .receipt_fixture import BLOCK_HASH, make_manifest

POOL_SHA = "ab" * 32
LADDER_BLOCK = 5000        # >= the test gate below ⇒ scored horizon ladder
LEGACY_BLOCK = 500         # < the gate ⇒ single-horizon rotating draw
BASE_SEED = 424242


@pytest.fixture
def rcfg(cfg):
    """chain.toml with a small, test-gated ladder so a handful of series fills it."""
    ev = replace(cfg.eval, scored_horizons=(64, 256, 720), scored_from_block=1000,
                 n_windows=30)
    return replace(cfg, eval=ev)


def _write_snapshot(root: Path, sha: str = POOL_SHA) -> Path:
    snap = root / "snapshots" / f"2026-09-20-block-{LADDER_BLOCK}"
    snap.mkdir(parents=True)
    rng = np.random.default_rng(11)
    md = {}
    for dom in ("energy", "nature", "web"):
        for i in range(8):
            sid = f"{dom}{i:02d}"
            np.save(snap / f"{sid}.npy", rng.normal(size=4200).cumsum())
            md[sid] = {"freq": "h", "seasonal_period": 24, "domain": dom, "source": f"{dom}-feed"}
    (snap / "metadata.json").write_text(json.dumps(md))
    (snap / replay_mod.SHA_MARKER).write_text(sha + "\n")
    return snap


def _scores(windows, scale: float, seed: int) -> list[WindowScore]:
    rng = np.random.default_rng(seed)
    return [WindowScore(series_id=w.series_id, mase=float(rng.uniform(0.5, 1.5) * scale),
                        qloss_per_q=rng.uniform(0.1, 1.0, 9) * scale,
                        abs_target=float(rng.uniform(5.0, 10.0)),
                        source=f"src{i % 12}")
            for i, w in enumerate(windows)]


def _receipt(cfg, windows, king_scores, *, block: int, warm_start: str = "",
             era_start: int = 0, window_ids=None):
    params = replace(cfg.koth_params(), min_windows=10, min_clusters=0)
    manifest = replace(make_manifest(cfg, base_seed=BASE_SEED),
                       eval_pool_sha256=POOL_SHA, eval_pool_key="pool/snapshots/x.tar",
                       warm_start_ckpt=warm_start)
    size = cfg.training.arch_preset
    chal = [WindowScore(s.series_id, s.mase * 1.01, s.qloss_per_q * 1.01, s.abs_target,
                        source=s.source) for s in king_scores]
    result = evaluate_round(king_scores, chal, params, seed=BASE_SEED, king_tenure_rounds=0)
    transition = state_mod.apply_round(
        state_mod.genesis("king_hk", 0), challenger_hotkey="chal_hk", challenger_uid=1,
        result=result, dethrone_cp=params.dethrone_cp, keep_former_kings=1)
    seeds = RoundSeeds.derive(BASE_SEED, cfg.training)
    return build_receipt(
        round_id=str(BASE_SEED), status="scored", epoch_start_block=block,
        epoch_block_hash=BLOCK_HASH, base_seed=BASE_SEED, seeds=seeds, manifest=manifest,
        eval_context=EvalContext(
            pool_ref="pool/snapshots/x.tar", pool_digest=POOL_SHA,
            window_ids=tuple(window_ids if window_ids is not None
                             else (w.series_id for w in windows)),
            n_windows=len(windows), num_samples=cfg.eval.num_samples),
        entry_scores=(
            EntryScores("king", size, "king_hk", 0,
                        tuple(WindowScoreRecord.from_score(s) for s in king_scores)),
            EntryScores("challenger", size, "chal_hk", 1,
                        tuple(WindowScoreRecord.from_score(s) for s in chal)),
        ),
        verdict=VerdictRecord.from_round(result, transition, params=params,
                                         bootstrap_seed=BASE_SEED, king_tenure_rounds=0),
        era_start_block=era_start,
    )


@pytest.mark.parametrize("block", [LADDER_BLOCK, LEGACY_BLOCK])
def test_replay_rebuilds_the_rounds_exact_windows(rcfg, tmp_path, block):
    snap = _write_snapshot(tmp_path)
    windows = replay_mod.verdict_windows(rcfg, snap, BASE_SEED, block)
    assert windows
    if block == LADDER_BLOCK:   # the ladder: one rung per horizon
        assert {w.series_id.split("-")[0] for w in windows} == {"h64", "h256", "h720"}
    king = _scores(windows, 1.0, 0)
    receipt = _receipt(rcfg, windows, king, block=block, era_start=block - 100)

    rr = replay_mod.load_replay_round(rcfg, receipt, tmp_path)
    assert rr.snapshot_dir == snap
    assert [w.series_id for w in rr.windows] == [w.series_id for w in windows]
    assert rr.seeds.generation_seed == receipt.generation_seed
    assert rr.seeds.training_seed == receipt.training_seed
    assert rr.contract_block == block - 100
    assert [s.series_id for s in rr.king_scores] == [w.series_id for w in windows]


def test_replay_judges_against_the_kings_receipt_scores(rcfg, tmp_path):
    snap = _write_snapshot(tmp_path)
    windows = replay_mod.verdict_windows(rcfg, snap, BASE_SEED, LADDER_BLOCK)
    king = _scores(windows, 1.0, 0)
    rr = replay_mod.load_replay_round(
        rcfg, _receipt(rcfg, windows, king, block=LADDER_BLOCK), tmp_path)

    much_better = [WindowScore(s.series_id, s.mase * 0.6, s.qloss_per_q * 0.6,
                               s.abs_target, source=s.source) for s in king]
    assert replay_mod.judge(rr, much_better, rcfg).challenger_wins_round
    assert not replay_mod.judge(rr, king, rcfg).challenger_wins_round


def test_replay_refuses_windows_that_do_not_match_the_receipt(rcfg, tmp_path):
    snap = _write_snapshot(tmp_path)
    windows = replay_mod.verdict_windows(rcfg, snap, BASE_SEED, LADDER_BLOCK)
    king = _scores(windows, 1.0, 0)
    wrong_ids = [w.series_id for w in windows][::-1]   # a different draw
    receipt = _receipt(rcfg, windows, king, block=LADDER_BLOCK, window_ids=wrong_ids)
    with pytest.raises(ReplayError, match="window_ids"):
        replay_mod.load_replay_round(rcfg, receipt, tmp_path)


def test_replay_refuses_a_missing_snapshot(rcfg, tmp_path):
    snap = _write_snapshot(tmp_path, sha="cd" * 32)   # revealed folder of ANOTHER pool
    windows = replay_mod.verdict_windows(rcfg, snap, BASE_SEED, LADDER_BLOCK)
    receipt = _receipt(rcfg, windows, _scores(windows, 1.0, 0), block=LADDER_BLOCK)
    with pytest.raises(ReplayError, match="POOL_SHA256"):
        replay_mod.load_replay_round(rcfg, receipt, tmp_path)


def test_find_snapshot_dir_accepts_folder_snapshots_or_checkout(tmp_path):
    snap = _write_snapshot(tmp_path)
    for root in (snap, snap.parent, tmp_path):
        assert replay_mod.find_snapshot_dir(root, "sha256:" + POOL_SHA) == snap


def test_score_replay_full_budget_judges_and_short_budget_does_not(rcfg, tmp_path, monkeypatch):
    snap = _write_snapshot(tmp_path)
    windows = replay_mod.verdict_windows(rcfg, snap, BASE_SEED, LADDER_BLOCK)
    king = _scores(windows, 1.0, 0)
    rr = replay_mod.load_replay_round(
        rcfg, _receipt(rcfg, windows, king, block=LADDER_BLOCK), tmp_path)

    seen = []

    def fake_train_and_evaluate(repo, cfg, *, contract, token_budget, seeds, windows,
                                warm_start_dir, **kw):
        from cascade.miner.score import TrainEvalRun
        seen.append((token_budget, seeds, warm_start_dir))
        better = [WindowScore(s.series_id, s.mase * 0.6, s.qloss_per_q * 0.6,
                              s.abs_target, source=s.source) for s in king]
        return TrainEvalRun(scores=better, corpus_digest="d" * 64, n_series=3,
                            train_seconds=1.0)

    monkeypatch.setattr(replay_mod, "train_and_evaluate", fake_train_and_evaluate)

    full = replay_mod.score_replay("scripts/example_generator", rcfg, rr, cache_dir=tmp_path)
    assert full.full_budget and full.verdict is not None
    assert full.verdict.challenger_wins_round
    assert full.geomean < full.king_geomean
    assert full.init_label == "random init"          # the round pinned no warm start
    budget, seeds, ws = seen[-1]
    assert budget == rcfg.training.at_block(rr.contract_block).primary_size.train_tokens
    assert seeds == rr.seeds and ws is None

    short = replay_mod.score_replay("scripts/example_generator", rcfg, rr,
                                    train_hours=0.05, cache_dir=tmp_path)
    assert not short.full_budget and short.verdict is None
    assert seen[-1][0] < budget


def test_cli_replay_rejects_flags_the_round_supplies(tmp_path, capsys):
    from cascade.miner.cli import main

    rc = main(["score", "scripts/example_generator", "--replay-round", "latest",
               "--snapshot-root", str(tmp_path), "--pool-dir", str(tmp_path)])
    assert rc == 2
    assert "--pool-dir" in capsys.readouterr().err
    rc = main(["score", "scripts/example_generator", "--replay-round", "latest"])
    assert rc == 2
    assert "--snapshot-root" in capsys.readouterr().err


def test_replay_trains_under_the_rounds_signed_contract(rcfg):
    """The manifest's contract_body wins over this checkout's chain.toml (live
    rounds ran points+mv20 billing while the repo still read series_points)."""
    from cascade.shared.manifest import contract_digest, contract_payload

    live = "points+mv30" if rcfg.training.budget_denomination == "points+mv20" else "points+mv20"
    body = {**contract_payload(rcfg.training), "budget_denomination": live}
    manifest = replace(make_manifest(rcfg, base_seed=BASE_SEED), contract_body=body,
                       contract_digest=contract_digest(body))
    contract, match, over = replay_mod.round_contract(rcfg, manifest, None)
    assert contract.budget_denomination == live and match
    assert over == {"budget_denomination": live}
    # a body that does not hash to the manifest digest is flagged, not trusted silently
    _, match, _ = replay_mod.round_contract(
        rcfg, replace(manifest, contract_digest="0" * 64), None)
    assert not match
