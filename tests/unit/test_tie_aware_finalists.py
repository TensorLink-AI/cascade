"""Tie-aware finalist cohort (DEC-CA-0012, trainer half).

The advance rule: a leader the screen's own statistic separated advances
ALONE, whatever the cap; a statistically tied top is re-scored on a larger
eval slice (the CPU run-off) and the survivors advance, capped at
``[round] max_finalists``; ``duel_rank`` stamps the cohort's record order.
At the shipped defaults (``max_finalists = 1``, ``tie_runoff_windows = 0``)
every path here must be bit-identical to the pre-DEC-CA-0012 trainer — one
finalist, ``duel_rank`` absent from the serialised manifest — which is the
hard requirement the first tests pin.

GPU/registry boundaries are faked exactly as in test_trainer_round; the
screeners return per-window scores so the joint-bootstrap tie criterion
(cascade.eval.heat) runs for real. Exact-equal score lists are used to make
ties deterministic: a zero per-window difference gives ``lcb_vs == 0.0``
(tied), and the ranking then falls to the ``(score, reveal_block, uid)``
tiebreak — a UID is not a seniority claim; the reveal block is.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cascade.eval.scoring import WindowScore, global_geomean
from cascade.provision.policy import ProvisionPolicy, StagePolicy, size_fleet
from cascade.shared.config import load_chain_config
from cascade.shared.manifest import dump_manifest
from cascade.trainer import loop as loop_mod
from cascade.trainer.loop import TrainerRunner
from tests.unit.test_trainer_round import (
    REF_A,
    REF_B,
    REF_C,
    REF_D,
    _commit,
    _FakeBaseTrainer,
    _patch_train_boundaries,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# A fourth challenger: the armed tests need a field LARGER than the cap, or the
# everyone-fits fast path advances the whole field without ever screening.
REF_E = "erin/gen-e@sha256:" + "1" * 64

N_HEAT = 60          # windows the fake screeners return for the heat slice
RUNOFF_TOTAL = 100   # [round] tie_runoff_windows in the run-off tests


def _pw(scale: float, seed: int, n: int = N_HEAT, offset: int = 0) -> list[WindowScore]:
    """Per-window scores on a FIXED window set: ``abs_target`` is seeded off the
    window index stream (``offset`` selects the slice), not the entrant, so every
    entrant is paired by construction — exactly how the heat and the run-off
    score (one shared slice per stage). Same ``(scale, seed)`` ⇒ byte-identical
    scores ⇒ a deterministic exact tie."""
    rng = np.random.default_rng(seed)
    win = np.random.default_rng(9000 + offset)
    return [
        WindowScore(
            series_id=f"w{offset}-{i}",
            mase=float(rng.uniform(0.5, 1.5) * scale),
            qloss_per_q=rng.uniform(0.1, 1.0, size=9) * scale,
            abs_target=float(win.uniform(5.0, 10.0)),
        )
        for i in range(n)
    ]


def _screen_for(score_lists: dict[str, list[WindowScore]]):
    def screen(ckpt_dir, gen, base_seed, block=None):
        return score_lists[gen.hotkey]

    return screen


def _round_cfg(cfg, **kw):
    return replace(cfg, round=replace(cfg.round, **kw))


def _never_runoff(*a, **k):
    raise AssertionError("runoff_fn must not be consulted on this path")


def _runner(cfg, tmp_path, screen_fn, runoff_fn=None):
    return TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                         use_sandbox=False, screen_fn=screen_fn, runoff_fn=runoff_fn)


def _commits(*blocks_by_hotkey: tuple[int, str, str, int]):
    return [_commit(uid, hk, ref, blk) for uid, hk, ref, blk in blocks_by_hotkey]


# ── config: both shipped tomls arm the cap; the run-off MUST stay off ────────


def test_round_keys_ship_armed_cap_and_runoff_off():
    """ARMED AT RELEASE (owner 2026-08-20): both tomls ship max_finalists = 3.
    tie_runoff_windows staying 0 is load-bearing, not a default: the run-off's
    incremental draw assumes the seeded-prefix property, which the jittered
    mix (DEC-CA-0019) does not provide — see the NOTE in chain.toml."""
    for name in ("chain.toml", "chain.testnet.toml"):
        r = load_chain_config(REPO_ROOT / name).round
        assert r.max_finalists == 3, f"{name} must ship the tie cohort armed"
        assert r.tie_runoff_windows == 0, f"{name} must ship the run-off OFF (jitter)"


def test_finalist_cap_follows_max_finalists(cfg):
    assert cfg.round.finalists == 1
    assert cfg.round.max_finalists == 3          # armed in the shipped toml
    assert cfg.round.finalist_cap == 3
    # max_finalists <= 1 is OFF: the legacy constant rules even if larger.
    assert replace(cfg.round, max_finalists=1).finalist_cap == 1
    assert replace(cfg.round, finalists=2, max_finalists=1).finalist_cap == 2
    assert replace(cfg.round, finalists=2, max_finalists=3).finalist_cap == 3


# ── inert defaults: bit-identical to the pre-DEC-CA-0012 trainer ─────────────


def test_inert_defaults_advance_one_even_when_statistically_tied(cfg, tmp_path, monkeypatch):
    """max_finalists = 1 / tie_runoff_windows = 0: an exact statistical tie at
    the top still advances exactly ONE finalist, the run-off is never consulted,
    and duel_rank never reaches the serialised manifest — so these manifests
    hash exactly as they did before the feature existed. (The shipped toml is
    armed since 2026-08-20, so inert is forced explicitly here.)"""
    _patch_train_boundaries(monkeypatch)
    cfg = _round_cfg(cfg, max_finalists=1)
    scores = {"b": _pw(1.0, seed=1), "c": _pw(1.0, seed=1),   # exact tie
              "d": _pw(1.5, seed=3)}
    runner = _runner(cfg, tmp_path, _screen_for(scores), runoff_fn=_never_runoff)
    commits = _commits((0, "a", REF_A, 4), (1, "b", REF_B, 5),
                       (2, "c", REF_C, 6), (3, "d", REF_D, 7))
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    chals = manifest.entries_for_role("challenger")
    assert len(chals) == 1                       # one finalist, exactly as today
    assert chals[0].miner_hotkey == "b"          # tie → earlier reveal wins
    assert all(e.duel_rank == 0 for e in manifest.entries)
    assert '"duel_rank"' not in dump_manifest(manifest)
    assert manifest.heat.finalists == 1


# ── the advance rule, armed ──────────────────────────────────────────────────


def test_separated_leader_advances_alone_even_with_cap_3(cfg, tmp_path, monkeypatch):
    _patch_train_boundaries(monkeypatch)
    cfg = _round_cfg(cfg, max_finalists=3)
    scores = {"b": _pw(1.0, seed=1), "c": _pw(1.5, seed=2),
              "d": _pw(1.5, seed=3), "e": _pw(1.5, seed=4)}
    runner = _runner(cfg, tmp_path, _screen_for(scores), runoff_fn=_never_runoff)
    commits = _commits((0, "a", REF_A, 4), (1, "b", REF_B, 5), (2, "c", REF_C, 6),
                       (3, "d", REF_D, 7), (4, "e", REF_E, 8))
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    chals = manifest.entries_for_role("challenger")
    assert [e.miner_hotkey for e in chals] == ["b"]   # separated ⇒ alone, cap ignored
    assert '"duel_rank"' not in dump_manifest(manifest)
    assert manifest.heat.finalists == 1               # the ACTUAL advanced count


def test_scalar_screener_with_cap_armed_degrades_to_leader_alone(cfg, tmp_path, monkeypatch):
    """No per-window components ⇒ no diagnostics ⇒ tied_set degrades to the
    leader — the pre-DEC-CA-0012 single finalist, never an uncontrolled cohort."""
    _patch_train_boundaries(monkeypatch)
    cfg = _round_cfg(cfg, max_finalists=3)
    scores = {"b": 0.9, "c": 0.2, "d": 0.5, "e": 0.7}
    runner = _runner(cfg, tmp_path,
                     lambda ckpt_dir, gen, base_seed, block=None: scores[gen.hotkey])
    commits = _commits((0, "a", REF_A, 4), (1, "b", REF_B, 5), (2, "c", REF_C, 6),
                       (3, "d", REF_D, 7), (4, "e", REF_E, 8))
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    assert [e.miner_hotkey for e in manifest.entries_for_role("challenger")] == ["c"]


def test_tied_top_advances_capped_without_runoff_and_stamps_duel_rank(
        cfg, tmp_path, monkeypatch):
    _patch_train_boundaries(monkeypatch)
    cfg = _round_cfg(cfg, max_finalists=2)          # run-off stays 0 = disabled
    tied = _pw(1.0, seed=1)
    scores = {"b": tied, "c": tied, "d": tied}      # three-way exact tie
    runner = _runner(cfg, tmp_path, _screen_for(scores))
    commits = _commits((0, "a", REF_A, 4), (1, "b", REF_B, 5),
                       (2, "c", REF_C, 6), (3, "d", REF_D, 7))
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    chals = sorted(manifest.entries_for_role("challenger"), key=lambda e: e.duel_rank)
    # Capped at 2 of the 3 tied, in (score, reveal_block, uid) order; best first.
    assert [(e.miner_hotkey, e.duel_rank) for e in chals] == [("b", 0), ("c", 1)]
    assert '"duel_rank"' in dump_manifest(manifest)  # the cohort carries its order
    assert manifest.heat.finalists == 2


# ── the run-off ──────────────────────────────────────────────────────────────


def _runoff_for(extra: dict[str, list[WindowScore]], calls: list):
    def runoff(ckpt_dir, gen, base_seed, block, start, stop):
        calls.append((gen.hotkey, start, stop))
        return extra[gen.hotkey]

    return runoff


def test_runoff_rescores_only_the_tied_set_and_separates(cfg, tmp_path, monkeypatch):
    """Tied {b, c}; the incremental windows separate them ⇒ the leader advances
    alone. Only the tied set is re-scored (d, already separated, is not), on
    exactly the windows the heat has not scored: [N_HEAT, RUNOFF_TOTAL)."""
    _patch_train_boundaries(monkeypatch)
    cfg = _round_cfg(cfg, max_finalists=3, tie_runoff_windows=RUNOFF_TOTAL)
    tied = _pw(1.0, seed=1)
    scores = {"b": tied, "c": tied,
              "d": _pw(1.5, seed=3), "e": _pw(1.5, seed=4)}
    n_extra = RUNOFF_TOTAL - N_HEAT
    extra = {"b": _pw(1.0, seed=4, n=n_extra, offset=1),
             "c": _pw(3.0, seed=5, n=n_extra, offset=1)}   # the run-off separates c
    calls: list = []
    runner = _runner(cfg, tmp_path, _screen_for(scores), _runoff_for(extra, calls))
    commits = _commits((0, "a", REF_A, 4), (1, "b", REF_B, 5), (2, "c", REF_C, 6),
                       (3, "d", REF_D, 7), (4, "e", REF_E, 8))
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    assert sorted(hk for hk, _, _ in calls) == ["b", "c"]         # never d / e
    assert all((start, stop) == (N_HEAT, RUNOFF_TOTAL) for _, start, stop in calls)
    assert [e.miner_hotkey for e in manifest.entries_for_role("challenger")] == ["b"]
    assert '"duel_rank"' not in dump_manifest(manifest)


def test_runoff_still_tied_advances_cohort_ranked_by_reveal(cfg, tmp_path, monkeypatch):
    """A run-off that cannot separate advances the survivors as a cohort,
    duel_rank in extended-score order — here an exact tie throughout, so the
    earlier REVEAL takes rank 0 despite its higher UID."""
    _patch_train_boundaries(monkeypatch)
    cfg = _round_cfg(cfg, max_finalists=3, tie_runoff_windows=RUNOFF_TOTAL)
    tied = _pw(1.0, seed=1)
    scores = {"b": tied, "c": tied,
              "d": _pw(1.5, seed=3), "e": _pw(1.5, seed=4)}
    n_extra = RUNOFF_TOTAL - N_HEAT
    still_tied = _pw(1.0, seed=6, n=n_extra, offset=1)
    extra = {"b": still_tied, "c": still_tied}
    runner = _runner(cfg, tmp_path, _screen_for(scores), _runoff_for(extra, []))
    # c (uid 2) revealed at block 5, BEFORE b (uid 1) at 6: reveal beats UID.
    commits = _commits((0, "a", REF_A, 4), (1, "b", REF_B, 6), (2, "c", REF_C, 5),
                       (3, "d", REF_D, 7), (4, "e", REF_E, 8))
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    chals = sorted(manifest.entries_for_role("challenger"), key=lambda e: e.duel_rank)
    assert [(e.miner_hotkey, e.duel_rank) for e in chals] == [("c", 0), ("b", 1)]
    assert manifest.heat.finalists == 2


def test_runoff_wall_clock_expiry_falls_back_to_the_tied_set(cfg, tmp_path, monkeypatch):
    """tie_runoff_phase_seconds = 0 expires before the first re-score: the
    pre-run-off tied set advances, capped — a screen that cannot finish must
    not sink the round it protects."""
    _patch_train_boundaries(monkeypatch)
    cfg = _round_cfg(cfg, max_finalists=2, tie_runoff_windows=RUNOFF_TOTAL,
                     tie_runoff_phase_seconds=0)
    tied = _pw(1.0, seed=1)
    scores = {"b": tied, "c": tied, "d": tied}
    runner = _runner(cfg, tmp_path, _screen_for(scores), runoff_fn=_never_runoff)
    commits = _commits((0, "a", REF_A, 4), (1, "b", REF_B, 5),
                       (2, "c", REF_C, 6), (3, "d", REF_D, 7))
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    chals = sorted(manifest.entries_for_role("challenger"), key=lambda e: e.duel_rank)
    assert [(e.miner_hotkey, e.duel_rank) for e in chals] == [("b", 0), ("c", 1)]


def test_runoff_scoring_failure_falls_back_to_the_tied_set(cfg, tmp_path, monkeypatch):
    _patch_train_boundaries(monkeypatch)
    cfg = _round_cfg(cfg, max_finalists=2, tie_runoff_windows=RUNOFF_TOTAL)
    tied = _pw(1.0, seed=1)
    scores = {"b": tied, "c": tied, "d": tied}

    def broken(ckpt_dir, gen, base_seed, block, start, stop):
        raise RuntimeError("pool unreachable")

    runner = _runner(cfg, tmp_path, _screen_for(scores), broken)
    commits = _commits((0, "a", REF_A, 4), (1, "b", REF_B, 5),
                       (2, "c", REF_C, 6), (3, "d", REF_D, 7))
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    chals = sorted(manifest.entries_for_role("challenger"), key=lambda e: e.duel_rank)
    assert [(e.miner_hotkey, e.duel_rank) for e in chals] == [("b", 0), ("c", 1)]


# ── the heat tiebreak: (score, reveal_block, uid) ────────────────────────────


def test_heat_tiebreak_prefers_earlier_reveal_over_lower_uid(cfg, tmp_path, monkeypatch):
    """Equal scores: the LOWER-uid challenger loses to the EARLIER-revealed one
    — a UID recycles, so it is not a seniority claim (DEC-CA-0012). Cap forced
    to 1 so this small field is actually screened (a field <= finalist_cap
    skips the heat and advances whole)."""
    _patch_train_boundaries(monkeypatch)
    cfg = _round_cfg(cfg, max_finalists=1)
    scores = {"b": 0.5, "c": 0.5, "d": 0.9}
    runner = _runner(cfg, tmp_path,
                     lambda ckpt_dir, gen, base_seed, block=None: scores[gen.hotkey])
    # b has uid 1 but revealed at 8; c has uid 2 and revealed at 6.
    commits = _commits((0, "a", REF_A, 4), (1, "b", REF_B, 8),
                       (2, "c", REF_C, 6), (3, "d", REF_D, 7))
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    assert manifest.entry_for_role("challenger").miner_hotkey == "c"


def test_heat_tiebreak_falls_to_uid_when_reveals_match(cfg, tmp_path, monkeypatch):
    _patch_train_boundaries(monkeypatch)
    cfg = _round_cfg(cfg, max_finalists=1)   # force a real screen (see above)
    scores = {"b": 0.5, "c": 0.5}
    runner = _runner(cfg, tmp_path,
                     lambda ckpt_dir, gen, base_seed, block=None: scores[gen.hotkey])
    commits = _commits((0, "a", REF_A, 4), (1, "b", REF_B, 6), (2, "c", REF_C, 6))
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    assert manifest.entry_for_role("challenger").miner_hotkey == "b"


# ── duel_rank carries record order, tolerating gaps ──────────────────────────


def test_duel_rank_survives_a_clone_drop_non_contiguous(cfg, tmp_path, monkeypatch):
    """Ranks are stamped off the advancing cohort BEFORE the final's
    content-clone drop, so a dropped entry leaves a gap — consumers sort,
    never index, and the validator tolerates non-contiguous ranks."""
    _patch_train_boundaries(monkeypatch)
    cfg = _round_cfg(cfg, max_finalists=3)
    tied = _pw(1.0, seed=1)
    scores = {"b": tied, "c": tied, "d": tied, "e": _pw(1.5, seed=4)}
    runner = _runner(cfg, tmp_path, _screen_for(scores))
    monkeypatch.setattr(
        loop_mod, "_drop_final_content_clones",
        lambda entries, jobs: [e for e in entries if e.miner_hotkey != "c"],
    )
    commits = _commits((0, "a", REF_A, 4), (1, "b", REF_B, 5), (2, "c", REF_C, 6),
                       (3, "d", REF_D, 7), (4, "e", REF_E, 8))
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    chals = sorted(manifest.entries_for_role("challenger"), key=lambda e: e.duel_rank)
    assert [(e.miner_hotkey, e.duel_rank) for e in chals] == [("b", 0), ("d", 2)]


# ── fleet sizing off the CAP ─────────────────────────────────────────────────


def _policy(*, heat_gpus=8, heat_max=4, final_gpus=2, final_max=4):
    return ProvisionPolicy(
        heat=StagePolicy(sku="NVIDIA RTX A6000", gpus_per_pod=heat_gpus,
                         max_pods=heat_max, providers=("lium",), max_price_hr=4.0),
        final=StagePolicy(sku="NVIDIA L40S", gpus_per_pod=final_gpus,
                          max_pods=final_max, providers=("lium",), max_price_hr=3.0),
        trigger_margin_blocks=25,
        max_spend_per_round=25.0,
    )


def test_final_fleet_sizes_off_the_cohort_cap():
    # king + up to 3 tie-cohort challengers = 4 slots → two 2-GPU pods: the
    # pre-phased fleet and within_budget must cover the WORST case the advance
    # rule can produce, not the single finalist the plan predicts.
    plan = size_fleet(20, 1, 0.5, 24.0, 3.0, _policy(), max_finalists=3)
    assert plan.final.slots == 4
    assert plan.final.pods == 2


def test_fleet_sizing_is_bit_identical_at_the_inert_defaults():
    base = size_fleet(20, 1, 0.5, 24.0, 3.0, _policy())
    assert size_fleet(20, 1, 0.5, 24.0, 3.0, _policy(), max_finalists=0) == base
    assert size_fleet(20, 1, 0.5, 24.0, 3.0, _policy(), max_finalists=1) == base


def test_field_within_the_cap_rents_no_heat_pods():
    # Everyone advances when the field fits inside the cap — nothing to screen,
    # mirroring the trainer's fast path.
    plan = size_fleet(3, 1, 0.5, 24.0, 3.0, _policy(), max_finalists=3)
    assert plan.heat.pods == 0 and plan.heat.slots == 0
    assert plan.final.slots == 4


def test_negative_max_finalists_is_rejected():
    with pytest.raises(ValueError):
        size_fleet(5, 1, 0.5, 24.0, 3.0, _policy(), max_finalists=-1)


# ── sanity: the exact-tie fixtures really are ties, and 1.5 really separates ─


def test_fixture_scales_behave_as_the_tests_assume():
    tied_a, tied_b = _pw(1.0, seed=1), _pw(1.0, seed=1)
    assert global_geomean(tied_a) == global_geomean(tied_b)
    assert global_geomean(_pw(1.5, seed=3)) > global_geomean(tied_a)
