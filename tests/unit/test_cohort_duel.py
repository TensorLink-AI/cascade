"""The tie-aware cohort duel (DEC-CA-0012).

Covers the three halves of the decision that are shippable ahead of the trainer's
run-off: the generalised heat tie statistic, the validator judging a whole cohort
under a family-wise alpha, and the audit verifying the *selection* rather than
just the verdict.

The invariant that matters most for rollout is that a cohort of ONE is
bit-identical to the pre-DEC-CA-0012 rule — that is what lets the validator ship
before the trainer ever advances more than one finalist.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.eval.heat import screen_diagnostics, tied_set
from cascade.eval.scoring import WindowScore
from cascade.shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    format_trained_pointer,
    load_manifest,
)
from cascade.validator.loop import ValidatorRunner
from cascade.validator.state import ChampionState, apply_round, genesis

CID = "alice/gen@sha256:" + "a" * 64
CID2 = "cascade/ckpt@sha256:" + "b" * 64


def _scores(scale, seed, n=300, spread=(0.5, 1.5)):
    rng = np.random.default_rng(seed)
    return [
        WindowScore(
            series_id=str(i),
            mase=float(rng.uniform(*spread) * scale),
            qloss_per_q=rng.uniform(0.1, 1.0, size=9) * scale,
            abs_target=float(rng.uniform(5.0, 10.0)),
        )
        for i in range(n)
    ]


def _rescale(base, factor):
    """A competitor that is uniformly ``factor`` times the base on every window —
    paired by construction (same ``abs_target``)."""
    return [
        WindowScore(s.series_id, s.mase * factor, s.qloss_per_q * factor, s.abs_target)
        for s in base
    ]


def _cohort_manifest(cfg, challengers, *, sizes=("",)):
    """``challengers`` is ``[(hotkey, uid, duel_rank)]``."""
    entries = []
    for size in sizes:
        entries.append(TrainedEntry("king_hk", 0, "king", CID,
                                    format_trained_pointer(CID2), "d", 10, size=size))
    for size in sizes:
        for hk, uid, rank in challengers:
            entries.append(TrainedEntry(hk, uid, "challenger", CID,
                                        format_trained_pointer(CID2), "d", 10,
                                        size=size, duel_rank=rank))
    return TrainingManifest(
        round_id="1", created_block=10,
        contract_digest=contract_digest(cfg.training),
        base_arch_digest=cfg.training.base_arch_digest,
        eval_dataset=cfg.eval.eval_dataset, entries=entries,
    )


# ── the manifest field is invisible until used ────────────────────────────────


def test_duel_rank_zero_leaves_the_signed_body_byte_identical(cfg):
    """The whole rollout rests on this: adding ``duel_rank`` must not re-serialise
    a single-finalist manifest, or every archived signature breaks."""
    m = _cohort_manifest(cfg, [("chal_hk", 1, 0)])
    body = m.canonical_body()
    assert b"duel_rank" not in body
    # And a ranked entry DOES carry it, so the order is inside what the trainer signs.
    ranked = _cohort_manifest(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1)])
    assert b"duel_rank" in ranked.canonical_body()


def test_duel_rank_round_trips_through_json(cfg):
    from cascade.shared.manifest import dump_manifest

    m = _cohort_manifest(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1)])
    back = load_manifest(dump_manifest(m))
    assert {e.miner_hotkey: e.duel_rank for e in back.entries_for_role("challenger")} == {
        "a_hk": 0, "b_hk": 1,
    }
    assert back.canonical_body() == m.canonical_body()


def test_cohort_is_ordered_by_duel_rank_not_manifest_order(cfg):
    # b_hk is listed first but ranked second; order must follow the rank.
    m = _cohort_manifest(cfg, [("b_hk", 2, 1), ("a_hk", 1, 0)])
    cohort, sizes = m.duel_cohort()
    assert [hk for hk, _ in cohort] == ["a_hk", "b_hk"]
    assert sizes == [""]


def test_legacy_all_zero_ranks_still_order_deterministically(cfg):
    m = _cohort_manifest(cfg, [("z_hk", 2, 0), ("a_hk", 1, 0)])
    assert [hk for hk, _ in m.duel_cohort()[0]] == ["a_hk", "z_hk"]


def test_cohort_keeps_only_the_maximal_common_size_set(cfg):
    """A challenger that failed to train where its peers succeeded does not get to
    compete against them — otherwise the crowning comparison is not like-for-like."""
    m = _cohort_manifest(cfg, [("a_hk", 1, 0)], sizes=("toto2-4m", "toto2-22m"))
    m.entries.append(TrainedEntry("b_hk", 2, "challenger", CID,
                                  format_trained_pointer(CID2), "d", 10,
                                  size="toto2-4m", duel_rank=1))
    cohort, sizes = m.duel_cohort()
    assert [hk for hk, _ in cohort] == ["a_hk"]
    assert sizes == ["toto2-4m", "toto2-22m"]


# ── the generalised tie statistic ─────────────────────────────────────────────


def _ranked(entrants):
    return [(k, v) for k, v in entrants]


def test_lcb_vs_reproduces_leader_lcb_at_the_runner_up():
    """``leader_lcb`` is the DEC-CA-0006 headline number; generalising it must not
    move it."""
    base = _scores(1.0, 3)
    ranked = _ranked([("a", _rescale(base, 0.5)), ("b", base), ("c", _rescale(base, 1.4))])
    d = screen_diagnostics(ranked, seed=1, B=400)
    assert d is not None
    assert d.runner_up_key == "b"
    assert d.lcb_vs["b"] == pytest.approx(d.leader_lcb)
    assert set(d.lcb_vs) == {"b", "c"}


def test_decisive_field_yields_a_tied_set_of_one():
    base = _scores(1.0, 4)
    # A clear leader: uniformly half the error of everyone else on every window.
    ranked = _ranked([("a", _rescale(base, 0.5)), ("b", base), ("c", _rescale(base, 1.1))])
    d = screen_diagnostics(ranked, seed=1, B=400)
    assert all(v > 0 for v in d.lcb_vs.values())
    assert tied_set(d, [k for k, _ in ranked], cap=3) == ["a"]


def test_identical_entrants_are_all_tied():
    base = _scores(1.0, 5)
    ranked = _ranked([("a", base), ("b", list(base)), ("c", list(base))])
    d = screen_diagnostics(ranked, seed=1, B=400)
    # Identical scores ⇒ zero relative improvement ⇒ the bound cannot clear 0.
    assert all(v <= 0 for v in d.lcb_vs.values())
    assert tied_set(d, [k for k, _ in ranked], cap=3) == ["a", "b", "c"]


def test_tied_set_respects_the_cap_and_rank_order():
    base = _scores(1.0, 6)
    ranked = _ranked([(k, list(base)) for k in ("a", "b", "c", "d", "e")])
    d = screen_diagnostics(ranked, seed=1, B=400)
    assert tied_set(d, [k for k, _ in ranked], cap=2) == ["a", "b"]
    assert tied_set(d, [k for k, _ in ranked], cap=0) == []


def test_tied_set_degrades_to_the_leader_without_diagnostics():
    """A scalar-only screener or an unpaired field must fall back to exactly the
    pre-DEC-CA-0012 behaviour, never to a wider cohort."""
    assert tied_set(None, ["a", "b", "c"], cap=3) == ["a"]


# ── the cohort duel ───────────────────────────────────────────────────────────


def _runner(cfg, eval_fn, state=None):
    return ValidatorRunner(cfg=cfg, state=state or genesis("king_hk", 0),
                           evaluate_fn=eval_fn, verify_signatures=False)


def _by_hotkey(king, mapping):
    """An evaluate_fn serving per-hotkey challenger scores."""
    calls: list[str] = []

    def fake_eval(entry, windows):
        calls.append(entry.miner_hotkey)
        return king if entry.role == "king" else mapping[entry.miner_hotkey]

    return fake_eval, calls


def test_whole_cohort_is_judged_and_the_king_is_evaluated_once(cfg):
    """The two properties that make cohort judging cheap and fair: no early stop,
    and the king's eval is reused."""
    king = _scores(1.0, 0)
    fake_eval, calls = _by_hotkey(king, {
        "a_hk": _rescale(king, 0.5),   # clears
        "b_hk": _rescale(king, 0.4),   # clears by MORE
    })
    m = _cohort_manifest(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1)])
    outcome = _runner(cfg, fake_eval).process_round(m, windows=[], base_seed=7)
    assert outcome is not None
    assert calls.count("king_hk") == 1
    assert outcome.duelled_hotkeys == ("a_hk", "b_hk")


def test_best_clearer_is_crowned_not_the_first(cfg):
    """The artifact the rejected sequential rule would have produced: a_hk is
    ranked first by the (noisy) screen but b_hk is genuinely better."""
    king = _scores(1.0, 0)
    fake_eval, _ = _by_hotkey(king, {
        "a_hk": _rescale(king, 0.7),
        "b_hk": _rescale(king, 0.4),
    })
    m = _cohort_manifest(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1)])
    outcome = _runner(cfg, fake_eval).process_round(m, windows=[], base_seed=7)
    assert outcome.result.challenger_wins_round
    assert outcome.decided_hotkey == "b_hk"
    assert outcome.transition.dethroned
    assert outcome.transition.new_king_hotkey == "b_hk"


def test_a_losing_leader_does_not_block_a_winning_runner_up(cfg):
    king = _scores(1.0, 0)
    fake_eval, _ = _by_hotkey(king, {
        "a_hk": _rescale(king, 1.2),   # worse than the king
        "b_hk": _rescale(king, 0.5),   # clears
    })
    m = _cohort_manifest(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1)])
    outcome = _runner(cfg, fake_eval).process_round(m, windows=[], base_seed=7)
    assert outcome.decided_hotkey == "b_hk"
    assert outcome.transition.dethroned


def test_all_fail_is_exactly_todays_outcome_king_holds(cfg):
    king = _scores(1.0, 0)
    fake_eval, _ = _by_hotkey(king, {
        "a_hk": _rescale(king, 1.1),
        "b_hk": _rescale(king, 1.3),
    })
    m = _cohort_manifest(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1)])
    runner = _runner(cfg, fake_eval)
    outcome = runner.process_round(m, windows=[], base_seed=7)
    assert not outcome.result.challenger_wins_round
    assert not outcome.transition.dethroned
    assert runner.state.king_hotkey == "king_hk"


def test_cohort_of_one_is_identical_to_the_pre_cohort_rule(cfg):
    """The rollout invariant: k=1 must not perturb the verdict at all, because the
    validator ships while the trainer still advances a single finalist."""
    king = _scores(1.0, 0)
    chal = _rescale(king, 0.9)
    fake_eval, _ = _by_hotkey(king, {"chal_hk": chal})
    m = _cohort_manifest(cfg, [("chal_hk", 1, 0)])
    outcome = _runner(cfg, fake_eval).process_round(m, windows=[], base_seed=7)

    from cascade.eval.koth import evaluate_round

    direct = evaluate_round(king, chal, cfg.koth_params(), seed=7, king_tenure_rounds=0)
    assert outcome.result.lcb == pytest.approx(direct.lcb)
    assert outcome.result.challenger_wins_round == direct.challenger_wins_round


def test_family_wise_alpha_tightens_the_bound_with_k(cfg):
    """k challengers get k draws at the tail, so each is held to alpha/k. The
    same challenger must therefore face a STRICTLY harder bound in a cohort."""
    king = _scores(1.0, 0)
    chal = _rescale(king, 0.9)

    def solo(entry, windows):
        return king if entry.role == "king" else chal

    lone = _runner(cfg, solo).process_round(
        _cohort_manifest(cfg, [("a_hk", 1, 0)]), windows=[], base_seed=7)

    fake_eval, _ = _by_hotkey(king, {"a_hk": chal, "b_hk": chal, "c_hk": chal})
    trio = _runner(cfg, fake_eval).process_round(
        _cohort_manifest(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1), ("c_hk", 3, 2)]),
        windows=[], base_seed=7)
    # A lower alpha quantile is a lower (more conservative) bound.
    assert trio.result.lcb < lone.result.lcb


def test_inconclusive_stops_the_cohort_and_holds_the_throne(cfg):
    """min_windows is a property of the slice, identical for every challenger — so
    one inconclusive duel means all would be, and the rest are not evaluated."""
    king = _scores(1.0, 0, n=3)          # far below [scoring] min_windows
    fake_eval, calls = _by_hotkey(king, {
        "a_hk": _rescale(king, 0.5), "b_hk": _rescale(king, 0.4),
    })
    m = _cohort_manifest(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1)])
    runner = _runner(cfg, fake_eval)
    outcome = runner.process_round(m, windows=[], base_seed=7)
    assert outcome.result.inconclusive
    assert not outcome.transition.dethroned
    assert "b_hk" not in calls          # stopped before the second challenger
    assert runner.state.king_hotkey == "king_hk"


# ── streak bookkeeping ────────────────────────────────────────────────────────


def _win(**kw):
    from cascade.eval.koth import RoundResult

    base = dict(challenger_wins_round=True, lcb=0.1, margin=0.02, n_windows=300,
                king_geomean=1.0, chal_geomean=0.5, inconclusive=False)
    return RoundResult(**{**base, **kw})


def test_defeated_cohort_members_lose_their_streaks():
    state = ChampionState(king_hotkey="king_hk", king_uid=0,
                          streaks={"a_hk": 1, "b_hk": 1, "other": 1})
    t = apply_round(state, challenger_hotkey="a_hk", challenger_uid=1,
                    result=_win(), dethrone_cp=3,
                    defeated_hotkeys=("a_hk", "b_hk"))
    assert t.state.streaks["a_hk"] == 2      # the crowned/decided one advances
    assert "b_hk" not in t.state.streaks     # a duelled loser resets
    assert t.state.streaks["other"] == 1     # nobody else is touched


def test_a_non_crowned_clearer_banks_nothing():
    """A streak is a claim on the throne and only one challenger holds it per
    round, so clearing the margin without being crowned must not accrue."""
    state = ChampionState(king_hotkey="king_hk", king_uid=0, streaks={"b_hk": 1})
    t = apply_round(state, challenger_hotkey="a_hk", challenger_uid=1,
                    result=_win(), dethrone_cp=3,
                    defeated_hotkeys=("a_hk", "b_hk"))
    assert "b_hk" not in t.state.streaks


def test_cohort_audit_verifies_the_selection(cfg):
    """Producer → consumer, on the real path: the validator scores a cohort, builds
    the receipt, and the audit re-derives alpha/k and confirms the crowned
    challenger really is the best clearer."""
    from cascade.audit import checks as C

    king = _scores(1.0, 0)
    fake_eval, _ = _by_hotkey(king, {
        "a_hk": _rescale(king, 0.7),
        "b_hk": _rescale(king, 0.4),
    })
    m = _cohort_manifest(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1)])
    runner = _runner(cfg, fake_eval)
    outcome = runner.process_round(m, windows=[], base_seed=7)
    receipt = runner.build_round_receipt(
        m, base_seed=7, epoch_start_block=10, epoch_block_hash="0x" + "ab" * 32,
        outcome=outcome, windows=[],
    )
    r = C.check_duel_cohort(receipt)
    assert r.status == C.PASS, r.detail
    # The correction it re-derived is the one the round was judged under, and the
    # receipt still records the UNMODIFIED config params (or check_koth_params
    # would fail against chain.toml).
    assert "/2" in r.detail
    assert receipt.verdict.params["bootstrap_alpha"] == cfg.scoring.bootstrap_alpha


def test_cohort_audit_catches_crowning_the_wrong_clearer(cfg):
    """The precise failure the rejected sequential rule would have produced: two
    challengers clear, and the weaker one takes the throne."""
    from dataclasses import replace as dc_replace

    from cascade.audit import checks as C

    king = _scores(1.0, 0)
    fake_eval, _ = _by_hotkey(king, {
        "a_hk": _rescale(king, 0.7),
        "b_hk": _rescale(king, 0.4),
    })
    m = _cohort_manifest(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1)])
    runner = _runner(cfg, fake_eval)
    outcome = runner.process_round(m, windows=[], base_seed=7)
    receipt = runner.build_round_receipt(
        m, base_seed=7, epoch_start_block=10, epoch_block_hash="0x" + "ab" * 32,
        outcome=outcome, windows=[],
    )
    # Re-order so the WEAKER clearer (a_hk) is the one the verdict attaches to.
    scores = list(receipt.entry_scores)
    chal = [s for s in scores if s.role == "challenger"]
    kings = [s for s in scores if s.role == "king"]
    swapped = dc_replace(receipt, entry_scores=tuple(kings + chal[::-1]))
    r = C.check_duel_cohort(swapped)
    assert r.status == C.FAIL
    assert "not the best clearer" in r.detail


def test_single_challenger_round_skips_the_cohort_check(cfg):
    from cascade.audit import checks as C

    king = _scores(1.0, 0)
    fake_eval, _ = _by_hotkey(king, {"chal_hk": _rescale(king, 0.6)})
    m = _cohort_manifest(cfg, [("chal_hk", 1, 0)])
    runner = _runner(cfg, fake_eval)
    outcome = runner.process_round(m, windows=[], base_seed=7)
    receipt = runner.build_round_receipt(
        m, base_seed=7, epoch_start_block=10, epoch_block_hash="0x" + "ab" * 32,
        outcome=outcome, windows=[],
    )
    assert C.check_duel_cohort(receipt).status == C.SKIP
    # ...and the ordinary verdict check still reproduces it.
    assert C.check_verdict(receipt).status == C.PASS


# ── what the receipt publishes ────────────────────────────────────────────────


def _receipt(cfg, challengers, mapping, king, seed=7):
    fake_eval, _ = _by_hotkey(king, mapping)
    m = _cohort_manifest(cfg, challengers)
    runner = _runner(cfg, fake_eval)
    outcome = runner.process_round(m, windows=[], base_seed=seed)
    return runner.build_round_receipt(
        m, base_seed=seed, epoch_start_block=10,
        epoch_block_hash="0x" + "ab" * 32, outcome=outcome, windows=[],
    ), outcome


def test_receipt_publishes_k_and_every_challenger_lcb(cfg):
    """The externally-checkable record: k, each challenger's bound under alpha/k,
    and the margin those bounds are compared against."""
    king = _scores(1.0, 0)
    receipt, outcome = _receipt(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1), ("c_hk", 3, 2)],
                               {"a_hk": _rescale(king, 0.7),
                                "b_hk": _rescale(king, 0.4),
                                "c_hk": _rescale(king, 1.2)}, king)
    v = receipt.verdict
    assert v.cohort_k == 3
    assert set(v.cohort_lcbs) == {"a_hk", "b_hk", "c_hk"}
    # Everything needed to re-check "who cleared" is present and self-consistent.
    cleared = {h for h, lcb in v.cohort_lcbs.items() if lcb >= v.margin}
    assert cleared == {"a_hk", "b_hk"}
    # The crown went to a clearer, and the headline verdict numbers ARE that
    # challenger's published bound. (The old assertion here short-circuited on
    # membership via `or`, leaving the numeric half decorative.)
    assert outcome.decided_hotkey in cleared
    assert v.lcb == pytest.approx(v.cohort_lcbs[outcome.decided_hotkey])


def test_alpha_over_k_moves_the_quantile_and_leaves_the_margin_alone(cfg):
    """The semantics ask: alpha/k tightens the BOOTSTRAP QUANTILE, not the margin.
    Pinned here because raising the margin instead would give different answers at
    3+ challengers, and a receipt replayed under the wrong lever won't reproduce."""
    king = _scores(1.0, 0)
    chal = _rescale(king, 0.9)
    solo, _ = _receipt(cfg, [("a_hk", 1, 0)], {"a_hk": chal}, king)
    trio, _ = _receipt(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1), ("c_hk", 3, 2)],
                       {"a_hk": chal, "b_hk": chal, "c_hk": chal}, king)
    # The margin is IDENTICAL — the correction did not touch it. (A fresh
    # runner judges at tenure 0, so under DEC-CA-0016's decay the bar is the
    # full win_margin_start.)
    assert trio.verdict.margin == solo.verdict.margin == cfg.scoring.win_margin_start
    # The bound moved down: a lower quantile of the same paired distribution.
    assert trio.verdict.lcb < solo.verdict.lcb
    # And the published alpha is the quantile actually taken.
    from cascade.shared.receipt import summarize_receipt

    s = summarize_receipt(trio)
    assert s["cohort_k"] == 3
    assert s["cohort_alpha"] == pytest.approx(cfg.scoring.bootstrap_alpha / 3)
    assert summarize_receipt(solo)["cohort_alpha"] is None


def test_single_challenger_receipt_body_is_byte_identical_to_pre_cohort(cfg):
    """The reason the cohort fields are safe to put in a SIGNED structure: at their
    defaults they must vanish from the canonical body, or every archived receipt
    re-serialises with bytes that were never signed."""
    king = _scores(1.0, 0)
    receipt, _ = _receipt(cfg, [("chal_hk", 1, 0)], {"chal_hk": _rescale(king, 0.6)}, king)
    body = receipt.canonical_body()
    assert b"cohort_k" not in body
    assert b"cohort_lcbs" not in body
    # A cohort round DOES carry them, so they are inside what the validator signs.
    cohort, _ = _receipt(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1)],
                         {"a_hk": _rescale(king, 0.7), "b_hk": _rescale(king, 0.4)}, king)
    assert b"cohort_k" in cohort.canonical_body()


def test_cohort_fields_round_trip_and_preserve_the_signed_bytes(cfg):
    from cascade.shared.receipt import dump_receipt, load_receipt

    king = _scores(1.0, 0)
    receipt, _ = _receipt(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1)],
                          {"a_hk": _rescale(king, 0.7), "b_hk": _rescale(king, 0.4)}, king)
    back = load_receipt(dump_receipt(receipt))
    assert back.canonical_body() == receipt.canonical_body()
    assert back.verdict.cohort_k == 2
    assert back.verdict.cohort_lcbs == receipt.verdict.cohort_lcbs


def test_audit_rejects_a_doctored_published_lcb(cfg):
    """A published bound that does not replay is a hard failure — those numbers are
    what an outside party checks the selection with."""
    from dataclasses import replace as dc_replace

    from cascade.audit import checks as C

    king = _scores(1.0, 0)
    receipt, _ = _receipt(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1)],
                          {"a_hk": _rescale(king, 0.7), "b_hk": _rescale(king, 0.4)}, king)
    assert C.check_duel_cohort(receipt).status == C.PASS
    bad = dict(receipt.verdict.cohort_lcbs)
    bad["a_hk"] = bad["a_hk"] + 0.5      # claim a_hk did far better than it did
    tampered = dc_replace(receipt, verdict=dc_replace(receipt.verdict, cohort_lcbs=bad))
    r = C.check_duel_cohort(tampered)
    assert r.status == C.FAIL
    assert "replays as" in r.detail


def test_audit_rejects_a_mismatched_k(cfg):
    """k sets the alpha, so a k that disagrees with the signed manifest means the
    round was not judged under the alpha it claims."""
    from dataclasses import replace as dc_replace

    from cascade.audit import checks as C

    king = _scores(1.0, 0)
    receipt, _ = _receipt(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1)],
                          {"a_hk": _rescale(king, 0.7), "b_hk": _rescale(king, 0.4)}, king)
    tampered = dc_replace(receipt, verdict=dc_replace(receipt.verdict, cohort_k=5))
    r = C.check_duel_cohort(tampered)
    assert r.status == C.FAIL
    assert "cohort_k=5" in r.detail


def test_inconclusive_leaves_the_whole_cohort_untouched():
    state = ChampionState(king_hotkey="king_hk", king_uid=0,
                          streaks={"a_hk": 1, "b_hk": 1})
    t = apply_round(state, challenger_hotkey="a_hk", challenger_uid=1,
                    result=_win(inconclusive=True, challenger_wins_round=False,
                                lcb=float("nan")),
                    dethrone_cp=3, defeated_hotkeys=("a_hk", "b_hk"))
    assert t.state.streaks == {"a_hk": 1, "b_hk": 1}


def test_mid_cohort_winner_is_recorded_last_and_every_consumer_agrees(cfg):
    """THE decided-last regression (DEC-CA-0012 convention): when the crowned
    challenger is not the last-ranked cohort member, the producer must move its
    records to the end — audit's replay (check_verdict / check_duel_cohort),
    the transition check, and the public summary all resolve "whose verdict is
    this" positionally. Before the reorder existed, this exact receipt failed
    check_verdict and check_duel_cohort and the summary credited the wrong
    challenger."""
    from cascade.audit import checks as C
    from cascade.shared.receipt import summarize_receipt

    king = _scores(1.0, 0)
    # Cohort order a, b, c — the winner (b, best geomean) sits in the MIDDLE
    # and a non-clearer (c) is ranked last.
    receipt, outcome = _receipt(cfg, [("a_hk", 1, 0), ("b_hk", 2, 1), ("c_hk", 3, 2)],
                                {"a_hk": _rescale(king, 0.7),
                                 "b_hk": _rescale(king, 0.4),
                                 "c_hk": _rescale(king, 1.2)}, king)
    assert outcome.decided_hotkey == "b_hk"
    # Producer honoured the convention: b's records are last, everyone else's
    # relative order is preserved.
    chal_order = [es.hotkey for es in receipt.entry_scores if es.role == "challenger"]
    assert chal_order == ["a_hk", "c_hk", "b_hk"]
    assert outcome.duelled_hotkeys[-1] == "b_hk"
    # Every positional consumer now attributes the round to the crowned clearer.
    r = C.check_verdict(receipt)
    assert r.status == C.PASS, r.detail
    r = C.check_duel_cohort(receipt)
    assert r.status == C.PASS, r.detail
    r = C.check_transition(receipt)
    assert r.status in (C.PASS, C.WARN), r.detail
    s = summarize_receipt(receipt)
    assert s["chal_hotkey"] == "b_hk"
    assert s["post_round_king_hotkey"] == "b_hk"
