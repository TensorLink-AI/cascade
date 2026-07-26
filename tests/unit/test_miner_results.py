"""``cascade results`` — the miner-centric report built from a public receipt.

Everything runs offline on the shared receipt fixture: the report logic is
pure (receipt + identity → dict), so these tests cover the exact JSON the
``--json`` flag emits and the semantics the rendered text is built from —
gap-to-cutoff math, status diagnoses, the challenger-relative verdict, and
the per-source duel breakdown.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from cascade.miner.results import (
    build_report,
    compact_row,
    render_report,
    resolve_identity,
)
from cascade.shared.manifest import HeatEntrant, HeatResult, dump_manifest
from cascade.shared.receipt import (
    EntryScores,
    WindowScoreRecord,
    dump_receipt,
    load_receipt,
)

from .receipt_fixture import make_rejected_receipt, make_scored_receipt

REF = "x/gen@sha256:" + "0" * 64


def _with_heat(receipt, heat: HeatResult):
    """Graft a heat block onto the receipt's embedded manifest, then round-trip
    the receipt through the wire like the CLI does."""
    m = replace(receipt.load_embedded_manifest(), heat=heat)
    receipt = replace(receipt, manifest=json.loads(dump_manifest(m)))
    return load_receipt(dump_receipt(receipt))


HEAT = HeatResult(
    screen_size="toto2-4m", finalists=2,
    entrants=(
        HeatEntrant(uid=1, hotkey="chal_hk", gen_ref=REF, status="advanced",
                    rank=1, rel_score=1.0, p_best=0.61, crps=0.41, mase=0.39),
        HeatEntrant(uid=7, hotkey="hk7", gen_ref=REF, status="advanced",
                    rank=2, rel_score=1.004),
        HeatEntrant(uid=9, hotkey="hk9", gen_ref=REF, status="screened",
                    rank=3, rel_score=1.009,
                    note="deadline_hit: wall-clock cap reached before the token "
                         "budget — trained on 41% of the budget"),
        HeatEntrant(uid=12, hotkey="hk12", gen_ref=REF, status="screened",
                    rank=4, rel_score=1.31),
        HeatEntrant(uid=15, hotkey="hk15", gen_ref=REF, status="failed_train",
                    note="generator_artifact_unreachable: HTTP 401 for x@sha256:00"),
        HeatEntrant(uid=18, hotkey="hk18", gen_ref=REF, status="duplicate",
                    note="corpus byte-identical to uid 9's earlier reveal — the "
                         "earliest reveal keeps the slot"),
        HeatEntrant(uid=21, hotkey="hk21", gen_ref=REF, status="failed_screen"),
    ),
    leader_lcb=-0.002, n_windows=96, n_clusters=14,
)


@pytest.fixture(scope="module")
def receipt():
    r, _, _ = make_scored_receipt()
    return _with_heat(r, HEAT)


# ── identity ─────────────────────────────────────────────────────────────────


def test_resolve_identity_by_uid_and_hotkey(receipt):
    assert resolve_identity(receipt, "1") == (1, "chal_hk")
    assert resolve_identity(receipt, "chal_hk") == (1, "chal_hk")
    assert resolve_identity(receipt, "9") == (9, "hk9")        # heat-only entrant
    assert resolve_identity(receipt, "424242") == (424242, None)


# ── heat section ─────────────────────────────────────────────────────────────


def test_screened_entrant_gets_gap_to_cutoff(receipt):
    rep = build_report(receipt, "9")
    e = rep["heat"]["entrant"]
    assert e["status"] == "screened" and e["rank"] == 3
    # gap to the LEADER and to the LAST FINALIST (rank 2, rel 1.004)
    assert e["pct_behind_leader"] == pytest.approx(0.9)
    assert e["cutoff_rel_score"] == pytest.approx(1.004)
    assert e["pct_behind_cutoff"] == pytest.approx(0.5)
    assert "gap to close" in e["diagnosis"]
    # leader_lcb ≤ 0 ⇒ the near-miss caveat is surfaced
    assert "not decisive" in e["diagnosis"]


def test_advanced_entrant_has_no_cutoff_gap(receipt):
    e = build_report(receipt, "1")["heat"]["entrant"]
    assert e["status"] == "advanced"
    assert e["pct_behind_cutoff"] is None
    assert "advanced to the duel" in e["diagnosis"]


def test_failure_notes_reach_diagnosis_and_advice(receipt):
    rep = build_report(receipt, "hk15")
    e = rep["heat"]["entrant"]
    assert e["status"] == "failed_train"
    assert "generator_artifact_unreachable" in e["diagnosis"]
    # the 401 gets the targeted fix, not the generic verify-locally tip
    assert any("visibility to PUBLIC" in t for t in rep["advice"])
    assert not any("cascade verify" in t for t in rep["advice"])

    rep21 = build_report(receipt, "21")  # failed_screen, note-less (pre-note round)
    assert "no reason was recorded" not in rep21["heat"]["entrant"]["diagnosis"]
    assert any("NaNs" in t for t in rep21["advice"])


def test_deadline_note_triggers_throughput_advice(receipt):
    rep = build_report(receipt, "hk9")
    assert any("wall-clock cap" in t for t in rep["advice"])
    # screened entrants also get the iterate-locally loop
    assert any("cascade fetch king" in t for t in rep["advice"])


def test_duplicate_status_names_the_owner(receipt):
    rep = build_report(receipt, "18")
    assert "uid 9" in rep["heat"]["entrant"]["diagnosis"]
    assert any("earliest reveal" in t for t in rep["advice"])


# ── duel section ─────────────────────────────────────────────────────────────


def test_challenger_win_report(receipt):
    rep = build_report(receipt, "chal_hk")
    duel = rep["duel"]
    assert duel["role"] == "challenger"
    v = duel["verdict"]
    assert v["challenger_wins_round"] and v["dethroned"]
    assert "WON the duel" in duel["diagnosis"]
    # per-size geomeans: the fixture's challenger is uniformly 0.6× the king
    s = duel["sizes"][0]
    assert s["own_geomean"] == pytest.approx(s["opp_geomean"] * 0.6, rel=1e-6)
    # winning challenger gets no "losses concentrate" tip
    assert not any("losses concentrate" in t for t in rep["advice"])


def test_king_view_is_symmetric(receipt):
    duel = build_report(receipt, "0")["duel"]
    assert duel["role"] == "king"
    assert "DETHRONED" in duel["diagnosis"]
    s = duel["sizes"][0]
    assert s["own_geomean"] == pytest.approx(s["opp_geomean"] / 0.6, rel=1e-6)


def test_close_loss_is_distinguished_from_plain_loss(receipt):
    # 0 < lcb ≤ margin is a keep-going signal, not "you were worse".
    v = replace(receipt.verdict, lcb=0.01, margin=0.02,
                challenger_wins_round=False, inconclusive=False, dethroned=False)
    r = replace(receipt, verdict=v)
    duel = build_report(r, "chal_hk")["duel"]
    assert "close loss" in duel["diagnosis"] and "BETTER than the king" in duel["diagnosis"]

    v2 = replace(v, lcb=-0.05)
    duel2 = build_report(replace(receipt, verdict=v2), "chal_hk")["duel"]
    assert "you lost the duel" in duel2["diagnosis"]


def test_source_breakdown_ranks_lost_feeds_worst_first():
    base, _, _ = make_scored_receipt()

    def rec(i, mase, source):
        return WindowScoreRecord(series_id=f"w{i}", channel=0, mase=mase,
                                 qloss_per_q=(mase,) * 3, abs_target=5.0,
                                 quantile_levels=(0.1, 0.5, 0.9), source=source)

    king = tuple(rec(i, 1.0, "feedA") for i in range(4)) + \
        tuple(rec(i + 4, 1.0, "feedB") for i in range(4))
    # challenger: much worse on feedA, better on feedB
    chal = tuple(rec(i, 2.0, "feedA") for i in range(4)) + \
        tuple(rec(i + 4, 0.5, "feedB") for i in range(4))
    r = replace(base, entry_scores=(
        EntryScores("king", "toto2-4m", "king_hk", 0, king),
        EntryScores("challenger", "toto2-4m", "chal_hk", 1, chal),
    ))
    duel = build_report(r, "chal_hk")["duel"]
    assert duel["sources_labeled"] is True
    worst = duel["worst_sources"]
    assert worst[0]["source"] == "feedA" and worst[0]["ratio"] > 1.0
    assert worst[-1]["source"] == "feedB" and worst[-1]["ratio"] < 1.0
    # a losing challenger is pointed at the lost feed — and only the lost one
    v = replace(r.verdict, challenger_wins_round=False)
    rep = build_report(replace(r, verdict=v), "chal_hk")
    tip = next(t for t in rep["advice"] if "losses concentrate" in t)
    assert "feedA" in tip and "feedB" not in tip


def test_mismatched_rows_skip_pairing_rather_than_guess():
    base, _, _ = make_scored_receipt()
    king_row = base.entry_scores[0]
    chal_row = base.entry_scores[1]
    shortened = replace(chal_row, scores=chal_row.scores[:10])
    r = replace(base, entry_scores=(king_row, shortened))
    duel = build_report(r, "chal_hk")["duel"]
    assert duel["worst_sources"] == [] and duel["best_sources"] == []


# ── whole-report shapes ──────────────────────────────────────────────────────


def test_non_participant_gets_reveal_advice(receipt):
    rep = build_report(receipt, "424242")
    assert not rep["participated"]
    assert "not in this round's participant set" in rep["participation_note"]
    assert any("reveal-status" in t for t in rep["advice"])


def test_rejected_round_is_round_level_not_miner_level():
    rep = build_report(load_receipt(dump_receipt(make_rejected_receipt())), "1")
    assert rep["status"] == "rejected"
    assert "round-level gate" in rep["participation_note"]
    text = render_report(rep)
    # no heat/duel noise on a round that scored no one
    assert "heat" not in text.split("note")[1].split("next steps")[0]
    assert compact_row(rep).endswith("rejected (signature_invalid)")


def test_report_is_strict_json(receipt):
    for who in ("1", "0", "9", "hk15", "424242"):
        dumped = json.dumps(build_report(receipt, who), allow_nan=False)
        assert json.loads(dumped)


def test_render_and_compact_smoke(receipt):
    for who, needle in (("1", "WON the duel"), ("9", "behind the last finalist"),
                        ("hk15", "trainer note"), ("18", "duplicate")):
        assert needle in render_report(build_report(receipt, who))
    row = compact_row(build_report(receipt, "9"))
    assert row.startswith("round ") and "heat screened" in row and "rank 3/4" in row


def test_heat_math_geomean_matches_components(receipt):
    e = build_report(receipt, "1")["heat"]["entrant"]
    assert math.sqrt(e["crps"] * e["mase"]) == pytest.approx(0.3999, abs=1e-3)
