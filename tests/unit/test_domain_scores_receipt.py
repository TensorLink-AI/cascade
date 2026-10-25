"""DEC-CA-0050 P2: signed window scores carry the pool domain (drop-when-default)
and the audit re-derives a lane-B domain verdict from them alone."""

from __future__ import annotations

import dataclasses
import json

import numpy as np

from cascade.eval.scoring import WindowScore
from cascade.shared.receipt import (
    EntryScores,
    WindowScoreRecord,
    dump_receipt,
    load_receipt,
)

from .receipt_fixture import make_scored_receipt

DOMAINS = ("energy", "web", "nature", "sales")


def _ws(i: int, scale: float, domain: str, rng) -> WindowScore:
    return WindowScore(
        series_id=f"w{i}",
        mase=float(rng.uniform(0.5, 1.5) * scale),
        qloss_per_q=(rng.uniform(0.1, 1.0, size=9) * scale).round(9),
        abs_target=float(round(rng.uniform(5.0, 10.0), 9)),
        domain=domain,
        source=f"feed-{i % 37}",
    )


def _labeled_receipt(effect: dict[str, float], n: int = 320):
    """The audit fixture's scored receipt with its entry_scores swapped for
    domain-labeled ones (king vs one challenger, ``effect`` = per-domain loss
    multiplier on the challenger)."""
    receipt, _, _ = make_scored_receipt()
    rng = np.random.default_rng(7)
    king = [_ws(i, 1.0, DOMAINS[i % len(DOMAINS)], rng) for i in range(n)]
    chal = [
        dataclasses.replace(
            k, mase=k.mase * effect.get(k.domain, 1.0),
            qloss_per_q=k.qloss_per_q * effect.get(k.domain, 1.0))
        for k in king
    ]
    size = receipt.entry_scores[0].size
    entries = (
        EntryScores("king", size, "king_hk", 0,
                    tuple(WindowScoreRecord.from_score(s, keep_domain=True) for s in king)),
        EntryScores("challenger", size, "chal_hk", 1,
                    tuple(WindowScoreRecord.from_score(s, keep_domain=True) for s in chal)),
    )
    return dataclasses.replace(receipt, entry_scores=entries), king, chal


# ── the field ────────────────────────────────────────────────────────────────


def test_unlabeled_scores_keep_canonical_bytes():
    receipt, _, _ = make_scored_receipt()
    assert all(s.domain is None for e in receipt.entry_scores for s in e.scores)
    body = json.loads(receipt.canonical_body())
    assert all("domain" not in s for e in body["entry_scores"] for s in e["scores"])


def test_domain_round_trips_through_the_signed_body():
    receipt, king, _ = _labeled_receipt({})
    body = json.loads(receipt.canonical_body())
    assert [s["domain"] for s in body["entry_scores"][0]["scores"][:4]] == list(DOMAINS)
    loaded = load_receipt(dump_receipt(receipt))
    assert loaded == receipt
    assert loaded.canonical_body() == receipt.canonical_body()
    assert loaded.entry_scores[0].scores[1].to_score().domain == "web"


def test_empty_domain_maps_to_none():
    rec = WindowScoreRecord.from_score(
        WindowScore("w0", 1.0, np.ones(9), 1.0, domain=""), keep_domain=True)
    assert rec.domain is None and rec.to_score().domain == ""


# ── the validator gate (audit finding 1) ─────────────────────────────────────


def _runner_with_lane_b(cfg, from_block, tmp_path):
    from cascade.shared.config import parse_lane_b
    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.state import genesis

    lb = parse_lane_b({"from_block": from_block, "domains": ["energy", "web"]})
    v = dataclasses.replace(cfg.validator, state_db_path=str(tmp_path / "s.json"))
    return ValidatorRunner(cfg=dataclasses.replace(cfg, lane_b=lb, validator=v),
                           state=genesis("king_hk", 0))


def test_lane_b_off_receipt_is_byte_identical_on_a_labeled_pool(cfg, tmp_path):
    """The deployed pool IS domain-labeled. With lane B off, the validator's
    signed window records must not carry it: the receipt hashes exactly as a
    pre-v2 validator's (scores without the label — the 838ac9b body)."""
    receipt, king, chal = make_scored_receipt()
    from cascade.shared.manifest import load_manifest

    m = load_manifest(json.dumps(receipt.manifest))
    labeled_k = [dataclasses.replace(s, domain=DOMAINS[i % 4]) for i, s in enumerate(king)]
    labeled_c = [dataclasses.replace(s, domain=DOMAINS[i % 4]) for i, s in enumerate(chal)]
    off = _runner_with_lane_b(cfg, 0, tmp_path)
    entries = tuple(
        dataclasses.replace(e, scores=off._window_records(sc, m))
        for e, sc in zip(receipt.entry_scores, (labeled_k, labeled_c), strict=True))
    gated = dataclasses.replace(receipt, entry_scores=entries)
    assert gated.canonical_body() == receipt.canonical_body()
    assert all(r.domain is None for e in entries for r in e.scores)
    # once lane B is active at the settlement's block, the label IS signed
    on = _runner_with_lane_b(cfg, 1, tmp_path)
    rec = on._window_records(labeled_k, m)
    assert [r.domain for r in rec[:4]] == list(DOMAINS)


def test_lane_b_pay_is_drop_when_default_and_round_trips():
    receipt, _, _ = make_scored_receipt()
    assert b"lane_b_pay" not in receipt.canonical_body()
    pay = {"block": 7200, "epoch_blocks": 900, "wins": [
        {"domain": "energy", "hotkey": "alice", "uid": 9, "merge_block": 7200, "age": 0,
         "multiplier": 1.0}]}
    r2 = dataclasses.replace(receipt, lane_b=(), lane_b_pay=pay)
    loaded = load_receipt(dump_receipt(r2))
    assert loaded.lane_b_pay == pay and loaded.lane_b == ()
    assert loaded.canonical_body() == r2.canonical_body()
