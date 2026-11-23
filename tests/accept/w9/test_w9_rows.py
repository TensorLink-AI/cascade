"""W9: `rows_sha256` over the canonical rows (plan rev6 §1.5 "(rev3) Rows
digest. Every verdict publishes `rows_sha256`, the sha256 of the exact
scored per-window rows (canonical serialisation, both models, frozen set
order)"; README "Rows").

Main checks the digest against the rows as listed (the tamper case passes on
main and stays minimal) but accepts any row order, so a permuted list with a
recomputed digest PASSes (an `assert`): two honest auditors holding the same
rows could then publish different digests.
"""

from __future__ import annotations

import copy

import pytest

from . import helpers as h


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return h.crown_chain(tmp_path_factory.mktemp("w9-rows"))


@pytest.fixture(scope="module")
def caps(tmp_path_factory):
    return h.caps(tmp_path_factory.mktemp("w9-rows-caps"))


def _canonical_order(report: dict) -> list[tuple]:
    """The pinned order: cohorts by index, each cohort's `window_ids` in
    order, and per window the models `[king_digest, house_digest (when
    present), *members]`."""
    out = []
    for c in sorted(report["cohorts"], key=lambda c: int(c["index"])):
        models = [c["king_digest"]] + ([c["house_digest"]] if "house_digest" in c else []) \
            + list(c["members"])
        out += [(int(c["index"]), w, m) for w in c["window_ids"] for m in models]
    return out


def _order(report: dict) -> list[tuple]:
    return [(int(r["cohort"]), r["window_id"], r["digest"]) for r in report["rows"]]


def test_honest_rows_are_in_the_pinned_order(run, caps):
    """The producer's order is the contract the audit re-derives."""
    for name, report in (("s1", run["s1"].report), ("s2", run["s2"].report),
                         ("cold", caps["cold"]["report"])):
        assert _order(report) == _canonical_order(report), f"{name}: rows out of order"


def test_a_tampered_rows_sha256_fails(run):
    report = copy.deepcopy(run["s1"].report)
    report["rows_sha256"] = h.text_sha("not the rows")
    from cascade.validator.lane_d_report import sign_report

    h.assert_fail(sign_report(report, h.signer()), run["cfg"], "rows_sha256 not over the rows")


@pytest.mark.parametrize("permute", ["reversed", "models_swapped", "windows_swapped"])
def test_permuted_rows_with_a_recomputed_digest_fail(run, caps, permute):
    src = caps["cold"] if permute == "models_swapped" else {"report": run["s1"].report,
                                                            "cfg": run["cfg"]}
    report = copy.deepcopy(src["report"])
    rows = report["rows"]
    if permute == "reversed":
        rows.reverse()
    elif permute == "models_swapped":
        # the two members of window 0 (a three-model window: king, A, C)
        rows[1], rows[2] = rows[2], rows[1]
    else:
        n = 1 if "house_digest" not in report["cohorts"][0] else 2
        n += len(report["cohorts"][0]["members"])
        rows[:2 * n] = rows[n:2 * n] + rows[:n]
    assert _order(report) != _canonical_order(report), "fixture: the permutation is a no-op"
    h.assert_fail(h.resign(report), src["cfg"],
                  f"rows {permute}: rows_sha256 over a non-canonical order")
