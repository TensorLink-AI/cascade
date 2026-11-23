"""W7d: UNVERIFIABLE settlements carry a typed reason (README "Verdicts"),
and the published state snapshots, when given, bind the posterior digest
(README "Posterior state").

On main the module is absent (`ModuleNotFoundError: cascade.satellite`) and
the subprocess case fails on its return-code assertion.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys

import pytest

from .helpers import (
    CHAIN_TIMEOUT,
    REPLAY_SCRIPT,
    REPO,
    UNVERIFIABLE_REASONS,
    boundary,
    build_all,
    canonical,
    chain_gated,
    describe,
    doctored,
    has_path,
    replay,
    replay_module,
    save_publication,
    sha,
)


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    return build_all(tmp_path_factory.mktemp("w7d-unverifiable"))


def _copy_reports(pub, dst):
    shutil.copytree(pub.reports_dir, dst)
    return dst


def _assert_unverifiable_from(res, n, reason):
    mod = replay_module()
    for r in res[:n - 1]:
        assert r.status == mod.MATCH, describe(res)
    assert res[n - 1].status == mod.UNVERIFIABLE, describe(res)
    assert res[n - 1].reason == reason, describe(res)
    assert res[n - 1].reason in mod.UNVERIFIABLE_REASONS
    for r in res[n:]:
        assert r.status == mod.UNVERIFIABLE and r.reason == "prior_unverified", describe(res)


def test_every_reason_is_exported():
    mod = replay_module()
    assert set(UNVERIFIABLE_REASONS) <= set(mod.UNVERIFIABLE_REASONS)


def test_an_unknown_report_schema_is_unverifiable(chains):
    c = chains["gated"]
    reports, receipts = doctored(c.pub, 2, lambda rep: rep.update(schema="lane-d-report/2"))
    res = replay(c.pub, reports=reports, receipts=receipts)
    _assert_unverifiable_from(res, 2, "unknown_schema")


def test_a_receipt_without_its_published_report_is_unverifiable(chains, tmp_path):
    c = chains["gated"]
    d = _copy_reports(c.pub, tmp_path / "reports")
    (d / f"lane_d_report_{boundary(2)}.json").unlink()
    res = replay(c.pub, reports=d)
    _assert_unverifiable_from(res, 2, "report_missing")


def test_a_report_mapping_without_the_boundary_is_unverifiable(chains):
    c = chains["house_bar"]
    reports = c.pub.reports()
    del reports[boundary(3)]
    res = replay(c.pub, reports=reports)
    _assert_unverifiable_from(res, 3, "report_missing")


@pytest.mark.parametrize("damage", ["duplicate_key", "truncated", "not_json"])
def test_an_unreadable_published_report_is_unverifiable(chains, tmp_path, damage):
    c = chains["gated"]
    d = _copy_reports(c.pub, tmp_path / "reports")
    f = d / f"lane_d_report_{boundary(2)}.json"
    raw = f.read_text()
    if damage == "duplicate_key":
        # The same key twice: never last-wins (W7c RV5 (c)).
        rep = json.loads(raw)
        text = json.dumps(rep, sort_keys=True)
        f.write_text(text[:-1] + ', "crowned": null}')
    elif damage == "truncated":
        f.write_text(raw[: len(raw) // 2])
    else:
        f.write_bytes(b"\xff\xfe not json")
    res = replay(c.pub, reports=d)
    _assert_unverifiable_from(res, 2, "report_unreadable")


def test_a_missing_signature_verifier_is_unverifiable_never_a_match(tmp_path):
    """DEC-CA-0051: a signature that cannot be checked is not a verdict on
    it. With the SDK unimportable every settlement is UNVERIFIABLE
    (`signature_unverifiable` first, `prior_unverified` after)."""
    c = chain_gated(tmp_path / "chain")
    saved = save_publication(c.pub, tmp_path / "pub")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    out = subprocess.run([sys.executable, "-P", "-c", REPLAY_SCRIPT, str(saved), "block-sdk"],
                         cwd=str(REPO), env=env, capture_output=True, text=True,
                         timeout=CHAIN_TIMEOUT)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])["results"]
    mod = replay_module()
    assert [r["status"] for r in got] == [mod.UNVERIFIABLE] * 3, got
    assert [r["reason"] for r in got] == ["signature_unverifiable", "prior_unverified",
                                          "prior_unverified"], got


# ── posterior state (README "Posterior state") ──────────────────────────────


def test_a_state_snapshot_off_the_receipt_posterior_is_a_mismatch(chains):
    mod = replay_module()
    c = chains["gated"]
    states = dict(c.pub.states)
    states[boundary(2)] = c.pub.states[boundary(1)]
    res = replay(c.pub, states=states)
    assert res[0].status == mod.MATCH, describe(res)
    assert res[1].status == mod.MISMATCH, describe(res)
    assert has_path(res[1], "receipt.posterior_state_sha256"), list(res[1].mismatches)


def test_a_hand_edited_state_with_reissued_receipts_is_a_mismatch_at_its_king(chains):
    """The state snapshot is re-canonicalised after a hand edit (KING_D back
    to the house) and both receipts that bind it are re-issued, so every
    digest links: the replay still re-derives the state's king from the
    signed report and refuses it."""
    mod = replay_module()
    c = chains["gated"]
    b2 = boundary(2)
    state = json.loads(c.pub.states[b2])
    rep = c.pub.report(2)
    state["king"] = {**rep["house"], "hotkey": None, "house": True}
    raw = canonical(state)
    states = {**c.pub.states, b2: raw}
    receipts = copy.deepcopy(c.pub.receipts)
    receipts[1]["posterior_state_sha256"] = sha(raw)
    receipts[2]["prior_state_sha256"] = sha(raw)
    res = replay(c.pub, receipts=receipts, states=states)
    assert res[0].status == mod.MATCH, describe(res)
    assert res[1].status == mod.MISMATCH, describe(res)
    assert has_path(res[1], "state.king"), list(res[1].mismatches)


def test_a_state_whose_last_report_binds_another_body_is_a_mismatch(chains):
    mod = replay_module()
    c = chains["house_bar"]
    b2 = boundary(2)
    state = json.loads(c.pub.states[b2])
    state["last_report"]["report_sha256"] = c.pub.receipts[0]["report_sha256"]
    raw = canonical(state)
    receipts = copy.deepcopy(c.pub.receipts)
    receipts[1]["posterior_state_sha256"] = sha(raw)
    receipts[2]["prior_state_sha256"] = sha(raw)
    res = replay(c.pub, receipts=receipts, states={**c.pub.states, b2: raw})
    assert [r.status for r in res[:1]] == [mod.MATCH], describe(res)
    assert res[1].status == mod.MISMATCH, describe(res)
    assert has_path(res[1], "state.last_report"), list(res[1].mismatches)


def test_a_non_canonical_state_snapshot_is_a_mismatch(chains):
    mod = replay_module()
    c = chains["gated"]
    b1 = boundary(1)
    raw = json.dumps(json.loads(c.pub.states[b1]), indent=1).encode()
    receipts = copy.deepcopy(c.pub.receipts)
    receipts[0]["posterior_state_sha256"] = sha(raw)
    receipts[1]["prior_state_sha256"] = sha(raw)
    res = replay(c.pub, receipts=receipts, states={**c.pub.states, b1: raw})
    assert res[0].status == mod.MISMATCH, describe(res)
    assert has_path(res[0], "state"), list(res[0].mismatches)
