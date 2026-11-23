"""W7c: the lane D receipt record and its prior/posterior digest binding
(plan rev6 §1.5 "State": "Its digest is in the receipt's prior and posterior
digests (DEC-CA-0058), and the report digest is bound into the receipt as the
bench source is"; RV5 W7b gaps (a)-(c)).

Every receipt comes from a real `settle_lane_d` run on the fixture chain.
README.md has the contract.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from .helpers import (
    LAST_REPORT_KEYS,
    RECEIPT_ERROR_CODES,
    RECEIPT_KEYS,
    STATE_ERROR_CODES,
    Chain,
    body_sha,
    boundary,
    canonical,
    is_hex64,
    other_crown,
    receipt_of,
    sha,
    state_sha,
    unarmed_cfg,
    write_canonical,
)


def _chain_run(tmp_path, upto: int):
    """A clean run: settlements, and the state file bytes after each."""
    chain = Chain(tmp_path)
    path = chain.state_path("run")
    runs, states = [], []
    for n in range(1, upto + 1):
        runs.append(chain.settle(n, path))
        states.append(path.read_bytes())
    return chain, path, runs, states


def _receipts(runs) -> list[dict]:
    out = [receipt_of(s) for s in runs]
    assert all(isinstance(r, dict) for r in out), f"settle_lane_d returned no receipt: {out!r}"
    return out


def _receipt_error(excinfo, code: str) -> None:
    from cascade.validator.lane_d_state import LaneDReceiptError

    assert isinstance(excinfo.value, LaneDReceiptError), repr(excinfo.value)
    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.code == code, (excinfo.value.code, str(excinfo.value))
    assert excinfo.value.code in RECEIPT_ERROR_CODES


def _resign(report: dict, key) -> dict:
    from cascade.validator.lane_d_report import sign_report

    return sign_report(report, key)


# ── invariant 1 / 10 pair ───────────────────────────────────────────────────


def test_receipt_unchanged_when_unarmed(tmp_path):
    chain = Chain(tmp_path)
    path = chain.state_path("unarmed")
    s = chain.settle(1, path, cfg=unarmed_cfg())
    assert s.report is None and s.pay is None
    assert getattr(s, "receipt", "absent") is None, \
        "an unarmed settlement carries receipt=None (no lane D receipt fields)"
    assert sorted(p.name for p in path.parent.rglob("*")) == [], \
        "an unarmed settlement writes no lane D file"


def test_receipt_armed_nonempty_lane_d_effect(tmp_path):
    from cascade.validator.lane_d_state import RECEIPT_SCHEMA, genesis

    chain, path, runs, states = _chain_run(tmp_path, 1)
    (rec,) = _receipts(runs)
    assert tuple(sorted(rec)) == tuple(sorted(RECEIPT_KEYS))
    assert rec["schema"] == RECEIPT_SCHEMA and isinstance(rec["schema"], str)
    assert rec["boundary"] == boundary(1) == runs[0].report["boundary"]
    assert type(rec["boundary"]) is int
    for k in ("prior_state_sha256", "posterior_state_sha256", "report_sha256"):
        assert is_hex64(rec[k]), (k, rec[k])
    # prior = the state as loaded before the settlement (here: genesis)
    assert rec["prior_state_sha256"] == state_sha(genesis(chain.house))
    # posterior = the state file as written
    assert rec["posterior_state_sha256"] == sha(states[0]) == state_sha(json.loads(states[0]))
    # report = the canonical body (not the signature bytes)
    assert rec["report_sha256"] == body_sha(runs[0].report)
    assert rec["prior_state_sha256"] != rec["posterior_state_sha256"]


# ── what the state and the receipt bind ─────────────────────────────────────


def test_state_binds_the_report_body_and_the_prior_never_the_signature(tmp_path):
    chain, path, runs, states = _chain_run(tmp_path, 2)
    recs = _receipts(runs)
    for s, rec, raw in zip(runs, recs, states, strict=True):
        state = json.loads(raw)
        last = state["last_report"]
        assert set(LAST_REPORT_KEYS) <= set(last), last
        assert "signature" not in last
        assert last["boundary"] == state["boundary"] == rec["boundary"]
        assert last["rows_sha256"] == s.report["rows_sha256"]
        assert last["report_sha256"] == rec["report_sha256"] == body_sha(s.report)
        assert last["prior_state_sha256"] == rec["prior_state_sha256"]
        assert s.report["signature"].encode() not in raw, \
            "the randomized signature must not enter the state (RV5 W7b gap b)"


def test_a_resigned_report_still_matches_its_receipt(tmp_path):
    from cascade.validator.lane_d_state import verify_receipt

    chain, path, runs, states = _chain_run(tmp_path, 1)
    (rec,) = _receipts(runs)
    again = _resign(runs[0].report, chain.signer)
    assert body_sha(again) == rec["report_sha256"]
    verify_receipt(rec, report=runs[0].report)
    verify_receipt(rec, report=again)
    verify_receipt(rec, report=again, state=json.loads(states[0]))
    # a JSON round trip with any layout is the same body
    verify_receipt(rec, report=json.loads(json.dumps(again, indent=3)))


# ── the chain ───────────────────────────────────────────────────────────────


def test_receipt_chain_links_each_posterior_to_the_next_prior(tmp_path):
    from cascade.validator.lane_d_state import verify_receipt, verify_receipt_chain

    chain, path, runs, states = _chain_run(tmp_path, 3)
    recs = _receipts(runs)
    for a, b in zip(recs, recs[1:], strict=False):
        assert b["prior_state_sha256"] == a["posterior_state_sha256"]
        assert b["boundary"] > a["boundary"]
    assert [r["boundary"] for r in recs] == [boundary(n) for n in (1, 2, 3)]
    verify_receipt_chain(recs)
    verify_receipt_chain(recs[1:])
    verify_receipt_chain(recs[:1])
    for s, rec, raw in zip(runs, recs, states, strict=True):
        verify_receipt(rec, report=s.report, state=json.loads(raw))
    # the crown at boundary 2 is in the posterior the chain binds
    post2 = json.loads(states[1])
    assert post2["king"]["digest"] == chain.x["digest"] == runs[1].report["crowned"]


def test_a_missing_link_is_refused(tmp_path):
    from cascade.validator.lane_d_state import verify_receipt_chain

    _, _, runs, _ = _chain_run(tmp_path, 3)
    r1, r2, r3 = _receipts(runs)
    with pytest.raises(ValueError) as e:
        verify_receipt_chain([r1, r3])
    _receipt_error(e, "chain_break")


@pytest.mark.parametrize("field", ["prior_state_sha256", "posterior_state_sha256",
                                   "report_sha256", "boundary", "schema"])
def test_a_receipt_missing_a_field_is_refused(tmp_path, field):
    from cascade.validator.lane_d_state import verify_receipt, verify_receipt_chain

    _, _, runs, _ = _chain_run(tmp_path, 2)
    r1, r2 = _receipts(runs)
    cut = {k: v for k, v in r2.items() if k != field}
    with pytest.raises(ValueError) as e:
        verify_receipt_chain([r1, cut])
    _receipt_error(e, "missing_field")
    with pytest.raises(ValueError) as e:
        verify_receipt(cut, report=runs[1].report)
    _receipt_error(e, "missing_field")


@pytest.mark.parametrize("field,value", [
    ("prior_state_sha256", ""), ("posterior_state_sha256", "A" * 64),
    ("report_sha256", "0x" + "ab" * 32), ("boundary", "1"), ("boundary", True),
    ("boundary", 1.0)])
def test_a_receipt_with_a_malformed_field_is_refused(tmp_path, field, value):
    from cascade.validator.lane_d_state import verify_receipt_chain

    _, _, runs, _ = _chain_run(tmp_path, 1)
    (rec,) = _receipts(runs)
    with pytest.raises(ValueError) as e:
        verify_receipt_chain([{**rec, field: value}])
    _receipt_error(e, "bad_value")


def test_receipt_structure_errors_are_typed(tmp_path):
    from cascade.validator.lane_d_state import RECEIPT_SCHEMA, verify_receipt_chain

    _, _, runs, _ = _chain_run(tmp_path, 1)
    (rec,) = _receipts(runs)
    with pytest.raises(ValueError) as e:
        verify_receipt_chain([json.dumps(rec)])
    _receipt_error(e, "not_object")
    with pytest.raises(ValueError) as e:
        verify_receipt_chain([{**rec, "schema": RECEIPT_SCHEMA + "-w7c"}])
    _receipt_error(e, "unknown_schema")
    with pytest.raises(ValueError) as e:
        verify_receipt_chain([{**rec, "w7c_extra": 1}])
    _receipt_error(e, "unknown_field")


@pytest.mark.parametrize("order", ["reversed", "repeated"])
def test_receipts_out_of_boundary_order_are_refused(tmp_path, order):
    from cascade.validator.lane_d_state import verify_receipt_chain

    _, _, runs, _ = _chain_run(tmp_path, 2)
    r1, r2 = _receipts(runs)
    seq = [r2, r1] if order == "reversed" else [r1, r1]
    with pytest.raises(ValueError) as e:
        verify_receipt_chain(seq)
    _receipt_error(e, "boundary_order")


# ── the report a receipt binds ──────────────────────────────────────────────


def test_a_replayed_old_report_is_refused_by_its_boundary(tmp_path):
    from cascade.validator.lane_d_state import verify_receipt

    chain, _, runs, _ = _chain_run(tmp_path, 2)
    r1, r2 = _receipts(runs)
    old = runs[0].report
    # verify_report would accept it: it binds no expected boundary (RV5 gap a)
    with pytest.raises(ValueError) as e:
        verify_receipt(r2, report=old)
    _receipt_error(e, "boundary_mismatch")
    # relabelled to the new boundary and re-signed: the body is still not the
    # one the receipt bound
    relabelled = _resign({**copy.deepcopy(old), "boundary": boundary(2)}, chain.signer)
    with pytest.raises(ValueError) as e:
        verify_receipt(r2, report=relabelled)
    _receipt_error(e, "report_mismatch")


def test_a_swapped_report_at_the_same_boundary_is_refused(tmp_path):
    from cascade.validator.lane_d_state import verify_receipt

    chain, _, runs, _ = _chain_run(tmp_path, 2)
    _, r2 = _receipts(runs)
    swapped = copy.deepcopy(runs[1].report)
    swapped["crowned"] = other_crown(swapped, chain)
    swapped = _resign(swapped, chain.signer)
    with pytest.raises(ValueError) as e:
        verify_receipt(r2, report=swapped)
    _receipt_error(e, "report_mismatch")


def test_a_posterior_that_is_not_the_given_state_is_refused(tmp_path):
    from cascade.validator.lane_d_state import verify_receipt

    _, _, runs, states = _chain_run(tmp_path, 2)
    r1, r2 = _receipts(runs)
    with pytest.raises(ValueError) as e:
        verify_receipt(r2, report=runs[1].report, state=json.loads(states[0]))
    _receipt_error(e, "state_mismatch")


# ── a tampered state file ───────────────────────────────────────────────────


def test_a_tampered_state_file_breaks_the_receipt_chain(tmp_path):
    from cascade.validator.lane_d_state import (
        LaneDStateError,
        load_state,
        verify_receipt_chain,
    )

    chain = Chain(tmp_path)
    path = chain.state_path("run")
    s1 = chain.settle(1, path)
    (r1,) = _receipts([s1])
    state = json.loads(path.read_bytes())
    (dg,) = sorted(state["entries"])
    state["entries"][dg]["context"] = 8192      # a hand edit, canonical bytes kept
    write_canonical(path, state)
    with pytest.raises(LaneDStateError) as e:
        load_state(path, chain.house, expect_sha256=r1["posterior_state_sha256"])
    assert e.value.code == "state_digest_mismatch" and e.value.code in STATE_ERROR_CODES
    try:
        s2 = chain.settle(2, path)
    except LaneDStateError as err:
        # an implementation that holds the last posterior refuses up front
        assert err.code == "state_digest_mismatch", err.code
        return
    (r2,) = _receipts([s2])
    assert r2["prior_state_sha256"] == state_sha(state) != r1["posterior_state_sha256"]
    with pytest.raises(ValueError) as e:
        verify_receipt_chain([r1, r2])
    _receipt_error(e, "chain_break")


def test_error_codes_are_exported():
    from cascade.validator.lane_d_state import RECEIPT_ERROR_CODES as codes

    # W7c pins that all of its codes are exported; later WPs (W14) may extend
    # the tuple, so this is a superset check, not an equality.
    exported = list(map(str, codes))
    assert len(exported) == len(set(exported))
    missing = sorted(set(RECEIPT_ERROR_CODES) - set(exported))
    assert not missing, missing


# ── the persisted report file ───────────────────────────────────────────────


def test_the_signed_report_is_persisted_before_the_state(tmp_path, monkeypatch):
    import os

    from cascade.validator.lane_d_state import load_report_file, report_path

    chain = Chain(tmp_path)
    path = chain.state_path("run")
    order = []
    real = os.replace

    def replace(src, dst, *a, **k):
        order.append(str(Path(dst).resolve()))
        return real(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", replace)
    s = chain.settle(1, path)
    monkeypatch.undo()
    rp = report_path(path, boundary(1))
    assert rp.resolve().is_relative_to(path.parent.resolve()), \
        "the report file lives with the state (it is snapshotted with it)"
    assert report_path(path, boundary(2)) != rp
    assert str(rp.resolve()) in order and str(path.resolve()) in order, order
    assert order.index(str(rp.resolve())) < order.index(str(path.resolve())), \
        "the report must land before the state that binds it"
    assert load_report_file(rp) == s.report


@pytest.mark.parametrize("where", ["top", "nested"])
def test_a_report_file_with_duplicate_keys_is_refused(tmp_path, where):
    from cascade.validator.lane_d_state import LaneDStateError, load_report_file

    _, _, runs, _ = _chain_run(tmp_path, 1)
    report = runs[0].report
    text = canonical(report).decode()
    if where == "top":
        bad = text.replace('"crowned":', '"crowned":"' + "e" * 64 + '","crowned":', 1)
    else:
        bad = text.replace('"pay":{', '"pay":{"share":1.0,', 1)
    assert bad != text
    f = tmp_path / "report.json"
    f.write_text(bad)
    with pytest.raises(LaneDStateError) as e:
        load_report_file(f)
    assert e.value.code == "duplicate_key"
    # the same report without the duplicate loads, whatever the layout
    f.write_text(json.dumps(report, indent=2))
    assert load_report_file(f) == report
