"""W3b2: the pod run's evidence verifier (plan §3.0 W3b2 "ci (recorded suite
outputs) + pod", §17.1 L4 "the verifier passes against the live resource").

The pod part is one live paired sweep of a candidate, KING_D and the house on
one bench host in one session, on a suite drawn by W3b1. Its evidence is
``evidence/w3b2.json`` (schema ``laned_w3b2/2``), verified by
``scripts/laned/accept W3b2 --evidence evidence/w3b2.json``, which exports it as
``$LANED_EVIDENCE``. The ``test_evidence_*`` tests SKIP when it is unset; when
it IS set, a missing, malformed, synthetic or hand-edited file fails hard.

RV5 (BLOCKING item 6): synthetic rows labelled ``source: "pod"`` with any block
hash and a self-consistent ``gate`` passed the old verifier. The evidence is
now bound to the repo's suite pin, to registry refs, to the shipped
``[lane_d]`` params, and to the owner's ``verified_live`` record (chain block
hash, the draw, KING_D / house pointers, registry resolution), hash-bound to
the body. The ``test_verifier_*`` tests prove offline that synthetic or
tampered evidence is refused. Contract: README.md, ``evidence_lib.py``."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from .evidence_lib import (
    MIN_ITEMS,
    ROLES,
    body_sha256,
    evidence_problems,
    lane_d_cfg,
    netuid,
    pinned_items,
    record_for,
    ref_for,
    repo_pin,
    repo_revisions,
    synthetic_body,
    verified_live_problems,
)
from .helpers import GATE_SUITE, SENTINEL, SESSION, gate_kwargs, triple


def _evidence() -> dict:
    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("LANED_EVIDENCE is unset: no pod evidence to verify yet. L4 needs "
                    "`scripts/laned/accept W3b2 --evidence evidence/w3b2.json` green on main "
                    "(a live paired sweep on one bench host, then the owner's `python -P -m "
                    "cascade.laned.bench_evidence verify`; see tests/accept/w3b2/README.md)")
    p = Path(path)
    assert p.is_file(), f"LANED_EVIDENCE={path} does not exist"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AssertionError(f"LANED_EVIDENCE={path} is not JSON: {e}") from None


def test_evidence_is_valid_pod_evidence():
    ev = _evidence()
    assert evidence_problems(ev) == []


def test_evidence_sweeps_drawn_suite_and_sentinel_in_one_session():
    ev = _evidence()
    sessions = {ev["rows"][r]["session"] for r in ROLES}
    assert sessions == {ev["session"]}
    assert ev["suite"] != ev["sentinel"]


# ── the repo pin (a) ─────────────────────────────────────────────────────────


def test_repo_pins_gift_and_boom_and_not_time_or_fev():
    revs = repo_revisions()
    assert {"gift-eval", "boom", "time"} <= set(revs), revs
    for s in ("gift", "boom"):
        pin = repo_pin(s)
        assert pin is not None and pin["n_items"] >= MIN_ITEMS, pin
        assert pin["version"] == revs["gift-eval" if s == "gift" else "boom"]
    assert repo_pin("time") is None and repo_pin("fev") is None


def test_repo_pin_is_w3b4_suite_identity_of_the_pinned_items():
    from cascade.eval.suite_identity import suite_identity

    for s in ("gift", "boom"):
        pin = repo_pin(s)
        entry = {"status": "ok", "data_revision": pin["version"],
                 "rows": [{"item": i, "crps": 1.0, "mase": 1.0}
                          for i in reversed(pinned_items(s))]}
        assert suite_identity(s, entry) == pin


# ── the verifier, offline ────────────────────────────────────────────────────


def _body() -> dict:
    return synthetic_body()


def _verified() -> dict:
    ev = _body()
    ev["verified_live"] = record_for(ev)
    return ev


def _offline(ev) -> list[str]:
    return evidence_problems(ev, check_git=False)


def test_verifier_body_checks_accept_a_self_consistent_body():
    """The body checks alone (never enough for L4) accept a consistent body;
    each tamper below is then refused by them, not by the missing record."""
    assert evidence_problems(_body(), check_git=False, require_live=False) == []


def test_verifier_refuses_self_consistent_synthetic_pod_evidence():
    """RV5's forgery: everything an offline writer controls is consistent
    (pinned items and revision, refs, block hash, gate recomputed). With no
    live verification it is refused."""
    probs = _offline(_body())
    assert probs and any(p.startswith("verified_live") for p in probs), probs


def test_verifier_refuses_rv5_helper_rows_relabelled_as_pod():
    """The old ``_good()``: helpers.triple() rows, ``source: "pod"``, any block
    hash, gate = paired_gate(...). Refused even with a record forged for it:
    its items and revision are not the repo pin, and it has no refs."""
    from cascade.validator.lane_d_bench import paired_gate

    from .evidence_lib import _block_hash_for

    cand, king, house = triple(king=(1.03, 1.03), house=(1.01, 1.01))
    kw = gate_kwargs()
    ev = {"schema": "laned_w3b2/2", "wp": "W3b2", "source": "pod", "host": "valpriv-4090",
          "tool": {"git_sha": "1" * 39 + "2", "git_dirty": False}, "session": SESSION,
          "draw": {"network": "finney", "netuid": netuid(), "block": 9612000,
                   "block_hash": _block_hash_for(GATE_SUITE, SENTINEL)},
          "suite": GATE_SUITE, "sentinel": SENTINEL, "pin": repo_pin(GATE_SUITE),
          "params": {k: kw[k] for k in ("alpha", "tau_public", "B", "seed")},
          "rows": {"candidate": cand, "king_d": king, "house": house},
          "gate": json.loads(json.dumps(paired_gate(cand, king, house, **kw))), "passed": True}
    ev["refs"] = {r: ref_for(ev["rows"][r]["digest"]) for r in ROLES}
    ev["verified_live"] = record_for(ev)
    probs = evidence_problems(ev, check_git=False)
    assert any("pinned item set" in p for p in probs), probs
    assert any("data_revision" in p for p in probs), probs


def test_verifier_accepts_a_consistent_verified_record():
    """The offline shape: a body plus a record bound to it (the real record
    comes from the owner's verify; test_w3b2_live_verify.py)."""
    assert _offline(_verified()) == []


def _tamper_gate(ev):
    ev["gate"]["lcb"]["house"]["crps"] += 0.01


def _tamper_row(ev):
    ev["rows"]["house"]["suites"][ev["suite"]]["rows"][5]["crps"] *= 0.5


def _tamper_session(ev):
    ev["rows"]["king_d"]["session"] = "valpriv-4090/older"


def _tamper_source(ev):
    ev["source"] = "fixture"


def _tamper_suite(ev):
    ev["suite"] = "boom"
    for r in ROLES:
        ev["rows"][r]["suites"]["boom"] = copy.deepcopy(ev["rows"][r]["suites"]["gift"])


def _tamper_fault(ev):
    ev["rows"]["candidate"]["suites"][ev["suite"]]["rows"][2]["mase"] = None
    ev["gate"] = None


def _tamper_wp(ev):
    ev["wp"] = "W3b1"


def _synthetic_items(ev):
    """Made-up item ids (the old helpers' "gift/task-000" shape)."""
    for r in ROLES:
        for i, row in enumerate(ev["rows"][r]["suites"][ev["suite"]]["rows"]):
            row["item"] = f"{ev['suite']}/task-{i:03d}"


def _drop_items(ev):
    """Every record misses the same items: invisible without a pin."""
    for r in ROLES:
        ent = ev["rows"][r]["suites"][ev["suite"]]
        ent["rows"] = ent["rows"][:-5]


def _other_revision(ev):
    for r in ROLES:
        ev["rows"][r]["suites"][ev["suite"]]["data_revision"] = "0" * 40


def _edited_pin(ev):
    ev["pin"]["items_sha256"] = "0" * 64


def _no_pin_suite(ev):
    ev["pin"] = None


def _bad_ref_format(ev):
    ev["refs"]["house"] = ev["rows"]["house"]["digest"]


def _ref_other_digest(ev):
    ev["refs"]["king_d"] = ref_for("e" * 64)


def _no_refs(ev):
    del ev["refs"]


def _block_hash_format(ev):
    ev["draw"]["block_hash"] = ev["draw"]["block_hash"][2:]


def _testnet(ev):
    ev["draw"]["network"] = "test"


def _other_netuid(ev):
    ev["draw"]["netuid"] = 1


def _other_alpha(ev):
    ev["params"]["alpha"] = 0.2


def _other_tau(ev):
    ev["params"]["tau_public"] = 0.5


def _small_b(ev):
    ev["params"]["B"] = 200


TAMPER = {f.__name__.lstrip("_"): f for f in (
    _tamper_gate, _tamper_row, _tamper_session, _tamper_source, _tamper_suite, _tamper_fault,
    _tamper_wp, _synthetic_items, _drop_items, _other_revision, _edited_pin, _no_pin_suite,
    _bad_ref_format, _ref_other_digest, _no_refs, _block_hash_format, _testnet, _other_netuid,
    _other_alpha, _other_tau, _small_b)}


@pytest.mark.parametrize("name", sorted(TAMPER))
def test_verifier_rejects_tampering(name):
    """Each tamper is refused by the body checks themselves, and again after
    the record is re-bound to the tampered body (a forger who rewrites
    ``body_sha256`` still fails the body)."""
    ev = _body()
    TAMPER[name](ev)
    assert evidence_problems(ev, check_git=False, require_live=False), (
        f"tampered evidence ({name}) passed the body checks")
    ev["verified_live"] = record_for(ev)
    assert _offline(ev), f"tampered evidence ({name}) was accepted"


def test_a_drawn_suite_without_a_repo_pin_is_refused():
    """TIME / FEV draws have no pinned item set in the repo: refused, not
    waved through on recorded identities."""
    cfg = lane_d_cfg()
    assert "time" in cfg.gate_suites
    ev = _verified()
    ev["suite"] = "time"
    probs = _offline(ev)
    assert any("no repo pin" in p for p in probs), probs


def test_a_reference_digest_equal_to_the_candidate_is_refused():
    ev = _body()
    ev["rows"]["candidate"]["digest"] = ev["rows"]["house"]["digest"]
    ev["refs"]["candidate"] = ev["refs"]["house"]
    assert any("candidate digest" in p for p in
               evidence_problems(ev, check_git=False, require_live=False))


# ── the verified_live record (c), offline ────────────────────────────────────


def test_editing_the_evidence_after_verification_is_refused():
    ev = _verified()
    for edit in (lambda e: e.update(host="other-host"),
                 lambda e: e["gate"].update(reason="edited"),
                 lambda e: e["rows"]["candidate"]["suites"]["time"]["rows"][0].update(crps=9.0),
                 lambda e: e["draw"].update(block=e["draw"]["block"] + 1)):
        e2 = copy.deepcopy(ev)
        edit(e2)
        assert any("body_sha256" in p for p in verified_live_problems(e2, check_git=False))


def _rec_block_hash(r, ev):
    r["block_hash"] = "0x" + "a" * 64


def _rec_block(r, ev):
    r["block"] += 1


def _rec_unfinalized(r, ev):
    r["finalized_block"] = ev["draw"]["block"] - 1


def _rec_suite(r, ev):
    r["suite"] = "boom"


def _rec_pointer_house(r, ev):
    r["pointers"]["house"] = ref_for("d" * 64)


def _rec_pointer_king(r, ev):
    r["pointers"]["king_d"] = ref_for("d" * 64)


def _rec_unresolved(r, ev):
    r["resolved"]["candidate"] = None


def _rec_resolved_other(r, ev):
    r["resolved"]["house"] = "sha256:" + "c" * 64


def _rec_refs(r, ev):
    r["refs"]["candidate"] = ref_for("b" * 64)


def _rec_network(r, ev):
    r["network"] = "test"


def _rec_verified(r, ev):
    r["verified"] = False


def _rec_schema(r, ev):
    r["schema"] = "laned_w3b2_verify/0"


def _rec_body(r, ev):
    r["body_sha256"] = "0" * 64


def _rec_future(r, ev):
    r["verified_at"] = "2999-01-01T00:00:00+00:00"


def _rec_naive_time(r, ev):
    r["verified_at"] = "2026-11-22T04:00:00"


def _rec_module(r, ev):
    r["tool"]["module"] = "cascade.laned.storage_probe"


def _rec_missing_key(r, ev):
    del r["pointers"]


RECORD_EDITS = {f.__name__.lstrip("_"): f for f in (
    _rec_block_hash, _rec_block, _rec_unfinalized, _rec_suite, _rec_pointer_house,
    _rec_pointer_king, _rec_unresolved, _rec_resolved_other, _rec_refs, _rec_network,
    _rec_verified, _rec_schema, _rec_body, _rec_future, _rec_naive_time, _rec_module,
    _rec_missing_key)}


@pytest.mark.parametrize("name", sorted(RECORD_EDITS))
def test_an_edited_record_is_refused(name):
    ev = _verified()
    RECORD_EDITS[name](ev["verified_live"], ev)
    assert verified_live_problems(ev, check_git=False), f"edited record ({name}) accepted"


def test_record_is_bound_to_the_body_hash():
    ev = _verified()
    assert ev["verified_live"]["body_sha256"] == body_sha256(ev)
    e2 = copy.deepcopy(ev)
    del e2["verified_live"]
    assert body_sha256(e2) == body_sha256(ev), "the hash excludes the record itself"
