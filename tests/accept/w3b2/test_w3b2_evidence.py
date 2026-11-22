"""W3b2: the pod run's evidence verifier (plan §3.0 W3b2 "ci (recorded suite
outputs) + pod", §17.1 L4).

The pod part is one live paired sweep of a candidate, KING_D and the house on
one bench host in one session, on a suite drawn by W3b1. Its evidence is
``evidence/w3b2.json`` (schema ``laned_w3b2/1``), verified by
``scripts/laned/accept W3b2 --evidence evidence/w3b2.json``, which exports it as
``$LANED_EVIDENCE``. The ``test_evidence_*`` tests SKIP when it is unset, so
the module can reach main (L2/L3) before the pod run; when it IS set, a
missing, malformed, fixture or hand-edited file fails hard. The
``test_verifier_*`` tests prove offline that the verifier rejects tampering.
Contract: README.md."""

from __future__ import annotations

import copy
import json
import os
import re
import tomllib
from pathlib import Path

import pytest

from .helpers import GATE_SUITE, SENTINEL, SESSION, gate_kwargs, triple

REPO = Path(__file__).resolve().parents[3]
SCHEMA = "laned_w3b2/1"
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")
HASH = re.compile(r"0x[0-9a-f]{64}")
MIN_ITEMS = 10
ROLES = ("candidate", "king_d", "house")


def _lane_d_cfg():
    from cascade.shared.config import parse_chain_config

    raw = tomllib.loads((REPO / "chain.toml").read_text(encoding="utf-8"))
    return parse_chain_config(raw).lane_d


def _draw(block_hash: str, sentinel):
    """W3b1's drawn gate suite for block_hash under the shipped chain.toml."""
    from cascade.validator.lane_d_gate import gate_suite_for

    cfg = _lane_d_cfg()
    return gate_suite_for(block_hash, gate_suites=cfg.gate_suites,
                          gate_weights=cfg.gate_weights, sentinel=sentinel)


def evidence_problems(ev) -> list[str]:
    """Every reason ``ev`` is not valid W3b2 pod evidence (empty = valid)."""
    from cascade.validator.lane_d_bench import paired_gate

    if not isinstance(ev, dict):
        return ["evidence is not a JSON object"]
    p: list[str] = []
    if ev.get("schema") != SCHEMA:
        p.append(f"schema != {SCHEMA}")
    if ev.get("wp") != "W3b2":
        p.append("wp != W3b2")
    if ev.get("source") != "pod":
        p.append("source != pod (fixture evidence is never accepted)")
    if ev.get("passed") is not True:
        p.append("passed is not true")
    if not isinstance(ev.get("host"), str) or not ev.get("host"):
        p.append("host missing")
    tool = ev.get("tool") or {}
    if not isinstance(tool.get("git_sha"), str) or not HEX40.fullmatch(tool["git_sha"]):
        p.append("tool.git_sha is not 40 hex")
    if tool.get("git_dirty") is not False:
        p.append("tool.git_dirty is not false")
    session = ev.get("session")
    if not isinstance(session, str) or not session:
        p.append("session missing")
    rows = ev.get("rows") or {}
    if sorted(rows) != sorted(ROLES):
        return p + [f"rows must be exactly {list(ROLES)}"]
    suite, sentinel = ev.get("suite"), ev.get("sentinel")
    for role in ROLES:
        rec = rows[role]
        if not isinstance(rec, dict):
            p.append(f"rows.{role} is not an object")
            continue
        if rec.get("session") != session:
            p.append(f"rows.{role}.session != session (one host session only)")
        if not isinstance(rec.get("digest"), str) or not HEX64.fullmatch(rec["digest"]):
            p.append(f"rows.{role}.digest is not 64 hex")
        entry = (rec.get("suites") or {}).get(suite) or {}
        if entry.get("status") != "ok" or len(entry.get("rows") or []) < MIN_ITEMS:
            p.append(f"rows.{role}: drawn suite {suite!r} not ok with >= {MIN_ITEMS} items")
        if sentinel is not None and sentinel not in (rec.get("suites") or {}):
            p.append(f"rows.{role}: sentinel {sentinel!r} not swept")
    digs = [rows[r].get("digest") for r in ROLES if isinstance(rows[r], dict)]
    if len(digs) == 3 and digs[0] in digs[1:]:
        p.append("candidate digest equals a reference digest")
    draw = ev.get("draw") or {}
    if not isinstance(draw.get("block_hash"), str) or not HASH.fullmatch(draw["block_hash"]):
        p.append("draw.block_hash is not 0x + 64 hex")
    else:
        want = _draw(draw["block_hash"], sentinel)
        if suite != want:
            p.append(f"suite {suite!r} is not the W3b1 draw {want!r} for draw.block_hash")
    if isinstance(draw.get("block"), bool) or not isinstance(draw.get("block"), int):
        p.append("draw.block is not an int")
    params = ev.get("params") or {}
    try:
        gate = paired_gate(rows["candidate"], rows["king_d"], rows["house"], suite=suite,
                           alpha=params.get("alpha"), tau_public=params.get("tau_public"),
                           B=params.get("B"), seed=params.get("seed"), sentinel=sentinel)
    except (ValueError, TypeError, KeyError) as e:
        return p + [f"paired_gate refused the recorded rows: {e}"]
    if json.loads(json.dumps(gate)) != ev.get("gate"):
        p.append("gate != paired_gate recomputed from the recorded rows (hand-edited)")
    if gate.get("outcome") not in ("pass", "gate_regression"):
        p.append(f"gate outcome {gate.get('outcome')!r}: the pod run must compute a verdict")
    if sentinel is not None and not (gate.get("sentinel") or {}).get("lcb"):
        p.append("the sentinel suite was not swept to numbers")
    return p


def _evidence() -> dict:
    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("LANED_EVIDENCE is unset: no pod evidence to verify yet. L4 needs "
                    "`scripts/laned/accept W3b2 --evidence evidence/w3b2.json` green on main "
                    "(a live paired sweep on one bench host; see tests/accept/w3b2/README.md)")
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


# ── the verifier, offline ────────────────────────────────────────────────────


def _good() -> dict:
    from cascade.validator.lane_d_bench import paired_gate

    i = 0
    while _draw("0x" + f"{i:064x}", SENTINEL) != GATE_SUITE:
        i += 1
    h = "0x" + f"{i:064x}"
    cand, king, house = triple(king=(1.03, 1.03), house=(1.01, 1.01))
    kw = gate_kwargs()
    gate = paired_gate(cand, king, house, **kw)
    return {"schema": SCHEMA, "wp": "W3b2", "source": "pod", "host": "valpriv-4090",
            "tool": {"git_sha": "1" * 39 + "2", "git_dirty": False}, "session": SESSION,
            "draw": {"block": 9612000, "block_hash": h}, "suite": GATE_SUITE,
            "sentinel": SENTINEL,
            "params": {k: kw[k] for k in ("alpha", "tau_public", "B", "seed")},
            "rows": {"candidate": cand, "king_d": king, "house": house},
            "gate": json.loads(json.dumps(gate)), "passed": True}


def test_verifier_accepts_consistent_evidence():
    assert evidence_problems(_good()) == []


def _tamper_gate(ev):
    ev["gate"]["lcb"]["house"]["crps"] += 0.01


def _tamper_row(ev):
    ev["rows"]["house"]["suites"][GATE_SUITE]["rows"][5]["crps"] *= 0.5


def _tamper_session(ev):
    ev["rows"]["king_d"]["session"] = "valpriv-4090/older"


def _tamper_source(ev):
    ev["source"] = "fixture"


def _tamper_suite(ev):
    ev["suite"] = "time"
    for r in ROLES:
        ev["rows"][r]["suites"]["time"] = copy.deepcopy(ev["rows"][r]["suites"][GATE_SUITE])


def _tamper_fault(ev):
    ev["rows"]["candidate"]["suites"][GATE_SUITE]["rows"][2]["mase"] = None
    ev["gate"] = None


def _tamper_wp(ev):
    ev["wp"] = "W3b1"


TAMPER = {"gate": _tamper_gate, "row": _tamper_row, "session": _tamper_session,
          "source": _tamper_source, "suite": _tamper_suite, "fault": _tamper_fault,
          "wp": _tamper_wp}


@pytest.mark.parametrize("name", sorted(TAMPER))
def test_verifier_rejects_tampering(name):
    ev = _good()
    TAMPER[name](ev)
    assert evidence_problems(ev), f"tampered evidence ({name}) was accepted"
