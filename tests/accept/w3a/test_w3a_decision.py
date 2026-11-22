"""W3a: the storage decision record ``decisions/STORAGE-LANE-D.md``.

Offline (always): the record exists with exactly one ``Chosen path:`` line and
one ``Evidence sha256:`` line, and names every probe and every path id.
With ``$LANED_EVIDENCE`` set: ``Chosen path:`` names exactly one path the
evidence supports and ``Evidence sha256:`` is the evidence file's sha256. That
test skips without evidence, so the template merges green and the owner fills
it after the live run. Contract: README.md.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from .helpers import (
    DECISION,
    PATH_IDS,
    PROBES,
    decision_problems,
    decision_template_problems,
)


def _decision_text() -> str:
    assert DECISION.is_file(), f"{DECISION} is missing (W3a PATHS: the decision template)"
    return DECISION.read_text(encoding="utf-8")


def _ev(results: dict) -> dict:
    return {"probes": [{"name": n, "result": results.get(n, "pass")} for n in PROBES]}


SHA = hashlib.sha256(b"w3a evidence bytes").hexdigest()


def _record(chosen: str, sha: str = SHA) -> str:
    return ("# Storage decision: lane D\n\n"
            f"Chosen path: {chosen}\n\nEvidence sha256: {sha}\n\n"
            "Evidence: evidence/w3a_probe.json\n\n"
            + "\n".join(f"| {n} | pass |" for n in PROBES) + "\n\n"
            "Options: " + ", ".join(PATH_IDS) + "\n")


# ── the record on this branch ────────────────────────────────────────────────


def test_decision_record_template_shape():
    problems = decision_template_problems(_decision_text())
    assert problems == [], "decision record:\n  " + "\n  ".join(problems)


def test_decision_names_exactly_one_path_with_evidence():
    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("LANED_EVIDENCE is unset: the owner fills 'Chosen path:' and 'Evidence "
                    "sha256:' in decisions/STORAGE-LANE-D.md after the live run; "
                    "`scripts/laned/accept W3a --evidence evidence/w3a_probe.json` checks it")
    import json

    raw = Path(path).read_bytes()
    problems = decision_problems(_decision_text(), json.loads(raw), raw)
    assert problems == [], "decision record vs evidence:\n  " + "\n  ".join(problems)


# ── the decision checker itself (offline) ────────────────────────────────────


def test_decision_checker_accepts_a_filled_supported_record():
    for chosen in ("presigned_multipart", "`presigned_multipart`", "**upload_relay**"):
        assert decision_problems(_record(chosen), _ev({}), b"w3a evidence bytes") == [], chosen


def test_decision_checker_rejects_an_unfilled_or_ambiguous_path():
    unfilled = "(owner fills: one of " + " | ".join(PATH_IDS) + ")"
    for chosen in (unfilled, "presigned_multipart or upload_relay", "", "presigned"):
        probs = decision_problems(_record(chosen), _ev({}), b"w3a evidence bytes")
        assert any("exactly one" in p for p in probs), (chosen, probs)


def test_decision_checker_rejects_a_path_the_evidence_does_not_support():
    ev = _ev({"presigned_upload_part": "fail", "single_put_limit": "fail"})
    for chosen in ("presigned_multipart", "single_presigned_put"):
        probs = decision_problems(_record(chosen), ev, b"w3a evidence bytes")
        assert any("not supported" in p for p in probs), (chosen, probs)
    assert decision_problems(_record("upload_relay"), ev, b"w3a evidence bytes") == []
    ev = _ev({"presigned_upload_part": "fail"})
    assert decision_problems(_record("single_presigned_put"), ev, b"w3a evidence bytes") == []


def test_decision_checker_rejects_a_wrong_evidence_sha():
    probs = decision_problems(_record("upload_relay"), _ev({}), b"other evidence bytes")
    assert any("Evidence sha256" in p for p in probs), probs


def test_decision_checker_rejects_two_chosen_lines():
    text = _record("upload_relay") + "\nChosen path: presigned_multipart\n"
    assert any("exactly one 'Chosen path:'" in p
               for p in decision_template_problems(text))
