"""W3: the testnet half of the Accept (plan §3.0 W3 "ci (fs adapter) +
testnet", §4 T2, T6, T7c, T16, T18d). Contract: README.md "Testnet evidence".

``test_evidence_testnet_run`` verifies ``$LANED_EVIDENCE`` and SKIPS while it
is unset, so the CI half can reach main first. The ``test_checker_*`` tests
prove the checker itself offline.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from .helpers import EVIDENCE_CASES, EVIDENCE_SCHEMA, chosen_upload_path, evidence_problems


def _good(chosen: str = "presigned_multipart") -> dict:
    return {
        "schema": EVIDENCE_SCHEMA, "wp": "W3", "source": "testnet", "network": "test",
        "netuid": 259, "upload_path": chosen, "intake_commit": "1f" * 20,
        "cases": [{"id": cid, "expected": want, "observed": want,
                   "http_status": 200 if want == "ok" else 409} for cid, want in EVIDENCE_CASES],
        "t7c": {"weights_bytes": 2_500_000_000, "box_rss_delta_mb": 12.5,
                "box_disk_delta_mb": 0},
        "passed": True,
    }


def _raw(ev: dict) -> bytes:
    return json.dumps(ev, indent=2).encode()


def test_evidence_testnet_run():
    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("LANED_EVIDENCE is unset: the W3 testnet run (netuid 259) has not been "
                    "recorded; see tests/accept/w3/README.md 'Testnet evidence'")
    p = Path(path)
    assert p.is_file(), f"LANED_EVIDENCE={path} does not exist"
    raw = p.read_bytes()
    try:
        ev = json.loads(raw)
    except ValueError as e:
        raise AssertionError(f"LANED_EVIDENCE={path} is not JSON: {e}") from None
    probs = evidence_problems(ev, raw, chosen_upload_path())
    assert probs == [], "W3 testnet evidence:\n  " + "\n  ".join(probs)


def test_checker_accepts_a_complete_run():
    ev = _good()
    assert evidence_problems(ev, _raw(ev), "presigned_multipart") == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda e: e.update(source="fixture"), "source"),
    (lambda e: e.update(netuid=91), "netuid"),
    (lambda e: e.update(upload_path="upload_relay"), "upload_path"),
    (lambda e: e.update(intake_commit="abc"), "intake_commit"),
    (lambda e: e["cases"].pop(), "case T7c"),
    (lambda e: e["cases"][0].update(observed="ok"), "case T2"),
    (lambda e: e["cases"][4].update(http_status=200), "case T18d"),
    (lambda e: e["t7c"].update(weights_bytes=10_000), "weights_bytes"),
    (lambda e: e["t7c"].update(box_rss_delta_mb=900), "box_rss_delta_mb"),
    (lambda e: e.update(passed=False), "passed"),
    (lambda e: e.update(note="https://x/?X-Amz-Signature=abc"), "credential"),
])
def test_checker_rejects_a_bad_run(mutate, needle):
    ev = copy.deepcopy(_good())
    mutate(ev)
    probs = evidence_problems(ev, _raw(ev), "presigned_multipart")
    assert any(needle in p for p in probs), probs


def test_checker_rejects_when_no_path_is_chosen():
    ev = _good()
    probs = evidence_problems(ev, _raw(ev), None)
    assert any("STORAGE-LANE-D" in p for p in probs), probs
