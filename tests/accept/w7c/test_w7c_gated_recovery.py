"""W7c × W5e (RV7 B1): crash recovery never wedges a GATED settlement.

boundary(2) is X's first verdict, a clear (rev6.11 (b): there is no
confirmation duel): a crowning candidate, so the §1.3b
gate sweeps the bench in the settlement's session, and the session id (and
every bench number) is part of the signed body. The crash lands between the
report write and the state write. A restarted validator has a NEW bench
session, and its bench may answer differently (the pod is down, or it
returns numerically different rows). Recovery therefore ADOPTS the persisted
report once it verifies (pinned signer, boundary, the prior state digest
bound in its signed body), derives the posterior state from it and never
sweeps the bench again. A persisted report that fails verification is a
typed refusal and is never adopted or overwritten. README.md "Recovery" has
the contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from .helpers import (
    REPO,
    STATE_ERROR_CODES,
    Chain,
    FlakyHost,
    body_sha,
    boundary,
    chain_fingerprint,
    crash_on_rename,
    fixture_key,
    other_crown,
    read_bytes,
    receipt_of,
    replace_string,
    sha,
    string_paths,
)

NEW_SESSION = "w7c-sess-b2-restarted"


def _refused(excinfo, code: str) -> None:
    from cascade.validator.lane_d_state import LaneDStateError

    assert isinstance(excinfo.value, LaneDStateError), repr(excinfo.value)
    assert excinfo.value.code == code, (excinfo.value.code, str(excinfo.value))
    assert excinfo.value.code in STATE_ERROR_CODES


def _gate_outcome(report: dict, dg: str) -> str | None:
    for r in (report.get("gate") or {}).get("results") or []:
        if r["digest"] == dg:
            return r["outcome"]
    return None


def _crash_at_b2(chain: Chain, path, monkeypatch, *, host=None) -> tuple[bytes, dict]:
    """Settle boundary(1), then crash boundary(2) after its report landed and
    before its state did. Returns the persisted report's bytes and parse."""
    from cascade.validator.lane_d_state import load_report_file, report_path

    chain.settle(1, path)
    before = read_bytes(path)
    crash_on_rename(monkeypatch, path)
    with pytest.raises(OSError):
        chain.settle(2, path, host=host)
    monkeypatch.undo()
    assert read_bytes(path) == before, "the crash must leave the boundary(1) state file"
    rp = report_path(path, boundary(2))
    written = read_bytes(rp)
    assert written is not None, "the signed report lands before the state write"
    return written, load_report_file(rp)


def _recover(chain: Chain, path, host, session=NEW_SESSION):
    try:
        return chain.settle(2, path, host=host, session=session)
    except ValueError as e:
        raise AssertionError(f"recovery at boundary(2) was refused: {e}") from e


RECOVERY_HOSTS = {
    # a restarted validator: same pod, a new session id
    "new_session": lambda chain: chain.passing_host(),
    # the pod is down now: a re-sweep would be bench_hold
    "pod_down": lambda chain: FlakyHost({}),
    # numerically different rows (GPU non-determinism): a re-sweep changes the lcb
    "different_rows": lambda chain: chain.passing_host(0.8),
    # rows that would now regress: a re-sweep would be gate_regression
    "regressed_rows": lambda chain: chain.passing_host(1.3),
}


@pytest.mark.parametrize("mode", sorted(RECOVERY_HOSTS))
def test_rv7_b1_recovery_adopts_the_persisted_crowning_report(tmp_path, monkeypatch, mode):
    from cascade.validator.lane_d_state import report_path

    chain = Chain(tmp_path)
    ref = chain_fingerprint(chain, chain.state_path("ref"), 3)
    path = chain.state_path("crash")
    written, persisted = _crash_at_b2(chain, path, monkeypatch)
    x = chain.x["digest"]
    assert persisted["crowned"] == x and _gate_outcome(persisted, x) == "pass"
    assert body_sha(persisted) == ref["bodies"][1]

    host = RECOVERY_HOSTS[mode](chain)
    s = _recover(chain, path, host)
    assert host.calls == [], "recovery adopts the persisted gate outcome: no bench sweep"
    assert s.report == persisted, "recovery returns the persisted report"
    assert read_bytes(report_path(path, boundary(2))) == written, "the report file is kept"
    assert receipt_of(s) == ref["receipts"][1], "the receipt the clean run gave"
    assert sha(path.read_bytes()) == ref["states"][1], "the clean run's state bytes"
    state = json.loads(path.read_bytes())
    assert state["king"]["digest"] == x and state["entries"][x]["status"] == "crowned"
    assert [k["digest"] for k in state["prior_kings"]].count(x) == 0

    s3 = chain.settle(3, path)    # the next boundary settles normally
    assert receipt_of(s3) == ref["receipts"][2]
    assert receipt_of(s3)["prior_state_sha256"] == receipt_of(s)["posterior_state_sha256"]
    assert s3.report["king_before"]["digest"] == x and s3.report["crowned"] is None
    assert sha(path.read_bytes()) == ref["states"][2]


def test_rv7_b1_recovery_adopts_a_persisted_bench_hold(tmp_path, monkeypatch):
    """The reverse: the crashed settlement held X (pod down). A restarted
    validator whose pod now passes must not crown X at boundary(2): the
    persisted hold is the record. X is re-gated at boundary(3) as in the
    clean run."""
    chain = Chain(tmp_path)
    ref_path = chain.state_path("ref-hold")
    ref = []
    for n in (1, 2, 3):
        s = chain.settle(n, ref_path, host=FlakyHost({}) if n == 2 else None)
        ref.append((receipt_of(s), body_sha(s.report), sha(ref_path.read_bytes()), s.report))
    x = chain.x["digest"]
    assert ref[1][3]["crowned"] is None and _gate_outcome(ref[1][3], x) == "bench_hold"
    assert ref[2][3]["crowned"] == x, "fixture: the held X is crowned at boundary(3)"

    path = chain.state_path("crash")
    _, persisted = _crash_at_b2(chain, path, monkeypatch, host=FlakyHost({}))
    assert body_sha(persisted) == ref[1][1]
    host = chain.passing_host()
    s = _recover(chain, path, host)
    assert host.calls == [], "recovery adopts the persisted bench_hold: no bench sweep"
    assert s.report == persisted and s.report["crowned"] is None
    assert receipt_of(s) == ref[1][0]
    assert sha(path.read_bytes()) == ref[1][2]
    state = json.loads(path.read_bytes())
    # rev6.11 (b): no `confirm_pending`. A held first-clear candidate stays
    # open (`pending`) with its clear kept in `gate_hold` (W5e / W5g).
    rec = state["entries"][x]
    assert state["king"]["digest"] != x and rec["status"] == "pending", rec["status"]
    assert rec.get("gate_hold"), "the held clear is kept for the re-gate"

    s3 = chain.settle(3, path)
    assert receipt_of(s3) == ref[2][0]
    assert s3.report["crowned"] == x
    assert sha(path.read_bytes()) == ref[2][2]


RECOVERY_SCRIPT = r"""
import json, sys
from pathlib import Path
from tests.accept.w7c.helpers import Chain, receipt_of
chain = Chain(Path(sys.argv[1]))
host = chain.passing_host(0.8)
s = chain.settle(2, chain.state_path("crash"), host=host, session=sys.argv[2])
print(json.dumps({"receipt": receipt_of(s), "calls": len(host.calls),
                  "crowned": s.report["crowned"]}, sort_keys=True))
"""


def test_rv7_b1_recovery_in_a_new_process(tmp_path, monkeypatch):
    """The restart is a new process with a new session and a bench that
    answers differently."""
    chain = Chain(tmp_path)
    ref = chain_fingerprint(chain, chain.state_path("ref"), 3)
    path = chain.state_path("crash")
    _crash_at_b2(chain, path, monkeypatch)
    env = {"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin",
           "HOME": os.environ.get("HOME", "/tmp")}
    proc = subprocess.run([sys.executable, "-P", "-c", RECOVERY_SCRIPT, str(tmp_path),
                           NEW_SESSION], capture_output=True, text=True, timeout=600,
                          check=False, env=env, cwd=str(REPO))
    assert proc.returncode == 0, proc.stderr[-4000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["calls"] == 0, "the restarted process never sweeps the bench for boundary(2)"
    assert out["crowned"] == chain.x["digest"]
    assert out["receipt"] == ref["receipts"][1]
    assert sha(path.read_bytes()) == ref["states"][1]
    assert receipt_of(chain.settle(3, path)) == ref["receipts"][2]


def test_rv7_b1_every_report_binds_its_prior_state_digest_under_the_signature(tmp_path):
    """Recovery checks the persisted report's prior digest against the state
    as loaded, so the signed body carries it (at a path the implementer
    chooses; W7a freezes the top-level keys, so `extensions` is the open
    place)."""
    chain = Chain(tmp_path)
    path = chain.state_path("run")
    for n in (1, 2, 3):
        s = chain.settle(n, path)
        prior = receipt_of(s)["prior_state_sha256"]
        body = {k: v for k, v in s.report.items() if k != "signature"}
        assert string_paths(body, prior), (
            f"boundary({n}): the signed report body does not carry prior_state_sha256")


# ── a persisted report that does not verify is refused, never adopted ──────


def _doctor(chain: Chain, report: dict, defect: str, prior: str) -> dict:
    from cascade.validator.lane_d_report import sign_report

    if defect == "foreign_signer":
        return sign_report(report, fixture_key("intruder"))
    if defect == "bad_signature":
        sig = report["signature"]
        return {**report, "signature": ("0" if sig[0] != "0" else "1") + sig[1:]}
    if defect == "edited_after_signing":
        return {**report, "crowned": other_crown(report, chain)}
    if defect == "wrong_boundary":
        return sign_report({**report, "boundary": boundary(3)}, chain.signer)
    if defect == "wrong_prior":
        paths = string_paths({k: v for k, v in report.items() if k != "signature"}, prior)
        assert paths, "the signed report body does not carry prior_state_sha256"
        other = hashlib.sha256(b"w7c some other prior state").hexdigest()
        return sign_report(replace_string(report, prior, other), chain.signer)
    raise AssertionError(defect)


REFUSALS = {"foreign_signer": "report_unverified", "bad_signature": "report_unverified",
            "edited_after_signing": "report_unverified", "wrong_boundary": "report_conflict",
            "wrong_prior": "report_conflict"}


@pytest.mark.parametrize("defect", sorted(REFUSALS))
def test_rv7_b1_an_unverified_persisted_report_is_refused(tmp_path, monkeypatch, defect):
    from cascade.validator.lane_d_state import report_path

    chain = Chain(tmp_path)
    path = chain.state_path("crash")
    _, persisted = _crash_at_b2(chain, path, monkeypatch)
    prior = hashlib.sha256(path.read_bytes()).hexdigest()
    rp = report_path(path, boundary(2))
    bad = json.dumps(_doctor(chain, persisted, defect, prior), sort_keys=True).encode()
    rp.write_bytes(bad)
    state_bytes = path.read_bytes()
    host = chain.passing_host()
    with pytest.raises(ValueError) as e:
        chain.settle(2, path, host=host, session=NEW_SESSION)
    _refused(e, REFUSALS[defect])
    assert host.calls == [], "a refused recovery never sweeps the bench"
    assert path.read_bytes() == state_bytes, "a refused recovery writes no state"
    assert rp.read_bytes() == bad, "a refused report is never silently overwritten"
