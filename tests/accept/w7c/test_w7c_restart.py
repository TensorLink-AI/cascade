"""W7c: restarts are idempotent (plan rev6 §3.2 invariant 5; T11 "Validator
restart mid-judgement: no re-score, no double crown").

A crash is simulated at the rename that would land a file: the signed report
file (`report_path`) or the state file. Recovery is a fresh `settle_lane_d`
call with the same inputs on the same state path. Every comparison is
against a clean run of the same chain on the same pool bytes. README.md has
the contract.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest

from .helpers import (
    REPO,
    STATE_ERROR_CODES,
    Chain,
    CountingForecasts,
    W7cChain,
    body_sha,
    boundary,
    chain_fingerprint,
    crash_on_rename,
    other_crown,
    read_bytes,
    receipt_of,
    sha,
)


def _resettle(chain, n, path, **kw):
    """A re-call that must succeed: a refusal is a contract failure (an
    AssertionError), not an error of the test."""
    try:
        return chain.settle(n, path, **kw)
    except ValueError as e:
        raise AssertionError(f"re-settling boundary({n}) was refused: {e}") from e


def _state_error(excinfo, code: str) -> None:
    from cascade.validator.lane_d_state import LaneDStateError

    assert isinstance(excinfo.value, LaneDStateError), repr(excinfo.value)
    assert excinfo.value.code == code, (excinfo.value.code, str(excinfo.value))
    assert excinfo.value.code in STATE_ERROR_CODES


def _reference(chain: Chain) -> dict:
    """The clean run: receipts, report bodies and state digests at 1..3."""
    ref = chain_fingerprint(chain, chain.state_path("ref"), 3)
    assert all(isinstance(r, dict) for r in ref["receipts"]), "settle_lane_d returned no receipt"
    return ref


@pytest.mark.parametrize("n", [1, 2, 3], ids=["floor_wait", "crowning", "idle"])
def test_inv5_t11_crash_after_the_report_lands_recovers_to_the_clean_run(tmp_path, monkeypatch,
                                                                       n):
    from cascade.validator.lane_d_state import load_report_file, report_path

    chain = W7cChain(tmp_path)
    ref = _reference(chain)
    path = chain.state_path("crash")
    for k in range(1, n):
        chain.settle(k, path)
    before = read_bytes(path)
    crash_on_rename(monkeypatch, path)
    with pytest.raises(OSError):
        chain.settle(n, path)
    monkeypatch.undo()
    assert read_bytes(path) == before, "the crash must leave the previous state file"
    rp = report_path(path, boundary(n))
    written = read_bytes(rp)
    assert written is not None, "the signed report lands before the state write"
    crashed_report = load_report_file(rp)
    assert body_sha(crashed_report) == ref["bodies"][n - 1]

    hosts = len(chain.hosts)
    s = _resettle(chain, n, path)
    assert all(h.calls == [] for h in chain.hosts[hosts:]), (
        "recovery adopts the persisted report: the bench is never swept again (RV7 B1)")
    assert read_bytes(rp) == written, "recovery keeps the report already written"
    assert s.report == crashed_report, "recovery returns the persisted report"
    assert body_sha(s.report) == ref["bodies"][n - 1]
    assert receipt_of(s) == ref["receipts"][n - 1]
    assert sha(path.read_bytes()) == ref["states"][n - 1]
    for k in range(n + 1, 4):
        s = chain.settle(k, path)
        assert receipt_of(s) == ref["receipts"][k - 1]
    assert sha(path.read_bytes()) == ref["states"][2]


@pytest.mark.parametrize("n", [1, 2], ids=["floor_wait", "crowning"])
def test_inv5_t11_crash_before_the_report_lands_recovers_to_the_clean_run(tmp_path, monkeypatch,
                                                                        n):
    from cascade.validator.lane_d_state import report_path

    chain = W7cChain(tmp_path)
    ref = _reference(chain)
    path = chain.state_path("crash")
    for k in range(1, n):
        chain.settle(k, path)
    before = read_bytes(path)
    rp = report_path(path, boundary(n))
    crash_on_rename(monkeypatch, rp)
    with pytest.raises(OSError):
        chain.settle(n, path)
    monkeypatch.undo()
    assert read_bytes(path) == before, "no state may land without its report"
    assert read_bytes(rp) is None
    s = _resettle(chain, n, path)
    assert body_sha(s.report) == ref["bodies"][n - 1]
    assert receipt_of(s) == ref["receipts"][n - 1]
    assert sha(path.read_bytes()) == ref["states"][n - 1]


def test_inv5_t11_restart_after_a_settled_boundary_does_not_rescore(tmp_path):
    from cascade.validator.lane_d_state import report_path

    chain = W7cChain(tmp_path)
    path = chain.state_path("run")
    s1 = chain.settle(1, path)
    s2 = chain.settle(2, path)
    assert s2.report["crowned"] == chain.x["digest"]
    state_bytes = path.read_bytes()
    rp = report_path(path, boundary(2))
    report_bytes = read_bytes(rp)
    assert report_bytes is not None
    hosts = len(chain.hosts)
    fc = CountingForecasts(chain.forecasts)
    again = _resettle(chain, 2, path, forecasts=fc)
    assert fc.reads == 0, "a settled boundary is never re-scored"
    assert all(h.calls == [] for h in chain.hosts[hosts:]), "a settled boundary is never re-gated"
    assert again.report == s2.report
    assert again.pay == s2.pay
    assert receipt_of(again) == receipt_of(s2) is not None
    assert path.read_bytes() == state_bytes
    assert read_bytes(rp) == report_bytes
    assert receipt_of(s1)["posterior_state_sha256"] == receipt_of(again)["prior_state_sha256"]


def test_inv5_t11_no_double_crown_across_repeated_restarts(tmp_path, monkeypatch):
    chain = W7cChain(tmp_path)
    ref = _reference(chain)
    path = chain.state_path("run")
    chain.settle(1, path)
    crash_on_rename(monkeypatch, path)
    with pytest.raises(OSError):
        chain.settle(2, path)
    monkeypatch.undo()
    first = _resettle(chain, 2, path)
    for _ in range(3):
        again = _resettle(chain, 2, path)
        assert again.report == first.report and receipt_of(again) == receipt_of(first)
    state = json.loads(path.read_bytes())
    x = chain.x["digest"]
    assert state["king"]["digest"] == x
    assert [k["digest"] for k in state["prior_kings"]].count(x) == 0
    assert state["entries"][x]["status"] == "crowned"
    s3 = chain.settle(3, path)
    assert receipt_of(s3) == ref["receipts"][2]
    assert receipt_of(s3)["prior_state_sha256"] == receipt_of(first)["posterior_state_sha256"]
    assert s3.report["king_before"]["digest"] == x and s3.report["crowned"] is None
    assert s3.pay["court_uids"][0] == chain.uid_of[chain.x["hotkey"]]
    assert sha(path.read_bytes()) == ref["states"][2]


def test_inv5_a_call_at_an_older_boundary_is_refused(tmp_path):
    from cascade.validator.lane_d_state import report_path

    chain = W7cChain(tmp_path)
    path = chain.state_path("run")
    chain.settle(1, path)
    chain.settle(2, path)
    state_bytes = path.read_bytes()
    r1 = read_bytes(report_path(path, boundary(1)))
    with pytest.raises(ValueError) as e:
        chain.settle(1, path)
    _state_error(e, "stale_boundary")
    assert path.read_bytes() == state_bytes
    assert read_bytes(report_path(path, boundary(1))) == r1


@pytest.mark.parametrize("phase", ["pending", "settled"])
@pytest.mark.parametrize("defect,code", [("body", "report_conflict"),
                                         ("duplicate", "duplicate_key")])
def test_inv5_a_defective_persisted_report_is_refused(tmp_path, monkeypatch, phase, defect, code):
    """RV7 B1: recovery (`pending`) adopts the persisted report only once its
    signature verifies, so a body edited after signing is `report_unverified`
    there. A re-call (`settled`) may refuse it as either code."""
    from cascade.validator.lane_d_state import LaneDStateError

    if defect == "body":
        codes = {"report_unverified"} if phase == "pending" else {"report_conflict",
                                                                  "report_unverified"}
    else:
        codes = {code}
    from cascade.validator.lane_d_state import report_path

    chain = W7cChain(tmp_path)
    path = chain.state_path("run")
    chain.settle(1, path)
    if phase == "pending":
        crash_on_rename(monkeypatch, path)
        with pytest.raises(OSError):
            chain.settle(2, path)
        monkeypatch.undo()
    else:
        chain.settle(2, path)
    rp = report_path(path, boundary(2))
    report = json.loads(rp.read_bytes())
    if defect == "body":
        report["crowned"] = other_crown(report, chain)   # a different judgement, same boundary
        bad = json.dumps(report, sort_keys=True).encode()
    else:
        bad = json.dumps(report, sort_keys=True).encode().replace(
            b'"boundary": ', b'"boundary": 0, "boundary": ', 1)
    rp.write_bytes(bad)
    state_bytes = path.read_bytes()
    with pytest.raises(ValueError) as e:
        chain.settle(2, path)
    assert isinstance(e.value, LaneDStateError), repr(e.value)
    assert e.value.code in codes, (e.value.code, str(e.value))
    assert e.value.code in STATE_ERROR_CODES
    assert path.read_bytes() == state_bytes
    assert rp.read_bytes() == bad, "a defective report file is never silently overwritten"


def test_inv5_a_resigned_persisted_report_is_accepted_and_kept(tmp_path, monkeypatch):
    from cascade.validator.lane_d_report import sign_report
    from cascade.validator.lane_d_state import load_report_file, report_path

    chain = W7cChain(tmp_path)
    ref = _reference(chain)
    path = chain.state_path("run")
    chain.settle(1, path)
    crash_on_rename(monkeypatch, path)
    with pytest.raises(OSError):
        chain.settle(2, path)
    monkeypatch.undo()
    rp = report_path(path, boundary(2))
    resigned = sign_report(load_report_file(rp), chain.signer)
    rp.write_text(json.dumps(resigned, indent=2))
    kept = rp.read_bytes()
    s = _resettle(chain, 2, path)
    assert s.report == resigned and rp.read_bytes() == kept
    assert receipt_of(s) == ref["receipts"][1]
    assert sha(path.read_bytes()) == ref["states"][1]


# ── determinism ─────────────────────────────────────────────────────────────


def test_inv5_two_clean_runs_are_byte_identical(tmp_path):
    from cascade.validator.lane_d_state import report_path

    chain = W7cChain(tmp_path)
    a, b = chain.state_path("a"), chain.state_path("b")
    for n in (1, 2, 3):
        sa, sb = chain.settle(n, a), chain.settle(n, b)
        assert a.read_bytes() == b.read_bytes(), f"state bytes differ after boundary({n})"
        assert receipt_of(sa) == receipt_of(sb) is not None
        assert body_sha(sa.report) == body_sha(sb.report)
        ra = json.loads(report_path(a, boundary(n)).read_bytes())
        assert body_sha(ra) == body_sha(sa.report)


HASHSEED_SCRIPT = r"""
import json, sys
from pathlib import Path
from tests.accept.w7c.helpers import W7cChain as Chain, chain_fingerprint
chain = Chain(Path(sys.argv[1]), pools_dir=Path(sys.argv[2]))
print(json.dumps(chain_fingerprint(chain, chain.state_path("run"), 3), sort_keys=True))
"""


def test_inv5_receipts_are_independent_of_the_hash_seed(tmp_path):
    pools = W7cChain(tmp_path / "shared").pools_dir
    outs = []
    for seed in ("0", "4242"):
        env = {"PYTHONPATH": str(REPO), "PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin",
               "HOME": os.environ.get("HOME", "/tmp")}
        proc = subprocess.run([sys.executable, "-P", "-c", HASHSEED_SCRIPT,
                               str(tmp_path / f"seed{seed}"), str(pools)],
                              capture_output=True, text=True, timeout=600, check=False,
                              env=env, cwd=str(REPO))
        assert proc.returncode == 0, proc.stderr[-4000:]
        outs.append(json.loads(proc.stdout))
    assert all(isinstance(r, dict) for r in outs[0]["receipts"]), outs[0]["receipts"]
    assert outs[0] == copy.deepcopy(outs[1])
