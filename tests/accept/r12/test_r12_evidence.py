"""R12: the live-run evidence ``evidence/r12_pod.json`` (plan §3.0 R12 Accept
"``--evidence evidence/r12_pod.json`` for a live run", §17.1 L4).

``scripts/laned/accept R12 --evidence evidence/r12_pod.json`` exports the
file as ``$LANED_EVIDENCE``; the ``test_evidence_*`` tests verify it with
:func:`helpers.evidence_problems`, which re-derives the on-rent verdict, the
host-key fingerprints, the suite-identity handshake and the code handshake
from the RECORDED captures, using the profile as committed at
``tool.git_sha`` — a hand-edited file fails. Without ``$LANED_EVIDENCE``
they SKIP with a clear reason, so the kit can reach main (L2/L3) before the
owner's pod run. The ``test_verifier_*`` tests prove the verifier, offline.
Contract: README.md.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from .helpers import (
    ADA,
    ADA_MIB,
    GATE_PROFILE,
    LAUNCH_PROFILE,
    NEW_HOST,
    PINNED_IDENTITIES,
    PORT,
    REPO,
    RTX4090,
    RTX4090_MIB,
    SECRET_ENV,
    Kit,
    as_clean_run,
    as_json,
    evidence_problems,
    gate_captures_dir,
    key_blob,
    output,
    require,
)

GOOD_IDS = {"suites": {k: v + "0a1b2c3d" * 7 for k, v in PINNED_IDENTITIES.items()},
            "bench_files": 17747}


def _evidence() -> dict:
    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("LANED_EVIDENCE is unset: no owner pod run to verify yet. L4 needs "
                    "`scripts/laned/accept R12 --evidence evidence/r12_pod.json` green on main "
                    "(the last step of `scripts/evalhost/setup_eval_host.sh` writes it; see "
                    "tests/accept/r12/README.md)")
    p = Path(path)
    assert p.is_file(), f"LANED_EVIDENCE={path} does not exist"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AssertionError(f"LANED_EVIDENCE={path} is not JSON: {e}") from None


# ── the owner's evidence ─────────────────────────────────────────────────────


def test_evidence_reproduced_from_its_captures(tmp_path):
    problems = evidence_problems(_evidence(), work=tmp_path)
    assert problems == [], "R12 pod evidence rejected:\n  " + "\n  ".join(problems)


def test_evidence_pod_passed():
    ev = _evidence()
    assert ev.get("wp") == "R12" and ev.get("passed") is True, (
        f"the eval host did not pass bring-up: on_rent={ev.get('on_rent')} "
        f"handshake={ev.get('handshake')}")


# ── the evidence subcommand + verifier (offline) ─────────────────────────────


@pytest.fixture
def kit(tmp_path):
    return Kit(tmp_path)


def _make(kit, *, profile=GATE_PROFILE, release=None, expect_rc=0, dry_run=True, **caps):
    require(profile)
    ids = caps.pop("identities", GOOD_IDS)
    d = gate_captures_dir(kit, identities=ids, **caps)
    out = kit.tmp / "evidence" / "r12_pod.json"
    args = ["evidence", "--profile", profile, "--pod", "valeval-blackwell4", "--host", NEW_HOST,
            "--port", str(PORT), "--captures", d, "--out", out]
    if release is not None:
        args += ["--release-dir", release]
    if dry_run:
        args.append("--dry-run")
    proc = kit.tool(*args)
    assert proc.returncode == expect_rc, output(proc)
    assert kit.shim_calls() == [], kit.shim_calls()
    ev = as_json(proc)
    if dry_run:
        assert not out.exists(), "evidence --dry-run must not write the file"
    return ev, out


def _good(kit) -> dict:
    ev, _ = _make(kit)
    return as_clean_run(ev)


def _launch(kit) -> dict:
    from cascade.eval.lane_d_handshake import worker_hello

    ev, _ = _make(kit, profile=LAUNCH_PROFILE, release=REPO, identities=None,
                  hello=worker_hello(root=REPO), model=RTX4090, vram=RTX4090_MIB,
                  driver="595.99.02", avail=500_000_000_000)
    return as_clean_run(ev)


def test_verifier_accepts_a_good_gate_run(kit):
    ev = _good(kit)
    assert ev["profile"] == "scripts/evalhost/profiles/bench_gate.toml", ev["profile"]
    assert ev["passed"] is True and ev["source"] == "pod" and ev["wp"] == "R12", ev
    assert evidence_problems(ev, work=kit.tmp) == []


def test_verifier_accepts_a_good_launch_host_run(kit):
    ev = _launch(kit)
    assert ev["profile"] == "scripts/evalhost/profiles/launch_eval.toml", ev["profile"]
    assert ev["passed"] is True, ev
    assert evidence_problems(ev, work=kit.tmp) == []


def test_evidence_without_dry_run_writes_the_file(kit):
    ev, out = _make(kit, dry_run=False)
    assert out.is_file()
    assert json.loads(out.read_text()) == ev


def test_a_failing_pod_records_passed_false_and_is_rejected(kit):
    ev, _ = _make(kit, expect_rc=1, model=ADA, vram=ADA_MIB)
    assert ev["passed"] is False and ev["on_rent"]["ok"] is False, ev
    assert evidence_problems(as_clean_run(ev), work=kit.tmp) != []


def _tamper(fn):
    def apply(ev):
        ev = copy.deepcopy(ev)
        fn(ev)
        return ev
    return apply


TAMPERS = {
    "schema": _tamper(lambda e: e.update(schema="laned_r12_pod/0")),
    "wp": _tamper(lambda e: e.update(wp="R13")),
    "source": _tamper(lambda e: e.update(source="fixture")),
    "checked_at_naive": _tamper(lambda e: e.update(checked_at="2026-11-22T09:00:00")),
    "unknown_sha": _tamper(lambda e: e["tool"].update(git_sha="0123456789abcdef" * 2 + "01234567")),
    "dirty": _tamper(lambda e: e["tool"].update(git_dirty=True)),
    "profile_outside": _tamper(lambda e: e.update(profile="tests/accept/r12/PATHS")),
    "profile_sha": _tamper(lambda e: e.update(profile_sha256="0" * 64)),
    "observed_driver": _tamper(lambda e: e["on_rent"]["observed"].update(driver="600.1.1")),
    "captured_driver": _tamper(lambda e: e["captures"].update(
        nvidia_smi_query=e["captures"]["nvidia_smi_query"].replace("595.99.02", "570.195.03"))),
    "captured_disk": _tamper(lambda e: e["captures"].update(
        df=e["captures"]["df"].replace("14800000000", "5700000000"))),
    "identity": _tamper(lambda e: e["captures"]["identities"]["suites"].update(boom="69325b55")),
    "fingerprint": _tamper(lambda e: e["host_key"].update(fingerprints=["SHA256:" + "A" * 43])),
    "keyscan_other_host": _tamper(lambda e: e["captures"].update(
        ssh_keyscan=e["captures"]["ssh_keyscan"].replace(NEW_HOST, "192.0.2.99"))),
    "secret": _tamper(lambda e: e["captures"].update(
        df=e["captures"]["df"] + "# " + SECRET_ENV["HF_TOKEN"] + "\n")),
    "passed_flag": _tamper(lambda e: e.update(passed=False)),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_verifier_rejects_tampering(kit, name):
    ev = TAMPERS[name](_good(kit))
    problems = evidence_problems(ev, work=kit.tmp)
    assert problems, f"tampered evidence ({name}) was accepted"


def test_verifier_rejects_a_failing_run_relabelled_as_passing(kit):
    ev, _ = _make(kit, expect_rc=1, model=ADA, vram=ADA_MIB)
    ev = as_clean_run(ev)
    ev["on_rent"]["ok"] = True
    ev["on_rent"]["reasons"] = []
    ev["passed"] = True
    assert evidence_problems(ev, work=kit.tmp) != []


def test_verifier_rejects_a_stale_worker_hello(kit):
    ev = _launch(kit)
    ev["captures"]["hello"]["package_hash"] = "e" * 64
    assert evidence_problems(ev, work=kit.tmp) != []


def test_verifier_rejects_a_different_host_key_capture(kit):
    ev = _good(kit)
    other = key_blob("someone-else")
    ev["captures"]["ssh_keyscan"] = f"[{NEW_HOST}]:{PORT} ssh-ed25519 {other}\n"
    assert evidence_problems(ev, work=kit.tmp) != []
