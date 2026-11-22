"""R2x: the S0b evidence verifier (plan §17.1 L4, §5 row 3 exit gate).

``scripts/laned/accept R2x --evidence evidence/s0b_notes.json`` exports the
file as ``$LANED_EVIDENCE``. The ``test_evidence_*`` tests verify it: schema
``laned_s0b_notes/1``, a live mainnet read, and every recorded decode and
simulate result re-derived by the R2 decoder from the recorded ``note_hex``
(a hand-edited file fails). The gate: ``trimmed_pending == []``.

Without ``$LANED_EVIDENCE`` they FAIL, by design: R2x is an owner WP whose
Accept is this verifier; there is nothing to accept before the owner's run.
The ``test_verifier_*`` tests prove the verifier rejects tampering, offline.
Contract: README.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .helpers import (
    PROBE_CFG,
    as_chain_evidence,
    dry_run,
    evidence_problems,
    load_fixture,
    write_json,
)

SECOND_HK = "5GsQ8nVr1kTzPq3MwXyLb7cH2dJf6eRtUa9oNiBvCx4mSe1D"
CRASH_NOTE = ("cascade-ready:1:rolling-era-king:9374400:9378000:prelaunch-95-r2:"
              "9486000:9486900:v2b-69630f92:0:0")


def _evidence() -> dict:
    path = os.environ.get("LANED_EVIDENCE")
    assert path, ("LANED_EVIDENCE is unset: R2x is accepted only with the owner's evidence, "
                  "`scripts/laned/accept R2x --evidence evidence/s0b_notes.json` "
                  "(run `python -P -m cascade.laned.s0b_collect ... --out evidence/s0b_notes.json` "
                  "first; see tests/accept/r2x/README.md)")
    p = Path(path)
    assert p.is_file(), f"LANED_EVIDENCE={path} does not exist"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AssertionError(f"LANED_EVIDENCE={path} is not JSON: {e}") from None


def _good(tmp_path: Path, mutate=None) -> dict:
    fx = load_fixture()
    if mutate:
        mutate(fx)
    snap = write_json(tmp_path / "snapshot.json", fx)
    _, ev = dry_run(tmp_path, fixture=snap, out=tmp_path / "ev.json",
                    expect_rc=1 if mutate else 0)
    return as_chain_evidence(ev)


# ── the owner's evidence ─────────────────────────────────────────────────────


def test_evidence_schema_and_reproduced_by_the_r2_decoder():
    problems = evidence_problems(_evidence())
    assert problems == [], "S0b evidence rejected:\n  " + "\n  ".join(problems)


def test_evidence_no_pending_segment_trimmed():
    ev = _evidence()
    assert ev.get("trimmed_pending") == [], (
        "S0b exit gate: a still-pending segment would be trimmed: "
        f"{ev.get('trimmed_pending')} (type its decided blocks in main first, then re-run)")
    assert ev.get("passed") is True, ev.get("passed")


# ── the verifier itself (offline) ────────────────────────────────────────────


def test_verifier_accepts_untampered_evidence(tmp_path):
    ev = _good(tmp_path)
    assert evidence_problems(ev, check_git=False) == []
    assert ev["passed"] is True and ev["simulate_params"]["cfg_feature"] == PROBE_CFG


def test_verifier_rejects_fixture_source(tmp_path):
    ev = _good(tmp_path)
    ev["source"] = "fixture"
    assert any(p.startswith("source") for p in evidence_problems(ev, check_git=False))


def test_verifier_rejects_wrong_network_or_netuid(tmp_path):
    for key, val in (("network", "test"), ("netuid", 259)):
        ev = _good(tmp_path)
        ev[key] = val
        assert any(p.startswith("network/netuid") for p in evidence_problems(ev, check_git=False))


def test_verifier_rejects_a_hidden_trim(tmp_path):
    ev = _good(tmp_path, lambda d: d["commitments"].__setitem__(SECOND_HK, CRASH_NOTE))
    assert ev["trimmed_pending"] and ev["passed"] is False
    ev["trimmed_pending"] = []
    ev["passed"] = True
    probs = evidence_problems(ev, check_git=False)
    assert any(p.startswith("trimmed_pending") for p in probs), probs
    assert any(p.startswith("passed") for p in probs), probs


def test_verifier_rejects_edited_simulate_or_decode(tmp_path):
    ev = _good(tmp_path, lambda d: d["commitments"].__setitem__(SECOND_HK, CRASH_NOTE))
    row = next(r for r in ev["hotkeys"] if r["hotkey"] == SECOND_HK)
    row["simulate"]["pending_trimmed"] = []
    row["simulate"]["refused"] = False
    ev["trimmed_pending"] = []
    ev["passed"] = True
    assert any("simulate does not match" in p for p in evidence_problems(ev, check_git=False))
    ev = _good(tmp_path)
    ev["hotkeys"][0]["decoded"]["segments"][0]["state"] = "pending"
    assert any("decoded does not match" in p for p in evidence_problems(ev, check_git=False))


def test_verifier_rejects_a_swapped_note(tmp_path):
    ev = _good(tmp_path, lambda d: d["commitments"].__setitem__(SECOND_HK, CRASH_NOTE))
    row = next(r for r in ev["hotkeys"] if r["hotkey"] == SECOND_HK)
    row["note_hex"] = ev["hotkeys"][0]["note_hex"]
    assert evidence_problems(ev, check_git=False)


def test_verifier_rejects_a_note_hidden_as_missing(tmp_path):
    ev = _good(tmp_path, lambda d: d["commitments"].__setitem__(SECOND_HK, CRASH_NOTE))
    row = next(r for r in ev["hotkeys"] if r["hotkey"] == SECOND_HK)
    row.update(note_state="missing", decoded=None, simulate=None, headroom_bytes=None)
    ev["trimmed_pending"] = []
    ev["passed"] = True
    assert evidence_problems(ev, check_git=False)


def test_verifier_rejects_a_dirty_or_unknown_tool_sha(tmp_path):
    ev = _good(tmp_path)
    ev["tool"]["git_dirty"] = True
    assert any(p.startswith("tool.git_dirty") for p in evidence_problems(ev, check_git=False))
    ev = _good(tmp_path)
    ev["tool"]["git_sha"] = "0123456789abcdef0123456789abcdef01234567"
    assert any("not a commit" in p for p in evidence_problems(ev, check_git=True))


def test_verifier_rejects_wrong_simulate_params(tmp_path):
    ev = _good(tmp_path)
    ev["simulate_params"]["worst_case"] = False
    assert any(p.startswith("simulate_params") for p in evidence_problems(ev, check_git=False))
    ev = _good(tmp_path)
    ev["target_cfg"] = "cfg-1a2b3c4d"
    assert any(p.startswith("simulate_params") for p in evidence_problems(ev, check_git=False))


def test_verifier_rejects_no_notes_at_all(tmp_path):
    def wipe(d):
        d["commitments"] = {}

    fx = load_fixture()
    wipe(fx)
    snap = write_json(tmp_path / "snapshot.json", fx)
    _, ev = dry_run(tmp_path, fixture=snap, out=tmp_path / "ev.json", expect_rc=1)
    ev = as_chain_evidence(ev)
    assert ev["passed"] is False
    assert any("nothing was checked" in p for p in evidence_problems(ev, check_git=False))
