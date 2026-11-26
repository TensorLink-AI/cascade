"""R10 / T25: the owner's live evidence goes through the REAL verifier.

``scripts/laned/accept R10 --evidence evidence/t25.json`` exports the file as
``$LANED_EVIDENCE``. ``test_r10_live_t25_evidence`` then runs
``python -P -m cascade.laned.restore verify-evidence --evidence $LANED_EVIDENCE``
(plus ``--bundle DIR`` when the evidence records a bundle dir that exists) and
requires exit 0. Without this, ``accept`` checks only ``"wp"`` and
``"passed"``, so a hand-written ``{"wp": "R10", "passed": true}`` would reach
L4 (RV2 BLOCKING R10).

Without ``$LANED_EVIDENCE`` (an ordinary PR run) the live test SKIPS. The
``test_live_check_*`` self-tests need no env: they point ``LANED_EVIDENCE`` at
fixture files and run the same test body, proving that a bare or hand-edited
file makes it fail. Contract: README.md.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from .helpers import REPO, build_source, output, run_cli, sha256_file, snapshot
from .test_r10_evidence import evidence

L4_COMMAND = "scripts/laned/accept R10 --evidence evidence/t25.json"

# Optional evidence keys that may name the snapshot bundle dir. verify-evidence
# ignores unknown keys; when one names an existing dir, --bundle is passed so
# bundle_sha256 / source are cross-checked against the real manifest.
BUNDLE_KEYS = ("bundle_dir", "bundle_path", "bundle")


def _bundle_arg(ev: object, ev_path: Path) -> list[object]:
    if not isinstance(ev, dict):
        return []
    for key in BUNDLE_KEYS:
        val = ev.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        p = Path(val)
        for cand in ([p] if p.is_absolute() else [ev_path.parent / p, REPO / p]):
            if (cand / "manifest.json").is_file():
                return ["--bundle", cand.resolve()]
    return []


def _live_check() -> None:
    """The body of the live test: skip when unset, fail hard otherwise."""
    raw = os.environ.get("LANED_EVIDENCE")
    if not raw:
        pytest.skip("LANED_EVIDENCE is unset (an ordinary PR run): the T25 evidence is "
                    f"verified only at L4, `{L4_COMMAND}`")
    path = Path(raw)
    assert path.is_file(), f"LANED_EVIDENCE={raw} does not exist"
    try:
        ev = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        ev = None  # verify-evidence itself reports the parse failure
    proc = run_cli("verify-evidence", "--evidence", path.resolve(), *_bundle_arg(ev, path))
    assert proc.returncode == 0, (
        f"verify-evidence refused LANED_EVIDENCE={raw} (exit {proc.returncode}):\n"
        f"{output(proc)}")
    assert "OK T25 evidence" in proc.stdout, output(proc)


# ── the owner's evidence ─────────────────────────────────────────────────────


def test_r10_live_t25_evidence():
    _live_check()


# ── self-tests of the live check (no env needed) ─────────────────────────────


def _write(tmp_path: Path, body: object) -> Path:
    path = tmp_path / "evidence" / "t25.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body if isinstance(body, str) else json.dumps(body, indent=2))
    return path


def _fails(monkeypatch, path: Path) -> str:
    monkeypatch.setenv("LANED_EVIDENCE", str(path))
    with pytest.raises(AssertionError) as exc:
        _live_check()
    return str(exc.value)


def _passes(monkeypatch, path: Path) -> None:
    monkeypatch.setenv("LANED_EVIDENCE", str(path))
    _live_check()


def test_live_check_skips_when_unset(monkeypatch):
    monkeypatch.delenv("LANED_EVIDENCE", raising=False)
    with pytest.raises(pytest.skip.Exception) as exc:
        _live_check()
    assert L4_COMMAND in str(exc.value)


def test_live_check_accepts_valid_fixture_evidence(tmp_path, monkeypatch):
    _passes(monkeypatch, _write(tmp_path, evidence()))


def test_live_check_fails_on_bare_wp_passed(tmp_path, monkeypatch):
    text = _fails(monkeypatch, _write(tmp_path, {"wp": "R10", "passed": True}))
    assert "missing key" in text, text


def test_live_check_fails_on_missing_file(tmp_path, monkeypatch):
    _fails(monkeypatch, tmp_path / "evidence" / "nope.json")


def test_live_check_fails_on_unparseable_file(tmp_path, monkeypatch):
    _fails(monkeypatch, _write(tmp_path, "{not json"))


def test_live_check_fails_on_hand_edited_verdict(tmp_path, monkeypatch):
    _passes(monkeypatch, _write(tmp_path / "control", evidence()))
    body = copy.deepcopy(evidence())
    body["first_verdict"]["outcome"] = "dethrone"  # verdict_identical stays true
    text = _fails(monkeypatch, _write(tmp_path, body))
    assert "verdict_identical" in text, text


def test_live_check_fails_on_hand_edited_netuid(tmp_path, monkeypatch):
    body = copy.deepcopy(evidence())
    body["source_netuid"] = 259
    _fails(monkeypatch, _write(tmp_path, body))


def test_live_check_cross_checks_a_recorded_bundle(tmp_path, monkeypatch):
    bundle, _ = snapshot(build_source(tmp_path), tmp_path / "bundles")
    real = sha256_file(bundle / "manifest.json")
    good = evidence(real)
    good["bundle_dir"] = str(bundle)
    _passes(monkeypatch, _write(tmp_path / "match", good))
    # hand-edited copy: a well-formed but different sha, so only the
    # --bundle cross-check (passed because the evidence names the bundle) catches it
    edited = copy.deepcopy(good)
    edited["bundle_sha256"] = hashlib.sha256(b"some other manifest").hexdigest()
    text = _fails(monkeypatch, _write(tmp_path / "edited", edited))
    assert "bundle_sha256" in text, text
    # a hand-edited source_block that the bundle manifest contradicts
    edited = copy.deepcopy(good)
    edited["source_block"] = good["source_block"] + 1
    text = _fails(monkeypatch, _write(tmp_path / "block", edited))
    assert "source_block" in text, text
