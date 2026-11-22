"""WB11: the owner "pin" step's evidence verifier (plan §3.0 WB11 "ci + owner
(pin)", §17.1 L4; the tests/accept/r2x pattern).

``scripts/laned/accept WB11 --evidence evidence/wb11.json`` exports the file
as ``$LANED_EVIDENCE``. The ``test_evidence_*`` tests check it: the artefact
built on the bench pod from the pinned suites (gift 30841734 / boom 69325b54 /
time 83e3d0b3 / fev 6de245af) is in the repo, its sha256 is the one pinned in
``chain.toml [lane_b] contamination_hashset``, and a second build reproduced
it. Without ``$LANED_EVIDENCE`` they SKIP, so the builder can merge (L3)
before the owner's run. The ``test_verifier_*`` tests prove the verifier
rejects tampering, offline from fixture suites. Contract: README.md."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from .helpers import (
    BOX_LOOKUP_CAP_BYTES,
    REPO,
    evidence_problems,
    good_evidence,
    lookup_budget,
    mod,
    write_pins,
)


def _evidence() -> dict:
    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("LANED_EVIDENCE is unset: no owner pin evidence yet. L4 needs "
                    "`scripts/laned/accept WB11 --evidence evidence/wb11.json` green on main "
                    "(build the artefact on the bench pod with `python -P -m "
                    "cascade.eval.contamination build ...`, pin its sha256 in [lane_b] "
                    "contamination_hashset; see tests/accept/wb11/README.md)")
    p = Path(path)
    assert p.is_file(), f"LANED_EVIDENCE={path} does not exist"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AssertionError(f"LANED_EVIDENCE={path} is not JSON: {e}") from None


# ── the owner's evidence ─────────────────────────────────────────────────────


def test_evidence_pin_matches_the_committed_artefact_and_its_rebuild():
    ev = _evidence()
    problems = evidence_problems(ev, REPO)
    assert problems == [], "WB11 pin evidence rejected:\n  " + "\n  ".join(problems)


def test_evidence_artefact_loads_with_both_sets():
    """The pinned artefact loads through the module and carries both sets."""
    ev = _evidence()
    m = mod()
    a = m.load_artefact(REPO / ev["artefact"])
    assert a["exact"] and a["normalised"], "empty exact or normalised set"
    assert m.artefact_sha256(a) == ev["sha256"]


# ── the verifier itself (offline) ────────────────────────────────────────────


def test_verifier_accepts_good_evidence(tmp_path):
    ev = good_evidence(tmp_path)
    assert evidence_problems(ev, tmp_path) == []


def test_verifier_accepts_testnet_pinned_to_the_same_hash_set(tmp_path):
    ev = good_evidence(tmp_path)
    write_pins(tmp_path, ev["sha256"], ev["sha256"])
    assert evidence_problems(ev, tmp_path) == []


def _edit(key, value):
    def f(ev, root):
        ev[key] = value
    return f


def _suite_version(suite, value):
    def f(ev, root):
        ev["suites"][suite]["version"] = value
    return f


def _drop_suite(suite):
    def f(ev, root):
        del ev["suites"][suite]
    return f


def _pins(main, testnet):
    def f(ev, root):
        write_pins(root, ev["sha256"] if main is None else main,
                   ev["sha256"] if testnet is None else testnet)
    return f


def _tamper_artefact(ev, root):
    p = root / ev["artefact"]
    art = json.loads(p.read_bytes())
    h = sorted(art["exact"])[0]
    del art["exact"][h]
    p.write_bytes(json.dumps(art, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True).encode("ascii"))


def _remove_artefact(ev, root):
    (root / ev["artefact"]).unlink()


# Passes the identity-prefix check (case-folded) but differs from the artefact.
GIFT_UPPER = "30841734AC5CFDDBD0C3BAD6D09D2B6B32BECBB0"
OTHER = "sha256:" + "0123456789abcdef" * 4

TAMPER = {
    "wrong_wp": _edit("wp", "WB13"),
    "not_passed": _edit("passed", False),
    "wrong_schema": _edit("schema", "laned_wb11_pin/0"),
    "fixture_source": _edit("source", "fixture"),
    "no_source": _edit("source", ""),
    "bad_sha_format": _edit("sha256", "0123456789abcdef" * 4),
    "rebuild_differs": _edit("rebuild_sha256", OTHER),
    "rebuild_missing": _edit("rebuild_sha256", None),
    "version_mismatch": _edit("version", 2),
    "artefact_outside_dir": _edit("artefact", "hashset-v1.json"),
    "artefact_zst_name": _edit("artefact", "benchmarks/contamination/hashset-v1.json.zst"),
    "missing_suite": _drop_suite("fev"),
    "wrong_identity": _suite_version("boom", "5c2c21f3" + "0" * 32),
    "suites_differ_from_artefact": _suite_version("gift", GIFT_UPPER),
    "mainnet_unpinned": _pins("", None),
    "mainnet_other_pin": _pins(OTHER, None),
    "testnet_other_pin": _pins(None, OTHER),
    "artefact_tampered": _tamper_artefact,
    "artefact_absent": _remove_artefact,
    # RV6 B8: the pin records scale, file size and the measured lookup peak
    "n_exact_missing": _edit("n_exact", None),
    "n_exact_wrong": lambda ev, root: ev.update(n_exact=ev["n_exact"] + 1),
    "n_normalised_wrong": lambda ev, root: ev.update(n_normalised=ev["n_normalised"] - 1),
    "file_bytes_wrong": lambda ev, root: ev.update(file_bytes=ev["file_bytes"] + 1),
    "lookup_peak_missing": _edit("lookup_peak_bytes", None),
    "lookup_peak_zero": _edit("lookup_peak_bytes", 0),
    "lookup_peak_over_budget": lambda ev, root: ev.update(
        lookup_peak_bytes=lookup_budget(max(ev["n_exact"], ev["n_normalised"])) + 1),
    "lookup_peak_over_box_cap": _edit("lookup_peak_bytes", BOX_LOOKUP_CAP_BYTES + 1),
}


@pytest.mark.parametrize("name", sorted(TAMPER))
def test_verifier_rejects_tampered_evidence(name, tmp_path):
    ev = good_evidence(tmp_path)
    ev = copy.deepcopy(ev)
    TAMPER[name](ev, tmp_path)
    problems = evidence_problems(ev, tmp_path)
    assert problems, f"verifier accepted tampered evidence ({name})"


def test_verifier_rejects_an_artefact_built_from_other_suite_versions(tmp_path):
    ev = good_evidence(tmp_path)
    m = mod()
    from .helpers import fixture_suites

    s = fixture_suites()
    s["gift"]["version"] = "4f2e9c01" + "0" * 32
    a = m.build_hashset(s, version=1)
    m.write_artefact(a, tmp_path / ev["artefact"])
    assert evidence_problems(ev, tmp_path), "an artefact of other suite versions was accepted"


def test_verifier_version_name_must_match(tmp_path):
    ev = good_evidence(tmp_path, version=2)
    assert evidence_problems(ev, tmp_path) == []
    assert ev["artefact"].endswith("hashset-v2.json")
    ev["artefact"] = "benchmarks/contamination/hashset-v1.json"
    assert evidence_problems(ev, tmp_path), "artefact name v1 with evidence version 2 accepted"
