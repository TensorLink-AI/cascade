"""W18: the owner's live-run evidence (plan §17.1 L4; W18 runs in "ci (fs
adapter) + owner").

``scripts/laned/accept W18 --evidence evidence/w18.json`` exports the file as
``$LANED_EVIDENCE``. The ``test_evidence_*`` tests verify it with
:func:`helpers.evidence_problems`, which re-derives every claim from the
embedded action log. Without ``$LANED_EVIDENCE`` they SKIP, so the sweeper
can reach main (L2/L3) before the owner's first live run. The
``test_verifier_*`` tests prove the verifier rejects bad evidence, offline.
Contract: README.md.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from .helpers import (
    ARCHIVE_BUCKET,
    VAULT_BUCKET,
    evidence_problems,
    ledger,
    make_evidence,
    run_sweep,
    s3_stores,
    scenario_entries,
    seeded_fake,
    vs,
)


def _evidence() -> dict:
    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("LANED_EVIDENCE is unset: no owner evidence to verify yet. L4 needs "
                    "`scripts/laned/accept W18 --evidence evidence/w18.json` green on main "
                    "(run `python -P -m cascade.laned.vault_sweeper --live ... --evidence-out "
                    "evidence/w18.json` against the live vault first; see "
                    "tests/accept/w18/README.md)")
    p = Path(path)
    assert p.is_file(), f"LANED_EVIDENCE={path} does not exist"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AssertionError(f"LANED_EVIDENCE={path} is not JSON: {e}") from None


def test_evidence_is_a_clean_live_run_on_the_vault():
    problems = evidence_problems(_evidence())
    assert problems == [], "W18 evidence rejected:\n  " + "\n  ".join(problems)


# ── the verifier itself (offline) ────────────────────────────────────────────


def _live_log(tmp_path: Path, *, live: bool = True, fail: bool = False) -> dict:
    entries = scenario_entries()
    fake = seeded_fake(entries)
    if fail:
        rel = entries["lost_due"].key("model.safetensors")
        fake.fail_put.add((ARCHIVE_BUCKET, f"lane-d-archive/{rel}"))
    vault, archive = s3_stores(fake)
    return run_sweep(vault, archive, ledger(entries), tmp_path, live=live,
                     plan="decided" if live else None)


def test_verifier_accepts_a_clean_live_s3_run(tmp_path):
    ev = make_evidence(_live_log(tmp_path))
    assert evidence_problems(ev) == []


def test_verifier_rejects_a_dry_run(tmp_path):
    assert any("live run" in p for p in evidence_problems(make_evidence(
        _live_log(tmp_path, live=False))))


def test_verifier_rejects_a_run_with_a_failed_action(tmp_path):
    ev = make_evidence(_live_log(tmp_path, fail=True))
    assert ev["passed"] is False
    assert evidence_problems(ev)


def test_verifier_rejects_tampering(tmp_path):
    base = make_evidence(_live_log(tmp_path))
    vault_keys = [a for a in base["log"]["actions"] if a["store"] == "vault"]

    def drop_archive(ev):
        k = next(a["key"] for a in ev["log"]["actions"]
                 if a["op"] == "delete" and a["outcome"] == "done" and a["store"] == "vault")
        ev["log"]["actions"] = [a for a in ev["log"]["actions"]
                                if not (a["key"] == k and a["op"] == "archive")]
        ev["log"]["counts"]["archived"] -= 1

    def counts(ev):
        ev["log"]["counts"]["deleted"] += 1

    def wp(ev):
        ev["wp"] = "W17"

    def vault(ev):
        ev["log"]["vault"]["bucket"] = "some-other-bucket"

    def overlap(ev):
        ev["log"]["archive"] = {"kind": "s3", "bucket": VAULT_BUCKET, "prefix": "lane-d/old"}

    def d16(ev):
        ev["log"]["d16"] = "open"

    def reason(ev):
        k = next(a for a in ev["log"]["actions"] if a["op"] == "keep")
        k["reason"] = "because"

    assert vault_keys
    for mutate in (drop_archive, counts, wp, vault, overlap, d16, reason):
        ev = copy.deepcopy(base)
        mutate(ev)
        assert evidence_problems(ev), f"{mutate.__name__}: tampered evidence accepted"


def test_verifier_needs_the_sweeper_module():
    assert hasattr(vs(), "sweep")
