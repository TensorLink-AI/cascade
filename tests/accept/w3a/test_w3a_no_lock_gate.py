"""W3a: the L4 gate on ``--no-object-lock`` evidence, and a clean ``tool`` record.

Owner rule 2026-11-23 (decision #717, plan rev6.4: Hippius has no Object
Lock; chosen path ``presigned_multipart``). The owner's no-lock run records
the two lock probes ``skipped`` (owner flag) and, on Hippius, some probes
that do not decide the upload path ended in ``error`` (``single_put_limit``,
``prefix_bucket_policy``, ``subtoken_scope``), so the body's ``passed`` is
false. The gate stays meaningful without demanding the impossible:

* lock-enabled evidence: unchanged (every probe pass/fail, ``passed: true``);
* no-lock evidence: both lock probes ``skipped``; every probe the decision's
  Chosen path depends on (``helpers.PATH_PROBES``: the path's own probe plus
  ``upload_part_copy_2_5gb`` for the king copy at crowning) is ``pass``; any
  other probe may be pass/fail/error, but its result is recorded exactly once,
  verbatim, in the decision's results table; and ``storage_probe verify``
  writes ``verified_live.path_verdict`` (``{path, required, passed}``,
  recomputed from the body), which must be ``passed: true``.

``storage_probe.tool_info()`` must not call a checkout dirty because of
untracked files under ``evidence/`` (where the probe writes, and where
``verify`` rewrites, the evidence), but still flags any tracked change and
any other untracked file. Contract: README.md "The L4 gate on
--no-object-lock evidence".
"""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest

from .helpers import (
    KING_COPY_PROBE,
    LOCK_PROBES,
    NO_LOCK_FLAG,
    PATH_PROBES,
    anchor_pair,
    dry_run,
    evidence_problems,
    fixture_evidence,
    gate_problems,
    live_pair,
    model_file,
    path_verdict,
    probe,
    probe_module,
    relabel_as_live,
    verified_live_problems,
)

# The owner's Hippius run, in fixture form: no lock, a pre-existing bucket
# policy (the policy probe refuses: error) and no sub-token (error), so the
# body is `passed: false` while the path probes pass.
OWNER_LIKE = {"bucket_lock_enabled": False, "bucket_has_policy": True, "subtoken": None}


def _nolock_live(tmp_path: Path, change: dict | None = None, name: str = "g.json") -> dict:
    model = model_file(tmp_path, lambda m: m.update({**OWNER_LIKE, **(change or {})}),
                       name=f"model-{name}")
    _, ev, _ = dry_run(tmp_path, NO_LOCK_FLAG, fixture=model, out=tmp_path / name, expect_rc=1)
    assert ev is not None, "a --no-object-lock dry-run with --fixture writes its evidence"
    return relabel_as_live(ev)


def _with_verdict(ev: dict) -> dict:
    """A stand-in record carrying only the verdict (the gate reads only it;
    the full record is checked by verified_live_problems)."""
    ev = copy.deepcopy(ev)
    ev["verified_live"] = {"path_verdict": path_verdict(ev)}
    return ev


def _decision(ev: dict, chosen: str = "presigned_multipart", table: dict | None = None,
              extra_rows: str = "") -> str:
    res = {p["name"]: p["result"] for p in ev["probes"]}
    res.update(table or {})
    rows = "\n".join(f"| `{n}` | `{r}` | a note |" for n, r in res.items() if r is not None)
    return (f"# Storage decision\n\n**Chosen path:** `{chosen}`\n\n"
            f"Evidence sha256: {'0' * 64}\n\n"
            "| Path id | What | Needs |\n|---|---|---|\n"
            "| `presigned_multipart` | multipart | `presigned_upload_part: pass` |\n\n"
            "| Probe | Result | Note |\n|---|---|---|\n" + rows + "\n" + extra_rows)


def _full(ev: dict) -> list[str]:
    return evidence_problems(ev, check_git=False, live_sizes=False)


# ── the verdict and the gate (offline, helpers only) ─────────────────────────


def test_owner_like_fixture_has_errors_and_a_false_body_passed(tmp_path):
    ev = _nolock_live(tmp_path)
    res = {p["name"]: p["result"] for p in ev["probes"]}
    assert res["prefix_bucket_policy"] == "error" and res["subtoken_scope"] == "error", res
    assert all(res[n] == "skipped" for n in LOCK_PROBES), res
    assert res["presigned_upload_part"] == "pass" and res[KING_COPY_PROBE] == "pass", res
    assert ev["passed"] is False
    assert evidence_problems(ev, check_git=False, live_sizes=False, require_verified=False) == []


def test_path_probes_table_is_pinned():
    assert PATH_PROBES == {
        "presigned_multipart": ("presigned_upload_part", "upload_part_copy_2_5gb"),
        "single_presigned_put": ("single_put_limit", "upload_part_copy_2_5gb"),
        "upload_relay": ("upload_part_copy_2_5gb",),
    }


def test_path_verdict_of_a_no_lock_run(tmp_path):
    ev = _nolock_live(tmp_path)
    assert path_verdict(ev) == {"path": "presigned_multipart",
                                "required": ["presigned_upload_part", "upload_part_copy_2_5gb"],
                                "passed": True}
    ev2 = _nolock_live(tmp_path, {"upload_part_copy": "rejected"}, name="g2.json")
    assert path_verdict(ev2)["passed"] is False
    ev3 = _nolock_live(tmp_path, {"presigned_upload_part": "rejected",
                                  "single_put_max_bytes": 600000}, name="g3.json")
    assert path_verdict(ev3) == {"path": "upload_relay", "required": ["upload_part_copy_2_5gb"],
                                 "passed": True}
    ev4 = copy.deepcopy(ev)
    ev4["cleanup"]["remaining_unlocked"] = ["x"]
    assert path_verdict(ev4)["passed"] is False
    ev5 = copy.deepcopy(ev)
    ev5["anchor_objects"] = []
    assert path_verdict(ev5)["passed"] is False


def test_path_verdict_never_passes_lock_enabled_evidence(tmp_path):
    ev = relabel_as_live(fixture_evidence(tmp_path))
    assert ev["passed"] is True
    assert path_verdict(ev)["passed"] is False


def test_gate_accepts_no_lock_evidence_whose_path_probes_pass_despite_errors(tmp_path):
    ev = _with_verdict(_nolock_live(tmp_path))
    assert gate_problems(ev, _decision(ev)) == [], gate_problems(ev, _decision(ev))


def test_gate_accepts_the_real_decision_tables_shape(tmp_path):
    """Rows as decisions/STORAGE-LANE-D.md writes them: backticked probe and
    result, a free-text note, and a path-id table that is not a result row."""
    ev = _with_verdict(_nolock_live(tmp_path))
    text = _decision(ev).replace("| a note |", "| the capability behaves as lane D needs. "
                                 "Two parts, 200 + ETag each |")
    assert gate_problems(ev, text) == []


@pytest.mark.parametrize("change,chosen,culprit", [
    ({"presigned_upload_part": "rejected"}, "upload_relay", None),
    ({"upload_part_copy": "rejected"}, "presigned_multipart", KING_COPY_PROBE),
    ({"upload_part_copy": "rejected"}, "upload_relay", KING_COPY_PROBE),
    ({"single_put_max_bytes": 600000}, "single_presigned_put", "single_put_limit"),
], ids=["relay-ok", "copy-fail", "relay-copy-fail", "single-put-fail"])
def test_gate_requires_the_chosen_paths_probes_to_pass(tmp_path, change, chosen, culprit):
    ev = _with_verdict(_nolock_live(tmp_path, change))
    probs = gate_problems(ev, _decision(ev, chosen))
    if culprit is None:
        assert probs == [], probs
    else:
        assert any(culprit in p and "must be 'pass'" in p for p in probs), probs


def test_gate_rejects_a_presigned_choice_without_presigned_pass(tmp_path):
    ev = _with_verdict(_nolock_live(tmp_path, {"presigned_upload_part": "rejected"}))
    probs = gate_problems(ev, _decision(ev, "presigned_multipart"))
    assert any("presigned_upload_part" in p and "must be 'pass'" in p for p in probs), probs


def test_gate_rejects_an_unfilled_chosen_path(tmp_path):
    ev = _with_verdict(_nolock_live(tmp_path))
    probs = gate_problems(ev, _decision(ev, "(owner fills)"))
    assert any("Chosen path" in p for p in probs), probs


@pytest.mark.parametrize("table", [
    {"subtoken_scope": "pass"},
    {"prefix_bucket_policy": "fail"},
    {"object_lock_multipart": "pass"},
    {"single_put_limit": None},
], ids=["error-as-pass", "error-as-fail", "skip-as-pass", "row-missing"])
def test_gate_cross_checks_every_result_against_the_decision_table(tmp_path, table):
    ev = _with_verdict(_nolock_live(tmp_path))
    name = next(iter(table))
    probs = gate_problems(ev, _decision(ev, table=table))
    assert any("results table" in p and name in p for p in probs), probs


def test_gate_rejects_a_duplicated_result_row(tmp_path):
    ev = _with_verdict(_nolock_live(tmp_path))
    text = _decision(ev, extra_rows="| `subtoken_scope` | `pass` | again |\n")
    probs = gate_problems(ev, text)
    assert any("results table" in p and "subtoken_scope" in p for p in probs), probs


def test_gate_rejects_a_lock_probe_that_is_not_skipped(tmp_path):
    ev = _nolock_live(tmp_path)
    row = probe(ev, "object_lock_multipart")
    row["result"], row["passed"] = "pass", True
    ev = _with_verdict(ev)
    probs = gate_problems(ev, _decision(ev))
    assert any("object_lock_multipart" in p and "skipped" in p for p in probs), probs


def test_gate_requires_the_verify_records_path_verdict(tmp_path):
    ev = _nolock_live(tmp_path)
    text = _decision(ev)
    probs = gate_problems(ev, text)
    assert any(p.startswith("verified_live.path_verdict") for p in probs), probs
    e2 = _with_verdict(ev)
    e2["verified_live"]["path_verdict"]["passed"] = False
    assert any(p.startswith("verified_live.path_verdict") for p in gate_problems(e2, text))
    e3 = _with_verdict(ev)
    e3["verified_live"]["path_verdict"]["required"] = ["presigned_upload_part"]
    assert any(p.startswith("verified_live.path_verdict") for p in gate_problems(e3, text))
    e4 = _with_verdict(_nolock_live(tmp_path, {"upload_part_copy": "rejected"}, name="g4.json"))
    probs = gate_problems(e4, _decision(e4))
    assert any(p.startswith("verified_live.path_verdict.passed") for p in probs), probs


def test_gate_on_lock_enabled_evidence_is_unchanged(tmp_path):
    ev = relabel_as_live(fixture_evidence(tmp_path))
    assert gate_problems(ev, "no decision needed") == []
    ev = relabel_as_live(fixture_evidence(tmp_path, lambda m: m.update(subtoken=None),
                                          expect_rc=1, name="e.json"))
    assert ev["passed"] is False
    probs = gate_problems(ev, _decision(ev))
    assert any("subtoken_scope" in p for p in probs), probs
    assert any(p.startswith("passed") for p in probs), probs
    ev["passed"] = True  # body-inconsistent, but the gate alone still refuses the error
    assert any("subtoken_scope" in p for p in gate_problems(ev, _decision(ev)))


# ── storage_probe verify writes the verdict ──────────────────────────────────


def _verify(sp, ev: dict) -> dict:
    ev, fake = anchor_pair(ev)
    rec = sp.verify_live(ev, fake)
    assert fake.forbidden == [], f"verify must be read-only, it called {fake.forbidden}"
    assert isinstance(rec, dict) and "path_verdict" in rec, (
        f"an unlocked_anchor verify record carries path_verdict, got keys {sorted(rec)}")
    ev["verified_live"] = rec
    return ev


def test_verify_records_the_path_verdict_of_a_no_lock_run(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev = _verify(sp, _nolock_live(tmp_path))
    assert ev["verified_live"]["path_verdict"] == path_verdict(ev) == {
        "path": "presigned_multipart",
        "required": ["presigned_upload_part", "upload_part_copy_2_5gb"], "passed": True}
    assert _full(ev) == [], _full(ev)
    assert gate_problems(ev, _decision(ev)) == [], gate_problems(ev, _decision(ev))


def test_verify_records_a_failing_path_verdict_without_refusing(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev = _verify(sp, _nolock_live(tmp_path, {"upload_part_copy": "rejected"}))
    assert ev["verified_live"]["path_verdict"]["passed"] is False
    assert ev["verified_live"]["verified"] is True
    assert _full(ev) == [], _full(ev)
    assert gate_problems(ev, _decision(ev)) != []


def test_verify_recomputes_the_path_rather_than_trusting_the_body(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev = _nolock_live(tmp_path)
    ev["recommended_path"] = "upload_relay"  # an edited body: the verdict still recomputes
    ev = _verify(sp, ev)
    assert ev["verified_live"]["path_verdict"]["path"] == "presigned_multipart"


def test_an_edited_path_verdict_is_refused_by_the_offline_verifier(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev = _verify(sp, _nolock_live(tmp_path, {"upload_part_copy": "rejected"}))
    for edit in (lambda pv: pv.update(passed=True),
                 lambda pv: pv.update(path="upload_relay"),
                 lambda pv: pv.update(required=["presigned_upload_part"]),
                 lambda pv: pv.update(extra=1)):
        e2 = copy.deepcopy(ev)
        edit(e2["verified_live"]["path_verdict"])
        probs = verified_live_problems(e2, check_git=False)
        assert any(p.startswith("verified_live.path_verdict") for p in probs), probs
    e2 = copy.deepcopy(ev)
    del e2["verified_live"]["path_verdict"]
    assert any(p.startswith("verified_live.path_verdict")
               for p in verified_live_problems(e2, check_git=False))


def test_a_lock_enabled_verify_record_is_unchanged(tmp_path, monkeypatch):
    sp = probe_module(monkeypatch)
    ev, fake = live_pair(relabel_as_live(fixture_evidence(tmp_path)))
    rec = sp.verify_live(ev, fake)
    assert "path_verdict" not in rec and "binding" not in rec, sorted(rec)


# ── tool_info: untracked evidence output is not a dirty checkout ─────────────


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=w3a", "-c", "user.email=w3a@test",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    (r / "evidence").mkdir(parents=True)
    (r / "cascade").mkdir()
    (r / "cascade" / "mod.py").write_text("x = 1\n")
    (r / "evidence" / "tracked.json").write_text("{}\n")
    _git(r, "init", "-q")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "init")
    sp = probe_module()
    monkeypatch.setattr(sp, "REPO", r)
    return sp, r


def test_tool_info_clean_checkout(repo):
    sp, _ = repo
    info = sp.tool_info()
    assert info["git_dirty"] is False and len(info["git_sha"]) == 40, info


@pytest.mark.parametrize("rel", ["evidence/w3a_probe.json", "evidence/sub/x.json",
                                 "evidence/w3a_probe.json.verify-tmp"])
def test_tool_info_ignores_untracked_files_under_evidence(repo, rel):
    sp, r = repo
    p = r / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}\n")
    info = sp.tool_info()
    assert info["git_dirty"] is False, (
        f"an untracked {rel} (the probe's --out / verify's file) must not make the checkout "
        f"dirty: {info}")


@pytest.mark.parametrize("change", ["tracked-code", "tracked-evidence", "staged-evidence",
                                    "untracked-code", "deleted-tracked"])
def test_tool_info_still_flags_real_changes(repo, change):
    sp, r = repo
    if change == "tracked-code":
        (r / "cascade" / "mod.py").write_text("x = 2\n")
    elif change == "tracked-evidence":
        (r / "evidence" / "tracked.json").write_text('{"edited": true}\n')
    elif change == "staged-evidence":
        (r / "evidence" / "new.json").write_text("{}\n")
        _git(r, "add", "evidence/new.json")
    elif change == "untracked-code":
        (r / "cascade" / "new.py").write_text("y = 1\n")
    else:
        (r / "cascade" / "mod.py").unlink()
    info = sp.tool_info()
    assert info["git_dirty"] is True, (change, info)


def test_tool_info_with_evidence_and_a_tracked_change_is_dirty(repo):
    sp, r = repo
    (r / "evidence" / "w3a_probe.json").write_text("{}\n")
    (r / "cascade" / "mod.py").write_text("x = 3\n")
    assert sp.tool_info()["git_dirty"] is True
