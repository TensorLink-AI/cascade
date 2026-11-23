"""WB19: the pod evidence (plan §12b.6 LB6; Accept `scripts/laned/accept WB19
--evidence evidence/wb19.json`; owner input OA8).

``accept`` exports the evidence file as ``$LANED_EVIDENCE``. The
``test_evidence_*`` tests verify it with ``evidence_problems`` below: every run
re-checked against the SHIPPED chain.toml (contract digest re-derived at the
recorded block, image pin), the pinned torch, the GPU-vs-SKU rule, the rate and
wall arithmetic, the table re-derived from the runs, coverage of
``[lane_b] funded_pod_skus`` and of the owner's OA8 GPU types, the full 8xH100
lane A leg when one is present (OPTIONAL since the owner skipped it
2026-11-23), the tool commit on this repo's history, and the WB22 reader.

Without ``$LANED_EVIDENCE`` they SKIP, so WB19 reaches L2/L3 before the pod
run; L4 needs a green ``accept WB19 --evidence ...`` on main. When it IS set,
any missing, malformed, dry-run or hand-edited evidence fails hard. The
``test_verifier_*`` tests prove the verifier refuses tampering, offline.

What no offline check can prove is that a GPU actually ran: the binding is
the owner's OA8 approval (``evidence/owner/oa8.json``, hashed into the
evidence) and the owner-committed evidence PR. Contract: README.md "Pod
evidence".
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest

from .helpers import (
    CHAIN_TOML,
    EVIDENCE_SCHEMA,
    LEG_SCHEMA,
    REPO,
    RUN_SCHEMA,
    TABLE_SCHEMA,
    TORCH_PIN,
    body_sha256,
    canonical,
    chain_cfg,
    leg_contract,
    leg_contract_digest,
    load_tool,
    make_pod_record,
    pinned_image,
    rehash,
    run_tool,
    split_layout,
    write_json,
)

OA8_PATH = REPO / "evidence" / "owner" / "oa8.json"
FULL_SKU = "8xH100"
REL = 1e-6
NOTE_MAX = 200     # a table row's optional operator note (combine --note)


def _evidence() -> tuple[dict, Path]:
    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("LANED_EVIDENCE is unset: no pod evidence to verify yet. L4 needs "
                    "`scripts/laned/accept WB19 --evidence evidence/wb19.json` (OA8 pod run)")
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    assert p.is_file(), f"LANED_EVIDENCE={path} does not exist"
    return json.loads(p.read_text()), p


def name_matches(sku: str, name: str) -> bool:
    """The SKU's base type equals one run of consecutive whitespace tokens of
    the device name (case-insensitive): `L40` never matches "NVIDIA L40S"."""
    base = split_layout(sku)[0].lower()
    toks = name.lower().split()
    return any("".join(toks[i:j]) == base for i in range(len(toks))
               for j in range(i + 1, len(toks) + 1))


def _finite_pos(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _leg_digest(rec: dict) -> str:
    leg = {"schema": LEG_SCHEMA, "mode": rec["mode"], "sku": rec["sku"],
           "contract_digest": rec["contract"]["digest"], "image_digest": rec["image_digest"],
           "steps": rec["steps"], "selection": rec["selection"]}
    return hashlib.sha256(canonical(leg)).hexdigest()


def _is_ancestor(sha: str) -> bool:
    if not re.fullmatch(r"[0-9a-f]{40}", sha or ""):
        return False
    p = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", sha, "HEAD"],
                       capture_output=True, timeout=60)
    if p.returncode != 0:
        return False
    q = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e",
                        f"{sha}:scripts/laned_wb19_measure.py"], capture_output=True, timeout=60)
    return q.returncode == 0


def run_problems(rec: dict, cfg, *, check_git: bool = True) -> list[str]:
    sku = rec.get("sku", "?")
    p: list[str] = []
    if rec.get("schema") != RUN_SCHEMA or rec.get("wp") != "WB19":
        return [f"{sku}: schema/wp"]
    if rec.get("body_sha256") != body_sha256(rec):
        p.append(f"{sku}: body_sha256 does not match the body (hand-edited)")
    if rec.get("source") != "pod":
        p.append(f"{sku}: source {rec.get('source')!r} is not 'pod'")
    try:
        block = int(rec["contract_block"])
        mode = rec["mode"]
        assert mode in ("expert", "full")
        if block < int(cfg.training.v2_from_block):
            p.append(f"{sku}: contract_block before [v2] from_block")
        if rec["contract"]["digest"] != leg_contract_digest(cfg, block, mode):
            p.append(f"{sku}: contract digest != the shipped chain.toml's {mode} leg contract")
        lc = leg_contract(cfg, block, mode)
        if rec["contract"]["micro_batches"] != lc.micro_batches:
            p.append(f"{sku}: micro_batches changed")
        if rec["image_digest"] != pinned_image(cfg):
            p.append(f"{sku}: image_digest != the pinned train image")
        if mode == "full" and rec["steps"] != lc.training_steps:
            p.append(f"{sku}: a full leg must run the contract's {lc.training_steps} steps")
        if mode == "expert" and not rec["selection"]:
            p.append(f"{sku}: an expert leg needs its selection")
        if rec["leg_config_digest"] != _leg_digest(rec):
            p.append(f"{sku}: leg_config_digest")
        g = rec["gpu"]
        base, n = split_layout(sku)
        if g["count"] != n or not name_matches(sku, g["name"]):
            p.append(f"{sku}: GPU {g['count']}x {g['name']!r} does not match the SKU")
        if not str(g.get("driver") or "").strip() or not str(g.get("cuda") or "").strip():
            p.append(f"{sku}: driver/cuda missing")
        if rec["torch"] != TORCH_PIN:
            p.append(f"{sku}: torch {rec['torch']!r} != the pinned {TORCH_PIN}")
        if rec["finite"] is not True:
            p.append(f"{sku}: not finite")
        w = rec["wall"]
        o = w["overhead_seconds"]
        parts = [w["train_seconds"], o["boot"], o["corpus"], o["harvest"]]
        if not _finite_pos(w["train_seconds"]) or not all(
                isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in parts):
            p.append(f"{sku}: wall split not finite / non-negative")
        elif w["total_seconds"] + 1e-6 < sum(parts):
            p.append(f"{sku}: total_seconds < train + boot + corpus + harvest")
        c = rec["contract"]
        if not math.isclose(rec["steps_per_s"], rec["steps"] / w["train_seconds"], rel_tol=REL):
            p.append(f"{sku}: steps_per_s != steps / train_seconds")
        tok = rec["steps"] * c["batch_size"] * c["context_length"] / w["train_seconds"]
        if not math.isclose(rec["tokens_per_s"], tok, rel_tol=REL):
            p.append(f"{sku}: tokens_per_s != steps x batch x context / train_seconds")
        if not (isinstance(rec["peak_memory_bytes"], int) and rec["peak_memory_bytes"] > 0):
            p.append(f"{sku}: peak_memory_bytes")
        if check_git and (rec["tool"].get("git_dirty") is not False
                          or not _is_ancestor(rec["tool"].get("git_sha", ""))):
            p.append(f"{sku}: tool commit is dirty or not on this repo's history")
    except (KeyError, TypeError, ValueError, AssertionError) as e:
        p.append(f"{sku}: malformed ({type(e).__name__}: {e})")
    return p


def derive_table(runs: list[dict], cfg) -> dict:
    exp = [r for r in runs if r["mode"] == "expert"]
    block = exp[0]["contract_block"]
    rows = {}
    for r in exp:
        o = r["wall"]["overhead_seconds"]
        rows[r["sku"]] = {
            "gpu_name": r["gpu"]["name"], "gpu_count": r["gpu"]["count"],
            "driver": r["gpu"]["driver"], "cuda": r["gpu"]["cuda"],
            "steps_per_s": r["steps_per_s"], "tokens_per_s": r["tokens_per_s"],
            "overhead_seconds": o["boot"] + o["corpus"] + o["harvest"],
            "peak_memory_bytes": r["peak_memory_bytes"],
            "memory_total_bytes": r["gpu"]["memory_total_bytes"],
            "run_body_sha256": r["body_sha256"]}
    return {"schema": TABLE_SCHEMA, "image_digest": pinned_image(cfg),
            "contract_digest": leg_contract_digest(cfg, block, "expert"),
            "contract_block": block, "rows": rows}


def _close(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(
            a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def failing_checks(ev: dict) -> list[str] | None:
    """Names of the failing checks, or None when ``checks`` is malformed."""
    checks = ev.get("checks")
    if not isinstance(checks, list) or not checks:
        return None
    out = []
    for c in checks:
        if not (isinstance(c, dict) and isinstance(c.get("name"), str)
                and isinstance(c.get("passed"), bool)):
            return None
        if not c["passed"]:
            out.append(c["name"])
    return out


def note_ok(note) -> bool:
    return isinstance(note, str) and bool(note.strip()) and len(note) <= NOTE_MAX


def evidence_problems(ev: dict, cfg, *, oa8: Path | None = OA8_PATH,
                      check_git: bool = True) -> list[str]:
    p: list[str] = []
    if ev.get("schema") != EVIDENCE_SCHEMA or ev.get("wp") != "WB19":
        return ["schema/wp"]
    # Partial WB19 evidence (owner 2026-11-23, plan rev6.16): coverage may
    # fail (unmeasured types stay gpu_rate_unmeasured); any other failing
    # check refuses the evidence. ``passed`` stays the honest conjunction.
    failing = failing_checks(ev)
    if failing is None:
        p.append("checks malformed")
    else:
        other = sorted(n for n in failing if n != "coverage")
        if other:
            p.append(f"failing check(s) other than coverage: {other}")
        if ev.get("passed") is not (not failing):
            p.append("passed is not the conjunction of the checks")
    if ev.get("body_sha256") != body_sha256(ev):
        p.append("evidence body_sha256 does not match (hand-edited)")
    runs = ev.get("runs") or []
    for r in runs:
        p += run_problems(r, cfg, check_git=check_git)
    if p:
        return p
    if len({r["contract_block"] for r in runs}) != 1:
        p.append("runs at more than one contract_block")
    exp = [r for r in runs if r["mode"] == "expert"]
    skus = [r["sku"] for r in exp]
    if len(skus) != len(set(skus)):
        p.append("duplicate expert SKU")
    if not exp:
        return p + ["no expert runs"]
    want_table = derive_table(runs, cfg)
    for sku, row in ((ev.get("table") or {}).get("rows") or {}).items():
        if isinstance(row, dict) and "note" in row:
            if not note_ok(row["note"]):
                p.append(f"{sku}: note must be a non-empty string of at most "
                         f"{NOTE_MAX} characters")
            elif sku in want_table["rows"]:
                want_table["rows"][sku]["note"] = row["note"]
    if not _close(ev.get("table"), want_table):
        p.append("table is not the one derived from the runs")
    need = set(cfg.lane_b.funded_pod_skus)
    if oa8 is not None:
        if not oa8.is_file():
            p.append(f"owner approval {oa8.relative_to(REPO) if oa8.is_relative_to(REPO) else oa8}"
                     " is missing (OA8)")
        else:
            need |= set(json.loads(oa8.read_text()).get("gpu_types") or [])
            if ev.get("oa8_sha256") != hashlib.sha256(oa8.read_bytes()).hexdigest():
                p.append("oa8_sha256 does not bind the committed OA8 approval")
    missing = sorted(s for s in need if s.lower() not in {k.lower() for k in skus})
    # a missing candidate is allowed (partial evidence) but must be declared:
    # the coverage check fails and its detail names every missing SKU
    cov = [c for c in ev.get("checks") or [] if isinstance(c, dict)
           and c.get("name") == "coverage"]
    if len(cov) != 1:
        p.append("exactly one coverage check")
    elif missing and (cov[0].get("passed") is not False
                      or any(s not in str(cov[0].get("detail", "")) for s in missing)):
        p.append(f"no expert run for {missing}, and the coverage check does not say so")
    elif not missing and cov[0].get("passed") is not True:
        p.append("coverage check fails although every candidate has an expert run")
    for r in exp:
        if r["peak_memory_bytes"] >= r["gpu"]["memory_total_bytes"]:
            p.append(f"{r['sku']}: does not fit memory at the contract micro-batching")
    # The full lane A leg is OPTIONAL (owner 2026-11-23, "Yes skip it"; plan
    # rev6.5): the measured [round] funded_sku_wall_seconds stands for the
    # full-leg wall. When a full run IS present it is validated as before.
    full = [r for r in runs if r["mode"] == "full" and r["sku"] == FULL_SKU]
    if len(full) > 1:
        p.append("more than one full lane A leg on 8xH100")
    elif not full:
        if ev.get("lane_a_full_leg") is not None:
            p.append("lane_a_full_leg is set but no full run backs it")
    else:
        f, la = full[0], ev.get("lane_a_full_leg") or {}
        o = f["wall"]["overhead_seconds"]
        want = {"sku": FULL_SKU, "train_seconds": f["wall"]["train_seconds"],
                "total_seconds": f["wall"]["total_seconds"],
                "overhead_seconds": o["boot"] + o["corpus"] + o["harvest"],
                "funded_sku_wall_seconds": dict(cfg.round.funded_sku_wall_seconds).get(FULL_SKU)}
        if not _close({k: la.get(k) for k in want}, want):
            p.append("lane_a_full_leg does not match the full run")
    return p


# ── the evidence (skips without $LANED_EVIDENCE) ───────────────────────────

def test_evidence_verifies_against_shipped_chain_toml():
    ev, _ = _evidence()
    problems = evidence_problems(ev, chain_cfg(CHAIN_TOML))
    assert problems == [], "WB19 evidence refused:\n" + "\n".join(problems)


def test_evidence_table_reads_with_wb22_reader():
    ev, path = _evidence()
    from cascade.funding.step_rates import load_step_rate_table

    table = load_step_rate_table(path, image_digest=pinned_image())
    exp = {r["sku"] for r in ev["runs"] if r["mode"] == "expert"}
    for r in ev["runs"]:
        if r["mode"] == "expert":
            assert table[r["sku"]].steps_per_s == pytest.approx(r["steps_per_s"], rel=1e-12)
    # partial evidence (owner 2026-11-23): the table holds the measured rows
    # only; a listed type with no run has no row (the WB22 gate refuses it
    # gpu_rate_unmeasured)
    assert set(table) == exp


# ── the verifier itself (offline) ──────────────────────────────────────────

_SYNTHETIC: dict = {}


def _synthetic(tmp_path_factory) -> dict:
    """A combine-built body from fixture leg results (what the verifier must
    still judge on its merits; it is NOT pod evidence). Built once per module."""
    if "ev" not in _SYNTHETIC:
        _SYNTHETIC["ev"] = _build_synthetic(tmp_path_factory)
    return copy.deepcopy(_SYNTHETIC["ev"])


def _build_synthetic(tmp_path_factory) -> dict:
    tool = load_tool()
    cfg = chain_cfg()
    recs = [make_pod_record(tool, tmp_path_factory.mktemp(f"v-{s}"), None, s)
            for s in cfg.lane_b.funded_pod_skus]
    recs.append(make_pod_record(tool, tmp_path_factory.mktemp("v-full"), None, FULL_SKU,
                                mode="full"))
    d = tmp_path_factory.mktemp("v-combine")
    files = [write_json(d / f"run{i}.json", r) for i, r in enumerate(recs)]
    out = d / "wb19.json"
    rc, _, _ = run_tool(tool, ["combine", "--chain-toml", CHAIN_TOML, "--runs", *files,
                               "--out", out], {"PATH": str(d)}, d, None)
    assert rc == 0
    return json.loads(out.read_text())


def test_verifier_arithmetic_holds_on_a_combine_body(tmp_path_factory):
    ev = _synthetic(tmp_path_factory)
    assert evidence_problems(ev, chain_cfg(), oa8=None, check_git=False) == []


def test_verifier_requires_the_owner_oa8_binding(tmp_path_factory, tmp_path):
    ev = _synthetic(tmp_path_factory)
    probs = evidence_problems(ev, chain_cfg(), oa8=tmp_path / "absent.json", check_git=False)
    assert any("OA8" in x for x in probs)
    oa8 = write_json(tmp_path / "oa8.json", {"gpu_types": []})
    probs = evidence_problems(ev, chain_cfg(), oa8=oa8, check_git=False)
    assert any("oa8_sha256" in x for x in probs)


@pytest.mark.parametrize("edit,needle", [
    (lambda ev: ev["runs"][0].update(steps_per_s=ev["runs"][0]["steps_per_s"] * 1.5),
     "hand-edited"),
    (lambda ev: ev["runs"][0].update(source="dry_run"), "source"),
    (lambda ev: ev["runs"][0].update(torch="2.5.0+cu124"), "torch"),
    (lambda ev: ev["runs"][0]["gpu"].update(count=1), "does not match the SKU"),
    (lambda ev: ev["runs"][0]["contract"].update(micro_batches=4), "micro_batches"),
    (lambda ev: ev["runs"][0].update(peak_memory_bytes=ev["runs"][0]["gpu"]["memory_total_bytes"]),
     "fit memory"),
    (lambda ev: ev["runs"][0].update(tokens_per_s=1.0), "tokens_per_s"),
])
def test_verifier_refuses_tampered_runs(tmp_path_factory, edit, needle):
    ev = copy.deepcopy(_synthetic(tmp_path_factory))
    edit(ev)
    if needle != "hand-edited":
        ev["runs"][0] = rehash(ev["runs"][0])
    ev = rehash(ev)
    probs = evidence_problems(ev, chain_cfg(), oa8=None, check_git=False)
    assert any(needle in x for x in probs), probs


def test_verifier_refuses_a_table_not_derived_from_runs(tmp_path_factory):
    ev = copy.deepcopy(_synthetic(tmp_path_factory))
    sku = next(iter(ev["table"]["rows"]))
    ev["table"]["rows"][sku]["steps_per_s"] *= 2
    probs = evidence_problems(rehash(ev), chain_cfg(), oa8=None, check_git=False)
    assert any("table" in x for x in probs), probs


def _without_full_leg(ev: dict) -> dict:
    ev["runs"] = [r for r in ev["runs"] if r["mode"] != "full"]
    ev["lane_a_full_leg"] = None
    return rehash(ev)


def test_verifier_accepts_evidence_without_a_full_leg(tmp_path_factory):
    """Owner 2026-11-23 ("Yes skip it", plan rev6.5): WB19 runs only the
    per-GPU expert legs; the full 8xH100 lane A leg is optional."""
    ev = _without_full_leg(copy.deepcopy(_synthetic(tmp_path_factory)))
    assert evidence_problems(ev, chain_cfg(), oa8=None, check_git=False) == []


def test_verifier_refuses_lane_a_full_leg_without_a_full_run(tmp_path_factory):
    ev = copy.deepcopy(_synthetic(tmp_path_factory))
    la = ev["lane_a_full_leg"]
    ev = _without_full_leg(ev)
    ev["lane_a_full_leg"] = la
    probs = evidence_problems(rehash(ev), chain_cfg(), oa8=None, check_git=False)
    assert any("no full run backs it" in x for x in probs), probs


def test_verifier_still_checks_a_present_full_leg(tmp_path_factory):
    ev = copy.deepcopy(_synthetic(tmp_path_factory))
    ev["lane_a_full_leg"]["train_seconds"] *= 2
    probs = evidence_problems(rehash(ev), chain_cfg(), oa8=None, check_git=False)
    assert any("lane_a_full_leg does not match" in x for x in probs), probs


def test_verifier_refuses_two_full_legs(tmp_path_factory):
    ev = copy.deepcopy(_synthetic(tmp_path_factory))
    ev["runs"].append(copy.deepcopy(next(r for r in ev["runs"] if r["mode"] == "full")))
    probs = evidence_problems(rehash(ev), chain_cfg(), oa8=None, check_git=False)
    assert any("more than one full lane A leg" in x for x in probs), probs


def test_verifier_refuses_unknown_tool_commit(tmp_path_factory):
    ev = copy.deepcopy(_synthetic(tmp_path_factory))
    ev["runs"][0]["tool"] = {"git_sha": "0123456789abcdef0123456789abcdef01234567",
                             "git_dirty": False}
    ev["runs"][0] = rehash(ev["runs"][0])
    probs = evidence_problems(rehash(ev), chain_cfg(), oa8=None, check_git=True)
    assert any("tool commit" in x for x in probs), probs


# ── partial evidence (owner 2026-11-23, plan rev6.16) ─────────────────────

def _partial(tmp_path_factory, drop: str) -> dict:
    """The synthetic body re-combined without ``drop``'s expert run."""
    tool = load_tool()
    ev = _synthetic(tmp_path_factory)
    runs = [r for r in ev["runs"] if not (r["mode"] == "expert" and r["sku"] == drop)]
    d = tmp_path_factory.mktemp("v-partial")
    files = [write_json(d / f"run{i}.json", r) for i, r in enumerate(runs)]
    out = d / "wb19.json"
    rc, _, _ = run_tool(tool, ["combine", "--chain-toml", CHAIN_TOML, "--runs", *files,
                               "--out", out], {"PATH": str(d)}, d, None)
    assert rc == 0
    return json.loads(out.read_text())


def test_verifier_accepts_partial_coverage(tmp_path_factory):
    drop = list(chain_cfg().lane_b.funded_pod_skus)[-1]
    ev = _partial(tmp_path_factory, drop)
    assert ev["passed"] is False and failing_checks(ev) == ["coverage"]
    assert evidence_problems(ev, chain_cfg(), oa8=None, check_git=False) == []


def test_verifier_refuses_partial_with_another_failing_check(tmp_path_factory):
    drop = list(chain_cfg().lane_b.funded_pod_skus)[-1]
    ev = _partial(tmp_path_factory, drop)
    for c in ev["checks"]:
        if c["name"] == "memory":
            c["passed"] = False
    probs = evidence_problems(rehash(ev), chain_cfg(), oa8=None, check_git=False)
    assert any("other than coverage" in x for x in probs), probs


def test_verifier_refuses_undeclared_missing_coverage(tmp_path_factory):
    """A missing candidate hidden behind coverage = passed is refused."""
    drop = list(chain_cfg().lane_b.funded_pod_skus)[-1]
    ev = _partial(tmp_path_factory, drop)
    for c in ev["checks"]:
        if c["name"] == "coverage":
            c["passed"], c["detail"] = True, "every candidate measured"
    ev["passed"] = True
    probs = evidence_problems(rehash(ev), chain_cfg(), oa8=None, check_git=False)
    assert any("coverage check does not say so" in x for x in probs), probs


def test_verifier_refuses_passed_not_the_conjunction(tmp_path_factory):
    drop = list(chain_cfg().lane_b.funded_pod_skus)[-1]
    ev = _partial(tmp_path_factory, drop)
    ev["passed"] = True
    probs = evidence_problems(rehash(ev), chain_cfg(), oa8=None, check_git=False)
    assert any("conjunction" in x for x in probs), probs


def test_verifier_accepts_a_row_note_and_refuses_a_bad_one(tmp_path_factory):
    ev = copy.deepcopy(_synthetic(tmp_path_factory))
    sku = next(iter(ev["table"]["rows"]))
    ev["table"]["rows"][sku]["note"] = "pre-WB25 single-GPU path: understates multi-GPU"
    assert evidence_problems(rehash(ev), chain_cfg(), oa8=None, check_git=False) == []
    ev["table"]["rows"][sku]["note"] = "x" * (NOTE_MAX + 1)
    probs = evidence_problems(rehash(ev), chain_cfg(), oa8=None, check_git=False)
    assert any("note" in x for x in probs), probs


def test_name_rule_examples():
    assert name_matches("8xH100", "NVIDIA H100 80GB HBM3")
    assert name_matches("RTX4090", "NVIDIA GeForce RTX 4090")
    assert name_matches("A6000", "NVIDIA RTX A6000")
    assert name_matches("L40S", "NVIDIA L40S")
    assert not name_matches("L40", "NVIDIA L40S")
    assert not name_matches("H100", "NVIDIA H200")
