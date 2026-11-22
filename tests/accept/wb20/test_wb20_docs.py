"""WB20 acceptance: the lane B docs (MINER, INTERFACE, AGENT_MINING, VALIDATOR).

Plan §3.0 WB20 row ("Lane B docs", ci), §11.2 "(rev4) Lane B docs", §12b.
These are doc CONTRACT tests, not prose review: every fact a doc must carry
is derived from the code on each run (see helpers.py), so a doc that drifts
from the code fails, and a doc that names a key, code or header the code
does not have fails. See README.md in this directory for the full contract.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

from . import helpers as h

DOC_IDS = [p.name for p in h.WB20_DOCS]


def _lane_b(path) -> str:
    text = h.lane_b_text(path)
    assert text.strip(), (f"{path.name}: no lane B section (a heading naming 'lane B'); "
                          "WB20 adds one to each of the four guides")
    return text


def _missing(text: str, tokens) -> list[str]:
    return [t for t in tokens if not h.has_token(text, t)]


# ── the code truth itself (passes on main; guards the derivation) ───────────

def test_code_truth_is_derivable():
    assert h.track_defaults(), "LANE_B_TRACK_DEFAULTS is empty"
    assert set(h.track_defaults()) <= h.lane_b_keys()
    assert "benchmark_series_embedded" in h.contamination_refusal_codes()
    assert h.declared_rows_codes() and h.deferral_codes() and h.gate_closed_reasons()
    assert h.expiry_codes() and h.receipt_keys() and h.open_rules() and h.suites()
    assert h.declared_rows_header() in h.lane_b_headers()
    assert h.experts_per_layer()


# ── a lane B section in each guide ──────────────────────────────────────────

@pytest.mark.parametrize("doc", h.WB20_DOCS, ids=DOC_IDS)
def test_each_doc_has_a_lane_b_section(doc):
    _lane_b(doc)


# ── MINER.md: every key + default, every code a miner can receive ───────────

@pytest.mark.parametrize("key", sorted(h.track_defaults()))
def test_miner_documents_every_lane_b_track_key_and_default(key):
    text = _lane_b(h.MINER)
    default = h.track_defaults()[key]
    if key == "funded_pod_skus" and h.funded_pod_skus_defaults_to_round():
        # D42: no literal default; the parser starts lane B's list from the
        # same file's [round] funded_pod_skus
        lines = [ln for ln in text.splitlines()
                 if any(h.names_key(s, key) for s in h.inline_spans(ln))]
        assert any(f"`[round] {key}`" in ln for ln in lines), (
            "MINER.md lane B: `funded_pod_skus` must be documented with its default, the "
            "same file's `[round] funded_pod_skus` (on the same line)")
        return
    assert h.documents_default(text, key, default), (
        f"MINER.md lane B: `[lane_b] {key}` must be documented with its default "
        f"{default!r} (a code span naming the key and, on the same line, a code span whose "
        "TOML value equals the parser default, or one span `key = <toml>`)")


MINER_CODE_GROUPS = {
    "contamination_refusals": h.contamination_refusal_codes,
    "declared_rows_refusals": h.declared_rows_codes,
    "deferrals": h.deferral_codes,
    "gate_closed_reasons": h.gate_closed_reasons,
    "funding_expiry": h.expiry_codes,
}


@pytest.mark.parametrize("group", sorted(MINER_CODE_GROUPS))
def test_miner_documents_every_lane_b_code(group):
    text = _lane_b(h.MINER)
    missing = _missing(text, MINER_CODE_GROUPS[group]())
    assert not missing, f"MINER.md lane B: undocumented {group} code(s): {missing}"


def test_miner_documents_deferral_status_form():
    text = _lane_b(h.MINER)
    missing = [c for c in h.deferral_codes()
               if not re.search(rf"deferred:\s*{re.escape(c)}\b", text)]
    assert not missing, ("MINER.md lane B: each deferral must be shown as it appears in "
                         f"status/lane_b.json (`deferred: <code>`); missing {missing}")
    assert h.has_token(text, "gate_closed_reason"), (
        "MINER.md lane B: the status field `gate_closed_reason` must be named")


def test_miner_sentinel_deferral_bound_documented():
    text = _lane_b(h.MINER)
    paras = [p for p in h.paragraphs(text) if h.has_token(p, "sentinel_target")]
    assert any(h.has_token(p, "sentinel_max_defer_eras") for p in paras), (
        "MINER.md lane B: the `sentinel_target` deferral must be explained together with its "
        "bound `sentinel_max_defer_eras` (same paragraph)")


# ── X-Lane-B-Declared-Rows (WB14) ───────────────────────────────────────────

@pytest.mark.parametrize("doc", [h.MINER, h.INTERFACE, h.AGENT_MINING],
                         ids=["MINER.md", "INTERFACE.md", "AGENT_MINING.md"])
def test_declared_rows_header_documented_with_a_valid_example(doc):
    text = _lane_b(doc)
    header = h.declared_rows_header()
    assert h.has_token(text, header), f"{doc.name} lane B: header `{header}` not documented"
    examples = h.declared_rows_examples(text)
    assert examples, (f"{doc.name} lane B: no `{header}: {{...}}` example (header, colon, "
                      "JSON object on one line, in a code span or a fenced block)")
    codes = h.declared_rows_codes()
    accepted = 0
    for line, raw in examples:
        named = [c for c in codes if c in line]
        rows, code = h.parse_rows(raw)
        if named:
            assert code in named, (f"{doc.name}: example {raw!r} is labelled {named} but the "
                                   f"WB14 parser returns {code or 'accepted'}")
        else:
            assert code is None, (f"{doc.name}: example {raw!r} is refused by the WB14 parser "
                                  f"with {code} under the default clamp; label it with that "
                                  "code on the same line or fix it")
            accepted += 1
    assert accepted, f"{doc.name}: no declared-rows example the WB14 parser accepts"


def test_interface_documents_the_signed_rows_binding():
    text = _lane_b(h.INTERFACE)
    tag = h.declared_rows_signed_tag()
    assert h.has_token(text, tag) or any(tag in s for s in h.spans(text)), (
        f"INTERFACE.md lane B: the signed ref suffix `|{tag}<sha256>` (the signature binds the "
        "declared rows) must be documented")
    assert h.has_token(text, "declared_rows"), (
        "INTERFACE.md lane B: the `declared_rows` field must be named")


@pytest.mark.parametrize("doc", [h.INTERFACE, h.AGENT_MINING],
                         ids=["INTERFACE.md", "AGENT_MINING.md"])
def test_declared_rows_ordinals_and_per_layer_clamp_documented(doc):
    text = _lane_b(doc)
    epl = h.experts_per_layer()
    assert h.ordinal_range_documented(text, len(epl)), (
        f"{doc.name} lane B: keys are MoE ordinals 0..{len(epl) - 1} (one line with "
        f"'ordinal' and the range 0..{len(epl) - 1})")
    missing = []
    for e in sorted(set(epl)):
        lo, hi = h.clamp(e)
        if not h.range_on_line(text, e, lo, hi):
            missing.append(f"{e} experts -> {lo}..{hi}")
    assert not missing, (f"{doc.name} lane B: per-layer clamp under the default "
                         f"min/max_experts_frac not documented for: {missing}")
    assert not _missing(text, ("min_experts_frac", "max_experts_frac")), (
        f"{doc.name} lane B: the clamp keys `min_experts_frac` / `max_experts_frac`")


def test_interface_documents_every_declared_rows_refusal():
    text = _lane_b(h.INTERFACE)
    missing = _missing(text, h.declared_rows_codes())
    assert not missing, f"INTERFACE.md lane B: undocumented declared-rows codes {missing}"


# ── the contamination ban (WB13) ────────────────────────────────────────────

def test_interface_documents_the_embedded_benchmark_ban_and_repo_cap():
    text = _lane_b(h.INTERFACE)
    missing = _missing(text, h.contamination_refusal_codes())
    assert not missing, f"INTERFACE.md lane B: undocumented intake refusal codes {missing}"
    cap = h.track_defaults()["suite_target_max_repo_mb"]
    paras = [p for p in h.paragraphs(text) if h.has_token(p, "repo_too_large")]
    assert any(h.has_token(p, "suite_target_max_repo_mb")
               and h.documents_default(p, "suite_target_max_repo_mb", cap) for p in paras), (
        "INTERFACE.md lane B: `repo_too_large` must be explained with "
        f"`suite_target_max_repo_mb` and its default {cap} (same paragraph)")


@pytest.mark.parametrize("doc", [h.INTERFACE, h.VALIDATOR], ids=["INTERFACE.md", "VALIDATOR.md"])
def test_contamination_receipt_field_documented(doc):
    text = _lane_b(doc)
    missing = _missing(text, ("contamination", *h.receipt_keys()))
    assert not missing, (f"{doc.name} lane B: the receipt field `contamination` and its keys "
                         f"{list(h.receipt_keys())}; missing {missing}")


def test_agent_mining_documents_refusals_and_deferrals():
    text = _lane_b(h.AGENT_MINING)
    missing = _missing(text, ("benchmark_series_embedded", *h.deferral_codes()))
    assert not missing, f"AGENT_MINING.md lane B: missing {missing}"


# ── suite targets (WB18) ────────────────────────────────────────────────────

@pytest.mark.parametrize("doc", [h.MINER, h.INTERFACE, h.AGENT_MINING],
                         ids=["MINER.md", "INTERFACE.md", "AGENT_MINING.md"])
def test_suite_target_names_and_commit_codes_documented(doc):
    text = _lane_b(doc)
    target, code = h.suite_prefixes()
    want = [f"{target}{s}" for s in h.suites()]
    if doc is not h.AGENT_MINING:
        want += [f"{code}{s}" for s in h.suites()]
    missing = _missing(text, want)
    assert not missing, f"{doc.name} lane B: suite targets / commit codes missing {missing}"
    if doc is h.MINER:
        assert h.documents_default(text, "suite_targets", h.track_defaults()["suite_targets"])


# ── the opening rule (WB17) ─────────────────────────────────────────────────

@pytest.mark.parametrize("doc", [h.MINER, h.VALIDATOR], ids=["MINER.md", "VALIDATOR.md"])
def test_open_rule_values_documented(doc):
    text = _lane_b(doc)
    missing = _missing(text, ("open_rule", *h.open_rules()))
    assert not missing, f"{doc.name} lane B: `open_rule` values missing {missing}"


def test_validator_documents_the_gate_record():
    text = _lane_b(h.VALIDATOR)
    missing = _missing(text, ("gate_rule", "gate_evidence", "gate_closed_reason",
                              *h.gate_closed_reasons()))
    assert not missing, f"VALIDATOR.md lane B: the WB17 gate record; missing {missing}"


# ── the hash set and declared rows (what validators check) ──────────────────

def test_validator_documents_the_pinned_hashset():
    text = _lane_b(h.VALIDATOR)
    # netconfig's release-only tuple is the rule's home (read as source)
    src = (h.CASCADE / "shared" / "netconfig.py").read_text(encoding="utf-8")
    assert '("lane_b", "contamination_hashset")' in src, "code truth: release-only hash set"
    paras = [p for p in h.paragraphs(text) if h.has_token(p, "contamination_hashset")]
    assert any(re.search(r"release[- ]only", p, re.I) for p in paras), (
        "VALIDATOR.md lane B: `[lane_b] contamination_hashset` is release-only (not "
        "overlay-settable); say so in the paragraph that names it")
    assert not _missing(text, ("declared_rows", "hashset_version")), (
        "VALIDATOR.md lane B: `declared_rows` and `hashset_version` must be named")


# ── the sentinel is neither guard nor target for its era (W-S2 / WB18) ──────

@pytest.mark.parametrize("doc", [h.MINER, h.VALIDATOR], ids=["MINER.md", "VALIDATOR.md"])
def test_sentinel_is_neither_guard_nor_target(doc):
    text = _lane_b(doc)
    ok = False
    for p in h.paragraphs(text):
        low = p.lower()
        if "sentinel" in low and "guard" in low and "target" in low \
                and re.search(r"\b(neither|nor|not|never)\b", low) and re.search(r"\bera\b", low):
            ok = True
            break
    assert ok, (f"{doc.name} lane B: one paragraph must state that the era's sentinel suite "
                "is neither a guard nor a target for that era (words: sentinel, guard, target, "
                "era and a negation)")


# ── no stale mentions ───────────────────────────────────────────────────────

SNAKE_WORD = re.compile(r"(?<![A-Za-z0-9_\-])[a-z][a-z0-9]*(?:_[a-z0-9]+)+(?![A-Za-z0-9_\-])")


@pytest.mark.parametrize("doc", h.WB20_DOCS, ids=DOC_IDS)
def test_no_lane_b_doc_names_a_key_or_code_missing_from_code(doc):
    full = h.read(doc)
    bad = []
    for m in re.finditer(r"\[lane_b\]\s+`?([a-z][a-z0-9_]*)", full):
        if m.group(1) not in h.lane_b_keys():
            bad.append(f"[lane_b] {m.group(1)} (no parser reads it)")
    for m in re.finditer(r"X-Lane-B-[A-Za-z][A-Za-z-]*[A-Za-z]", full):
        if m.group(0) not in h.lane_b_headers():
            bad.append(f"header {m.group(0)}")
    for m in re.finditer(r"deferred:\s*`?([a-z][a-z0-9_]*)", full):
        if m.group(1) not in h.deferral_codes():
            bad.append(f"deferred: {m.group(1)}")
    for m in re.finditer(r"open_rule\s*=\s*\"([^\"]*)\"", full):
        if m.group(1) not in h.open_rules():
            bad.append(f'open_rule = "{m.group(1)}"')
    target, code = h.suite_prefixes()
    text = h.lane_b_text(doc)
    for s in h.spans(text):
        for m in re.finditer(rf"(?<![A-Za-z0-9_\-]){re.escape(target)}([a-z0-9_]+)", s):
            if m.group(1) not in h.suites():
                bad.append(f"{target}{m.group(1)}")
        cm = re.fullmatch(rf"{re.escape(code)}([a-z0-9]+)", s)
        if cm and cm.group(1) not in h.suites():
            bad.append(f"{code}{cm.group(1)}")
    for line in text.splitlines():
        for s in h.inline_spans(line):
            for w in SNAKE_WORD.findall(s):
                if not h.code_has_token(w):
                    bad.append(f"`{w}` (in lane B section; not in code or the chain TOMLs)")
    assert not bad, f"{doc.name}: stale lane B mentions: {sorted(set(bad))}"


# ── links, secrets, check_docs ──────────────────────────────────────────────

@pytest.mark.parametrize("doc", h.WB20_DOCS, ids=DOC_IDS)
def test_every_relative_link_resolves(doc):
    assert not h.broken_links(doc)


@pytest.mark.parametrize("doc", h.WB20_DOCS, ids=DOC_IDS)
def test_no_secrets_hostnames_or_ips(doc):
    assert not h.leaks(doc)


def test_check_docs_passes():
    env = {**os.environ, "PYTHONPATH": str(h.REPO)}
    proc = subprocess.run([sys.executable, "-P", str(h.REPO / "scripts" / "check_docs.py")],
                          cwd=h.REPO, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"scripts/check_docs.py failed:\n{proc.stdout}\n{proc.stderr}"
