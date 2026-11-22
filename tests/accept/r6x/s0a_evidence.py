"""R6x / S0a evidence (`laned_s0a_diff/1`): collector and verifier.

Contract: tests/accept/r6x/README.md. Two commands, both read only:

    collect  (the owner, on the box, after the main PR merged and the deployed
             file was reconciled):
        env PYTHONPATH=<checkout> python -P tests/accept/r6x/s0a_evidence.py collect \\
            --deployed /root/cascade/chain.toml --repo <checkout> --main-ref origin/main \\
            --state-dir /root/cascade --effective-block <next era start> \\
            --out <checkout>/evidence/s0a_diff.json

    verify   (CI via LANED_EVIDENCE, or by hand):
        env PYTHONPATH=<checkout> python -P tests/accept/r6x/s0a_evidence.py verify \\
            --evidence evidence/s0a_diff.json --repo <checkout> [--deployed F]

The collector never prints or records a config VALUE: the consensus side is a
per-key sha256 fingerprint, so the evidence file can live in the public repo.
Operator-only keys (host paths, the [generation] / [lane_d_ops] tables,
trainer-only recovery policy) are never fingerprinted at all.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from cascade.laned import dedrift as R6
from cascade.shared import decided as D
from cascade.shared import netconfig as N
from cascade.shared.config import parse_chain_config

SCHEMA = "laned_s0a_diff/1"
WP = "R6x"
RULE = "r6x-consensus/1"
CLASSES = R6.CLASSES
ABSENT = {"__absent__": True}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
TOP_KEYS = frozenset({
    "schema", "wp", "passed", "deployed_sha256", "main_commit", "main_chain_sha256",
    "decided_votes_sha256", "effective_block", "overlay", "classification",
    "consensus_rule", "deployed_fingerprints", "consensus_diff", "audit_decided",
})
AUDIT_CMD = ("cascade-audit decided --config <deployed> --state-dir <state-dir> "
             "(in-process: decided.drift_report + decided.state_report, record from "
             "<main_commit>:decided_votes.toml)")


class EvidenceError(ValueError):
    """Evidence the verifier refuses."""


# ── the consensus-key rule ───────────────────────────────────────────────────


def is_consensus_key(table: str, key: str) -> bool:
    """r6x-consensus/1: a key the validator-side consensus reads.

    Every key except (a) R6's operator-only class (the [generation] and
    [lane_d_ops] tables, trainer-only recovery policy, the validator host
    path, log-only telemetry) and (b) the trainer's own [round] rental and
    payment knobs: a [round] key outside ``netconfig.ALLOWED["round"]`` (the
    schedule the satellites read) that is neither a decided block nor a
    decided pin. Top-level keys count."""
    if not table:
        return True
    cls, _reason = R6.classify(table, key, 0)
    if cls == "operator-only":
        return False
    top = table.split(".", 1)[0]
    dotted = f"{table}.{key}"
    trainer_knob = (top == "round" and key not in (N.ALLOWED["round"] or frozenset())
                    and dotted not in R6.decided_block_keys() and dotted not in D.PIN_KEYS)
    return not trainer_knob


def _leaves(node: dict, prefix: tuple[str, ...] = ()):
    for k, v in node.items():
        if isinstance(v, dict):
            yield from _leaves(v, prefix + (k,))
        else:
            yield ".".join(prefix), k


def _normal(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normal(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _normal(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, frozenset, set)):
        seq = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return [_normal(v) for v in seq]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def effective_value(cfg: object, raw: dict, table: str, key: str) -> object:
    """The value the loaded config uses: the parsed section attribute (so an
    absent key reads as its code default), else the raw TOML value, else the
    ABSENT sentinel."""
    node: object = cfg
    for part in table.split(".") if table else ():
        node = node.get(part) if isinstance(node, dict) else getattr(node, part, None)
        if node is None:
            break
    if node is not None:
        if isinstance(node, dict) and key in node:
            return _normal(node[key])
        if not isinstance(node, dict) and hasattr(node, key):
            return _normal(getattr(node, key))
    r: object = raw
    for part in table.split(".") if table else ():
        r = r.get(part, {}) if isinstance(r, dict) else {}
    if isinstance(r, dict) and key in r:
        return _normal(r[key])
    return ABSENT


def fingerprint(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def consensus_keys(raw: dict) -> set[tuple[str, str]]:
    return {(t, k) for t, k in _leaves(raw) if is_consensus_key(t, k)}


def fingerprints(raw: dict, keys: set[tuple[str, str]]) -> dict[tuple[str, str], str]:
    cfg = parse_chain_config(copy.deepcopy(raw))
    return {tk: fingerprint(effective_value(cfg, raw, *tk)) for tk in keys}


def main_with_overlay(main_raw: dict, body: dict) -> dict:
    return N.merge_raw(main_raw, body.get("overrides") or {})


# ── git (the release side always comes from a commit, never a work tree) ────


def _git(repo: Path, *args: str) -> bytes:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if proc.returncode != 0:
        raise EvidenceError(f"git {' '.join(args)}: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def resolve_commit(repo: Path, ref: str) -> str:
    return _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()


def show(repo: Path, commit: str, path: str) -> bytes:
    return _git(repo, "show", f"{commit}:{path}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── collect ──────────────────────────────────────────────────────────────────


def collect(deployed: Path, repo: Path, main_ref: str, state_dir: Path,
            effective_block: int) -> dict:
    dep_bytes = deployed.read_bytes()
    dep_raw = tomllib.loads(dep_bytes.decode("utf-8"))
    commit = resolve_commit(repo, main_ref)
    main_bytes = show(repo, commit, "chain.toml")
    votes_bytes = show(repo, commit, D.DECIDED_FILE)
    main_raw = tomllib.loads(main_bytes.decode("utf-8"))

    with tempfile.TemporaryDirectory(prefix="r6x-") as tmp:
        main_file = Path(tmp) / "chain.toml"
        main_file.write_bytes(main_bytes)
        votes_file = Path(tmp) / D.DECIDED_FILE
        votes_file.write_bytes(votes_bytes)
        files = R6.dedrift(deployed, main_file, effective_block)
        record = D.load_decided(votes_file)
    classification = json.loads(files["classification.json"])
    body = json.loads(files["overlay.json"])

    findings = D.drift_report(parse_chain_config(copy.deepcopy(dep_raw)), record)
    findings += D.state_report(D.read_state_records(state_dir), record)
    rendered = D.render(findings)
    fails = sorted({f"{f.kind} {f.feature} {f.key}".strip() for f in findings if f.fails})

    merged = main_with_overlay(main_raw, body)
    keys = consensus_keys(dep_raw) | consensus_keys(merged)
    dep_fp = fingerprints(dep_raw, keys)
    main_fp = fingerprints(merged, keys)
    diff = [{"table": t, "key": k, "deployed": dep_fp[(t, k)], "main": main_fp[(t, k)]}
            for t, k in sorted(keys) if dep_fp[(t, k)] != main_fp[(t, k)]]
    main_drift = sorted(f"{m['table']}.{m['key']}" for m in classification["markers"]
                        if m["class"] == "main" and m["drift"])
    audit_rc = 1 if fails else 0
    doc = {
        "schema": SCHEMA, "wp": WP,
        "passed": not diff and not main_drift and audit_rc == 0,
        "deployed_sha256": sha256(dep_bytes),
        "main_commit": commit,
        "main_chain_sha256": sha256(main_bytes),
        "decided_votes_sha256": sha256(votes_bytes),
        "effective_block": int(effective_block),
        "overlay": body,
        "classification": {
            "sha256": sha256(files["classification.json"].encode("utf-8")),
            "counts": classification["counts"],
            "markers": len(classification["markers"]),
            "main_drift": main_drift,
        },
        "consensus_rule": RULE,
        "deployed_fingerprints": [[t, k, dep_fp[(t, k)]] for t, k in sorted(keys)],
        "consensus_diff": diff,
        "audit_decided": {"command": AUDIT_CMD, "exit_code": audit_rc,
                          "output_sha256": sha256(rendered.encode("utf-8")), "fails": fails},
    }
    return doc


# ── verify ───────────────────────────────────────────────────────────────────


def _need(cond: bool, why: str) -> None:
    if not cond:
        raise EvidenceError(why)


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def check_fields(ev: object) -> None:
    _need(isinstance(ev, dict), "evidence is not a JSON object")
    assert isinstance(ev, dict)
    _need(set(ev) == TOP_KEYS, f"top-level keys {sorted(set(ev) ^ TOP_KEYS)} wrong")
    _need(ev["schema"] == SCHEMA, f"schema {ev['schema']!r} != {SCHEMA!r}")
    _need(ev["wp"] == WP, f"wp {ev['wp']!r} != {WP!r}")
    _need(ev["passed"] is True, "passed is not true")
    for k in ("deployed_sha256", "main_chain_sha256", "decided_votes_sha256"):
        _need(isinstance(ev[k], str) and bool(HEX64.match(ev[k])), f"{k} is not a sha256 hex")
    _need(isinstance(ev["main_commit"], str) and bool(HEX40.match(ev["main_commit"])),
          "main_commit is not a full commit sha")
    _need(_is_int(ev["effective_block"]) and ev["effective_block"] > 0,
          "effective_block is not a positive int")
    _need(ev["consensus_rule"] == RULE, f"consensus_rule {ev['consensus_rule']!r} != {RULE!r}")
    c = ev["classification"]
    _need(isinstance(c, dict) and set(c) == {"sha256", "counts", "markers", "main_drift"},
          "classification block malformed")
    _need(isinstance(c["sha256"], str) and bool(HEX64.match(c["sha256"])),
          "classification.sha256 is not a sha256 hex")
    _need(isinstance(c["counts"], dict) and set(c["counts"]) == set(CLASSES)
          and all(_is_int(v) and v >= 0 for v in c["counts"].values()),
          "classification.counts must hold the three classes as ints")
    _need(_is_int(c["markers"]) and sum(c["counts"].values()) == c["markers"],
          "classification.counts do not add up to the marker count")
    _need(c["main_drift"] == [], f"main-class drift remains: {c['main_drift']}")
    _need(ev["consensus_diff"] == [], f"consensus-key diff not empty: {len(ev['consensus_diff'])} key(s)"
          if isinstance(ev["consensus_diff"], list) else "consensus_diff is not a list")
    a = ev["audit_decided"]
    _need(isinstance(a, dict) and set(a) == {"command", "exit_code", "output_sha256", "fails"},
          "audit_decided block malformed")
    _need(_is_int(a["exit_code"]) and a["exit_code"] == 0, "cascade-audit decided did not exit 0")
    _need(a["fails"] == [], f"cascade-audit decided findings: {a['fails']}")
    _need(isinstance(a["output_sha256"], str) and bool(HEX64.match(a["output_sha256"])),
          "audit_decided.output_sha256 is not a sha256 hex")
    fps = ev["deployed_fingerprints"]
    _need(isinstance(fps, list) and fps, "deployed_fingerprints empty")
    for row in fps:
        _need(isinstance(row, list) and len(row) == 3 and all(isinstance(x, str) for x in row)
              and bool(HEX64.match(row[2])), f"bad fingerprint row {row!r}")
    _need(len({(r[0], r[1]) for r in fps}) == len(fps), "duplicate fingerprint keys")
    _need(fps == sorted(fps, key=lambda r: (r[0], r[1])), "fingerprints are not sorted")
    for t, k, _fp in fps:
        _need(is_consensus_key(t, k), f"{t}.{k} is not a consensus key (rule {RULE})")
    body = ev["overlay"]
    try:
        N.check_body(copy.deepcopy(body))
    except N.NetConfigError as e:
        raise EvidenceError(f"overlay body refused: {e}") from e
    _need(body.get("effective_block") == ev["effective_block"],
          "overlay.effective_block != effective_block")
    for table, kv in (body.get("overrides") or {}).items():
        for key, value in kv.items():
            cls, _r = R6.classify(table, key, value)
            _need(cls == "overlay", f"overlay carries {table}.{key} of class {cls}")


def rederive_main(ev: dict, repo: Path) -> None:
    """The release side from the recorded commit: hashes match, the overlay
    validates on it, and every consensus key's fingerprint equals the
    recorded deployed one (so the diff is empty by recomputation)."""
    commit = ev["main_commit"]
    main_bytes = show(repo, commit, "chain.toml")
    _need(sha256(main_bytes) == ev["main_chain_sha256"], "main_chain_sha256 != the commit's chain.toml")
    _need(sha256(show(repo, commit, D.DECIDED_FILE)) == ev["decided_votes_sha256"],
          "decided_votes_sha256 != the commit's decided_votes.toml")
    main_raw = tomllib.loads(main_bytes.decode("utf-8"))
    try:
        N.validate_overlay(parse_chain_config(copy.deepcopy(main_raw)), copy.deepcopy(ev["overlay"]))
    except N.NetConfigError as e:
        raise EvidenceError(f"overlay does not validate on main: {e}") from e
    merged = main_with_overlay(main_raw, ev["overlay"])
    recorded = {(t, k): fp for t, k, fp in ev["deployed_fingerprints"]}
    missing = sorted(consensus_keys(merged) - set(recorded))
    _need(not missing, f"main consensus keys with no deployed fingerprint: {missing[:5]}")
    main_fp = fingerprints(merged, set(recorded))
    diff = sorted(f"{t}.{k}" for (t, k), fp in recorded.items() if main_fp[(t, k)] != fp)
    _need(not diff, f"re-derived consensus-key diff not empty: {diff}")


def rederive_deployed(ev: dict, deployed: Path) -> None:
    """With the deployed file at hand (the box): its bytes are the recorded
    ones and its fingerprints are the recorded ones."""
    data = deployed.read_bytes()
    _need(sha256(data) == ev["deployed_sha256"], "deployed_sha256 != the deployed file")
    raw = tomllib.loads(data.decode("utf-8"))
    recorded = {(t, k): fp for t, k, fp in ev["deployed_fingerprints"]}
    missing = sorted(consensus_keys(raw) - set(recorded))
    _need(not missing, f"deployed consensus keys not fingerprinted: {missing[:5]}")
    fp = fingerprints(raw, set(recorded))
    bad = sorted(".".join(tk) for tk, v in recorded.items() if fp[tk] != v)
    _need(not bad, f"deployed fingerprints do not match the file: {bad}")


def verify(ev: object, repo: Path, deployed: Path | None = None) -> None:
    check_fields(ev)
    assert isinstance(ev, dict)
    rederive_main(ev, repo)
    if deployed is not None:
        rederive_deployed(ev, deployed)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="s0a_evidence", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--deployed", type=Path, required=True)
    c.add_argument("--repo", type=Path, required=True)
    c.add_argument("--main-ref", default="origin/main")
    c.add_argument("--state-dir", type=Path, required=True)
    c.add_argument("--effective-block", type=int, required=True)
    c.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--evidence", type=Path, required=True)
    v.add_argument("--repo", type=Path, required=True)
    v.add_argument("--deployed", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "collect":
            doc = collect(args.deployed, args.repo, args.main_ref, args.state_dir,
                          args.effective_block)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"s0a_evidence: passed={doc['passed']} counts={doc['classification']['counts']} "
                  f"main_drift={len(doc['classification']['main_drift'])} "
                  f"consensus_diff={len(doc['consensus_diff'])} "
                  f"audit_exit={doc['audit_decided']['exit_code']}")
            for row in doc["consensus_diff"]:
                print(f"  diff {row['table']}.{row['key']}")
            for f in doc["audit_decided"]["fails"]:
                print(f"  audit {f}")
            return 0 if doc["passed"] else 1
        verify(json.loads(args.evidence.read_text(encoding="utf-8")), args.repo, args.deployed)
        print("s0a_evidence: ok")
        return 0
    except (EvidenceError, R6.DedriftError, N.NetConfigError, ValueError, OSError) as e:
        print(f"s0a_evidence: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
