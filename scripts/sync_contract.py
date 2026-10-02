#!/usr/bin/env python3
"""Bring chain.toml's training contract in line with the LIVE one.

The operator's deployed config can run ahead of this repo (observed
2026-10-02: live rounds billed ``points+mv20`` on worker v0.13.0 while
chain.toml still read ``series_points`` / v0.9.0). The live contract is
public and signed: the anchor validator's latest receipt embeds the round's
manifest, whose ``contract_body`` hashes to its ``contract_digest``.

This script reads that receipt (signature verified), compares the body's
scalar fields with chain.toml's ``[training]``, rewrites the values that
differ (plus ``[round] funded_pod_image`` when the worker image moved), and
checks that the edited file now reproduces the live ``contract_digest``.

Usage:
    scripts/sync_contract.py                  # report + edit chain.toml
    scripts/sync_contract.py --check          # report only (exit 10 when stale)
    scripts/sync_contract.py --chain-toml chain.testnet.toml

Exit codes: 0 in sync, 10 drift found (edited unless --check), 2 error.
The contract-sync workflow turns an edit into a PR for an owner to review:
chain.toml is consensus config, and a receipt shows the EFFECTIVE contract,
not how the deployment expresses it (e.g. a scheduled switch).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _toml_literal(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v)
    return repr(v)


def set_key(text: str, section: str, key: str, value) -> tuple[str, bool]:
    """Replace ``key = …`` inside ``[section]``, keeping its trailing comment."""
    lines = text.splitlines(keepends=True)
    cur = None
    pat = re.compile(rf'^(\s*{re.escape(key)}\s*=\s*)("(?:[^"\\]|\\.)*"|[^#\s][^#]*?)(\s*(#.*)?\n?)$')
    for i, line in enumerate(lines):
        h = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
        if h:
            cur = h.group(1).strip()
            continue
        if cur == section:
            m = pat.match(line)
            if m:
                lines[i] = m.group(1) + _toml_literal(value) + m.group(3)
                return "".join(lines), True
    return text, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-toml", default="chain.toml")
    ap.add_argument("--check", action="store_true", help="report only, do not edit")
    ap.add_argument("--summary", default="", help="write a markdown summary here (PR body)")
    args = ap.parse_args()

    from cascade.miner.harness.images import resolve_worker_image
    from cascade.miner.harness.king import anchor_receipt_text
    from cascade.shared.config import load_chain_config
    from cascade.shared.manifest import contract_digest, contract_payload

    path = (REPO / args.chain_toml) if not Path(args.chain_toml).is_absolute() \
        else Path(args.chain_toml)
    cfg = load_chain_config(path)
    try:
        receipt = json.loads(anchor_receipt_text(cfg))           # signature-verified
    except Exception as e:  # noqa: BLE001
        print(f"cannot read the anchor's signed receipt: {e}", file=sys.stderr)
        return 2
    manifest = receipt.get("manifest") or {}
    body = manifest.get("contract_body")
    if not isinstance(body, dict):
        print("the latest receipt's manifest carries no contract_body", file=sys.stderr)
        return 2
    live_digest = manifest.get("contract_digest", "")
    if contract_digest(body) != live_digest:
        print("contract_body does not hash to the manifest's contract_digest", file=sys.stderr)
        return 2

    local = contract_payload(cfg.training)
    diff = {k: (local.get(k), v) for k, v in body.items()
            if isinstance(v, (str, int, float, bool)) and local.get(k) != v}
    round_id = receipt.get("round_id")
    block = receipt.get("epoch_start_block")
    if not diff:
        print(f"chain.toml [training] matches the live contract (round {round_id}).")
        return 0

    lines = [f"Live contract from the anchor validator's signed receipt, round `{round_id}` "
             f"(block {block}); manifest `contract_digest` `{live_digest[:16]}…`.", "",
             "| field | chain.toml | live |", "|---|---|---|"]
    lines += [f"| `{k}` | `{a}` | `{b}` |" for k, (a, b) in sorted(diff.items())]
    text = path.read_text(encoding="utf-8")
    unexpressed = []
    for k, (_, v) in sorted(diff.items()):
        text, ok = set_key(text, "training", k, v)
        if not ok:
            unexpressed.append(k)
    if "train_image_digest" in diff:
        ref = resolve_worker_image(diff["train_image_digest"][1])
        if ref:
            text, ok = set_key(text, "round", "funded_pod_image", ref)
            if ok:
                lines.append(f"| `[round] funded_pod_image` | | `{ref}` |")
    print("\n".join(lines))
    if args.check:
        return 10

    path.write_text(text, encoding="utf-8")
    edited = contract_digest(load_chain_config(path).training)
    lines += ["", f"Edited `{path.name}` reproduces the live `contract_digest`: "
                  f"**{'yes' if edited == live_digest else 'NO'}**."]
    if unexpressed:
        lines.append(f"Not present in `{path.name}` (add by hand): {', '.join(unexpressed)}.")
    if edited != live_digest:
        lines.append("The deployment may express this differently (e.g. a scheduled "
                     "`budget_denomination_after`); review before merging.")
    lines += ["", "chain.toml is consensus config: review before merging. Validators already "
                  "run this contract (they accepted the round), so this only brings the repo "
                  "in line."]
    if args.summary:
        Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-4:]))
    return 10


if __name__ == "__main__":
    raise SystemExit(main())
