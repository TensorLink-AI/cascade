#!/usr/bin/env python3
"""Fork detector: replay recent MAINNET receipts through THIS checkout's
satellite/audit code (``cascade.audit.main.audit_receipt``, tier 0).

Every satellite validator re-derives the main validator's verdicts and weight
vector from its signed receipts with exactly this code. A PR that changes what
that code computes for an already-settled round (an ungated consensus change)
makes a FAIL here — before it ships and forks the fleet (2026-11-01 contract
change mid-seed-epoch; 2026-09-22 ungated tenure divergence; 2026-11-02 vote
feature-name hash trap).

Exit status:
  0  every fetched receipt replayed with no FAIL
  1  a FAIL (consensus divergence), or receipts fetched but none replayable
  0  (with a GitHub warning) when storage/chain is unreachable — an outage is
     not a fork; the job says it could not verify.

Usage: replay_mainnet_receipts.py [--last N] [--validator SS58] [--config PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MAIN = "5FcCsoB2R5M55RdrNrvPQv7xmLJRFzs7R7C73UUSxgUCyfsw"   # [netconfig] owner_hotkey
# Checks whose PASS proves a receipt was actually re-derived (not skipped).
REPLAYED = ("verdict", "duel-cohort", "weights", "transition", "bench-gate", "margin-v2",
            "king-check")


def _gh(level: str, msg: str) -> None:
    print(f"::{level}::{msg}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--last", type=int, default=12)
    ap.add_argument("--validator", default="")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--network", default="finney")
    args = ap.parse_args(argv)

    from cascade.audit.checks import FAIL, PASS
    from cascade.audit.main import _chain_client, _fetch_text, audit_receipt, load_receipt
    from cascade.shared.config.loader import load_chain_config, require_vote_pin
    from cascade.shared.hippius import RECEIPT_INDEX_KEY

    require_vote_pin()                      # exactly as cascade-audit / satellites load
    cfg = load_chain_config(args.config)
    main_hk = args.validator or str(getattr(cfg.netconfig, "owner_hotkey", "") or DEFAULT_MAIN)
    try:
        doc = json.loads(_fetch_text(cfg, RECEIPT_INDEX_KEY))
    except Exception as e:  # noqa: BLE001 — an outage is not a fork
        _gh("warning", f"receipt replay NOT verified: receipt index unreachable ({e})")
        return 0
    rows = [r for r in doc.get("rounds", []) if isinstance(r, dict)
            and str(r.get("receipt_key", "")).startswith(f"receipts/{main_hk}/")]
    rows.sort(key=lambda r: str(r.get("published_at") or ""), reverse=True)
    seen, picked = set(), []
    for r in rows:
        rid = str(r.get("round_id"))
        if rid not in seen:
            seen.add(rid)
            picked.append(r)
        if len(picked) >= args.last:
            break
    if not picked:
        _gh("error", f"receipt index lists no receipts for {main_hk[:12]}")
        return 1
    try:
        client = _chain_client(cfg, args.network)
    except Exception as e:  # noqa: BLE001
        _gh("warning", f"chain client unavailable ({e}); dated-launch receipts will not replay")
        client = None

    fails, replayed, unreachable = [], 0, 0
    for r in picked:
        rid = str(r["round_id"])
        try:
            rc = load_receipt(_fetch_text(cfg, str(r["receipt_key"])))
        except Exception as e:  # noqa: BLE001
            unreachable += 1
            print(f"round {rid}: receipt unreadable ({e})")
            continue
        results = audit_receipt(rc, cfg, tier=0, client=client)
        bad = [x for x in results if x.status == FAIL]
        ok = any(x.status == PASS and x.name in REPLAYED for x in results)
        replayed += ok
        status = "FAIL" if bad else ("replayed" if ok else "not replayed")
        print(f"round {rid}: {status} — " + ", ".join(f"{x.name}={x.status}" for x in results))
        for x in bad:
            fails.append(f"round {rid} {x.name}: {x.detail}")

    for f in fails:
        _gh("error", f"CONSENSUS DIVERGENCE vs a settled mainnet receipt — {f}")
    if fails:
        print(f"\n{len(fails)} FAIL(s): this change alters what satellites derive for settled "
              "rounds. Gate it behind an activation block (release-then-activate) or fix it.")
        return 1
    if replayed == 0:
        if unreachable == len(picked):
            _gh("warning", "receipt replay NOT verified: no receipt fetchable")
            return 0
        _gh("error", "no receipt was actually re-derived (every one skipped/WARN-only) — the "
                     "replay is blind; fix the replay inputs before trusting it")
        return 1
    print(f"\nOK: {replayed}/{len(picked)} mainnet receipt(s) re-derived identically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
