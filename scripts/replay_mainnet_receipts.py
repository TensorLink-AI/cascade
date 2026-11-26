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


# Settled receipts KNOWN to lack the emission record because the signing
# validator ran pre-ab0b412 code (cascade/validator/weights.py stamped
# ``_last_emission`` only AFTER an accepted push; a settlement push that hit the
# 100-block weight rate limit shipped a receipt with a king check but no
# ``emission`` block, so the audit cannot read the reign clock). The receipt is
# genuinely defective, the audit rule is right, and satellites' cascade-audit
# still reports FAIL on it. Here (CI only) such a receipt is re-derived with the
# reign clock taken from the PRECEDING settlement's signed receipt, which must
# verify exactly as pinned below; only a PASS of that replay downgrades the FAIL
# to a WARN. Any other king-check FAIL — or this one with a different detail —
# still fails. Never add an entry without the validator log line proving the
# rejected push and the preceding receipt's clock.
NO_CLOCK = "no reign clock recorded on the ladder (due-ness not derivable)"
KNOWN_UNSTAMPED_EMISSION: dict[str, dict] = {
    # 2026-11-22 13:35:03 era-2656 start settlement (boundary 9561600): the
    # 13:19:17 weight re-assert landed 78 blocks earlier, so the settlement
    # push was rejected ("rate limit is 100 blocks") and the receipt shipped
    # without `emission`. The preceding settlement (9560700, round
    # 9550553995117568107, ladder event "none", no king check) records the
    # clock 9547200; nothing between the two could move it.
    "14666242199155766185": {"clock_round": "9550553995117568107",
                             "clock_boundary": 9560700, "reign_start_block": 9547200},
    # 2026-11-24 13:42:50 settlement (boundary 9576000), signed by validator
    # rc37 (70cbc33c, still pre-ab0b412): the 13:25:08 weight re-assert landed
    # 88 blocks earlier, so the push was rejected ("rate limit is 100 blocks")
    # and the receipt shipped a king check (fail, LCB -0.2936 < 0.2000) without
    # `emission`. The clock is borrowed from 9574200 (round
    # 9495195076473738382, ladder event "none", no king check, clock 9561600 =
    # the 2026-11-22 king-check restart). The settlement between them (9575100,
    # round 15040697815483672835) also shipped without `emission` (07:31:33
    # nonce-rejected push) but logged "v2 ladder: none", recorded no king check
    # and no verdict, and kept king_since_block 9531900; no "reign clock
    # restarted" was logged between 2026-11-22 13:35 and 2026-11-24 11:56.
    "13058784948867351860": {"clock_round": "9495195076473738382",
                             "clock_boundary": 9574200, "reign_start_block": 9561600},
}


def _gh(level: str, msg: str) -> None:
    print(f"::{level}::{msg}")


def known_clock_problem(rc, src, pin: dict) -> str | None:
    """Why the pinned clock cannot be borrowed from ``src`` for ``rc`` (None =
    it can). Pure, so the unit tests pin it."""
    em = src.emission or {}
    lad = em.get("ladder") or {}
    if rc.emission is not None:
        return "the receipt carries an emission record (not the unstamped defect)"
    if tuple(rc.weights or ()) != ():
        return "the receipt records a landed weight vector (not a rejected push)"
    if rc.king_check is None:
        return "the receipt records no king check"
    if str(src.round_id) != str(pin["clock_round"]):
        return f"clock receipt {src.round_id} != pinned {pin['clock_round']}"
    if int(src.epoch_start_block) != int(pin["clock_boundary"]) or not (
            int(src.epoch_start_block) < int(rc.epoch_start_block)):
        return (f"clock receipt boundary {src.epoch_start_block} != pinned "
                f"{pin['clock_boundary']} (or not before {rc.epoch_start_block})")
    if lad.get("reign_start_block") is None or int(lad["reign_start_block"]) != int(
            pin["reign_start_block"]):
        return (f"clock receipt reign_start_block {lad.get('reign_start_block')} != pinned "
                f"{pin['reign_start_block']}")
    if lad.get("event") != "none" or src.king_check is not None:
        return "the clock receipt's ladder step could have moved the clock"
    ksb = (rc.state_post or {}).get("king_since_block")
    if ksb is None or int(ksb) != int(em.get("king_since_block", -1)):
        return f"king_since_block {ksb} != the clock receipt's {em.get('king_since_block')}"
    return None


def with_borrowed_clock(rc, src):
    """``rc`` with the emission fields the king check reads (the reign clock
    and the crowning block) taken from ``src`` — for re-derivation only; the
    signed bytes are untouched."""
    from dataclasses import replace

    em = src.emission or {}
    return replace(rc, emission={
        "king_since_block": em.get("king_since_block"),
        "ladder": {"reign_start_block": int(em["ladder"]["reign_start_block"])}})


def _replay_known_unstamped(rc, results, pin, rows, cfg, client):
    """Swap a pinned receipt's ``king-check`` "no reign clock" FAIL for the
    re-derivation under the preceding receipt's verified clock: PASS there ⇒
    WARN here (the defect is still printed); anything else stays FAIL."""
    from cascade.audit.checks import FAIL, KING_CHECK_BENCH, PASS, WARN, CheckResult
    from cascade.audit.main import _fetch_text, audit_receipt, load_receipt

    i = next((k for k, x in enumerate(results)
              if x.name == "king-check" and x.status == FAIL and x.detail == NO_CLOCK), None)
    if i is None:
        return results
    key = next((str(r["receipt_key"]) for r in rows
                if str(r.get("round_id")) == str(pin["clock_round"])), None)
    try:
        if key is None:
            raise LookupError(f"clock receipt {pin['clock_round']} not in the receipt index")
        src = load_receipt(_fetch_text(cfg, key))
    except Exception as e:  # noqa: BLE001 — unverifiable ⇒ the FAIL stands
        why = f"clock receipt unreadable ({e})"
    else:
        why = known_clock_problem(rc, src, pin)
    if why is None:
        # The re-derived row is "king-check" when the bench half replays (or
        # is not needed), or KING_CHECK_BENCH WARN when due-ness, the pool
        # half and the recorded bench means all verify but the signed bench
        # task rows were not fetchable (satellites retry; never a FAIL).
        kc = next((x for x in audit_receipt(with_borrowed_clock(rc, src), cfg, tier=0,
                                            client=client)
                   if x.name in ("king-check", KING_CHECK_BENCH)), None)
        if kc is None:
            why = "borrowed-clock replay produced no king-check row"
        elif (kc.name, kc.status) in (("king-check", PASS), (KING_CHECK_BENCH, WARN)):
            out = list(results)
            out[i] = CheckResult("king-check", WARN,
                                 f"KNOWN DEFECT (pre-ab0b412 validator: rate-limited push, "
                                 f"no emission record): {NO_CLOCK}; re-derived with reign "
                                 f"clock {pin['reign_start_block']} from signed receipt "
                                 f"{pin['clock_round']}: {kc.detail}")
            _gh("warning", f"round {rc.round_id}: {out[i].detail}")
            return out
        else:
            why = f"borrowed-clock replay {kc.name}={kc.status}: {kc.detail}"
    out = list(results)
    out[i] = CheckResult("king-check", FAIL, f"{NO_CLOCK} (known-defect pin rejected: {why})")
    return out


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
    loaded = []
    for r in picked:
        rid = str(r["round_id"])
        try:
            rc = load_receipt(_fetch_text(cfg, str(r["receipt_key"])))
        except Exception as e:  # noqa: BLE001
            unreachable += 1
            print(f"round {rid}: receipt unreadable ({e})")
            continue
        loaded.append(rc)
        results = audit_receipt(rc, cfg, tier=0, client=client)
        pin = KNOWN_UNSTAMPED_EMISSION.get(rid)
        if pin is not None:
            results = _replay_known_unstamped(rc, results, pin, rows, cfg, client)
        bad = [x for x in results if x.status == FAIL]
        ok = any(x.status == PASS and x.name in REPLAYED for x in results)
        replayed += ok
        status = "FAIL" if bad else ("replayed" if ok else "not replayed")
        print(f"round {rid}: {status} — " + ", ".join(f"{x.name}={x.status}" for x in results))
        for x in bad:
            fails.append(f"round {rid} {x.name}: {x.detail}")

    from cascade.shared.era import image_epoch_from

    if image_epoch_from(cfg) and loaded:
        # W-IMG: one image per seed epoch across the replayed settlements
        from cascade.audit.checks import check_image_epoch_chain

        row = check_image_epoch_chain(
            sorted(loaded, key=lambda x: int(x.epoch_start_block)), cfg)
        print(f"image-epoch-chain={row.status} — {row.detail}")
        if row.status == FAIL:
            fails.append(f"image-epoch-chain: {row.detail}")

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
