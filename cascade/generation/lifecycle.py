"""Storage maintenance from an anchored validator receipt, never dashboard claims."""
from __future__ import annotations

import time

from . import dataset


def apply_verified_receipt(store, public_store, receipt, gcfg, *, now=None):
    """Caller must verify the receipt signature. Ambiguous/missing bindings hold.

    Match source + hotkey + generation seed, not the consumed-prefix digest:
    training may consume less than the full stored corpus.
    """
    verdict = receipt.verdict
    if receipt.status != "scored" or verdict is None or getattr(verdict, "inconclusive", True):
        return []
    rows = dataset.read_dataset_index(store, gcfg.storage_prefix)
    changed = []
    for entry in receipt.manifest.get("entries", []):
        if not isinstance(entry, dict):
            continue
        matches = [r for r in rows if not r.get("expired")
                   and r.get("hotkey") == entry.get("miner_hotkey")
                   and r.get("generator_ref") == entry.get("gen_ref")
                   and r.get("generation_seed") == receipt.generation_seed]
        if len(matches) != 1:
            continue
        row = matches[0]
        if int(row.get("settlement_block") or -1) > int(receipt.epoch_start_block):
            continue
        king = (entry.get("role") == "king" or
                (getattr(verdict, "dethroned", False) and entry.get("miner_hotkey") == verdict.king_hotkey))
        dataset.record_scores(store, row["round_id"], row["hotkey"], prefix=gcfg.storage_prefix,
            verdict="king" if king else "not_crowned", crowned=bool(king or row.get("crowned")),
            settled_at=float(time.time() if now is None else now),
            settlement_round=str(receipt.round_id), settlement_block=int(receipt.epoch_start_block))
        # Crown is durable BEFORE public copies; a failed copy cannot make it
        # eligible for expiry. Idempotent publication is retried on later upkeep.
        if king and row.get("published") != gcfg.king_publish:
            dataset.publish_king_corpus(store, public_store, row["round_id"], row["hotkey"], gcfg)
        changed.append((str(row["round_id"]), row["hotkey"]))
    return changed


def maintain(store, public_store, receipt, gcfg, *, now=None, keep=frozenset()):
    """Expire confirmed settled non-kings only; unresolved work stays available."""
    apply_verified_receipt(store, public_store, receipt, gcfg, now=now)
    rows = dataset.read_dataset_index(store, gcfg.storage_prefix)
    protected = set(keep)
    for row in rows:
        if not row.get("settled_at") or row.get("hotkey") == getattr(receipt.verdict, "king_hotkey", None):
            protected.add((str(row["round_id"]), row["hotkey"]))
        if row.get("crowned") and row.get("published") != gcfg.king_publish:
            dataset.publish_king_corpus(store, public_store, row["round_id"], row["hotkey"], gcfg)
    return dataset.expire_corpora(store, time.time() if now is None else now, gcfg, keep=protected)
