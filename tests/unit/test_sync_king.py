"""scripts/sync_king.py ``_anchor_receipt_king``: the anchor's latest receipt
can be a rejected (king re-sync) or no-contest receipt that names no king —
2026-11-04 it superseded the crowning's scored receipt 21 s after it was
written and king-sync kept the dethroned king. The sync walks the
``prev_round_id`` chain back to the newest scored, anchor-signed receipt."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import sync_king  # noqa: E402

from cascade.shared.config import load_chain_config  # noqa: E402
from cascade.shared.hippius import (  # noqa: E402
    manifest_round_key,
    receipt_latest_key,
    receipt_round_key,
)

ANCHOR = str(load_chain_config(REPO / "chain.toml").manifest.validator_hotkey)
OLD, NEW = "5Gsa19SF" + "O" * 40, "5CMfsDbJ" + "N" * 40
NEW_REF = "cascade/gen-new@sha256:" + "1" * 64


def _crown(rid: str) -> dict:
    return {"status": "scored", "round_id": rid, "epoch_start_block": 9_464_000,
            "verdict": {"king_hotkey": NEW, "king_uid": 220, "dethroned": True},
            "manifest": {"entries": [
                {"role": "king", "miner_hotkey": OLD, "miner_uid": 31,
                 "gen_ref": "cascade/gen-old@sha256:" + "0" * 64},
                {"role": "challenger", "miner_hotkey": NEW, "miner_uid": 220, "gen_ref": NEW_REF}]}}


@pytest.fixture
def verify_all(monkeypatch):
    """Signature checks pass unless the body carries ``"forged"``."""
    import cascade.shared.receipt as rc

    monkeypatch.setattr(rc, "load_receipt", lambda text: text)
    monkeypatch.setattr(rc, "verify_receipt_signature", lambda text, anchor: '"forged"' not in text)


def _run(bucket: dict):
    def get(key):
        if key not in bucket:
            raise FileNotFoundError(key)
        return bucket[key]
    args = SimpleNamespace(cascade_tree=REPO, chain_toml="chain.toml")
    return sync_king._anchor_receipt_king(args, get_text=get)


def test_rejected_latest_walks_back_to_the_crowning_receipt(verify_all):
    resync = {"status": "rejected", "round_id": "17758994360717789943",
              "reject_reason": "king_resyncing"}
    bucket = {
        receipt_latest_key(ANCHOR): json.dumps(resync),
        manifest_round_key("17758994360717789943"): json.dumps({"prev_round_id": "10447861256687716398"}),
        receipt_round_key("10447861256687716398", ANCHOR): json.dumps(_crown("10447861256687716398")),
    }
    k = _run(bucket)
    assert k["hotkey"] == NEW and k["uid"] == 220 and k["gen_ref"] == NEW_REF
    assert k["round_id"] == "10447861256687716398" and k["crowned_here"]


def test_scored_latest_is_used_directly(verify_all):
    k = _run({receipt_latest_key(ANCHOR): json.dumps(_crown("r1"))})
    assert k["hotkey"] == NEW and k["round_id"] == "r1"


def test_no_scored_receipt_found_keeps_current_behaviour(verify_all):
    bucket = {receipt_latest_key(ANCHOR): json.dumps({"status": "rejected", "round_id": "r2"})}
    assert _run(bucket) is None                 # broken chain ⇒ caller keeps the archive
    bucket[manifest_round_key("r2")] = json.dumps({"prev_round_id": "r1"})
    bucket[receipt_round_key("r1", ANCHOR)] = json.dumps({**_crown("r1"), "forged": True})
    assert _run(bucket) is None                 # an unsigned receipt is never trusted


def test_unverified_latest_still_raises(verify_all):
    with pytest.raises(ValueError, match="signature"):
        _run({receipt_latest_key(ANCHOR): json.dumps({**_crown("r1"), "forged": True})})
