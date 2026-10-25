"""DEC-CA-0058 receipt stamps written by the full validator: the judged
manifest's digest and the settlement state before / after — absent (receipt
bytes unchanged) while ``[netconfig] owner_hotkey`` is empty."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from cascade.shared.receipt import dump_receipt, load_receipt
from cascade.shared.settlement_state import paid_hotkeys, settlement_state, state_digest
from cascade.validator.receipts import ReceiptMixin
from cascade.validator.state import ChampionState

from .receipt_fixture import make_scored_receipt


class _Runner(ReceiptMixin):
    def __init__(self, cfg, state):
        self.cfg, self.state, self._settlement_prior = cfg, state, None


def _owned(cfg):
    return replace(cfg, netconfig=replace(cfg.netconfig, owner_hotkey="5Main"))


def test_no_stamps_while_inert(cfg):
    r = _Runner(cfg, ChampionState(king_hotkey="k", last_handled_round_id="1",
                                   last_handled_manifest_sha="ab" * 32))
    r._capture_settlement_prior("1")
    assert r._settlement_prior is None
    assert r._satellite_receipt_kw(SimpleNamespace(round_id="1")) == {}


def test_stamps_prior_and_post(cfg):
    before = ChampionState(king_hotkey="old", tenure_rounds=3, former_kings=("older",))
    r = _Runner(_owned(cfg), before)
    r._capture_settlement_prior("7")
    r.state = replace(before, king_hotkey="new", tenure_rounds=0,
                      former_kings=("old", "older"), last_handled_round_id="7",
                      last_handled_manifest_sha="cd" * 32)
    kw = r._satellite_receipt_kw(SimpleNamespace(round_id="7"))
    assert kw["manifest_sha256"] == "cd" * 32
    assert kw["state_prior_sha256"] == state_digest(settlement_state(before))
    assert kw["state_post"]["king"] == "new"
    assert paid_hotkeys(kw["state_post"]) == {"new", "old", "older"}
    # another round's prior is never stamped
    assert "state_prior_sha256" not in r._satellite_receipt_kw(SimpleNamespace(round_id="8"))


def test_fields_round_trip_and_drop_when_default(cfg):
    receipt, _, _ = make_scored_receipt(cfg)
    assert "manifest_sha256" not in dump_receipt(receipt)          # bytes unchanged
    stamped = replace(receipt, manifest_sha256="ab" * 32, state_prior_sha256="cd" * 32,
                      state_post={"king": "k", "former_kings": []})
    back = load_receipt(dump_receipt(stamped))
    assert (back.manifest_sha256, back.state_prior_sha256, back.state_post) == (
        "ab" * 32, "cd" * 32, {"king": "k", "former_kings": []})
    assert back.canonical_body() == stamped.canonical_body()
