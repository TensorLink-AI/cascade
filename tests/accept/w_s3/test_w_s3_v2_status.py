"""W-S3: the sentinel on ``status/v2.json``, the lane A status board (plan §16
rule 4: "on a per-suite board in status/lane_d.json and the lane A status").

* ``build_v2_status(..., sentinel=board)`` carries the era's sentinel board
  (``build_sentinel_board``) as ``doc["sentinel"]``, verbatim; without it the
  document keeps main's bytes.
* ``lane_a_view(receipt)`` names the era's sentinel draw the receipt records
  (W-S1's ``sentinel`` field); a receipt without one keeps main's view.

README.md states the contract."""

from __future__ import annotations

import pytest

from . import helpers as h


@pytest.fixture(autouse=True)
def _card(monkeypatch):
    h.no_torch_model_card(monkeypatch)


def _board(kings=(h.K1, h.K2, h.K3), store=None):
    if store is None:
        store, _ = h.three_kings_store()
    return h.board(h.cfg(), kings=list(kings), store=store)


# ── doc["sentinel"] ─────────────────────────────────────────────────────────


def test_v2_status_unchanged_when_unarmed():
    """No board (W-S1 unarmed: ``build_sentinel_board`` returns None): the
    document is main's, key for key."""
    assert h.v2_doc() == h.GOLDEN_V2_DOC
    doc = h.v2_doc(sentinel=None)
    assert doc == h.GOLDEN_V2_DOC
    assert "sentinel" not in doc


def test_v2_status_armed_nonempty_lane_d_effect():
    b = _board()
    doc = h.v2_doc(sentinel=b)
    assert doc["sentinel"] == b
    assert {k: v for k, v in doc.items() if k != "sentinel"} == h.GOLDEN_V2_DOC
    assert doc["sentinel"]["gating"] is False
    assert [r["king_digest"] for r in doc["sentinel"]["kings"]] == [h.K1, h.K2, h.K3]


def test_v2_status_shows_missing_and_faulty_records_as_such():
    store = h.MemoryStore()
    h.publish_record(store, king=h.K1, previous=h.K3, nan_role="previous_king")
    doc = h.v2_doc(sentinel=_board(kings=(h.K1, h.K2), store=store))
    rows = {r["king_digest"]: r for r in doc["sentinel"]["kings"]}
    assert rows[h.K1]["state"] == "fault"
    assert rows[h.K1]["models"]["previous_king"]["crps"] is None
    assert rows[h.K2]["state"] == "missing"
    assert rows[h.K2]["models"] is None


def test_v2_status_board_is_published_byte_identically():
    from cascade.shared import v2_status

    b = _board()
    texts = []
    for _ in range(2):
        store = h.MemoryStore()
        key = v2_status.publish_v2_status(store, h.v2_doc(sentinel=b))
        assert key == v2_status.V2_STATUS_KEY == "status/v2.json"
        texts.append(store.objects[key])
    assert texts[0] == texts[1]
    assert h.strict_loads(texts[0])["sentinel"] == b


def test_v2_status_board_is_a_copy_not_an_alias():
    """The trainer keeps building documents; a later board must not reach
    into an earlier document."""
    b = _board()
    doc = h.v2_doc(sentinel=b)
    snapshot = h.canonical(doc["sentinel"])
    b["kings"].clear()
    b["suite"] = "changed"
    assert h.canonical(doc["sentinel"]) == snapshot


# ── lane_a_view: the era's sentinel draw from the receipt ───────────────────


def test_lane_a_view_unchanged_when_unarmed():
    from cascade.shared import v2_status

    view = v2_status.lane_a_view(h.LANE_A_RECEIPT)
    assert h.canonical(view) == h.GOLDEN_LANE_A_VIEW
    assert "sentinel" not in view


def test_lane_a_view_armed_nonempty_lane_d_effect():
    from cascade.shared import v2_status

    receipt = h.armed_receipt()
    view = v2_status.lane_a_view(receipt)
    assert "sentinel" in view, "lane_a_view drops the receipt's sentinel draw"
    sen = view["sentinel"]
    assert sen["suite"] == receipt["sentinel"]["suite"] == h.draw(h.era_hash("armed"))
    assert sen["block"] == h.ERA_BLOCK
    assert sen["block_hash"] == h.era_hash("armed")
    rest = {k: v for k, v in view.items() if k != "sentinel"}
    assert h.canonical(rest) == h.GOLDEN_LANE_A_VIEW


def test_both_lanes_name_the_same_era_sentinel():
    from cascade.shared import v2_status

    doc = h.v2_doc(lane_a=v2_status.lane_a_view(h.armed_receipt()), sentinel=_board())
    assert doc["lane_a"]["sentinel"]["suite"] == doc["sentinel"]["suite"]
    assert doc["lane_a"]["sentinel"]["block"] == doc["sentinel"]["era"]["block"]
