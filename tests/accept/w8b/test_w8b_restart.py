"""W8b restart-safety: a re-call after a restart is idempotent (W7c): no
re-score, no re-sweep once the report landed, no double crown, the same
published bytes. Contract: README.md."""

from __future__ import annotations

import pytest

from . import helpers as h


def _rig(tmp_path):
    cfg = h.chain_cfg(tmp_path / "v")
    return h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path))


def test_recall_after_a_clean_settlement_changes_nothing(tmp_path):
    rig = _rig(tmp_path)
    rig.step(h.B1)
    rig.step(h.B2)
    swept = len(rig.host.calls)
    objects = dict(rig.store.objects)
    files = h.dir_bytes(rig.state_dir)
    for _ in range(2):
        rig.restart()
        rig.step(h.B2)
        assert len(rig.host.calls) == swept, "a restart re-swept the bench"
        assert rig.store.objects == objects, "a restart changed a published object"
        assert h.dir_bytes(rig.state_dir) == files
        assert rig.step.court_for(h.B2) == [h.CHALLENGER_UID], \
            "the re-call restores the published court without a re-score"
    rig.step(h.B3)
    rep = rig.published_report(h.B3)
    assert rep["crowned"] is None, "X was crowned twice"
    assert rep["king_after"]["digest"] == rig.fx["x"]


def test_crash_between_report_and_state_adopts_the_persisted_report(tmp_path, monkeypatch):
    import cascade.validator.lane_d as lane_d

    w = h.wiring()
    rig = _rig(tmp_path)
    rig.step(h.B1)
    real = lane_d.save_state

    def crash(path, state):
        raise RuntimeError("w8b: process killed between the report and the state write")

    monkeypatch.setattr(lane_d, "save_state", crash)
    with pytest.raises(Exception):  # noqa: B017 — the crash itself, any type
        rig.step(h.B2)
    monkeypatch.setattr(lane_d, "save_state", real)
    assert rig.report_file(h.B2).is_file(), "fixture: the report landed before the crash"
    persisted = rig.report_file(h.B2).read_bytes()
    assert w.report_key(h.B2) not in rig.store.objects
    assert rig.step.court_for(h.B2) == [None]
    swept = len(rig.host.calls)
    rig.restart()
    rig.step(h.B2)
    assert len(rig.host.calls) == swept, "recovery re-swept the bench (RV7 B1: adopt)"
    assert rig.report_file(h.B2).read_bytes() == persisted
    assert rig.store.objects[w.report_key(h.B2)] == persisted.decode("ascii")
    assert rig.step.court_for(h.B2) == [h.CHALLENGER_UID]
    from cascade.validator.lane_d_state import verify_receipt, verify_receipt_chain

    receipts = [rig.published_receipt(h.B1), rig.published_receipt(h.B2)]
    verify_receipt_chain(receipts)
    verify_receipt(receipts[1], report=rig.published_report(h.B2))


def test_crash_after_state_before_publish_publishes_on_the_next_tick(tmp_path):
    """The state is saved at B2, the publish fails, the process restarts:
    the latest published receipt is still B1's. The next tick publishes B2
    without re-scoring and the receipt chain stays linked."""
    w = h.wiring()
    rig = _rig(tmp_path)
    rig.step(h.B1)
    fresh = h.Rig(tmp_path / "clean", rig.cfg, rig.fx)
    fresh.step(h.B1)
    clean = fresh.step(h.B2)
    rig.store.fail_put = lambda key: key.startswith("lane-d/")
    with pytest.raises(Exception) as ei:
        rig.step(h.B2)
    h.assert_wiring_error(ei.value, "publish_failed")
    assert rig.state_path.read_bytes() == fresh.state_path.read_bytes(), \
        "fixture: the state was saved at B2 before the publish failed"
    rig.store.fail_put = None
    swept = len(rig.host.calls)
    rig.restart()
    rig.step(h.B2)
    assert len(rig.host.calls) == swept
    assert rig.published_receipt(h.B2) == clean.receipt
    assert rig.store.objects[w.RECEIPT_LATEST_KEY] == rig.store.objects[w.receipt_key(h.B2)]
    assert rig.step.court_for(h.B2) == [h.CHALLENGER_UID]
    rig.step(h.B3)
    from cascade.validator.lane_d_state import verify_receipt_chain

    verify_receipt_chain([rig.published_receipt(b) for b in (h.B1, h.B2, h.B3)])


def test_an_older_boundary_after_a_newer_one_never_resettles(tmp_path):
    rig = _rig(tmp_path)
    rig.step(h.B1)
    rig.step(h.B2)
    objects = dict(rig.store.objects)
    files = h.dir_bytes(rig.state_dir)
    try:
        rig.step(h.B1)
    except Exception as e:  # noqa: BLE001 — refused (stale) or a no-op re-publish
        h.assert_wiring_error(e, "settle_refused")
    assert rig.store.objects == objects and h.dir_bytes(rig.state_dir) == files
    assert rig.step.court_for(h.B2) == [h.CHALLENGER_UID]
