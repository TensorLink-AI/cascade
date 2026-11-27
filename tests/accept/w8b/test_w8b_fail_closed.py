"""W8b fail-closed: any unreadable input ⇒ no settlement this tick (retry next
tick), never a guessed input; lane D pay moves only once the report and its
receipt are published. Contract: README.md."""

from __future__ import annotations

import contextlib
import shutil

import pytest

from . import helpers as h

# After a restart the court is either not yet known or the published B1 one
# (whether a fresh step restores it before its first settlement is not pinned).
RESTARTED = (None, [None])


def _frozen(rig) -> tuple:
    return (h.dir_bytes(rig.state_dir), dict(rig.store.objects), len(rig.host.calls))


def _refused(rig, boundary: int, code: str, *, court_before) -> None:
    before = _frozen(rig)
    with pytest.raises(Exception) as ei:
        rig.step(boundary)
    h.assert_wiring_error(ei.value, code)
    assert _frozen(rig) == before, "a refused tick wrote a file, published or swept the bench"
    allowed = court_before if isinstance(court_before, tuple) else (court_before,)
    assert rig.step.court_for(boundary) in allowed


def _healthy_retry(rig) -> None:
    rig.step(h.B2)
    assert rig.published_report(h.B2)["crowned"] == rig.fx["x"]
    assert rig.step.court_for(h.B2) == [h.CHALLENGER_UID]


def _after_b1(tmp_path, **kw):
    cfg = h.chain_cfg(tmp_path / "v", sentinel=kw.pop("sentinel", False))
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path), **kw)
    rig.step(h.B1)
    assert rig.step.court_for(h.B1) == [None]
    return rig


# ── unreadable chain inputs ─────────────────────────────────────────────────


def test_boundary_hash_unreadable_is_no_settlement(tmp_path):
    rig = _after_b1(tmp_path)
    rig.factory.reader.fail = {h.B2}
    _refused(rig, h.B2, "block_hash_unreadable", court_before=None)
    rig.factory.reader.fail = set()
    _healthy_retry(rig)


def test_malformed_boundary_hash_is_never_used(tmp_path):
    rig = _after_b1(tmp_path)
    rig.factory.reader.malformed = {h.B2}
    _refused(rig, h.B2, "block_hash_unreadable", court_before=None)
    rig.factory.reader.malformed = set()
    _healthy_retry(rig)


def test_no_chain_endpoint_is_no_settlement(tmp_path):
    cfg = h.chain_cfg(tmp_path / "v")
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path))
    rig.factory.fail = True
    _refused(rig, h.B1, "block_hash_unreadable", court_before=None)
    rig.factory.fail = False
    rig.step(h.B1)
    assert rig.step.court_for(h.B1) == [None]


def test_era_start_hash_unreadable_with_ws1_armed_is_no_settlement(tmp_path):
    cfg = h.chain_cfg(tmp_path / "v", sentinel=True)
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path))
    rig.factory.reader.fail = {h.ERA_START}
    _refused(rig, h.B1, "era_hash_unreadable", court_before=None)
    rig.factory.reader.fail = set()
    rig.step(h.B1)
    _healthy_retry(rig)


# ── unreadable settlement inputs ────────────────────────────────────────────


def test_inputs_unreadable_is_no_settlement(tmp_path):
    rig = _after_b1(tmp_path)
    rig.inputs.fail = True
    _refused(rig, h.B2, "inputs_unavailable", court_before=None)
    rig.inputs.fail = False
    _healthy_retry(rig)


def test_no_inputs_source_configured_never_settles(tmp_path):
    cfg = h.chain_cfg(tmp_path / "v")
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path), with_inputs=False)
    for _ in range(2):
        _refused(rig, h.B1, "inputs_unavailable", court_before=None)


def test_last_published_receipt_unreadable_is_no_settlement(tmp_path):
    w = h.wiring()
    rig = _after_b1(tmp_path)
    rig.restart()
    rig.store.fail_get = lambda key: key == w.RECEIPT_LATEST_KEY
    _refused(rig, h.B2, "receipt_unreadable", court_before=RESTARTED)
    rig.store.fail_get = None
    _healthy_retry(rig)


def test_malformed_last_published_receipt_is_no_settlement(tmp_path):
    w = h.wiring()
    rig = _after_b1(tmp_path)
    good = rig.store.objects[w.RECEIPT_LATEST_KEY]
    rig.restart()
    rig.store.objects[w.RECEIPT_LATEST_KEY] = good[:-7]
    _refused(rig, h.B2, "receipt_unreadable", court_before=RESTARTED)
    rig.store.objects[w.RECEIPT_LATEST_KEY] = good
    _healthy_retry(rig)


# ── expect_prior_sha256 from the last published receipt (W7c) ──────────────


def _swap_in_foreign_state(tmp_path, rig_store_receipt_from: str):
    """Validator A settled B1 with X listed (crowned there at its first
    verdict, rev6.11 (b)) and published; a different,
    self-consistent B1 state (no entries) replaces A's state dir. With A's
    published receipt the swap is refused; with the foreign settlement's own
    published receipt (the control) it settles."""
    cfg = h.chain_cfg(tmp_path / "v")
    fx = h.fixture(tmp_path)
    a = h.Rig(tmp_path / "a", cfg, fx, inputs=h.Inputs(fx, listed_from=h.B1))
    a.step(h.B1)
    c = h.Rig(tmp_path / "c", cfg, fx, inputs=h.Inputs(fx, entries=[]))
    c.step(h.B1)
    assert a.published_receipt(h.B1)["posterior_state_sha256"] != \
        c.published_receipt(h.B1)["posterior_state_sha256"]
    shutil.rmtree(a.state_dir)
    shutil.copytree(c.state_dir, a.state_dir)
    store = a.store if rig_store_receipt_from == "a" else c.store
    return h.Rig(tmp_path / "a", cfg, fx, store=store, state_dir=a.state_dir)


def test_state_not_bound_to_the_last_published_receipt_is_refused(tmp_path):
    from cascade.validator.lane_d_state import LaneDStateError

    rig = _swap_in_foreign_state(tmp_path, "a")
    before = _frozen(rig)
    with pytest.raises(Exception) as ei:
        rig.step(h.B2)
    h.assert_wiring_error(ei.value, "settle_refused")
    cause = ei.value.__cause__
    assert isinstance(cause, LaneDStateError) and cause.code == "state_digest_mismatch", cause
    assert _frozen(rig) == before
    assert rig.step.court_for(h.B2) in RESTARTED


def test_state_bound_to_the_last_published_receipt_settles(tmp_path):
    """The control: the same swapped state with its own published receipt."""
    rig = _swap_in_foreign_state(tmp_path, "c")
    rig.step(h.B2)
    assert h.wiring().report_key(h.B2) in rig.store.objects


def test_tampered_state_file_is_refused(tmp_path):
    rig = _after_b1(tmp_path)
    raw = rig.state_path.read_bytes()
    rig.state_path.write_bytes(raw + b" ")
    _refused(rig, h.B2, "settle_refused", court_before=None)


# ── pay moves only via the published report ─────────────────────────────────


def test_report_publish_failure_withholds_the_new_court(tmp_path):
    w = h.wiring()
    rig = _after_b1(tmp_path)
    rig.store.fail_put = lambda key: key == w.report_key(h.B2)
    with pytest.raises(Exception) as ei:
        rig.step(h.B2)
    h.assert_wiring_error(ei.value, "publish_failed")
    assert rig.step.court_for(h.B2) is None, "pay moved without a published report"
    assert w.receipt_key(h.B2) not in rig.store.objects
    swept = len(rig.host.calls)
    rig.store.fail_put = None
    rig.step(h.B2)                    # the next tick: publish only, nothing re-swept
    assert len(rig.host.calls) == swept
    assert rig.step.court_for(h.B2) == [h.CHALLENGER_UID]
    assert rig.store.objects[w.report_key(h.B2)] == \
        rig.report_file(h.B2).read_text(encoding="ascii")


def test_receipt_publish_failure_withholds_the_new_court(tmp_path):
    w = h.wiring()
    rig = _after_b1(tmp_path)
    b1_latest = rig.store.objects[w.RECEIPT_LATEST_KEY]
    rig.store.fail_put = lambda key: key == w.receipt_key(h.B2)
    with pytest.raises(Exception) as ei:
        rig.step(h.B2)
    h.assert_wiring_error(ei.value, "publish_failed")
    assert rig.step.court_for(h.B2) is None
    assert rig.store.objects[w.RECEIPT_LATEST_KEY] == b1_latest
    rig.store.fail_put = None
    rig.step(h.B2)
    assert rig.step.court_for(h.B2) == [h.CHALLENGER_UID]


def test_status_publish_failure_never_withholds_pay(tmp_path):
    from cascade.validator.lane_d_status import LANE_D_STATUS_KEY

    rig = _after_b1(tmp_path)
    rig.store.fail_put = lambda key: key == LANE_D_STATUS_KEY
    # raising (a retry) or not is the builder's choice
    with contextlib.suppress(Exception):
        rig.step(h.B2)
    assert rig.step.court_for(h.B2) == [h.CHALLENGER_UID], \
        "status/lane_d.json is telemetry: its failure never blocks the paid court"


def test_bench_host_unreachable_never_crowns(tmp_path):
    rig = _after_b1(tmp_path)
    rig.bench.fail = True
    before = h.dir_bytes(rig.state_dir)
    try:
        rig.step(h.B2)
    except Exception:  # noqa: BLE001 — no settlement this tick is allowed
        assert h.dir_bytes(rig.state_dir) == before
    else:
        assert rig.published_report(h.B2)["crowned"] is None
    assert rig.step.court_for(h.B2) == [None]
