"""W8b weights: lane D's pay reaches the vector (W8 emission) only via the
published report; no published court ⇒ lane D's share burns and lanes A/B/C
are byte-identical; unarmed ⇒ byte-identical vectors and receipts.
Contract: README.md."""

from __future__ import annotations

import pytest

from . import helpers as h

AT_B2 = h.B2 + 5                  # a re-assert inside the B2 settlement (boundary B2)


def _nonzero(vec) -> dict[int, float]:
    return {i: w for i, w in enumerate(vec) if w}


# ── invariant 1: unarmed is byte-identical (goldens recorded on main) ───────


def test_inv1_weights_and_receipt_unchanged_when_unarmed(tmp_path):
    for block, receipt_sha in h.UNARMED_RECEIPT_SHA256.items():
        for step in (None, h.CourtStep([h.CHALLENGER_UID])):
            cfg = h.shipped_cfg(tmp_path / f"v{block}")
            vec, r = h.vector_at(cfg, block, step=step)
            assert h.vector_sha256(vec) == h.UNARMED_VECTOR_SHA256, _nonzero(vec)
            assert h.unarmed_receipt_sha256(r, vec, block) == receipt_sha
            if step is not None:
                assert step.court_calls == [] and step.step_calls == [], \
                    "an unarmed weight set consulted lane D"
    assert not list(tmp_path.rglob("lane_d*")), "unarmed wrote a lane D file"


def test_inv1_weights_and_receipt_armed_nonempty_lane_d_effect(tmp_path):
    cfg = h.chain_cfg(tmp_path / "v")
    step = h.CourtStep([h.CHALLENGER_UID])
    vec, r = h.vector_at(cfg, AT_B2, step=step)
    assert h.vector_sha256(vec) != h.UNARMED_VECTOR_SHA256
    assert vec[h.CHALLENGER_UID] == pytest.approx(0.25, rel=1e-12)
    assert [b for b, _ in step.court_calls] == [h.B2]
    assert h.unarmed_receipt_sha256(r, vec, AT_B2) != h.UNARMED_RECEIPT_SHA256[h.ARM_BLOCK + 1805]


# ── the fold (lanes B and C off: the RV3 note 10 trap) ──────────────────────


@pytest.mark.parametrize("court", [[h.CHALLENGER_UID], [None], [h.CHALLENGER_UID, h.OTHER_UID],
                                   [None, h.OTHER_UID]],
                         ids=["king", "house", "king+prior", "house+prior"])
def test_lane_d_court_folds_exactly_as_w8_specifies(tmp_path, court):
    cfg = h.chain_cfg(tmp_path / "v", lanes_bc=False)
    vec, r = h.vector_at(cfg, AT_B2, step=h.CourtStep(court))
    want = h.want_vector(r, cfg, h.B2, court)
    assert h.close(vec, want), (_nonzero(vec), _nonzero(want))
    assert sum(vec) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("case", ["no_step", "no_court", "court_raises", "plain_callable"])
def test_no_published_court_burns_lane_d_share(tmp_path, case):
    cfg = h.chain_cfg(tmp_path / "v", lanes_bc=False)
    step = {"no_step": None, "no_court": h.CourtStep(None),
            "court_raises": h.CourtStep(raises=RuntimeError("w8b: lane D state unreadable")),
            "plain_callable": (lambda boundary: None)}[case]
    vec, r = h.vector_at(cfg, AT_B2, step=step)
    want = h.want_vector(r, cfg, h.B2, [])
    assert h.close(vec, want), (_nonzero(vec), _nonzero(want))
    assert vec[cfg.scoring.burn_uid] > 0.25


def test_lanes_abc_byte_identical_whatever_lane_d_pays(tmp_path):
    """The shipped lanes (C live) under the shipped policy: only lane D's own
    0.25 moves between the court and the burn slot."""
    cfg = h.chain_cfg(tmp_path / "v")
    paid, _ = h.vector_at(cfg, AT_B2, step=h.CourtStep([h.CHALLENGER_UID]))
    burnt, _ = h.vector_at(cfg, AT_B2, step=h.CourtStep(None))
    burn = cfg.scoring.burn_uid
    for uid, (p, b) in enumerate(zip(paid, burnt, strict=True)):
        if uid not in (burn, h.CHALLENGER_UID):
            assert p == b, f"uid {uid}: lane A/B/C weight moved with lane D ({p!r} != {b!r})"
    assert paid[h.CHALLENGER_UID] == pytest.approx(0.25, rel=1e-12)
    assert burnt[h.CHALLENGER_UID] == 0.0
    assert burnt[burn] - paid[burn] == pytest.approx(0.25, rel=1e-12)


def test_court_is_read_for_the_vectors_boundary_on_the_poll_thread(tmp_path):
    import threading

    cfg = h.chain_cfg(tmp_path / "v")
    for block, boundary in ((AT_B2, h.B2), (h.B3 + 17, h.B3), (h.B1 + 899, h.B1)):
        step = h.CourtStep([h.CHALLENGER_UID])
        h.vector_at(cfg, block, step=step)
        assert step.court_calls, f"block {block}: the court was never read"
        assert set(step.court_calls) == {(boundary, threading.get_ident())}


def test_inactive_boundary_is_byte_identical_to_unarmed(tmp_path):
    """Armed from ARM_BLOCK: a vector shaped at boundary ARM_BLOCK (the
    era before) carries no lane D part."""
    step = h.CourtStep([h.CHALLENGER_UID])
    armed, _ = h.vector_at(h.chain_cfg(tmp_path / "a"), h.ARM_BLOCK + 5, step=step)
    unarmed, _ = h.vector_at(h.shipped_cfg(tmp_path / "u"), h.ARM_BLOCK + 5)
    assert h.vector_sha256(armed) == h.vector_sha256(unarmed)
    assert step.court_calls == []


def test_force_burn_armed_is_the_pure_burn(tmp_path):
    armed, _ = h.vector_at(h.chain_cfg(tmp_path / "a", force_burn=True), AT_B2,
                           step=h.CourtStep([h.CHALLENGER_UID]))
    unarmed_cfg = h.shipped_cfg(tmp_path / "u")
    from dataclasses import replace

    unarmed_cfg = replace(unarmed_cfg, validator=replace(unarmed_cfg.validator, force_burn=True))
    unarmed, _ = h.vector_at(unarmed_cfg, AT_B2)
    assert h.vector_sha256(armed) == h.vector_sha256(unarmed)
    assert _nonzero(armed) == {unarmed_cfg.scoring.burn_uid: 1.0}


# ── end to end: the published report is the only source ─────────────────────


def _settled_rig(tmp_path, **kw):
    cfg = h.chain_cfg(tmp_path / "v", lanes_bc=False)
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path), **kw)
    rig.step(h.B1)
    return rig


def test_weights_pay_the_published_court_and_the_audit_rederives_them(tmp_path):
    from dataclasses import replace

    from tests.unit.receipt_fixture import make_scored_receipt

    rig = _settled_rig(tmp_path)
    rig.step(h.B2)
    vec, r = h.vector_at(rig.cfg, AT_B2, step=rig.step)
    want = h.want_vector(r, rig.cfg, h.B2, [h.CHALLENGER_UID])
    assert h.close(vec, want), (_nonzero(vec), _nonzero(want))
    pytest.importorskip("bittensor.sp_core")
    from cascade.audit.checks import derive_weights

    receipt, _, _ = make_scored_receipt(None)
    receipt = replace(receipt, epoch_start_block=h.B2, reward_uids=(h.KING_UID,),
                      weights=tuple(vec), emission=r._last_emission, lane_b_pay=None,
                      lane_c_pay=None, lane_b=None, lane_c=None)
    record = {"report": rig.published_report(h.B2), "receipt": rig.published_receipt(h.B2)}
    got, why = derive_weights(receipt, rig.cfg, lane_d=record)
    assert why == "" and h.close(got, vec, rel=1e-9), why


def test_an_unpublished_settlement_never_reaches_the_weights(tmp_path):
    w = h.wiring()
    rig = _settled_rig(tmp_path)
    rig.store.fail_put = lambda key: key == w.report_key(h.B2)
    with pytest.raises(Exception):  # noqa: B017 — publish_failed (test_w8b_fail_closed)
        rig.step(h.B2)
    vec, r = h.vector_at(rig.cfg, AT_B2, step=rig.step)
    assert vec[h.CHALLENGER_UID] == 0.0, "X's crown reached the weights before its report"
    assert h.close(vec, h.want_vector(r, rig.cfg, h.B2, [None]))
    rig.store.fail_put = None
    rig.step(h.B2)
    vec, r = h.vector_at(rig.cfg, AT_B2, step=rig.step)
    assert h.close(vec, h.want_vector(r, rig.cfg, h.B2, [h.CHALLENGER_UID]))
