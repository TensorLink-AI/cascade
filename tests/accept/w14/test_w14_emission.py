"""W14: the lane D-only burn (`cascade/shared/emission.py`, `lane_parts`).

Plan W14 row: "a lane D-only burn"; §5: "Stop publishing ⇒ only lane D
burns". An absent settlement burns lane D's whole share to `burn_uid`
whatever court the state holds; lanes A, B and C are unchanged byte for
byte. README.md "The lane D-only burn".
"""

from __future__ import annotations

import pytest

from tests.accept.w8 import helpers as w8

from .helpers import has_param

N = 12
D_COURT = (9, 10, 11)       # disjoint from lane A (3, 4), B (1, 2) and C (5, 7)
REL = 1e-12


def _lane_parts():
    from cascade.shared.emission import lane_parts

    return lane_parts


def _parts(*, d_cfg=None, d_court=(), lanes_bc=False, **extra):
    kw = dict(extra)
    if lanes_bc:
        kw.update(lane_b_cfg=w8.lane_b_cfg(), wins=w8.full_b_wins(),
                  lane_c_cfg=w8.lane_c_cfg(), c_wins=w8.full_c_wins())
    if d_cfg is not None:
        kw.update(lane_d_cfg=d_cfg, d_court_uids=tuple(d_court))
    return _lane_parts()(list(w8.LANE_A_COURT), N, king_decay=w8.LANE_A_KING_DECAY,
                         burn_uid=w8.BURN_UID, **kw)


def _require_flag():
    assert has_param(_lane_parts(), "d_absent"), "lane_parts takes no d_absent keyword"


# ── invariant 1 ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("lanes_bc", [False, True])
def test_inv1_weights_unchanged_when_unarmed(lanes_bc):
    """No lane D (``lane_d_cfg=None``): the flag changes nothing, and the
    default is today's vector."""
    base = _parts(lanes_bc=lanes_bc)
    if has_param(_lane_parts(), "d_absent"):
        for flag in (False, True):
            a, b = _parts(lanes_bc=lanes_bc, d_absent=flag)
            assert w8.as_bytes(a) == w8.as_bytes(base[0])
            assert w8.as_bytes(b) == w8.as_bytes(base[1])
        a, b = _parts(lanes_bc=lanes_bc, d_cfg=w8.lane_d_cfg(), d_court=D_COURT,
                      d_absent=False)
        p = _parts(lanes_bc=lanes_bc, d_cfg=w8.lane_d_cfg(), d_court=D_COURT)
        assert (w8.as_bytes(a), w8.as_bytes(b)) == (w8.as_bytes(p[0]), w8.as_bytes(p[1]))


@pytest.mark.parametrize("lanes_bc", [False, True])
def test_inv1_weights_armed_nonempty_lane_d_effect(lanes_bc):
    _require_flag()
    cfg = w8.lane_d_cfg()
    pa, pb = _parts(lanes_bc=lanes_bc, d_cfg=cfg, d_court=D_COURT)
    aa, ab = _parts(lanes_bc=lanes_bc, d_cfg=cfg, d_court=D_COURT, d_absent=True)
    # Lane A: byte-identical (lane D's share is still carved from it).
    assert w8.as_bytes(aa) == w8.as_bytes(pa)
    # Lane D's court is paid nothing; every other non-burn uid is unchanged.
    for u in range(N):
        if u in D_COURT:
            assert pb[u] > 0.0 and ab[u] == 0.0, u
        elif u != w8.BURN_UID:
            assert w8.as_bytes([ab[u]]) == w8.as_bytes([pb[u]]), u
    # The whole share lands on burn_uid.
    assert ab[w8.BURN_UID] == pytest.approx(pb[w8.BURN_UID] + float(cfg.share), rel=REL)
    assert sum(aa) + sum(ab) == pytest.approx(1.0, rel=REL)


# ── whatever the court ──────────────────────────────────────────────────────


@pytest.mark.parametrize("court", [D_COURT, (9,), (None, 10), ()])
def test_absent_burns_lane_d_whatever_the_court(court):
    _require_flag()
    cfg = w8.lane_d_cfg()
    want = _parts(lanes_bc=True, d_cfg=cfg, d_court=(), d_absent=True)
    got = _parts(lanes_bc=True, d_cfg=cfg, d_court=court, d_absent=True)
    assert (w8.as_bytes(got[0]), w8.as_bytes(got[1])) == (w8.as_bytes(want[0]),
                                                          w8.as_bytes(want[1]))


def test_absent_lane_d_part_alone_is_the_share_on_burn():
    """Lanes B and C off: the second element is exactly lane D's share on
    burn_uid and nothing else."""
    _require_flag()
    _, b = _parts(d_cfg=w8.lane_d_cfg(), d_court=D_COURT, d_absent=True)
    assert b[w8.BURN_UID] == pytest.approx(0.25, rel=REL)
    assert all(b[u] == 0.0 for u in range(N) if u != w8.BURN_UID)


def test_absent_keeps_lane_a_through_the_emission_policy():
    """The policy (ladder, cap, slash) shapes lane A only: lane A's uids get
    byte-identical weight whether lane D is present or absent."""
    from cascade.shared.emission import EmissionPolicy

    _require_flag()
    cfg = w8.lane_d_cfg()
    present = _parts(lanes_bc=True, d_cfg=cfg, d_court=D_COURT)
    gone = _parts(lanes_bc=True, d_cfg=cfg, d_court=D_COURT, d_absent=True)
    for policy in (EmissionPolicy(), EmissionPolicy(king_share=0.8, former_share=0.1),
                   EmissionPolicy(lane_a_cap=0.5), EmissionPolicy(king_slash=0.3)):
        vp, va = w8.vector(*present, policy), w8.vector(*gone, policy)
        for u in w8.LANE_A_COURT + (1, 2, 5, 7):
            assert w8.as_bytes([va[u]]) == w8.as_bytes([vp[u]]), (policy, u)
        assert all(va[u] == 0.0 for u in D_COURT)
        assert sum(va) == pytest.approx(sum(vp), rel=REL)
