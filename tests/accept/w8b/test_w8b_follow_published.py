"""#910 (testnet b274626d, 2026-11-24): a node rebuilt from EMPTY lane D
state. The main validator's lost ``lane_d_state.json`` made every lane D
settlement refuse (``state_digest_mismatch``); a node that does not produce
lane D (no inputs) never had a court at all.

Chosen fix (documented in docs/VALIDATOR.md "Lane D: rebuilt from empty
state"): the step never invents state. A node that cannot settle a boundary
itself pays the court of the SIGNED PUBLISHED settlement at that boundary
when the audit's lane D checker verifies it (as a satellite does);
otherwise lane D's share burns, lanes A/B/C unchanged. Producing new
settlements after a state loss still needs the operator restore
(``restore_state``), which fails closed per lane meanwhile."""

from __future__ import annotations

import json

import pytest

from . import helpers as h

pytest.importorskip("bittensor.sp_core")


def _published(tmp_path):
    cfg = h.chain_cfg(tmp_path / "v")
    fx = h.fixture(tmp_path)
    main = h.Rig(tmp_path / "main", cfg, fx)
    main.step(h.B1)
    main.step(h.B2)
    assert main.step.court_for(h.B2) == [h.CHALLENGER_UID], "fixture: X holds KING_D at B2"
    return cfg, fx, main


def test_910_rebuilt_from_empty_state_follows_the_published_court(tmp_path):
    cfg, fx, main = _published(tmp_path)
    objects = dict(main.store.objects)
    rebuilt = h.Rig(tmp_path / "rebuilt", cfg, fx, store=main.store,
                    state_dir=tmp_path / "empty")
    with pytest.raises(Exception) as ei:
        rebuilt.step(h.B2)
    h.assert_wiring_error(ei.value, "settle_refused")          # never genesis
    assert rebuilt.step.court_for(h.B2) == [h.CHALLENGER_UID], \
        "the rebuilt node does not pay the signed published court"
    assert main.store.objects == objects, "following published something"
    assert not (tmp_path / "empty" / "lane_d_state.json").exists(), \
        "following wrote a lane D state"


def test_910_a_node_without_inputs_follows_and_completes(tmp_path):
    cfg, fx, main = _published(tmp_path)
    follower = h.Rig(tmp_path / "follower", cfg, fx, store=main.store,
                     state_dir=tmp_path / "follower_state", with_inputs=False)
    assert follower.step(h.B2) is None
    assert follower.step.court_for(h.B2) == [h.CHALLENGER_UID]


@pytest.mark.parametrize("tamper", ["court_uid", "unsigned_body"])
def test_910_an_unverified_published_settlement_is_not_followed(tmp_path, tamper):
    """A published report the audit refuses (court edited after signing):
    nothing is followed, lane D's share burns (empty court)."""
    w = h.wiring()
    cfg, fx, main = _published(tmp_path)
    rep = json.loads(main.store.objects[w.report_key(h.B2)])
    if tamper == "court_uid":
        rep["pay"]["court"][0]["uid"] = h.OTHER_UID
    else:
        rep["seed"] = "lane-d/1:0:" + "0" * 64
    main.store.objects[w.report_key(h.B2)] = json.dumps(rep)
    follower = h.Rig(tmp_path / "follower", cfg, fx, store=main.store,
                     state_dir=tmp_path / "follower_state", with_inputs=False)
    with pytest.raises(Exception) as ei:
        follower.step(h.B2)
    h.assert_wiring_error(ei.value, "inputs_unavailable")
    assert follower.step.court_for(h.B2) is None
    assert follower.step.court_for(h.B1) is None


def test_910_nothing_published_burns_lane_d_only(tmp_path):
    """Empty state and nothing published at the boundary: no court (lane D's
    share burns through the weight fold's empty court)."""
    cfg = h.chain_cfg(tmp_path / "v")
    fx = h.fixture(tmp_path)
    rig = h.Rig(tmp_path / "rig", cfg, fx, with_inputs=False)
    with pytest.raises(Exception) as ei:
        rig.step(h.B1)
    h.assert_wiring_error(ei.value, "inputs_unavailable")
    assert rig.step.court_for(h.B1) is None
