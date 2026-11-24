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


def test_910_a_node_without_inputs_follows_only_a_verified_settlement(tmp_path):
    """A full validator that does not produce lane D (no inputs): an
    unverified published settlement (its court edited after signing) is not
    followed and lane D's share burns (no court); the verified one is
    followed and the step completes."""
    w = h.wiring()
    cfg, fx, main = _published(tmp_path)
    key = w.report_key(h.B2)
    honest = main.store.objects[key]
    rep = json.loads(honest)
    rep["pay"]["court"][0]["uid"] = h.OTHER_UID
    main.store.objects[key] = json.dumps(rep)
    follower = h.Rig(tmp_path / "follower", cfg, fx, store=main.store,
                     state_dir=tmp_path / "follower_state", with_inputs=False)
    with pytest.raises(Exception) as ei:
        follower.step(h.B2)
    h.assert_wiring_error(ei.value, "inputs_unavailable")
    assert follower.step.court_for(h.B2) is None, "an unverified court was followed"
    main.store.objects[key] = honest
    try:
        got = follower.step(h.B2)
    except Exception as e:  # noqa: BLE001 — base: the step cannot follow
        got = e
    assert got is None, f"the step did not follow the verified settlement: {got!r}"
    assert follower.step.court_for(h.B2) == [h.CHALLENGER_UID]
