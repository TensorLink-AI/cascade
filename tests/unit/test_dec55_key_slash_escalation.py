"""DEC-CA-0055 escalating king key slash (owner 2026-10-20): the first
operator-paid lapse of a reign holds the slash for ``[v2] key_slash_min_eras``
eras (then until a king-funded leg), a second lapse holds it for the rest of
the reign, a crowning resets it, ``operator_fault`` never counts — derived by
the validator from manifests alone, replayed by the audit and bound across
receipts for satellites. Plus the trainer's persisted failed-key digest."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.audit.checks import check_weights, key_slash_mismatch, ladder_prior_mismatch
from cascade.shared.emission import KeySlash, key_slash_step
from cascade.shared.settlement_state import settlement_state
from cascade.validator.state import ChampionState, apply_forfeit, dumps, loads

from .test_dec55_validator import KING, ROOT, _Client, _manifest, _receipt, _runner

LANE_A = 0.5            # lane A's pool on this config (lane B live)


@pytest.fixture
def env(tmp_path):
    """The DEC-CA-0055 validator fixture (``test_dec55_validator.env``)."""
    from cascade.shared.config import load_chain_config
    from cascade.shared.era import era_length_blocks

    cfg = load_chain_config(ROOT / "chain.toml")
    era = era_length_blocks(cfg.round, 0)
    gate = 10_000 * era
    cfg = replace(cfg, v2=replace(cfg.v2, from_block=gate),
                  training=replace(cfg.training, v2_from_block=gate),
                  lane_b=replace(cfg.lane_b, from_block=gate + 1, share=0.5),
                  validator=replace(cfg.validator, state_db_path=str(tmp_path / "s.json")))
    return cfg, gate, era


class _Chain:
    """Drive one reign settlement by settlement, auditing each receipt and
    binding each ladder prior to the previous receipt's posterior."""

    def __init__(self, cfg, gate, era, *, state=None):
        self.cfg, self.era = cfg, era
        self.b = gate + 20 * era
        self.r = _runner(cfg, state or ChampionState(
            king_hotkey=KING, king_uid=3, king_since_block=gate + era, king_share_bp=9000))
        self.prev_post = settlement_state(self.r.state)

    def settle(self, funded_by, *, step=None):
        b = self.b
        m = _manifest(b, funded_by=funded_by)
        self.r._v2_settlement_economics(m, [], 1, None, KING)
        lad = self.r._settlement_ladder[1]
        self.r._lane_b_block = b
        vec = self.r._apply_weights(_Client(b, {KING: 3}), m.round_id, [3], boundary=b)
        rec = _receipt(self.r, m, b, [3], vec)
        assert check_weights(rec, self.cfg).status == "PASS"
        assert ladder_prior_mismatch(lad, self.prev_post) == ""
        self.prev_post = settlement_state(self.r.state)
        self.b += self.cfg.round.epoch_blocks if step is None else step
        return lad, vec[3]

    def settle_eras(self, funded_by, eras):
        n = int(eras * self.era // self.cfg.round.epoch_blocks)
        return [self.settle(funded_by) for _ in range(n)]


def _paid(vec3, bp):
    return vec3 == pytest.approx(bp / 10000.0 * LANE_A)


# ── the pure step ────────────────────────────────────────────────────────────


def test_step_rules(env):
    cfg, _gate, era = env
    k = cfg.v2.key_slash_min_eras
    assert k == 6
    s = key_slash_step(funded_by="operator", boundary=1000, lapses=0, lapse_open=False,
                       slash_until_block=0, cfg=cfg)
    assert s == KeySlash(1, True, 1000 + k * era, "operator_paid")
    # consecutive operator-paid settlements are ONE lapse
    s2 = key_slash_step(funded_by="operator", boundary=2000, lapses=1, lapse_open=True,
                        slash_until_block=s.slash_until_block, cfg=cfg)
    assert s2.lapses == 1 and s2.reason == "operator_paid"
    # operator_fault (our side) neither opens nor closes a lapse
    f = key_slash_step(funded_by="operator_fault", boundary=1000, lapses=0, lapse_open=False,
                       slash_until_block=0, cfg=cfg)
    assert f == KeySlash()
    # a closed first lapse, then another: rest of the reign
    s3 = key_slash_step(funded_by="operator", boundary=10**9, lapses=1, lapse_open=False,
                        slash_until_block=1, cfg=cfg)
    assert s3.lapses == 2 and s3.reason == "repeat_lapse"


# ── the validator, audited settlement by settlement ──────────────────────────


def test_first_lapse_holds_exactly_min_eras_even_after_rotation(env):
    cfg, gate, era = env
    c = _Chain(cfg, gate, era)
    lad, v = c.settle("operator")
    assert lad["slash_reason"] == "operator_paid" and _paid(v, 100)
    assert lad["slash_until"] == lad["slash_until_block"]
    until = lad["slash_until_block"]
    # rotated to a working key at once: still 1 % inside the window
    n_window = int(cfg.v2.key_slash_min_eras * era // cfg.round.epoch_blocks) - 1
    for _ in range(n_window):
        lad, v = c.settle("king")
        assert c.b - cfg.round.epoch_blocks < until
        assert lad["slash_reason"] == "min_window" and _paid(v, 100)
    # the first boundary at the window's end: lifted (exactly min_eras eras)
    assert c.b == until
    lad, v = c.settle("king")
    assert not lad["slashed"] and lad["slash_reason"] == "" and _paid(v, 9000)
    assert c.r.state.king_key_lapses == 1


def test_first_lapse_without_rotation_stays_slashed_past_the_minimum(env):
    cfg, gate, era = env
    c = _Chain(cfg, gate, era)
    rows = c.settle_eras("operator", cfg.v2.key_slash_min_eras + 2)
    assert all(lad["slashed"] and _paid(v, 100) for lad, v in rows)
    assert rows[-1][0]["slash_reason"] == "operator_paid" and rows[-1][0]["slash_until"] is None
    assert c.r.state.king_key_lapses == 1                 # one long lapse, not many
    lad, v = c.settle("king")                              # fixed: lifts at once
    assert not lad["slashed"] and _paid(v, 9000)


def test_second_lapse_slashes_the_rest_of_the_reign(env):
    cfg, gate, era = env
    c = _Chain(cfg, gate, era)
    c.settle("operator")
    c.settle_eras("king", cfg.v2.key_slash_min_eras)      # window over, key fixed
    assert not c.settle("king")[0]["slashed"]
    lad, _ = c.settle("operator")                          # second lapse
    assert lad["slash_reason"] == "repeat_lapse" and lad["slash_until"] is None
    rows = c.settle_eras("king", cfg.v2.key_slash_min_eras * 3)
    assert all(lad["slash_reason"] == "repeat_lapse" and _paid(v, 100) for lad, v in rows)


def test_dethrone_and_forfeit_reset_the_lapse_count(env):
    cfg, gate, era = env
    c = _Chain(cfg, gate, era, state=ChampionState(
        king_hotkey="5New", king_uid=3, king_since_block=gate + 20 * era, king_slashed=True,
        king_key_lapses=2, king_lapse_open=True, king_slash_until_block=10**12))
    out = SimpleNamespace(transition=SimpleNamespace(dethroned=True, new_king_hotkey="5New"),
                          result=SimpleNamespace(lcb=0.12), lcb_units="increment")
    c.r._v2_settlement_economics(_manifest(c.b, funded_by="operator"), [], 1, out, KING)
    st = c.r.state
    assert (st.king_slashed, st.king_key_lapses, st.king_lapse_open,
            st.king_slash_until_block) == (False, 0, False, 0)
    lad = c.r._settlement_ladder[1]
    assert key_slash_mismatch(lad, cfg, boundary=c.b, funded_by="operator",
                              dethroned=True) == ""
    # the new reign's first lapse is a FIRST lapse again
    f = apply_forfeit(replace(st, king_hotkey=KING, king_key_lapses=2, king_slashed=True),
                      forfeited={KING}, successor="5Succ", crowned_block=c.b)
    assert f.state.king_key_lapses == 0 and not f.state.king_slashed


def test_operator_fault_never_counts(env):
    cfg, gate, era = env
    c = _Chain(cfg, gate, era)
    rows = c.settle_eras("operator_fault", 2)
    assert all(not lad["slashed"] and _paid(v, 9000) for lad, v in rows)
    assert c.r.state.king_key_lapses == 0
    c.settle("operator")
    c.settle("operator_fault")                             # does not close the lapse
    c.settle_eras("king", cfg.v2.key_slash_min_eras)
    lad, _ = c.settle("operator")
    assert lad["slash_reason"] == "repeat_lapse"


def test_min_eras_zero_is_the_old_first_lapse_behaviour(env):
    cfg, gate, era = env
    cfg = replace(cfg, v2=replace(cfg.v2, key_slash_min_eras=0))
    c = _Chain(cfg, gate, era)
    lad, v = c.settle("operator")
    assert lad["slashed"] and _paid(v, 100)
    lad, v = c.settle("king")                              # lifts at once, as before
    assert not lad["slashed"] and _paid(v, 9000)


def test_audit_and_satellite_catch_a_tampered_slash(env):
    cfg, gate, era = env
    c = _Chain(cfg, gate, era)
    c.settle("operator")
    post = c.prev_post
    lad, _ = c.settle("king")
    # a validator that lifted the slash early is caught by the replay
    bad = dict(lad, slashed=False, slash_reason="", slash_until=None)
    assert key_slash_mismatch(bad, cfg, boundary=c.b - cfg.round.epoch_blocks,
                              funded_by="king", dethroned=False)
    # … and one that forged a clean prior is caught by the receipt chain
    forged = dict(lad, prev_key_lapses=0, prev_lapse_open=False, prev_slash_until_block=0)
    assert ladder_prior_mismatch(forged, post)
    assert ladder_prior_mismatch(lad, post) == ""


def test_state_round_trips_and_drops_defaults():
    st = ChampionState(king_hotkey=KING)
    assert "king_key_lapses" not in dumps(st)
    full = replace(st, king_key_lapses=2, king_lapse_open=True, king_slash_until_block=99)
    assert loads(dumps(full)) == full


def test_key_slash_min_eras_is_validated_and_overlay_settable():
    from cascade.shared.config.v2 import parse_v2_economics
    from cascade.shared.netconfig import ALLOWED

    assert parse_v2_economics({})["key_slash_min_eras"] == 6
    assert parse_v2_economics({"key_slash_min_eras": 0})["key_slash_min_eras"] == 0
    for bad in (-1, "6", 1.5, True):
        with pytest.raises(ValueError):
            parse_v2_economics({"key_slash_min_eras": bad})
    assert "key_slash_min_eras" in ALLOWED["v2"] and "key_slash_bp" in ALLOWED["v2"]


# ── the trainer: the failed-key digest survives a restart ────────────────────


def test_failed_key_digest_survives_a_trainer_restart(tmp_path, monkeypatch):
    import requests

    from cascade.trainer.rolling import RollingState, load_state, save_state
    from cascade.trainer.runner_rolling import _RollingMixin

    calls = []
    monkeypatch.setattr(requests, "get", lambda url, headers=None, timeout=None: (
        calls.append(headers["X-API-KEY"]) or SimpleNamespace(ok=True)))
    path = tmp_path / "era_state.json"

    def runner(key):
        state = load_state(path)
        sched = SimpleNamespace(state=state, _lock=threading.RLock(),
                                _save=lambda: save_state(path, state))
        r = SimpleNamespace(_rolling_sched=sched)
        r._payer_vault = lambda: SimpleNamespace(get_shared=lambda hk: key)
        return r

    save_state(path, RollingState())
    assert "king_failed_keys" not in path.read_text()       # dropped when empty
    _RollingMixin._note_king_key_failed(runner("old-key"), "hkK")
    assert load_state(path).king_failed_keys == {"hkK": hashlib.sha256(b"old-key").hexdigest()}
    # a restart: the same failed key still cannot lift the slash
    assert not _RollingMixin._rolling_king_key_ok(runner("old-key"), "hkK")
    assert calls == []
    # a rotated working key lifts it and clears the persisted digest
    assert _RollingMixin._rolling_king_key_ok(runner("new-key"), "hkK")
    assert load_state(path).king_failed_keys == {}
    assert "king_failed_keys" not in path.read_text()
