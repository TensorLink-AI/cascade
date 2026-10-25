"""DEC-CA-0058 satellite × DEC-CA-0055 king check: a main validator that
records a king-check PASS its own recorded rows do not support is caught by
the satellite's tier-0 replay (``king-check``) and the satellite burns.

Reuses the signed-chain fixtures of ``test_satellite_replay``; the settlement
is marked a v2 ladder settlement for the check (``_king_check_live``) so the
rest of the chain fixture stays the pre-v2 one."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cascade.shared.era import min_effective_era
from cascade.shared.receipt import EntryScores, WindowScoreRecord
from cascade.validator import satellite as sat_mod

from .test_satellite_replay import (
    BOUNDARIES,
    DictStore,
    FakeClient,
    _burn,
    _runner,
    make_settlement,
)
from .test_satellite_replay import _no_anon_s3 as _no_anon_s3  # noqa: F401 — autouse
from .test_satellite_replay import scfg as scfg  # noqa: F401 — fixture

pytest.importorskip("bittensor.sp_core")

SIX = {k: 1.0 for k in ("gifteval_crps", "gifteval_mase", "boom_crps", "boom_mase",
                        "time_crps", "time_mase")}


def _forge_pass(cfg, prev_round):
    def tamper(rc):
        king = next(e for e in rc.entry_scores if e.role == "king")
        # the init the check compared against is BETTER than the king
        init = EntryScores("init", king.size, "baseline", -1, tuple(
            replace(s, mase=s.mase * 0.7, qloss_per_q=tuple(q * 0.7 for q in s.qloss_per_q))
            for s in king.scores))
        assert isinstance(init.scores[0], WindowScoreRecord)
        b = int(rc.epoch_start_block)
        kc = {"result": "pass", "round_id": rc.round_id, "block": b,
              "king_hotkey": king.hotkey, "king_pointer": "ckpt", "reasons": [],
              "bench_round": prev_round,
              "warm_start_era": int(min_effective_era(cfg.round, b)),
              "lcb": 0.5, "margin": 0.2, "tenure_rounds": 0,
              "bench": {"king": SIX, "init": {k: 2.0 for k in SIX}, "init_pointer": "init"}}
        emission = {"boundary": b, "king_since_block": None, "policy": {},
                    "ladder": {"prev_bp": None, "bp": None, "event": "carried",
                               "court_hotkey": None, "court_until_block": 0,
                               "slashed": False, "funded_by": "", "reign_start_block": 0}}
        return replace(rc, king_check=kc, king_check_scores=(init,), emission=emission)
    return tamper


def test_forged_king_check_pass_burns(scfg, tmp_path, monkeypatch):  # noqa: F811
    import cascade.audit.checks as checks

    monkeypatch.setattr(checks, "_king_check_live", lambda cfg, boundary: True)
    store = DictStore()
    _, l0 = make_settlement(scfg, store, 0, chal_scale=1.0)
    runner = _runner(scfg, store, tmp_path)
    client = FakeClient(BOUNDARIES[0] + 300)
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "", doc
    rc, _ = make_settlement(scfg, store, 1, prev=l0, chal_scale=1.0,
                            tamper=_forge_pass(scfg, l0["round_id"]))
    row = checks.check_king_check(rc, scfg)
    assert row.status == checks.FAIL and "give fail" in row.detail, row
    client.block = BOUNDARIES[1] + 300
    doc = runner.poll(client)
    assert doc["mode"] == sat_mod.BURNING
    assert doc["last_failing_check"] == "king-check", doc
    assert client.pushed[-1] == _burn(scfg)
    # an unverifiable bench half is "could not verify" (retry), never a pass
    assert "king-check-bench" in sat_mod.UNVERIFIABLE_WARN

