"""W8b satellite path, #790 (testnet E2E 2026-11-23): a satellite's
activation startup must compute and signal the DEC-CA-0068 shared-ladder
vote, exactly like the full validator.

`ActivationMixin` reads and writes `self._shared_ladder` /
`self._shared_ladder_tally`. The full validator (`cascade/validator/loop.py`)
declares both; on main `SatelliteRunner` does not, so its startup raised
`AttributeError: 'SatelliteRunner' object has no attribute '_shared_ladder'`
and the satellite never signalled (nor tracked) the shared-ladder vote: a
consensus-divergence risk once the vote locks in.

The chain is a fake (validator stakes and plain commitments only); nothing
here imports a PATHS module at import time. No torch, no bittensor.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
PRIMARY = "rolling-era-king"
B0 = 9046800 + 3600
HORIZONS = (64, 256, 720)


def _cfg():
    """The shipped chain.toml with only the shared-ladder vote open: rolling
    eras typed (3600 -> 900 grid at B0), the other on-chain votes disarmed,
    the horizon ladder and the shared count set, no typed ladder block (so
    the rule is decided on chain)."""
    from cascade.shared.config import load_chain_config

    c = load_chain_config(REPO / "chain.toml")
    c = replace(c, scoring=replace(c.scoring, forfeit_hotkeys=(), forfeit_from_block=0,
                                   forfeit_successor_hotkey="", margin_v2_from_block=0,
                                   win_margin_start_v2=0.0, win_margin_end_v2=0.0,
                                   margin_warmup_blocks_v2=0),
                v2=replace(c.v2, prelaunch_burn=0.0, launch_at_utc="", from_block=0),
                eval=replace(c.eval, scored_horizons=HORIZONS, scored_from_block=B0,
                             shared_ladder_windows=40, shared_ladder_from_block=0))
    return replace(c, round=replace(c.round, rolling_from_block=B0, epoch_blocks=900,
                                    epoch_blocks_prev=3600, epoch_activation_block=B0),
                   scoring=replace(c.scoring, era_king_from_block=B0,
                                   tenure_blocks_from_block=B0,
                                   cohort_maxt_increment_from_block=B0))


class FakeChain:
    def __init__(self, validators, signals, *, block, hotkey="v1"):
        self.validators = list(validators)
        self.signals = dict(signals)
        self.block = block
        self.hotkey = hotkey
        self.written: list[str] = []

    def current_block(self):
        return self.block

    def validator_stakes(self, block=None):
        return list(self.validators)

    def read_plain_commitments(self, block=None):
        return dict(self.signals)

    def set_plain_commitment(self, payload):
        self.written.append(payload)
        self.signals[self.hotkey] = payload

    def hotkey_ss58(self):
        return self.hotkey


def _fleet(*stakes):
    import cascade.shared.activation as A

    return [A.ValidatorStake(hotkey=f"v{i}", stake=s, permit=True, last_update=0)
            for i, s in enumerate(stakes, 1)]


def _satellite(cfg, tmp_path):
    import cascade.shared.activation as A
    from cascade.validator.satellite import SatelliteRunner

    return SatelliteRunner(cfg=cfg, state_dir=tmp_path,
                           activation_store=A.ActivationStore(tmp_path / "activation_state.json"))


def test_satellite_declares_every_activation_record_the_validator_declares():
    """Every ActivationRecord field (and its tally) of the full validator is a
    SatelliteRunner field too, so no future vote is missing on satellites."""
    from cascade.shared.activation import ActivationRecord
    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.satellite import SatelliteRunner

    sat = {f.name for f in dataclasses.fields(SatelliteRunner)}
    records = [f.name for f in dataclasses.fields(ValidatorRunner)
               if f.type in (ActivationRecord, "ActivationRecord")]
    assert "_shared_ladder" in records
    missing = [n for n in records if n not in sat]
    missing += [f"{n}_tally" for n in records if f"{n}_tally" not in sat]
    assert missing == [], f"SatelliteRunner lacks {missing}"


def test_satellite_startup_signals_locks_persists_and_restores_the_shared_ladder(tmp_path):
    import cascade.shared.activation as A
    from cascade.validator.windows import shared_ladder

    cfg = _cfg()
    spec = A.shared_ladder_feature(cfg)
    assert spec is not None and spec.decided_on_chain
    sat = _satellite(cfg, tmp_path)
    assert hasattr(sat, "_shared_ladder"), "SatelliteRunner has no _shared_ladder (#790)"
    chain = FakeChain(_fleet(60, 40), {}, block=B0 + 3600 + 1, hotkey="v1")
    sat._activation_startup(chain)
    # the satellite casts the vote: its note carries the lad- segment ...
    assert chain.written, "the satellite wrote no readiness note"
    assert chain.written[0] == f"cascade-ready:1:{PRIMARY}:0:0:{spec.name}:0:0"
    # ... and, with the stake majority ready, locks it in like the validator
    assert chain.written[-1] == (f"cascade-ready:1:{PRIMARY}:0:0:{spec.name}:"
                                 f"{B0 + 3600}:{B0 + 7200}")
    assert sat._shared_ladder.locked
    assert sat._shared_ladder_tally is not None
    assert sat.cfg.eval.shared_ladder_from_block == B0 + 7200
    assert sat.shared_ladder_block == B0 + 7200
    assert shared_ladder(sat.cfg.eval, B0 + 6300) == 0
    assert shared_ladder(sat.cfg.eval, B0 + 7200) == 40
    saved = A.ActivationStore(tmp_path / "activation_shared_ladder_state.json").load()
    assert saved.activation_block == B0 + 7200
    # a restart with the chain down restores the decided block from disk
    again = _satellite(cfg, tmp_path)
    dead = SimpleNamespace(current_block=lambda: (_ for _ in ()).throw(RuntimeError("down")),
                           hotkey_ss58=lambda: "v1")
    again._activation_startup(dead)
    assert again.cfg.eval.shared_ladder_from_block == B0 + 7200


def test_satellite_is_silent_on_a_typed_shared_ladder_block(tmp_path):
    from cascade.validator.windows import shared_ladder

    cfg = _cfg()
    cfg = replace(cfg, eval=replace(cfg.eval, shared_ladder_from_block=B0 + 3600))
    sat = _satellite(cfg, tmp_path)
    assert hasattr(sat, "_shared_ladder"), "SatelliteRunner has no _shared_ladder (#790)"
    chain = FakeChain(_fleet(60, 40), {}, block=B0 + 3600 + 1)
    sat._activation_startup(chain)
    assert chain.written == []
    assert sat.shared_ladder_block == 0
    assert shared_ladder(sat.cfg.eval, B0 + 3600) == 40
