"""2026-11-26 launch incident: the owner's overlay turned lane B on, the trainer
logged "overlay seq 2 in force", but `LaneBDriver.cfg` kept the startup config
(lane B off there), so `lane_b.active(block)` stayed False and a funded lane B
entry was never seated. Adopting an overlay must re-point the lane drivers."""
from __future__ import annotations

from types import SimpleNamespace

from cascade.trainer.loop import TrainerRunner


def _runner(**attrs):
    r = SimpleNamespace(**attrs)
    r.__dict__.setdefault("_rolling_sched", None)
    return r


def test_overlay_adoption_repoints_lane_b_and_lane_c_drivers():
    startup = SimpleNamespace(round=object(), scoring=object(), tag="startup")
    in_force = SimpleNamespace(round=object(), scoring=object(), tag="overlay")
    lane_b = SimpleNamespace(cfg=startup)
    lane_c = SimpleNamespace(cfg=startup)
    sched_seen = []
    sched = SimpleNamespace(update_config=sched_seen.append)
    r = _runner(lane_b_driver=lane_b, lane_c_driver=lane_c, promotion=None)
    r.__dict__["_rolling_sched"] = sched

    TrainerRunner._point_cfg_consumers(r, in_force)

    assert lane_b.cfg is in_force
    assert lane_c.cfg is in_force
    assert sched_seen == [in_force]          # the existing consumer is unchanged


def test_absent_or_cfgless_drivers_are_left_alone():
    in_force = SimpleNamespace(round=object(), scoring=object())
    bare = SimpleNamespace()                  # a driver stand-in without a cfg attribute
    r = _runner(lane_b_driver=None, lane_c_driver=bare, promotion=None)

    TrainerRunner._point_cfg_consumers(r, in_force)   # must not raise

    assert not hasattr(bare, "cfg")
