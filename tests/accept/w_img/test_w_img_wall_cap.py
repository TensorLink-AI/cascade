"""W-IMG unmeasured-image wall cap (owner 2026-11-24: "YES to the 5 h cap").

A funded or king rent on an image with NO measured pod walls is rented with
the leg wall capped at ``[round] unmeasured_image_wall_cap_seconds`` (18000
s) instead of refused; the leg's record carries ``walls_measured: false`` and
the cap. A measured image behaves exactly as before."""

from __future__ import annotations

import re
import threading
import types
from dataclasses import replace
from types import SimpleNamespace

import pytest

from .helpers import REPO, base_cfg, require_w_img, v2_setup

MEASURED = "sha256:" + "5" * 64
OTHER = "sha256:" + "6" * 64


def _round(cfg, *, pod_image, cap=18000):
    return replace(cfg.round, funded_sku_wall_measured_image=MEASURED,
                   funded_pod_image=f"repo@{pod_image}",
                   unmeasured_image_wall_cap_seconds=cap,
                   funded_sku_wall_seconds=(("8xH200", 12270),))


def test_measured_image_is_unchanged(tmp_path):
    require_w_img()
    from cascade.trainer.cost_guards import (
        measured_wall_refusal,
        unmeasured_wall_cap,
        wall_record,
        walls_measured,
    )

    rnd = _round(base_cfg(tmp_path), pod_image=MEASURED)
    assert walls_measured(rnd)
    assert measured_wall_refusal(rnd) == ""
    assert unmeasured_wall_cap(rnd) == 0.0
    assert wall_record(rnd) == {}


def test_unmeasured_image_rents_under_the_cap_instead_of_refusing(tmp_path):
    require_w_img()
    from cascade.trainer.cost_guards import (
        measured_wall_refusal,
        unmeasured_wall_cap,
        wall_record,
        walls_measured,
    )

    rnd = _round(base_cfg(tmp_path), pod_image=OTHER)
    assert not walls_measured(rnd)
    assert measured_wall_refusal(rnd) == ""
    assert unmeasured_wall_cap(rnd) == 18000.0
    assert wall_record(rnd) == {"walls_measured": False, "wall_cap_seconds": 18000}


def test_cap_zero_keeps_the_old_refusal(tmp_path):
    require_w_img()
    from cascade.trainer.cost_guards import measured_wall_refusal, unmeasured_wall_cap

    rnd = _round(base_cfg(tmp_path), pod_image=OTHER, cap=0)
    assert unmeasured_wall_cap(rnd) == 0.0
    assert "rent refused" in measured_wall_refusal(rnd)


def _runner(cfg, contract, **loc):
    local = threading.local()
    local.funded_contract = contract
    for k, v in loc.items():
        setattr(local, k, v)
    r = SimpleNamespace(cfg=cfg)
    r.__dict__["_leg_local_obj"] = local
    return r


def test_leg_wall_is_the_cap_on_an_unmeasured_image_only(tmp_path):
    require_w_img()
    from cascade.trainer.runner_capacity import _FundedCapacityMixin

    cfg = base_cfg(tmp_path)
    contract = replace(cfg.training, max_train_seconds=20000)
    measured = replace(cfg, round=_round(cfg, pod_image=MEASURED))
    unmeasured = replace(cfg, round=_round(cfg, pod_image=OTHER))
    wall = _FundedCapacityMixin._leg_wall_seconds
    assert wall(_runner(measured, contract), "8xH200") == 12270.0       # the table, as today
    assert wall(_runner(unmeasured, contract), "8xH200") == 18000.0     # the cap
    short = replace(contract, max_train_seconds=10000)
    assert wall(_runner(unmeasured, short), "8xH200") == 10000.0        # never above the contract


def test_watchdog_holds_an_unmeasured_leg_to_the_cap(tmp_path):
    require_w_img()
    from cascade.trainer.cost_guards import leg_watchdog, note_unmeasured_wall

    cfg = base_cfg(tmp_path)
    cfg = replace(cfg, round=replace(_round(cfg, pod_image=OTHER),
                                     progress_watchdog_grace_seconds=600))
    contract = replace(cfg.training, max_train_seconds=20000)
    r = _runner(cfg, contract)
    assert note_unmeasured_wall(r, cfg.round, "test") == {
        "walls_measured": False, "wall_cap_seconds": 18000}
    dog = leg_watchdog(r, gen=SimpleNamespace(hotkey="hk" * 8), pod=None, contract=contract)
    assert dog.max_train_seconds == 18000.0
    # a measured image clears the mark: the watchdog uses the contract wall
    note_unmeasured_wall(r, replace(cfg.round, funded_pod_image=f"repo@{MEASURED}"), "test")
    dog = leg_watchdog(r, gen=SimpleNamespace(hotkey="hk" * 8), pod=None, contract=contract)
    assert dog.max_train_seconds == 20000.0


def test_completed_leg_record_carries_walls_measured_false(tmp_path):
    require_w_img()
    import json

    from cascade.trainer.cost_guards import note_unmeasured_wall
    from cascade.trainer.runner_funded_pods import _FundedPodMixin

    _, m, v2c, cfg = v2_setup(tmp_path, rule=False)
    entry = m.entries[0]
    rnd = _round(cfg, pod_image=OTHER)
    for pod_image, want in ((OTHER, {"walls_measured": False, "wall_cap_seconds": 18000}),
                            (MEASURED, {})):
        r = _runner(cfg, v2c, work_root=None)
        r.work_root = tmp_path / pod_image[-4:]
        r._completed_leg_path = types.MethodType(_FundedPodMixin._completed_leg_path, r)
        note_unmeasured_wall(r, replace(rnd, funded_pod_image=f"repo@{pod_image}"), "test")
        _FundedPodMixin._persist_completed_leg(r, entry, round_id=1, contract=v2c,
                                               role="king", hotkey="hk")
        payload = json.loads(r._completed_leg_path(1, v2c.arch_preset, "king", "hk").read_text())
        assert {k: payload[k] for k in ("walls_measured", "wall_cap_seconds")
                if k in payload} == want


def test_config_key_defaults_to_five_hours_and_round_trips(tmp_path):
    require_w_img()
    from cascade.shared.config import load_chain_config

    cfg = load_chain_config(REPO / "chain.toml")
    assert cfg.round.unmeasured_image_wall_cap_seconds == 18000

    def _with(value):
        text = (REPO / "chain.toml").read_text()
        text = re.sub(r"(?m)^\[round\]$", f"[round]\nunmeasured_image_wall_cap_seconds = {value}",
                      text, count=1)
        p = tmp_path / "chain.toml"
        p.write_text(text)
        return p

    assert load_chain_config(_with(0)).round.unmeasured_image_wall_cap_seconds == 0
    assert load_chain_config(_with(7200)).round.unmeasured_image_wall_cap_seconds == 7200
    with pytest.raises(ValueError, match="unmeasured_image_wall_cap_seconds"):
        load_chain_config(_with(-1))
