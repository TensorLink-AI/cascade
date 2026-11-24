"""WB13 production wiring: ``--lane-b/--lane-c auto`` follow the ADOPTED overlay.

Launch-readiness audit 2026-11-24: the intake decided lane B / lane C
armed-ness from the static ``chain.toml`` once at start, ignoring the adopted
owner-signed netconfig overlay (DEC-CA-0057). The Wed 2026-11-25 launch arms
``[lane_b] from_block`` by overlay (``deploy/overlays/launch-20261125.pre.
mainnet.toml``), so the intake would have kept answering 503
``lane_b_disabled``. ``cascade.funding.main.EffectiveLanes`` resolves the
overlay read-only with the same ``NetConfigResolver`` the trainer uses and
re-points the intake's lanes per poll:

* an overlay arming lane B at block N: 503 before N, open at and after N,
  with no restart;
* no overlay (or no owner key): exactly the static wiring.

The chain and the manifest store are in-memory fakes (the netconfig unit
fixtures); nothing touches the network and no torch is loaded.
"""

from __future__ import annotations

from dataclasses import replace

from cascade.funding.intake import FundingIntake
from cascade.funding.main import EffectiveLanes
from cascade.funding.queue import (
    FundedQueue,
    LaneBQueue,
    LaneCQueue,
    lane_b_queue_path,
    lane_c_queue_path,
)
from cascade.funding.vault import PayerKeyVault
from cascade.shared import netconfig as nc
from cascade.shared.config import parse_chain_config
from tests.unit.test_activation import FakeChain, _v
from tests.unit.test_netconfig_core import E1, E2, ERA, MemStore, R, base_raw, publish


class Clock:
    def __init__(self, t: float = 1_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _base(owner=None):
    # lanes B and C run on the v2 era schedule: [v2] from_block typed (the
    # launch release has it), the lanes themselves shipped inert. The
    # DEC-CA-0068 lad- vote is disarmed in memory, as the netconfig runtime
    # tests do: only the overlay's own cfg- segment is voted on.
    raw = base_raw(owner)
    raw["v2"]["from_block"] = R
    cfg = parse_chain_config(raw)
    cfg = replace(cfg, eval=replace(cfg.eval, shared_ladder_windows=0,
                                    shared_ladder_feature=""))
    assert cfg.lane_b.from_block == 0 and cfg.lane_c.from_block == 0
    return cfg


def _wire(tmp_path, cfg, *, store=None, chain=None, lane_b="auto", lane_c="auto"):
    """``main()``'s wiring, minus the chain client and the HTTP server."""
    clock = Clock()
    qpath = tmp_path / "queue.json"
    res = None
    if store is not None:
        res = nc.NetConfigResolver(cfg, state_path=tmp_path / "intake_netconfig_state.json",
                                   store=store, signal=False)
    lanes = EffectiveLanes(
        cfg, lane_b_mode=lane_b, lane_c_mode=lane_c,
        open_lane_b_queue=lambda: LaneBQueue(lane_b_queue_path(qpath)),
        open_lane_c_queue=lambda: LaneCQueue(lane_c_queue_path(qpath)),
        client=chain, netconfig=res, poll_seconds=12.0, clock=clock)
    intake = FundingIntake(
        FundedQueue(qpath), PayerKeyVault(dir=None), resolve_reveal=lambda hk, ref: None,
        require_signature=False, lane_b_cfg=lanes.lane_b, lane_refresh=lanes)
    lanes.apply(intake, cfg)
    return intake, lanes, clock


def _lane_b_status(intake) -> tuple[int, str]:
    status, body = intake.lane_b_fund({})
    return status, body.get("code", "")


def _lock_note(body) -> str:
    feat = nc.cfg_feature_name(body)
    return f"cascade-ready:1:rolling-era-king:0:0:{feat}:{E1}:{E1 + ERA}"


def test_overlay_arms_lane_b_at_its_block_without_restart(tmp_path):
    cfg = _base()
    store = MemStore()
    body = publish(store, {"lane_b": {"from_block": E2}}, E2)
    chain = FakeChain([_v("v1", 60.0), _v("v2", 40.0)], {"v1": _lock_note(body)},
                      block=E1 + 10, hotkey="intake")
    intake, lanes, clock = _wire(tmp_path, cfg, store=store, chain=chain)

    # before N: the overlay is adopted (locked in) but not yet in force
    assert _lane_b_status(intake) == (503, "lane_b_disabled")
    assert intake.lane_b_queue_view()[0] == 503
    assert [h.apply_block for h in lanes.netconfig.state.history] == [E2]
    assert chain.written == []                     # read-only: never signals

    chain.block = E2 - 1
    clock.t += 13
    assert _lane_b_status(intake) == (503, "lane_b_disabled")

    # at N: open (the request reaches the target check, not the lane gate)
    chain.block = E2
    clock.t += 13
    status, code = _lane_b_status(intake)
    assert (status, code) == (400, "missing_targets")
    assert intake.lane_b_queue_view()[0] == 200
    assert isinstance(intake.lane_b_queue, LaneBQueue)
    assert intake.lane_b_cfg is lanes.view.lane_b and intake.lane_b_cfg.from_block > 0
    assert intake.lane_b_targets == tuple(intake.lane_b_cfg.targets)
    assert intake.lane_b_max_targets == int(intake.lane_b_cfg.max_domains_per_leg)

    # after N: stays open; lane C (unarmed by the overlay) stays closed
    chain.block = E2 + 5 * ERA
    clock.t += 13
    assert _lane_b_status(intake)[0] == 400
    assert intake.lane_c_queue_view() == (503, {"code": "lane_c_disabled"})


def test_view_is_rate_limited_per_poll(tmp_path):
    cfg = _base()
    store = MemStore()
    body = publish(store, {"lane_b": {"from_block": E2}}, E2)
    chain = FakeChain([_v("v1", 60.0), _v("v2", 40.0)], {"v1": _lock_note(body)},
                      block=E1 + 10, hotkey="intake")
    intake, _, clock = _wire(tmp_path, cfg, store=store, chain=chain)
    assert _lane_b_status(intake)[0] == 503
    chain.block = E2
    # inside the poll window the view in place answers
    assert _lane_b_status(intake)[0] == 503
    clock.t += 12
    assert _lane_b_status(intake)[0] == 400


def test_lane_c_follows_an_overlay_too(tmp_path):
    cfg = _base()
    store = MemStore()
    body = publish(store, {"lane_c": {"from_block": E2}}, E2)
    chain = FakeChain([_v("v1", 60.0), _v("v2", 40.0)], {"v1": _lock_note(body)},
                      block=E2 - 1, hotkey="intake")
    intake, _, clock = _wire(tmp_path, cfg, store=store, chain=chain)
    assert intake.lane_c_queue_view()[0] == 503
    chain.block = E2
    clock.t += 13
    assert intake.lane_c_queue_view()[0] == 200
    assert intake.lane_b_queue_view()[0] == 503


def test_no_overlay_is_the_static_wiring(tmp_path):
    cfg = _base()
    chain = FakeChain([_v("v1", 60.0)], {}, block=E2 + 10, hotkey="intake")
    # owner key set, nothing published: the base config, lanes off
    intake, lanes, clock = _wire(tmp_path, cfg, store=MemStore(), chain=chain)
    for _ in range(3):
        clock.t += 13
        assert _lane_b_status(intake) == (503, "lane_b_disabled")
        assert intake.lane_c_queue_view()[0] == 503
    assert lanes.view is cfg and intake.lane_b_cfg is cfg.lane_b

    # no owner key (main() builds no resolver): no chain read at all
    inert = cfg
    intake, lanes, clock = _wire(tmp_path / "inert", inert)
    clock.t += 13
    assert _lane_b_status(intake) == (503, "lane_b_disabled")
    assert lanes.view is inert

    # chain.toml arming lane B: open from the start, exactly as before
    armed = replace(inert, lane_b=replace(inert.lane_b, from_block=E2))
    intake, _, _ = _wire(tmp_path / "armed", armed)
    assert _lane_b_status(intake) == (400, "missing_targets")
    assert intake.lane_b_targets == tuple(armed.lane_b.targets)

    # --lane-b on / off keep their meaning
    intake, _, _ = _wire(tmp_path / "on", inert, lane_b="on")
    assert _lane_b_status(intake)[0] == 400
    intake, _, _ = _wire(tmp_path / "off", armed, lane_b="off")
    assert _lane_b_status(intake) == (503, "lane_b_disabled")


def test_chain_read_failure_keeps_the_lanes(tmp_path):
    cfg = _base()
    store = MemStore()
    body = publish(store, {"lane_b": {"from_block": E2}}, E2)
    chain = FakeChain([_v("v1", 60.0), _v("v2", 40.0)], {"v1": _lock_note(body)},
                      block=E2, hotkey="intake")
    intake, _, clock = _wire(tmp_path, cfg, store=store, chain=chain)
    assert _lane_b_status(intake)[0] == 400

    def down():
        raise RuntimeError("substrate down")

    chain.current_block = down
    clock.t += 13
    assert _lane_b_status(intake)[0] == 400       # never closes on a flake
