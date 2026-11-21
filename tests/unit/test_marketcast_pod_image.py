"""Owner 2026-11-20: MarketCast legs boot their OWN worker image
(``[marketcast] pod_image`` + ``worker_code_fingerprint``) so arming MarketCast
never repins lane A. Unset = the fleet's pins (byte-identical)."""
from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cascade.shared.activation as A
from cascade.shared.config import load_chain_config
from cascade.shared.config.marketcast import parse_marketcast
from cascade.trainer.leg_pins import leg_contract_scope, leg_round, mkt_leg_scope
from cascade.trainer.marketcast_runner import mkt_contract_block_kw
from tests.unit.test_image_after_repin import SW, acfg  # noqa: F401,F811 — the shared fixture

REPO = Path(__file__).resolve().parents[2]
MKT = "sha256:" + "a" * 64
MKT_REF = "ghcr.io/tensorlink-ai/cascade-worker:worker-v2.1.0-mkt@" + MKT
MKT_FP = "b" * 64
BASE = {"assets": {"crypto": ["BTC"]}, "suites": ["returns"],
        "frequencies": {"returns": ["5m"]}, "horizons": {"returns": ["5m"]}}


def test_parse_both_or_neither_and_digest_pinned():
    mc = parse_marketcast({**BASE, "pod_image": MKT_REF, "worker_code_fingerprint": MKT_FP.upper()})
    assert (mc.pod_image, mc.worker_code_fingerprint) == (MKT_REF, MKT_FP)
    assert (parse_marketcast(BASE).pod_image, parse_marketcast(BASE).worker_code_fingerprint) == ("", "")
    for bad, why in (({"pod_image": MKT_REF}, "both or neither"),
                     ({"worker_code_fingerprint": MKT_FP}, "both or neither"),
                     ({"pod_image": "ghcr.io/x/y:tag", "worker_code_fingerprint": MKT_FP}, "digest"),
                     ({"pod_image": MKT_REF, "worker_code_fingerprint": "abc"}, "64 hex")):
        with pytest.raises(ValueError, match=why):
            parse_marketcast({**BASE, **bad})


def _cfg(pod_image=MKT_REF, fp=MKT_FP):
    c = load_chain_config(REPO / "chain.toml")
    return replace(c, marketcast=replace(c.marketcast, pod_image=pod_image,
                                         worker_code_fingerprint=fp))


def test_only_a_marketcast_leg_uses_its_image():
    cfg = _cfg()
    runner = SimpleNamespace(cfg=cfg)
    assert leg_round(runner) is cfg.round                        # lane A / king: the fleet's
    with mkt_leg_scope(runner):
        rnd = leg_round(runner)
        assert (rnd.funded_pod_image, rnd.worker_code_fingerprint) == (MKT_REF, MKT_FP)
        assert rnd.funded_sku_wall_measured_image == ""
        assert rnd.funded_pod_sku == cfg.round.funded_pod_sku   # everything else unchanged
        seen = []
        t = threading.Thread(target=lambda: seen.append(leg_round(runner)))
        t.start()
        t.join()
        assert seen[0] is cfg.round                              # thread-local
    assert leg_round(runner) is cfg.round                        # scope restored


def test_unset_is_the_fleet_image():
    cfg = _cfg("", "")
    runner = SimpleNamespace(cfg=cfg)
    with mkt_leg_scope(runner):
        assert leg_round(runner) is cfg.round


def test_its_own_image_wins_over_the_fleet_after_repin(acfg, monkeypatch):  # noqa: F811
    """A MarketCast leg on its own release never takes the fleet's DEC-CA-0067
    after pins: no contract_block for the image hook, its own pins at rent."""
    monkeypatch.setattr("cascade.trainer.leg_pins.after_image_refusal", lambda cfg: "")
    acfg = replace(acfg, marketcast=replace(acfg.marketcast, pod_image="",
                                            worker_code_fingerprint=""))      # the fleet image
    armed = A.apply_image_decoupled_activation(acfg, SW)
    assert mkt_contract_block_kw(armed, SimpleNamespace(start_block=SW)) == {"contract_block": SW}
    own = replace(armed, marketcast=replace(armed.marketcast, pod_image=MKT_REF,
                                            worker_code_fingerprint=MKT_FP))
    assert mkt_contract_block_kw(own, SimpleNamespace(start_block=SW)) == {}
    runner = SimpleNamespace(cfg=own)
    with leg_contract_scope(runner, own.training.at_block(SW)):
        assert leg_round(runner).funded_pod_image != MKT_REF    # a lane-A leg: the after pins
        with mkt_leg_scope(runner):
            assert leg_round(runner).funded_pod_image == MKT_REF


def test_shipped_release_pins_its_own_image_apart_from_the_fleet():
    cfg = load_chain_config(REPO / "chain.toml")
    mc = cfg.marketcast
    assert mc.pod_image and len(mc.worker_code_fingerprint) == 64
    assert mc.pod_image != cfg.round.funded_pod_image
    assert mc.worker_code_fingerprint != cfg.round.worker_code_fingerprint
