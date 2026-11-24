"""W-OV (plan rev6.24): a satellite on THIS code, with no code or config
change, adopts an owner-signed overlay that changes lane A, lane C (its
``n_windows`` and its MarketCast asset universe) and lane D tunables (court_weights, d_margin_start), takes
it only from the overlay's block, and derives the main validator's vector."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from .helpers import fixture_raw

pytest.importorskip("bittensor.sp_core")


def _overrides(raw: dict) -> dict:
    assets = {k: list(v) for k, v in raw["marketcast"]["assets"].items()}
    assets["crypto"] = [a for a in assets["crypto"] if a != "JUP"]
    return {
        "scoring": {"king_decay": 0.7},                      # lane A court split
        "lane_c": {"n_windows": 10_000},                     # owner "10k" (see plan)
        "lane_d": {"court_weights": [0.7, 0.2, 0.1], "d_margin_start": 0.04},
        "marketcast": {"assets": assets, "windows_per_cell": 96},
    }


@pytest.fixture
def scfg(tmp_path):
    from tests.unit.test_satellite_replay import _keyed

    return _keyed(SimpleNamespace(raw=fixture_raw(tmp_path)))


def test_w_ov_satellite_adopts_overlay_and_derives_main_vector(scfg, tmp_path):
    from cascade.audit.main import apply_receipt_overlay
    from cascade.shared import netconfig as nc
    from cascade.shared import netconfig_bounds as nb
    from tests.unit import test_satellite_replay as t

    ov = _overrides(scfg.raw)
    assert nb.value_refusals(ov) == [] and nb.key_refusals(ov) == []
    body = nc.next_overlay(ov, t.BOUNDARIES[0], None, network=nc.network_identity(scfg))
    store = t.DictStore()
    store.put_text(nc.seq_key(1), json.dumps(nc.sign_overlay(body, t.MAIN_KP)))
    main_cfg = nc.build_config(scfg, body)                      # the main validator's
    assert main_cfg.scoring.king_decay == 0.7 != scfg.scoring.king_decay
    receipt, _ = t.make_settlement(main_cfg, store, 0, netconfig=(1, nc.body_digest(body)))

    runner = t._runner(scfg, store, tmp_path)                    # base config only
    client = t.FakeClient(t.BOUNDARIES[0] + 300)
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "", doc
    assert client.pushed[-1] == [float(w) for w in receipt.weights]
    assert client.pushed[-1] != t._burn(scfg)

    # the satellite's re-derived config carries every overlay value
    sat_cfg, row = apply_receipt_overlay(receipt, scfg, store=store)
    assert row is None or row.status != "fail"
    assert sat_cfg.lane_c.n_windows == 10_000 == main_cfg.lane_c.n_windows
    assert tuple(sat_cfg.lane_d.court_weights) == (0.7, 0.2, 0.1) and sat_cfg.lane_d.d_margin_start == 0.04
    assert sat_cfg.marketcast.windows_per_cell == 96
    assert "JUP" not in sat_cfg.marketcast.assets_map["crypto"]
    assert "JUP" in scfg.marketcast.assets_map["crypto"]

    # lane D's court vector under the adopted values is the main's
    from cascade.shared.emission import lane_d_vector

    def court(c):
        return lane_d_vector(4, share=0.25, court_uids=[1, 2, 3], burn_uid=0,
                             court_weights=c.lane_d.court_weights)
    assert court(sat_cfg) == court(main_cfg) != court(scfg)


def test_w_ov_overlay_applies_only_from_its_block(scfg):
    from cascade.shared import netconfig as nc
    from cascade.shared import netconfig_bounds as nb

    ov = _overrides(scfg.raw)
    assert nb.value_refusals(ov) == []
    apply = 9 * 3600
    body = nc.next_overlay(ov, apply, None, network=nc.network_identity(scfg))
    hist = [nc.Adopted(apply_block=apply, seq=1, digest=nc.body_digest(body),
                       envelope={"body": body})]
    assert nc.config_from_history(scfg, hist, apply - 1) is scfg
    at = nc.config_from_history(scfg, hist, apply)
    assert at.lane_c.n_windows == 10_000 and tuple(at.lane_d.court_weights) == (0.7, 0.2, 0.1)
    assert at.netconfig.seq == 1 and at.netconfig.digest == nc.body_digest(body)
