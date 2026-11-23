"""WB22: ``cascade.funding.lane_b_gpus`` (catalog, step-rate gate, image pin,
rent scope, audit pin). Stdlib only."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from .helpers import (
    OWNER_LANE_B_SKUS,
    REPO,
    STEP_RATES_ENV,
    UNMEASURED,
    chain_cfg,
    cfg_with_lane_b,
    gpus,
    norm,
    pinned_image,
    table,
    write_table,
)


def _choice(gpu="", price=0.0, fallback=False):
    from cascade.funding.gpu_choice import GpuChoice

    return GpuChoice(gpu=gpu, allow_fallback=fallback, max_price_per_hour=price)


def _rates(skus) -> dict:
    """``{sku: StepRate}`` through the WB19 reader (the real parser)."""
    from cascade.funding.step_rates import _row

    return {s: _row(s, row) for s, row in table(list(skus))["rows"].items()}


def _gpus(rates_for=None, cfg=None):
    """Lane B GPUs of the shipped config; ``rates_for`` None = no table."""
    rates = None if rates_for is None else _rates(rates_for)
    return gpus().lane_b_gpus(cfg or chain_cfg(), rates)


# ── constants and paths ──────────────────────────────────────────────────────

def test_constants():
    m = gpus()
    assert m.STEP_RATES_ENV == STEP_RATES_ENV
    assert m.DEFAULT_STEP_RATES == "evidence/wb19.json"
    assert m.UNMEASURED == UNMEASURED


def test_step_rates_path_env_and_default(tmp_path):
    m = gpus()
    assert m.step_rates_path({STEP_RATES_ENV: str(tmp_path / "x.json")}) == tmp_path / "x.json"
    assert m.step_rates_path({}) == REPO / "evidence" / "wb19.json"
    assert m.step_rates_path({STEP_RATES_ENV: ""}) == REPO / "evidence" / "wb19.json"


def test_image_pin_is_the_v2_contract_image():
    m = gpus()
    cfg = chain_cfg()
    assert m.lane_b_image_pin(cfg) == pinned_image(cfg)
    b = int(cfg.training.v2_from_block)
    assert m.lane_b_image_pin(cfg, b) == pinned_image(cfg)


def test_load_missing_file_is_nothing_measured(tmp_path):
    m = gpus()
    assert m.load_lane_b_rates(tmp_path / "absent.json", image_digest=pinned_image()) == {}


def test_load_refuses_a_table_of_another_image(tmp_path):
    from cascade.funding.step_rates import StepRateTableError

    m = gpus()
    p = write_table(tmp_path / "t.json", ["8xH100"], image="sha256:" + "0" * 64)
    with pytest.raises(StepRateTableError) as e:
        m.load_lane_b_rates(p, image_digest=pinned_image())
    assert e.value.code == "wrong_image"


# ── the catalog ──────────────────────────────────────────────────────────────

def test_catalog_is_lane_b_list_not_round_list():
    g = _gpus()
    cfg = chain_cfg()
    assert norm(g.skus) == OWNER_LANE_B_SKUS
    assert norm(g.catalog.skus) == OWNER_LANE_B_SKUS
    assert g.catalog.walls == ()
    assert g.default_max_price_per_hour == pytest.approx(
        float(cfg.round.funded_max_price_per_hour))
    assert g.catalog.default_max_price_per_hour == pytest.approx(g.default_max_price_per_hour)
    assert g.catalog.canonical("8XRTX4090") == "8xRTX4090"
    assert g.catalog.canonical("RTX4090") is None


def test_catalog_follows_the_lane_b_key():
    m = gpus()
    cfg = cfg_with_lane_b(funded_pod_skus=["8xH100"])
    assert norm(m.lane_b_gpus(cfg, None).skus) == ("8xH100",)
    assert norm(cfg.round.funded_pod_skus) != ("8xH100",)


def test_no_table_wired_means_no_gate():
    g = _gpus(None)
    assert g.rates is None
    assert all(g.measured(s) for s in OWNER_LANE_B_SKUS)
    assert norm(g.rent_skus()) == OWNER_LANE_B_SKUS
    assert g.rate("8xH100") is None
    assert g.refusal(_choice("8xH100")) is None and g.refusal(_choice()) is None


def test_rate_gate_filters_in_list_order():
    g = _gpus(["8xRTX4090", "8xH100"])
    assert norm(g.rent_skus()) == ("8xH100", "8xRTX4090")
    assert g.measured("8xrtx4090") and not g.measured("8xH200")
    assert g.rate("8xH100").steps_per_s > 0
    assert g.refusal(_choice("8xRTX4090")) is None
    assert g.refusal(_choice()) is None


def test_unmeasured_choice_is_refused_with_the_typed_code():
    g = _gpus(["8xH100"])
    body = g.refusal(_choice("8xH200"))
    assert body and body["code"] == UNMEASURED and "8xH200" in body["message"]


def test_nothing_measured_refuses_the_default_path_too():
    g = gpus().lane_b_gpus(chain_cfg(), {})
    assert g.rent_skus() == ()
    body = g.refusal(_choice())
    assert body and body["code"] == UNMEASURED


def test_from_env_loads_the_table(tmp_path):
    m = gpus()
    p = write_table(tmp_path / "t.json", ["8xRTX4090"])
    g = m.lane_b_gpus_from_env(chain_cfg(), env={STEP_RATES_ENV: str(p)})
    assert norm(g.rent_skus()) == ("8xRTX4090",)


@pytest.mark.parametrize("doc", ["{not json", json.dumps({"schema": "nope"})])
def test_from_env_fails_closed_on_a_bad_table(tmp_path, doc):
    m = gpus()
    p = tmp_path / "bad.json"
    p.write_text(doc, encoding="utf-8")
    g = m.lane_b_gpus_from_env(chain_cfg(), env={STEP_RATES_ENV: str(p)})
    assert g.rates == {} and g.rent_skus() == ()


def test_from_env_fails_closed_on_a_stale_image(tmp_path):
    m = gpus()
    p = write_table(tmp_path / "t.json", ["8xH100"], image="sha256:" + "1" * 64)
    g = m.lane_b_gpus_from_env(chain_cfg(), env={STEP_RATES_ENV: str(p)})
    assert g.rent_skus() == ()


def test_from_env_missing_default_file_is_nothing_measured(tmp_path):
    m = gpus()
    g = m.lane_b_gpus_from_env(chain_cfg(), env={STEP_RATES_ENV: str(tmp_path / "none.json")})
    assert g.rates == {}


def test_module_is_torch_free():
    gpus()
    src = (REPO / "cascade" / "funding" / "lane_b_gpus.py").read_text(encoding="utf-8")
    assert "import torch" not in src


# ── the rent scope ───────────────────────────────────────────────────────────

def test_rent_scope_sets_and_restores_the_thread_local():
    m = gpus()
    g = _gpus(["8xRTX4090", "8xH200"])
    runner = SimpleNamespace()
    with m.rent_scope(runner, g):
        loc = runner.__dict__["_leg_local_obj"]
        assert isinstance(loc, threading.local)
        assert norm(loc.lane_b_rent_skus) == ("8xH200", "8xRTX4090")
        seen = []
        t = threading.Thread(target=lambda: seen.append(
            getattr(runner.__dict__["_leg_local_obj"], "lane_b_rent_skus", None)))
        t.start()
        t.join()
        assert seen == [None], "another leg's thread must not see lane B's list"
    assert getattr(runner.__dict__["_leg_local_obj"], "lane_b_rent_skus", None) is None


def test_rent_scope_restores_on_exception_and_nests():
    m = gpus()
    runner = SimpleNamespace()
    outer, inner = _gpus(["8xH100"]), _gpus(["8xH200"])
    with pytest.raises(RuntimeError):
        with m.rent_scope(runner, outer):
            with m.rent_scope(runner, inner):
                assert norm(runner._leg_local_obj.lane_b_rent_skus) == ("8xH200",)
            assert norm(runner._leg_local_obj.lane_b_rent_skus) == ("8xH100",)
            raise RuntimeError("leg failed")
    assert getattr(runner._leg_local_obj, "lane_b_rent_skus", None) is None


# ── the audit pin (recorded gpu_name AND gpu_n) ──────────────────────────────

@pytest.mark.parametrize("rec_name,rec_n,loc_name,loc_n,ok", [
    ("NVIDIA H100 80GB HBM3", 8, "NVIDIA H100 80GB HBM3", 8, True),
    ("NVIDIA H100 80GB HBM3", 8, "NVIDIA H100 80GB HBM3", 4, False),
    ("NVIDIA H100 80GB HBM3", 8, "NVIDIA H200", 8, False),
    ("", 8, "", 8, False),
    ("NVIDIA GeForce RTX 4090", 0, "NVIDIA GeForce RTX 4090", 1, True),
    ("NVIDIA GeForce RTX 4090", 0, "NVIDIA GeForce RTX 4090", 8, False),
    ("NVIDIA GeForce RTX 4090", 8, "NVIDIA GeForce RTX 4090", 8, True),
])
def test_exact_rederive_pin(rec_name, rec_n, loc_name, loc_n, ok):
    m = gpus()
    assert m.exact_rederive_ok(recorded_gpu_name=rec_name, recorded_gpu_n=rec_n,
                               local_gpu_name=loc_name, local_gpu_n=loc_n) is ok


def test_table_helper_matches_the_wb19_reader(tmp_path):
    """Guard on this test file's own fixture: the WB19 reader accepts it."""
    from cascade.funding.step_rates import load_step_rate_table

    p = tmp_path / "t.json"
    p.write_text(json.dumps(table(list(OWNER_LANE_B_SKUS))), encoding="utf-8")
    assert set(load_step_rate_table(p, image_digest=pinned_image())) == set(OWNER_LANE_B_SKUS)
