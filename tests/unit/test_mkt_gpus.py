"""Owner 2026-11-24: MarketCast (lane C) legs may rent ANY Lium GPU type.

``[marketcast] funded_pod_skus`` is the recommended list (separate from lane
A's ``[round] funded_pod_skus``, which is unchanged); any other well-formed
type is accepted at the miner's own risk (warned, Lium key only, no refund);
the trainer rents the miner's chosen type and never on Shadeform. Trainer,
intake and miner CLI only — no validator / audit path reads any of it.

No rentals, no network, no torch: every provider call is mocked."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import cascade.provision.funded as funded_mod
from cascade.funding.gpu_choice import GPU_HEADER, GpuCatalog, GpuChoice
from cascade.funding.mkt_gpus import (
    LIUM_ONLY,
    lium_only_key,
    mkt_catalog,
    mkt_rent_scope,
    mkt_rent_skus,
    mkt_skus,
    own_risk_gpu,
)
from cascade.funding.providers import (
    CREDENTIALS_HEADER,
    PayerCredential,
    encode_credentials,
)
from cascade.shared.config import load_chain_config
from cascade.shared.config.marketcast import parse_marketcast
from cascade.trainer.faults import _FundedLegSkip
from cascade.trainer.loop import TrainerRunner

REPO = Path(__file__).resolve().parents[2]
SHIPPED = ("1xRTX4090", "1xA100", "1xH100", "8xH100")


# ── config ───────────────────────────────────────────────────────────────────


def test_shipped_chain_toml_lists_the_recommended_types_and_keeps_lane_a():
    cfg = load_chain_config(REPO / "chain.toml")
    assert cfg.marketcast.funded_pod_skus == SHIPPED
    assert tuple(cfg.round.funded_pod_skus) == ("8xH100", "8xH200")   # lane A untouched
    assert mkt_skus(cfg) == SHIPPED
    assert mkt_catalog(cfg).skus == SHIPPED


def test_parse_default_is_empty_and_falls_back_to_round():
    mc = parse_marketcast({})
    assert mc.funded_pod_skus == ()
    from types import SimpleNamespace

    cfg = SimpleNamespace(marketcast=mc, round=SimpleNamespace(funded_pod_skus=("8xH100",)))
    assert mkt_skus(cfg) == ("8xH100",)


@pytest.mark.parametrize("bad", ["1xH100", ["a;b"], ["H100", "h100"], [""], [3]])
def test_parse_refuses_malformed_lists(bad):
    with pytest.raises(ValueError, match="funded_pod_skus"):
        parse_marketcast({"funded_pod_skus": bad})


def test_parse_round_trips_a_list():
    assert parse_marketcast({"funded_pod_skus": [" 1xH100", "RTX3090"]}).funded_pod_skus == (
        "1xH100", "RTX3090")


def test_spec_instances_never_inherit_the_rent_list():
    from cascade.shared.competition_spec import _NOT_INHERITED

    assert "funded_pod_skus" in _NOT_INHERITED


# ── helpers ──────────────────────────────────────────────────────────────────

CAT = GpuCatalog(skus=SHIPPED, default_max_price_per_hour=40.0)


def test_own_risk_gpu_rule():
    assert own_risk_gpu("1xRTX3090", CAT, bundle=False) == "1xRTX3090"
    assert own_risk_gpu("1xh100", CAT, bundle=False) == ""          # listed: not own risk
    assert own_risk_gpu("1xRTX3090", CAT, bundle=True) == ""        # Lium key only
    assert own_risk_gpu("a;b", CAT, bundle=False) == ""             # malformed
    assert own_risk_gpu("", CAT, bundle=False) == ""


def test_rent_list_adds_only_the_own_risk_choice():
    from types import SimpleNamespace

    cfg = SimpleNamespace(marketcast=SimpleNamespace(funded_pod_skus=SHIPPED),
                          round=SimpleNamespace(funded_pod_skus=("8xH200",)))
    assert mkt_rent_skus(cfg, GpuChoice()) == SHIPPED
    assert mkt_rent_skus(cfg, GpuChoice("1xh100")) == SHIPPED
    assert mkt_rent_skus(cfg, GpuChoice("1xRTX3090")) == (*SHIPPED, "1xRTX3090")


def test_lium_only_key_drops_every_other_provider():
    assert lium_only_key("sk-lium") == "sk-lium"
    both = encode_credentials([PayerCredential("shadeform", "sf"), PayerCredential("lium", "lk")])
    from cascade.funding.providers import decode_credentials

    assert decode_credentials(lium_only_key(both)) == (PayerCredential("lium", "lk"),)
    assert lium_only_key(encode_credentials([PayerCredential("shadeform", "sf")])) == ""


# ── intake ───────────────────────────────────────────────────────────────────


def _mkt_intake(tmp_path, catalog=CAT):
    from tests.unit.test_miner_chosen_gpu import _intake

    intake = _intake(tmp_path, lane_c=True)
    intake.mkt_gpu_catalog = catalog
    return intake


def _headers(gpu, **kw):
    from cascade.miner.lane_c import build_lane_c_headers
    from tests.unit.test_miner_chosen_gpu import HK, REF, _clock, _sign

    return build_lane_c_headers("lane-c-fund", HK, "mkt:returns", REF, "sk", _sign,
                                now=_clock, gpu=gpu, **kw)


def test_recommended_single_gpu_type_is_accepted_without_warning(tmp_path):
    from tests.unit.test_miner_chosen_gpu import HK

    intake = _mkt_intake(tmp_path)
    status, body = intake.lane_c_fund(_headers("1xrtx4090"))
    assert status == 202 and body["gpu"] == "1xRTX4090"     # the list's spelling
    assert "gpu_own_risk" not in body
    assert intake.lane_c_queue.get(HK).gpu == "1xRTX4090"


def test_any_other_lium_type_is_accepted_at_the_miners_own_risk(tmp_path):
    from tests.unit.test_miner_chosen_gpu import HK

    intake = _mkt_intake(tmp_path)
    status, body = intake.lane_c_fund(_headers("1xRTX3090", max_price=1.5))
    assert status == 202 and body["gpu"] == "1xRTX3090" and body["gpu_own_risk"] is True
    assert "OWN RISK" in body["warning"] and "max_leg_hours" in body["warning"]
    assert "nothing is refunded" in body["warning"]
    e = intake.lane_c_queue.get(HK)
    assert (e.gpu, e.gpu_max_price_per_hour) == ("1xRTX3090", 1.5)


def test_own_risk_type_with_a_multi_provider_bundle_is_refused(tmp_path):
    intake = _mkt_intake(tmp_path)
    h = _headers("1xRTX3090")
    h.pop("X-Lium-Api-Key", None)
    h[CREDENTIALS_HEADER] = encode_credentials([PayerCredential("shadeform", "sf")])
    status, body = intake.lane_c_fund(h)
    assert status == 400 and body["code"] == LIUM_ONLY


def test_malformed_type_is_refused(tmp_path):
    intake = _mkt_intake(tmp_path)
    h = _headers("1xRTX4090")
    h[GPU_HEADER] = "x;y"
    assert intake.lane_c_fund(h)[1]["code"] == "unknown_gpu"


def test_without_a_mkt_catalog_lane_c_keeps_lane_a_catalog(tmp_path):
    intake = _mkt_intake(tmp_path, catalog=None)
    assert intake.lane_c_fund(_headers("1xRTX3090"))[1]["code"] == "unknown_gpu"


# ── trainer rent path (rent_funded_pod mocked) ───────────────────────────────


def _runner(tmp_path, key="sk_test123"):
    from tests.unit.test_funded_pod_wiring import _runner as base
    from tests.unit.test_funded_pod_wiring import _vault
    from tests.unit.test_miner_chosen_gpu import SKUS, WALLS

    r = base(tmp_path, funded_pod_skus=SKUS, funded_sku_wall_seconds=WALLS,
             funded_sku_per_leg=True)
    r._chosen_gpu_rent = TrainerRunner._chosen_gpu_rent.__get__(r)
    _vault(tmp_path, "hkA", key)
    return r


def _challenger():
    from tests.unit.test_funded_pod_wiring import _challenger as c

    return c("hkA")


@pytest.mark.usefixtures("operator_legs_preapproved")
def test_mkt_leg_rents_the_own_risk_type_exactly(tmp_path, monkeypatch):
    from tests.unit.test_funded_pod_wiring import _rent_ok

    r = _runner(tmp_path)
    calls = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: calls.append(kw) or _rent_ok())
    with mkt_rent_scope(r, (*SHIPPED, "1xRTX3090")):
        r._rent_funded_host("777", _challenger(), gpu_choice=GpuChoice("1xrtx3090", False, 1.0))
    (kw,) = calls
    assert (kw["sku"], kw["skus"]) == ("1xRTX3090", ("1xRTX3090",))
    assert kw["price_caps"]["max_price_per_hour"] == 1.0


@pytest.mark.usefixtures("operator_legs_preapproved")
def test_mkt_leg_default_rents_from_the_mkt_list_not_lane_a(tmp_path, monkeypatch):
    from tests.unit.test_funded_pod_wiring import _rent_ok

    r = _runner(tmp_path)
    calls = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: calls.append(kw) or _rent_ok())
    with mkt_rent_scope(r, SHIPPED):
        assert r._funded_skus_for_rent() == SHIPPED
        assert r._sku_per_leg_active()
        r._rent_funded_host("777", _challenger(), gpu_choice=GpuChoice())
    assert set(calls[0]["skus"]) <= set(SHIPPED)
    assert not hasattr(r.__dict__["_leg_local_obj"], "mkt_rent_skus")   # scope restored
    from tests.unit.test_miner_chosen_gpu import SKUS

    assert r._funded_skus_for_rent() == SKUS                            # lane A unchanged


@pytest.mark.usefixtures("operator_legs_preapproved")
def test_mkt_leg_never_rents_on_shadeform(tmp_path, monkeypatch):
    from cascade.funding.providers import decode_credentials
    from tests.unit.test_funded_pod_wiring import _rent_ok

    both = encode_credentials([PayerCredential("shadeform", "sf"), PayerCredential("lium", "lk")])
    r = _runner(tmp_path, key=both)
    calls = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: calls.append(kw) or _rent_ok())
    with mkt_rent_scope(r, SHIPPED):
        r._rent_funded_host("777", _challenger(), gpu_choice=GpuChoice("1xH100"))
    assert decode_credentials(calls[0]["api_key"]) == (PayerCredential("lium", "lk"),)


@pytest.mark.usefixtures("operator_legs_preapproved")
def test_mkt_leg_with_no_lium_key_is_not_rented(tmp_path, monkeypatch):
    r = _runner(tmp_path, key=encode_credentials([PayerCredential("shadeform", "sf")]))
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: pytest.fail("rented without a Lium key"))
    with mkt_rent_scope(r, SHIPPED), pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger(), gpu_choice=GpuChoice("1xH100"))
    msg, miner_fault, error_class, burn = r._funded_leg_failures["hkA"]
    assert "Lium only" in msg and error_class == "auth" and not burn


# ── one GPU is enough: the MarketCast worker is a single process ─────────────


def test_mkt_worker_is_single_process_on_cuda0():
    from cascade.trainer import worker

    src = inspect.getsource(worker.run_mkt_worker)
    for word in ("torchrun", "world_size", "init_process_group", "WORLD_SIZE", "nproc"):
        assert word not in src
    leg = (REPO / "cascade/trainer/marketcast_leg.py").read_text(encoding="utf-8")
    for word in ("torchrun", "init_process_group", "DistributedDataParallel", "world_size"):
        assert word not in leg


# ── miner CLI ────────────────────────────────────────────────────────────────


def test_lane_c_fund_gpu_help_lists_the_recommended_types_and_own_risk_rule():
    from cascade.miner.cli import LANE_C_GPU_HELP

    for sku in SHIPPED:
        assert sku in LANE_C_GPU_HELP
    for words in ("OWN RISK", "max_leg_hours", "never Shadeform", "nothing is refunded"):
        assert words in LANE_C_GPU_HELP
    import argparse

    from cascade.miner.cli import add_gpu_args

    p = argparse.ArgumentParser()
    add_gpu_args(p, lane_c=True)
    (act,) = [a for a in p._actions if "--gpu" in a.option_strings]
    assert act.help == LANE_C_GPU_HELP
    src = (REPO / "cascade/miner/lane_c.py").read_text(encoding="utf-8")
    assert "add_gpu_args(f, lane_c=True)" in src


# ── validators untouched ─────────────────────────────────────────────────────


def test_no_validator_or_audit_code_reads_the_mkt_gpu_choice():
    for sub in ("cascade/validator", "cascade/audit"):
        for f in (REPO / sub).rglob("*.py"):
            text = f.read_text(encoding="utf-8")
            assert "mkt_gpus" not in text, f
            assert "funded_pod_skus" not in text, f
