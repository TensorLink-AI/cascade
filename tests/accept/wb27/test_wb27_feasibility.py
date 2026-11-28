"""One pre-rent feasibility check for every paid leg (retro 2026-11-28 #5).

Tonight's paid failures were all knowable before the rent:

* 01:27 the lane B judge pod was a Blackwell (sm_120) card; the pinned
  ``torch 2.4.1+cu124`` has no kernels for it ("no kernel image is
  available"), a settlement slot lost;
* 2026-11-27 22:10 a MarketCast recipe OOMed a recommended 1xRTX4090 (#989);
* 2026-11-27 23:24 a recipe's data range could not cut a 15m sample after
  30 min of paid prep (#990).

``cascade.funding.feasibility`` is the one check: arch vs the pinned torch's
recorded arch list (no torch import), memory vs estimate, data geometry,
image qualified + walls measured, target disk vs pull + headroom. The
intake refuses at fund time with a typed code; the trainer re-runs it right
before each rent and refuses BEFORE any provider call; the lane B judge pod
checks its observed compute capability before any fetch or model load.

Torch-free: torch is made unimportable for every test here.
"""

from __future__ import annotations

import copy
import json
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIX = ROOT / "tests" / "accept" / "w_cmp" / "fixtures"


@pytest.fixture(autouse=True)
def _no_torch(monkeypatch):
    """Every test here runs with ``import torch`` failing (box rule)."""
    if "torch" not in sys.modules:
        monkeypatch.setitem(sys.modules, "torch", None)
    yield


# ── 1. GPU architecture vs the pinned torch ─────────────────────────────────


def test_sm120_and_blackwell_refused_for_the_pinned_torch():
    from cascade.funding import feasibility as fz

    for gpu in ("1xRTX5090", "RTXPRO6000", "8xB200", "4xB300"):
        r = fz.gpu_arch_refusal(gpu)
        assert r is not None and r.code == "gpu_arch_unsupported", gpu
        assert r.actor == "miner" and "2.4.1+cu124" in r.message
    # the observed compute capability of the 01:27 judge pod
    r = fz.gpu_arch_refusal(compute_cap=fz.parse_compute_cap("12.0\n12.0\n"))
    assert r is not None and r.detail["compute_capability"] == "12.0"


def test_ampere_ada_hopper_pass():
    from cascade.funding import feasibility as fz

    for gpu in ("1xRTX4090", "L40S", "1xA100", "8xH100", "8xH200", "RTX3090", "A6000"):
        assert fz.gpu_arch_refusal(gpu) is None, gpu
    for cap in ((8, 9), (8, 0), (8, 6), (9, 0)):
        assert fz.gpu_arch_refusal(compute_cap=cap) is None, cap
    assert fz.gpu_arch_refusal("SOMENEWTYPE") is None      # unknown: the pod checks


def test_default_list_refused_only_when_no_type_runs():
    from cascade.funding import feasibility as fz

    assert fz.gpu_arch_refusal(chosen=False, candidates=("8xB200", "8xH100")) is None
    r = fz.gpu_arch_refusal(chosen=False, candidates=("8xB200", "4xB300"))
    assert r is not None and r.code == "no_supported_gpu" and r.actor == "operator"


def test_the_arch_table_tracks_the_locked_torch():
    """A torch repin must record its arch list (the table is data)."""
    from cascade.funding import feasibility as fz

    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    versions = set(re.findall(r'name = "torch"\nversion = "([^"]+)"', lock))
    assert fz.PINNED_TORCH in versions, versions
    assert fz.PINNED_TORCH in fz.TORCH_CUDA_ARCH


def test_intake_gpu_header_refuses_a_blackwell_choice_free():
    from cascade.funding.gpu_choice import GPU_HEADER, GpuCatalog, parse_gpu_headers

    cat = GpuCatalog(skus=("1xRTX5090", "1xRTX4090"))
    choice, seg, err = parse_gpu_headers({GPU_HEADER: "1xRTX5090"}, cat)
    assert err is not None and err["code"] == "gpu_arch_unsupported", err
    assert seg == "" and not choice.chosen
    choice, _seg, err = parse_gpu_headers({GPU_HEADER: "1xRTX4090"}, cat)
    assert err is None and choice.gpu == "1xRTX4090"


# ── 2. memory, geometry, image, walls, disk ─────────────────────────────────


def test_vram_unfit_recipe_refused_on_a_4090_and_fits_a100():
    from cascade.funding import feasibility as fz

    recipe = json.loads((FIX / "mkt_oom_recipe.json").read_text())
    rec = ("1xRTX4090", "1xA100", "1xH100", "8xH100")
    r = fz.feasibility(fz.LegPlan(lane="c", gpu="1xRTX4090", mkt_recipe=recipe,
                                  candidates=rec))
    assert r is not None and r.code == "recipe_exceeds_gpu_memory" and r.actor == "miner"
    assert r.detail["fitting_gpus"] == ["1xA100", "1xH100", "8xH100"]
    assert fz.feasibility(fz.LegPlan(lane="c", gpu="1xA100", mkt_recipe=recipe,
                                     candidates=rec)) is None


def test_lane_b_expert_context_must_fit_measured_memory():
    from cascade.funding import feasibility as fz

    small = SimpleNamespace(peak_memory_bytes=20 << 30, memory_total_bytes=24 << 30)
    big = SimpleNamespace(peak_memory_bytes=10 << 30, memory_total_bytes=80 << 30)
    assert fz.lane_b_vram_refusal(small, 4096) is None            # measured at 4096
    r = fz.lane_b_vram_refusal(small, 8192)
    assert r is not None and r.code == "context_length_unfit"
    assert fz.lane_b_vram_refusal(big, 8192) is None
    assert fz.lane_b_vram_refusal(None, 8192) is not None         # unmeasured type
    assert fz.lane_b_vram_refusal(None, 8192, own_risk=True) is None


def test_uncuttable_data_range_refused():
    from cascade.funding import feasibility as fz
    from cascade.shared.config import load_chain_config

    recipe = json.loads((FIX / "mkt_geom_recipe.json").read_text())
    index = json.loads((FIX / "mkt_public_index_20261127.json").read_text())
    mc = load_chain_config(ROOT / "chain.toml").marketcast
    r = fz.feasibility(fz.LegPlan(lane="c", gpu="1xA100", mkt_recipe=recipe, mkt_cfg=mc,
                                  mkt_history=index, candidates=("1xA100",)))
    assert r is not None and r.code == "recipe_data_range_uncuttable", r
    assert r.actor == "miner" and r.detail["timeframe"] == "15m"
    fixed = copy.deepcopy(recipe)
    fixed["data"]["start"] = r.detail["fits_from_start"]
    assert fz.feasibility(fz.LegPlan(lane="c", gpu="1xA100", mkt_recipe=fixed, mkt_cfg=mc,
                                     mkt_history=index, candidates=("1xA100",))) is None


def test_unqualified_image_and_unmeasured_walls_refused_as_operator():
    from cascade.funding import feasibility as fz

    r = fz.feasibility(fz.LegPlan(gpu="8xH100", image_reason="no qualification record"))
    assert r is not None and (r.code, r.actor) == ("image_unqualified", "operator")
    r = fz.feasibility(fz.LegPlan(gpu="8xH100", walls_reason="walls measured on another image"))
    assert r is not None and (r.code, r.actor) == ("walls_unmeasured", "operator")
    # operator-side refusals come first: never tell a miner about their choice
    # while our own image is the blocker
    r = fz.feasibility(fz.LegPlan(gpu="1xRTX5090", image_reason="x"))
    assert r.code == "image_unqualified"
    assert fz.feasibility(fz.LegPlan(gpu="8xH100")) is None


def test_low_target_disk_refused():
    from cascade.funding import feasibility as fz

    gb = 10**9
    r = fz.feasibility(fz.LegPlan(gpu="8xH100", disk_free_bytes=30 * gb,
                                  disk_need_bytes=25 * gb, disk_headroom_bytes=10 * gb))
    assert r is not None and (r.code, r.actor) == ("target_disk_short", "operator")
    assert fz.feasibility(fz.LegPlan(gpu="8xH100", disk_free_bytes=40 * gb,
                                     disk_need_bytes=25 * gb,
                                     disk_headroom_bytes=10 * gb)) is None


def test_public_reason_uses_the_code_template_never_the_text():
    from cascade.funding.feasibility import Refusal, failure_message
    from cascade.funding.public_reason import public_reason

    msg = failure_message(Refusal("gpu_arch_unsupported", "pod 10.1.2.3 secret", "arch",
                                  "miner"))
    code, text = public_reason({"status": "failed", "last_error_class": "infeasible",
                                "last_error": msg})
    assert code == "gpu_arch_unsupported" and "nothing rented" in text
    assert "10.1.2.3" not in text and "secret" not in text
    code, text = public_reason({"status": "queued", "last_error_class": "operator_blocked",
                                "last_error": failure_message(Refusal(
                                    "no_supported_gpu", "x", "arch", "operator"))})
    assert code == "no_supported_gpu" and "no cost to you" in text


# ── 3. the rent paths refuse before any provider call ───────────────────────


def _call(fn):
    """The exception ``fn`` raised (None if none). The Accept on BASE needs an
    assertion failure, not a pytest.raises "DID NOT RAISE"."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return e
    return None


def test_funded_rent_refuses_a_blackwell_choice_before_renting(tmp_path, monkeypatch):
    import cascade.provision.funded as funded_mod
    from cascade.funding.gpu_choice import GpuChoice
    from cascade.trainer.faults import _FundedLegSkip
    from cascade.trainer.loop import TrainerRunner
    from tests.unit.test_funded_pod_wiring import _challenger, _runner, _vault

    skus = ("RTX5090", "RTX4090")
    r = _runner(tmp_path, funded_pod_skus=skus,
                funded_sku_wall_seconds={"RTX5090": 9000, "RTX4090": 13500})
    r._chosen_gpu_rent = TrainerRunner._chosen_gpu_rent.__get__(r)
    _vault(tmp_path, "hkA")
    calls = []

    def _rent(**kw):
        calls.append(kw)
        raise AssertionError("rented before the feasibility check refused")

    monkeypatch.setattr(funded_mod, "rent_funded_pod", _rent)
    raised = _call(lambda: r._rent_funded_host("777", _challenger("hkA"),
                                               gpu_choice=GpuChoice("RTX5090")))
    assert calls == [], "the pod was rented before the feasibility refusal"
    assert isinstance(raised, _FundedLegSkip), f"expected _FundedLegSkip, got {raised!r}"
    msg, miner_fault, cls, burn = r._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (True, "infeasible", False)
    assert msg.startswith("feasibility gpu_arch_unsupported:")
    assert r._load_funded_ledger() == []          # not even a write-ahead row


def test_funded_rent_default_list_all_blackwell_is_an_operator_block(tmp_path, monkeypatch):
    import cascade.provision.funded as funded_mod
    from cascade.trainer.faults import _FundedLegSkip
    from tests.unit.test_funded_pod_wiring import _challenger, _runner, _vault

    r = _runner(tmp_path, funded_pod_skus=("8xB200", "4xB300"),
                funded_sku_wall_seconds={"8xB200": 6866, "4xB300": 10299})
    _vault(tmp_path, "hkA")
    calls = []

    def _rent(**kw):
        calls.append(kw)
        raise AssertionError("rented before the feasibility check refused")

    monkeypatch.setattr(funded_mod, "rent_funded_pod", _rent)
    raised = _call(lambda: r._rent_funded_host("777", _challenger("hkA")))
    assert calls == [], "the pod was rented before the feasibility refusal"
    assert isinstance(raised, _FundedLegSkip), f"expected _FundedLegSkip, got {raised!r}"
    assert "hkA" in r._funded_leg_failures, "no verdict recorded for the refused leg"
    _msg, miner_fault, cls, burn = r._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "operator_blocked", False)


def test_mkt_leg_refuses_an_unfit_recipe_before_the_funded_leg(monkeypatch):
    from cascade.trainer import marketcast_runner as mr
    from cascade.trainer import money_guard
    from cascade.trainer.lane_c_driver import LegFailure

    recipe = json.loads((FIX / "mkt_oom_recipe.json").read_text())
    rec = mr.MktLegRecipe(target="mkt:returns", payload=json.dumps(recipe),
                          has_generator=False)
    monkeypatch.setattr(mr, "mkt_leg_contract", lambda cfg, base, hours: object())
    monkeypatch.setattr(mr, "mkt_leg_hours", lambda cfg, hours=0.0: 1.0)
    monkeypatch.setattr(money_guard, "operator_pinned_skus", lambda *a, **k: ())
    cfg = SimpleNamespace(
        throne_contracts_at=lambda b: [object()],
        marketcast=SimpleNamespace(funded_pod_skus=("1xRTX4090", "1xA100"),
                                   horizons_map={}, micro_batch_tokens=0, base_ref=""),
        round=SimpleNamespace(funded_pod_skus=()))
    ran = []
    r = SimpleNamespace(cfg=cfg, _funded_leg_failures={}, _mkt_harvested={},
                        _run_funded_leg=lambda *a, **k: ran.append(1),
                        _dispatcher=lambda: None)
    ops = SimpleNamespace(r=r, _seeds=lambda: (SimpleNamespace(base_seed=1),
                                               SimpleNamespace(start_block=1, index=1)),
                          queue=lambda: None)
    leg = {"hotkey": "hkC", "uid": 7, "ref": "x@sha256:" + "a" * 64, "gpu": "1xRTX4090",
           "target": "mkt:returns", "started_block": 1}
    with pytest.raises(LegFailure) as ei:
        mr.train_mkt_leg(ops, leg, rec)
    f = ei.value
    assert (f.miner_fault, f.error_class, f.burn) == (True, "infeasible", False)
    assert f.msg.startswith("feasibility recipe_exceeds_gpu_memory:")
    assert ran == []
    # the same leg on a Blackwell choice: refused on arch, still nothing run
    with pytest.raises(LegFailure) as ei:
        mr.train_mkt_leg(ops, {**leg, "gpu": "1xRTX5090"}, rec)
    assert ei.value.msg.startswith("feasibility gpu_arch_unsupported:") and ran == []


def test_mkt_pre_rent_keeps_the_own_risk_memory_policy():
    from cascade.trainer import marketcast_runner as mr

    recipe = json.loads((FIX / "mkt_oom_recipe.json").read_text())
    rec = mr.MktLegRecipe(target="mkt:returns", payload=json.dumps(recipe),
                          has_generator=False)
    mc = SimpleNamespace(horizons_map={}, micro_batch_tokens=0)
    own = SimpleNamespace(gpu="1xRTX3090")          # own-risk: accepted with a warning
    assert mr.mkt_pre_rent_refusal(rec, ("1xRTX3090",), own, mc,
                                   recommended=("1xRTX4090", "1xA100")) is None


def test_lane_b_judge_pod_refuses_sm120_before_any_fetch(tmp_path, monkeypatch):
    from cascade.trainer import lane_b_offload as lbo

    assert lbo.pod_gpu_arch_refusal("cpu", query=lambda: "12.0") is None
    assert lbo.pod_gpu_arch_refusal("cuda", query=lambda: "8.9\n") is None
    assert lbo.pod_gpu_arch_refusal("cuda", query=lambda: "") is None
    r = lbo.pod_gpu_arch_refusal("cuda:0", query=lambda: "12.0\n")
    assert r is not None and r.code == "gpu_arch_unsupported"

    monkeypatch.setattr(lbo, "_nvidia_compute_cap", lambda: "12.0\n")
    def _fetch(*a, **k):
        raise AssertionError("fetched before the arch refusal")

    monkeypatch.setattr(lbo, "_pod_fetch", _fetch)
    run = tmp_path / "run"
    run.mkdir()
    (run / "payload.pkl").write_bytes(pickle.dumps({"device": "cuda", "windows": []}))
    out = lbo._pod_op("eval", {"pointer": "p"}, run, tmp_path / "root")
    assert set(out) == {"offload_unavailable"}
    assert out["offload_unavailable"].startswith("feasibility gpu_arch_unsupported:")
