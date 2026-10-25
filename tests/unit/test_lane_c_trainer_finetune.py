"""DEC-CA-0056 lane C — the generic king fine-tune mode, UNREACHABLE since
MarketCast-only lane C (owner 2026-10-20: no generic target loads, commits or
funds; the driver routes MarketCast legs only) and kept until the
``role="finetune"`` plumbing is removed: the leg contract (budget / LR / schedule from the recipe, no
architecture change, no finished-form branch), the weights-only staged base
(fresh optimizer), the worker/dispatch plumbing, and a tiny MoE fine-tune
that loads the base strictly."""

from __future__ import annotations

import dataclasses
import json

import pytest

from cascade.shared.config.lane_c import LaneCConfig
from cascade.trainer.finetune import (
    LegRecipe,
    finetune_contract,
    leg_contract,
    leg_suffix,
    stage_finetune_base,
)
from cascade.trainer.remote import RemoteHost, receipt_role, worker_argv
from cascade.trainer.toto2_trainer import _lr_at

LC = LaneCConfig(targets=("domain:energy", "horizon:720"), max_lr_scale=2.0,
                 max_steps_fraction=0.5, leg_hours=1.0)
GOOD = {"target": "domain:energy", "lr_scale": 0.5, "steps_fraction": 0.5,
        "warmup_fraction": 0.1, "schedule": "linear"}


def _tree(tmp_path, obj):
    d = tmp_path / "tree"
    d.mkdir(exist_ok=True)
    (d / "generator.py").write_text("def generate(): ...\n")
    (d / "recipe.json").unlink(missing_ok=True)
    if obj is not None:
        (d / "recipe.json").write_text(obj if isinstance(obj, str) else json.dumps(obj))
    return d


def test_leg_contract_scales_budget_and_lr_never_the_architecture(cfg):
    base = cfg.training.primary_size
    rec = LegRecipe("domain:energy", 0.5, 0.5, 0.1, "linear")
    c = finetune_contract(base, rec, leg_hours=1.0, guard_factor=1.0, guard_floor_seconds=0)
    assert c.train_tokens == base.tokens_for_hours(0.5)
    assert c.max_train_seconds <= 3600
    assert c.base_lr == pytest.approx(base.base_lr * 0.5)
    assert (c.lr_schedule, c.warmup_fraction) == ("warmup_linear", 0.1)
    assert c.ema_decay == 0.0 and c.anneal_fraction == 0.0 and c.warm_lr_scale == 1.0
    for f in ("arch_preset", "base_arch_digest", "d_model", "num_layers", "num_heads",
              "patch_size", "context_length", "corpus_mode", "budget_denomination"):
        assert getattr(c, f) == getattr(base, f), f
    assert finetune_contract(base, dataclasses.replace(rec, schedule="constant"),
                             leg_hours=1.0).lr_schedule == "warmup_flat"
    with pytest.raises(ValueError):
        finetune_contract(base, dataclasses.replace(rec, lr_scale=3.0), leg_hours=1.0,
                          max_lr_scale=2.0)
    with pytest.raises(ValueError):
        finetune_contract(base, dataclasses.replace(rec, steps_fraction=0.9), leg_hours=1.0,
                          max_steps_fraction=0.5)
    # the chain-config wrapper reads [lane_c] leg_hours + bounds
    cc = dataclasses.replace(cfg, lane_c=dataclasses.replace(LC, leg_hours=2.0))
    assert leg_contract(cc, base, rec).train_tokens == base.tokens_for_hours(1.0)


def test_linear_schedule():
    assert _lr_at(0, 1000, 100, 1.0, schedule="warmup_linear") == 0.0
    assert _lr_at(100, 1000, 100, 1.0, schedule="warmup_linear") == 1.0
    assert _lr_at(550, 1000, 100, 1.0, schedule="warmup_linear") == pytest.approx(0.5)
    assert _lr_at(1000, 1000, 100, 1.0, schedule="warmup_linear") == 0.0
    # the shipped schedules are unchanged
    assert _lr_at(550, 1000, 100, 1.0) == pytest.approx(0.5)


def test_staged_base_is_weights_only(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    for f in ("weights.safetensors", "config.json", "model.py", "forecast_wrapper.py",
              "optimizer.safetensors", "power_ema.safetensors", "weights_stable.safetensors"):
        (base / f).write_bytes(f.encode())
    out = stage_finetune_base(base, tmp_path / "staged")
    assert sorted(p.name for p in out.iterdir()) == [
        "config.json", "forecast_wrapper.py", "model.py", "weights.safetensors"]
    assert (out / "weights.safetensors").read_bytes() == b"weights.safetensors"
    with pytest.raises(FileNotFoundError):
        stage_finetune_base(tmp_path / "staged" / "nope", tmp_path / "x")


def test_worker_argv_and_receipt_role():
    host = RemoteHost(name="p", host="1.2.3.4")
    argv = worker_argv(host, gen_ref="a/b@sha256:" + "1" * 64, uid=3, hotkey="H",
                       role="finetune", base_seed=1, block=2, trainer_spec="m:T",
                       warm_start_ref="hub://k@sha256:" + "2" * 64,
                       repo_suffix=leg_suffix("domain:energy", 3),
                       finetune_recipe=json.dumps(GOOD))
    assert argv[argv.index("--recipe-json") + 1] == json.dumps(GOOD)
    assert "--repo-suffix=-cdomain_energy-u3" in argv
    with pytest.raises(ValueError):
        worker_argv(host, gen_ref="a/b@sha256:" + "1" * 64, uid=3, hotkey="H",
                    role="finetune", base_seed=1, block=2, trainer_spec="m:T")
    # every other role's argv is unchanged
    plain = worker_argv(host, gen_ref="a/b@sha256:" + "1" * 64, uid=3, hotkey="H",
                        role="challenger", base_seed=1, block=2, trainer_spec="m:T")
    assert "--recipe-json" not in plain
    assert receipt_role("finetune") == receipt_role("expert") == "challenger"
    assert receipt_role("king") == "king"


def test_worker_rejects_a_bad_recipe_as_a_miner_fault():
    from cascade.trainer.worker import main

    base = ["--gen-ref", "a/b@sha256:" + "1" * 64, "--uid", "1", "--hotkey", "H",
            "--role", "finetune", "--base-seed", "1", "--block", "1",
            "--trainer", "cascade.trainer.toto2_trainer:Toto2Trainer"]
    assert main(base) == 2                                     # no recipe / base
    bad = json.dumps({**GOOD, "lr_scale": 50.0})
    assert main(base + ["--recipe-json", bad, "--warm-start-ref",
                        "hub://k@sha256:" + "2" * 64]) == 3


# ── a tiny MoE fine-tune (torch) ─────────────────────────────────────────────


def test_finetune_loads_the_base_strictly_with_a_fresh_optimizer(tmp_path):
    pytest.importorskip("torch")
    from safetensors.torch import load_file

    from .test_moe_trainer import _C, _train

    c = _C()
    base = tmp_path / "base"
    r0 = _train(base, c)
    assert (base / "optimizer.safetensors").is_file()          # the lineage branch exists
    staged = stage_finetune_base(base, tmp_path / "staged")
    ft = dataclasses.replace(c, lr_schedule="warmup_linear", power_ema_sigma_rels=(),
                             power_ema_eval=-1, base_lr=1e-30, warmup_tokens=0)
    r1 = _train(tmp_path / "ft", ft, budget=200, warm=staged)
    assert r1.metrics["optim_state_resumed"] is False
    assert r1.metrics["lr_schedule"] == "warmup_linear"
    assert not (tmp_path / "ft" / "optimizer.safetensors").exists()
    w0 = load_file(str(base / "weights.safetensors"))
    w1 = load_file(str(tmp_path / "ft" / "weights.safetensors"))
    assert w0.keys() == w1.keys()
    # at a vanishing LR the fine-tune is the base's scored weights (routing
    # biases aside, which loss-free balancing moves without the LR)
    drift = max(float((w0[k].float() - w1[k].float()).abs().max())
                for k in w0 if "bias" not in k and "route" not in k)
    assert drift < 1e-3, drift
    assert r0.param_count == r1.param_count
    # an architecture change cannot load the base: the leg aborts
    with pytest.raises(RuntimeError):
        _train(tmp_path / "bad", dataclasses.replace(ft, d_model=64, umup_base_d_model=64),
               budget=200, warm=staged)
