"""Toto2-aligned optimizer constants + warm-start LR scale (DEC-CA-0035).

Measured 2026-08-27 (docs/notes/2026-08-27-toto2-alignment.md): the full
Toto 2.0 constants bundle (momentum 0.96, row-EMA β₂ 0.999, clip 7.0,
wd 2e-8, AdamW β 0.91/0.972, 54:1 matrix:AdamW split) at warm LR
base_lr×0.125 BEAT the converged warm-start init by 0.0038 — the campaign
best; every knob here ships inert at the previously hardcoded default.
These tests freeze (a) the unarmed byte-compatibility, (b) each knob's
application, (c) the loader round-trip — the DEC-CA-0033 fields shipped
without loader parsing, so a TOML arming would have silently no-oped;
that regression is pinned here too.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cascade.trainer.toto2_trainer import Toto2Trainer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _contract(**kw) -> SimpleNamespace:
    base = dict(
        context_length=16, horizon=8, patch_size=4, d_model=16, num_layers=1,
        num_heads=1, head_dim=16, mlp_expansion=2, num_quantiles=9,
        batch_size=4, max_train_seconds=300, base_lr=1e-3,
        weight_decay=0.0, optimizer="adamw", warmup_tokens=0,
        input_transform="arcsinh_causal", lr_schedule="wsd",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _series(n: int):
    rng = np.random.default_rng(0)
    for _ in range(n):
        yield rng.normal(size=32).cumsum()


def _weights(path: Path) -> dict:
    from safetensors.torch import load_file

    return load_file(str(path))


def _same_state(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


class _TinyNet(torch.nn.Module):
    """One matrix param (muon route) + an embed + a bias (adamw route)."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(4, 8)
        self.proj = torch.nn.Linear(8, 8)


# ── optimizer constants ──────────────────────────────────────────────────────


def test_build_optimizer_defaults_match_previous_hardcoded_values():
    opt = Toto2Trainer(device="cpu")._build_optimizer(
        _TinyNet(), _contract(optimizer="normuon_adamw"))
    assert opt.momentum == 0.95
    assert opt.beta2 == 0.95
    for grp in opt.adamw.param_groups:
        assert grp["betas"] == (0.9, 0.999)
        assert grp["lr_scale"] == 1.0


def test_build_optimizer_applies_toto2_constants():
    opt = Toto2Trainer(device="cpu")._build_optimizer(
        _TinyNet(), _contract(optimizer="normuon_adamw",
                              muon_momentum=0.96, muon_row_beta2=0.999,
                              adamw_beta1=0.91, adamw_beta2=0.972,
                              adamw_lr_scale=1.0 / 54.0))
    assert opt.momentum == 0.96
    assert opt.beta2 == 0.999
    for grp in opt.adamw.param_groups:
        assert grp["betas"] == (0.91, 0.972)
        assert grp["lr_scale"] == pytest.approx(1.0 / 54.0)


@pytest.mark.parametrize("bad", [
    {"muon_momentum": 1.0}, {"muon_row_beta2": 0.0},
    {"adamw_beta1": 1.5}, {"adamw_beta2": -0.1}, {"adamw_lr_scale": 0.0},
])
def test_optimizer_constant_bounds(bad):
    with pytest.raises(ValueError, match="must be in"):
        Toto2Trainer(device="cpu")._build_optimizer(
            _TinyNet(), _contract(optimizer="normuon_adamw", **bad))


# ── grad clip ────────────────────────────────────────────────────────────────


def test_grad_clip_reaches_the_call_site(tmp_path: Path, monkeypatch):
    seen: list[float] = []
    orig = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, **kw):
        seen.append(float(max_norm))
        return orig(params, max_norm, **kw)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    trainer.train(_series(16), _contract(grad_clip=7.0), training_seed=1,
                  token_budget=128, out_dir=tmp_path / "a")
    assert seen and set(seen) == {7.0}
    seen.clear()
    trainer.train(_series(16), _contract(), training_seed=1,
                  token_budget=128, out_dir=tmp_path / "b")
    assert seen and set(seen) == {1.0}      # unarmed = deployed behaviour


def test_grad_clip_bounds(tmp_path: Path):
    with pytest.raises(ValueError, match="grad_clip must be"):
        Toto2Trainer(device="cpu", deterministic=False).train(
            _series(4), _contract(grad_clip=-1.0), training_seed=1,
            token_budget=128, out_dir=tmp_path / "c")


# ── warm-start LR scale ──────────────────────────────────────────────────────


def test_warm_lr_scale_is_inert_from_scratch(tmp_path: Path):
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    r = trainer.train(_series(32), _contract(warm_lr_scale=0.125),
                      training_seed=1, token_budget=256, out_dir=tmp_path / "a")
    assert "warm_lr_scale" not in r.metrics
    # and the from-scratch weights are byte-identical to a no-knob run
    r2 = trainer.train(_series(32), _contract(), training_seed=1,
                       token_budget=256, out_dir=tmp_path / "b")
    assert r2.metrics["tokens_seen"] == r.metrics["tokens_seen"]
    assert _same_state(_weights(tmp_path / "a" / "weights.safetensors"),
                       _weights(tmp_path / "b" / "weights.safetensors"))


def test_warm_lr_scale_scales_the_warm_started_run(tmp_path: Path):
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    init = tmp_path / "init"
    trainer.train(_series(32), _contract(), training_seed=1,
                  token_budget=256, out_dir=init)
    full = tmp_path / "full"
    scaled = tmp_path / "scaled"
    trainer.train(_series(32), _contract(), training_seed=2,
                  token_budget=256, out_dir=full, warm_start_dir=init)
    r = trainer.train(_series(32), _contract(warm_lr_scale=0.125),
                      training_seed=2, token_budget=256, out_dir=scaled,
                      warm_start_dir=init)
    assert r.metrics["warm_lr_scale"] == 0.125
    assert not _same_state(_weights(full / "weights.safetensors"),
                           _weights(scaled / "weights.safetensors"))


def test_warm_lr_scale_bounds(tmp_path: Path):
    with pytest.raises(ValueError, match="warm_lr_scale must be in"):
        Toto2Trainer(device="cpu", deterministic=False).train(
            _series(4), _contract(warm_lr_scale=1.5), training_seed=1,
            token_budget=128, out_dir=tmp_path / "c")


# ── digest + loader ──────────────────────────────────────────────────────────


def test_constants_are_digest_bound_drop_when_default():
    from cascade.shared.manifest import contract_digest

    base = {"base_lr": 4e-3, "lr_schedule": "wsd"}
    inert = dict(base, muon_momentum=0.95, muon_row_beta2=0.95, grad_clip=1.0,
                 adamw_beta1=0.9, adamw_beta2=0.999, adamw_lr_scale=1.0,
                 warm_lr_scale=1.0)
    assert contract_digest(base) == contract_digest(inert)
    for armed in (dict(base, muon_momentum=0.96),
                  dict(base, muon_row_beta2=0.999),
                  dict(base, grad_clip=7.0),
                  dict(base, adamw_beta1=0.91),
                  dict(base, adamw_beta2=0.972),
                  dict(base, adamw_lr_scale=1.0 / 54.0),
                  dict(base, warm_lr_scale=0.125)):
        assert contract_digest(armed) != contract_digest(base)


def test_loader_round_trips_bundle_and_constants(tmp_path: Path):
    """Regression: the DEC-CA-0033 fields landed with dataclass defaults but
    no loader parsing — arming them in chain.toml silently no-oped. Every
    armable [training] knob must survive load_chain_config."""
    from cascade.shared.config import load_chain_config

    import re

    text = (REPO_ROOT / "chain.toml").read_text()
    # The shipped file ARMS the DEC-CA-0035 constants (2026-08-27 release), so
    # the round trip is proven by REPLACING each armed line with a different
    # value and reading it back; the two still-absent knobs are inserted.
    swaps = {
        "ema_decay": "0.5", "muon_momentum": "0.97", "muon_row_beta2": "0.998",
        "grad_clip": "5.0", "adamw_beta1": "0.92", "adamw_beta2": "0.973",
        "adamw_lr_scale": "0.02", "warm_lr_scale": "0.25",
    }
    patched = text
    for key, val in swaps.items():
        patched, n = re.subn(rf"(?m)^{key}\s*=\s*\S+", f"{key} = {val}", patched, count=1)
        assert n == 1, f"expected exactly one armed {key} line in chain.toml"
    assert "\nbase_lr" in patched
    patched = patched.replace(
        "\nbase_lr", "\ngen_seed_mix = 3\nrewarmup_fraction = 0.02\nbase_lr", 1)
    p = tmp_path / "chain.toml"
    p.write_text(patched)
    t = load_chain_config(p).training
    assert t.ema_decay == 0.5
    assert t.gen_seed_mix == 3
    assert t.rewarmup_fraction == 0.02
    assert t.muon_momentum == 0.97
    assert t.muon_row_beta2 == 0.998
    assert t.grad_clip == 5.0
    assert (t.adamw_beta1, t.adamw_beta2) == (0.92, 0.973)
    assert t.adamw_lr_scale == 0.02
    assert t.warm_lr_scale == 0.25


def test_armed_constants_refuse_non_normuon_optimizer():
    """An armed optimizer knob on the plain-AdamW fallback path would bump
    contract_digest while changing no numerics — refused instead."""
    with pytest.raises(ValueError, match="require optimizer="):
        Toto2Trainer(device="cpu")._build_optimizer(
            _TinyNet(), _contract(optimizer="adamw", muon_momentum=0.96))
    opt = Toto2Trainer(device="cpu")._build_optimizer(_TinyNet(), _contract())
    assert isinstance(opt, torch.optim.AdamW)   # defaults: fallback still fine
