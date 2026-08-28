"""Fork-anneal: finished-form duel checkpoints (DEC-CA-0029).

The deferred "D" of the wsd schedule: past the stable token budget the run
forks — mid-stable weights + optimizer state are retained as the lineage
branch (``weights_stable.safetensors`` + ``optimizer.safetensors``) and
training continues under a cosine decay to 0, with the ANNEALED weights
becoming ``weights.safetensors`` (the artifact every scoring layer loads).
Warm-starts must resume from the stable branch, never the decayed endpoint.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cascade.trainer.toto2_trainer import (  # noqa: E402
    OPTIM_STATE_FILE,
    STABLE_WEIGHTS_FILE,
    Toto2Trainer,
    _anneal_lr,
)


def _contract(max_secs: int = 300, **kw) -> SimpleNamespace:
    base = dict(
        context_length=16, horizon=8, patch_size=4, d_model=16, num_layers=1,
        num_heads=1, head_dim=16, mlp_expansion=2, num_quantiles=9,
        batch_size=4, max_train_seconds=max_secs, base_lr=1e-3,
        weight_decay=0.0, optimizer="adamw", warmup_tokens=0,
        input_transform="arcsinh_causal", lr_schedule="wsd",
        anneal_fraction=0.25,
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


# ── the anneal LR curve ───────────────────────────────────────────────────────


def test_anneal_lr_cosine_from_base_to_zero():
    assert _anneal_lr(0, 1000, 4e-3) == pytest.approx(4e-3)
    assert _anneal_lr(500, 1000, 4e-3) == pytest.approx(2e-3)
    assert _anneal_lr(1000, 1000, 4e-3) == pytest.approx(0.0, abs=1e-12)
    assert _anneal_lr(2000, 1000, 4e-3) == pytest.approx(0.0, abs=1e-12)  # clamped
    lrs = [_anneal_lr(p, 1000, 4e-3) for p in range(0, 1001, 50)]
    assert all(x >= y for x, y in zip(lrs, lrs[1:], strict=False))  # monotone decay


# ── the fork ──────────────────────────────────────────────────────────────────


def test_anneal_run_ships_annealed_weights_plus_lineage_branch(tmp_path: Path):
    out = tmp_path / "ckpt"
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    result = trainer.train(_series(64), _contract(), training_seed=1,
                           token_budget=256, out_dir=out)
    m = result.metrics
    # anneal ran: tokens past the stable budget, telemetry present
    assert m["anneal_tokens"] == 64                      # 0.25 × 256
    assert m["anneal_tokens_seen"] > 0
    assert m["deadline_hit"] is False
    # both branches shipped, and they differ (decayed steps moved the weights)
    annealed = _weights(out / "weights.safetensors")
    stable = _weights(out / STABLE_WEIGHTS_FILE)
    assert (out / OPTIM_STATE_FILE).is_file()
    assert not _same_state(annealed, stable)


def test_wall_expiry_inside_stable_phase_ships_matching_branches(tmp_path: Path):
    # An already-expired wall stops the run before the fork: the endpoint IS
    # mid-stable, so the two weight files must match and the anneal telemetry
    # must show zero decayed tokens (deadline_hit marks the under-budget run).
    out = tmp_path / "ckpt"
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    result = trainer.train(_series(64), _contract(max_secs=0), training_seed=1,
                           token_budget=1_000_000, out_dir=out)
    m = result.metrics
    assert m["deadline_hit"] is True
    assert m["anneal_tokens_seen"] == 0
    assert _same_state(_weights(out / "weights.safetensors"),
                       _weights(out / STABLE_WEIGHTS_FILE))


def test_anneal_requires_wsd_schedule(tmp_path: Path):
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    with pytest.raises(ValueError, match="anneal_fraction.*requires"):
        trainer.train(_series(4), _contract(lr_schedule="warmup_cosine"),
                      training_seed=1, token_budget=256, out_dir=tmp_path / "c")


def test_anneal_fraction_bounds(tmp_path: Path):
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    with pytest.raises(ValueError, match="anneal_fraction must be in"):
        trainer.train(_series(4), _contract(anneal_fraction=1.5),
                      training_seed=1, token_budget=256, out_dir=tmp_path / "c")


# ── lineage continuity ────────────────────────────────────────────────────────


def test_warm_start_resumes_from_stable_branch_not_annealed(tmp_path: Path):
    # Round 1: annealed checkpoint with both branches.
    gen1 = tmp_path / "gen1"
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    trainer.train(_series(64), _contract(), training_seed=1,
                  token_budget=256, out_dir=gen1)
    stable = _weights(gen1 / STABLE_WEIGHTS_FILE)
    annealed = _weights(gen1 / "weights.safetensors")
    assert not _same_state(stable, annealed)
    # Round 2 warm-starts from gen1 with an EMPTY corpus: no batch ever trains,
    # so the checkpoint it writes is exactly the init it loaded — which must be
    # the stable branch.
    gen2 = tmp_path / "gen2"
    result = trainer.train(iter(()), _contract(anneal_fraction=0.0),
                           training_seed=2, token_budget=256, out_dir=gen2,
                           warm_start_dir=gen1)
    assert result.metrics["optim_state_resumed"] is True
    loaded = _weights(gen2 / "weights.safetensors")
    assert _same_state(loaded, stable)
    assert not _same_state(loaded, annealed)


def test_pre_anneal_checkpoint_still_warm_starts(tmp_path: Path):
    # A checkpoint from before the anneal cut has no stable file; the loader
    # falls back to its only weights — the pre-DEC-CA-0029 behaviour.
    gen1 = tmp_path / "gen1"
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    trainer.train(_series(64), _contract(anneal_fraction=0.0), training_seed=1,
                  token_budget=256, out_dir=gen1)
    assert not (gen1 / STABLE_WEIGHTS_FILE).exists()
    w1 = _weights(gen1 / "weights.safetensors")
    gen2 = tmp_path / "gen2"
    trainer.train(iter(()), _contract(anneal_fraction=0.0), training_seed=2,
                  token_budget=256, out_dir=gen2, warm_start_dir=gen1)
    assert _same_state(_weights(gen2 / "weights.safetensors"), w1)


# ── contract digest ───────────────────────────────────────────────────────────


def test_anneal_fraction_is_digest_bound_drop_when_default():
    from cascade.shared.manifest import contract_digest

    base = {"base_lr": 4e-3, "lr_schedule": "wsd"}
    with_default = dict(base, anneal_fraction=0.0)
    armed = dict(base, anneal_fraction=0.15)
    # inert default is digest-invisible; setting it is the deliberate bump
    assert contract_digest(base) == contract_digest(with_default)
    assert contract_digest(armed) != contract_digest(base)
