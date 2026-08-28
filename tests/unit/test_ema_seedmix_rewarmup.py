"""The measured variance bundle (DEC-CA-0033): EMA finished-form checkpoints,
N-seed generation mix, and the warm-started re-warmup.

Measured 2026-08-26 (docs/notes/2026-08-26-seed-variance-ema.md): EMA-0.999
shrinks entrant-specific generation-seed noise 4–11× and improves the
absolute geomean ~7%; the seed mix averages N corpus realizations (~√N on
the residual); base_lr against fresh optimizer state on a warm-started wsd
step 1 costs +0.11 geomean (the u158 probe). All three knobs ship inert
behind drop-when-default — these tests freeze both the armed behaviour and
the unarmed byte-compatibility.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cascade.trainer.stream import _SeedMixStream, open_round_stream  # noqa: E402
from cascade.trainer.toto2_trainer import (  # noqa: E402
    STABLE_WEIGHTS_FILE,
    Toto2Trainer,
    _lr_at,
)


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


# ── the re-warmup LR curve ────────────────────────────────────────────────────


def test_rewarmup_ramps_warm_started_wsd_from_zero():
    lr = lambda pos, rw: _lr_at(pos, 1000, 100, 1e-3, schedule="wsd",  # noqa: E731
                                warm_started=True, rewarmup=rw)
    assert lr(0, 200) == pytest.approx(0.0)
    assert lr(100, 200) == pytest.approx(5e-4)
    assert lr(200, 200) == pytest.approx(1e-3)
    assert lr(900, 200) == pytest.approx(1e-3)          # flat after the ramp


def test_rewarmup_zero_preserves_deployed_first_step_behaviour():
    # rewarmup=0 (every deployed config): warm-started wsd is base_lr from
    # token 0 — the exact behaviour the u158 probe measured.
    assert _lr_at(0, 1000, 100, 1e-3, schedule="wsd",
                  warm_started=True, rewarmup=0) == pytest.approx(1e-3)


def test_rewarmup_never_touches_the_from_scratch_warmup():
    # A from-scratch run keeps warmup-once semantics whatever rewarmup says.
    assert _lr_at(50, 1000, 100, 1e-3, schedule="wsd",
                  warm_started=False, rewarmup=200) == pytest.approx(5e-4)


def test_rewarmup_fraction_only_arms_on_warm_started_wsd(tmp_path: Path):
    # Set on a from-scratch run: inert (no ramp key in metrics).
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    result = trainer.train(_series(32), _contract(rewarmup_fraction=0.1),
                           training_seed=1, token_budget=256,
                           out_dir=tmp_path / "c")
    assert "rewarmup_tokens" not in result.metrics


def test_rewarmup_fraction_bounds(tmp_path: Path):
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    with pytest.raises(ValueError, match="rewarmup_fraction must be in"):
        trainer.train(_series(4), _contract(rewarmup_fraction=1.5),
                      training_seed=1, token_budget=256, out_dir=tmp_path / "c")


# ── EMA finished-form checkpoints ─────────────────────────────────────────────


def test_ema_run_ships_ema_artifact_plus_raw_lineage(tmp_path: Path):
    out = tmp_path / "ckpt"
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    result = trainer.train(_series(64), _contract(ema_decay=0.99),
                           training_seed=1, token_budget=256, out_dir=out)
    assert result.metrics["ema_decay"] == 0.99
    ema = _weights(out / "weights.safetensors")
    raw = _weights(out / STABLE_WEIGHTS_FILE)
    assert not _same_state(ema, raw)        # the average is not the endpoint


def test_warm_start_resumes_from_raw_endpoint_not_ema(tmp_path: Path):
    gen1 = tmp_path / "gen1"
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    trainer.train(_series(64), _contract(ema_decay=0.99), training_seed=1,
                  token_budget=256, out_dir=gen1)
    raw = _weights(gen1 / STABLE_WEIGHTS_FILE)
    ema = _weights(gen1 / "weights.safetensors")
    # Round 2 warm-starts with an EMPTY corpus: the checkpoint it writes is
    # exactly the init it loaded — which must be the raw lineage branch.
    gen2 = tmp_path / "gen2"
    trainer.train(iter(()), _contract(), training_seed=2, token_budget=256,
                  out_dir=gen2, warm_start_dir=gen1)
    loaded = _weights(gen2 / "weights.safetensors")
    assert _same_state(loaded, raw)
    assert not _same_state(loaded, ema)


def test_ema_and_anneal_are_mutually_exclusive(tmp_path: Path):
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    with pytest.raises(ValueError, match="arm exactly one"):
        trainer.train(_series(4), _contract(ema_decay=0.999, anneal_fraction=0.2),
                      training_seed=1, token_budget=256, out_dir=tmp_path / "c")


def test_ema_decay_bounds(tmp_path: Path):
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    with pytest.raises(ValueError, match="ema_decay must be in"):
        trainer.train(_series(4), _contract(ema_decay=1.5),
                      training_seed=1, token_budget=256, out_dir=tmp_path / "c")


def test_unarmed_run_is_byte_identical_to_before(tmp_path: Path):
    # Neither knob set: no stable file, no new metric keys — the deployed
    # checkpoint shape, exactly.
    out = tmp_path / "ckpt"
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    result = trainer.train(_series(64), _contract(), training_seed=1,
                           token_budget=256, out_dir=out)
    assert not (out / STABLE_WEIGHTS_FILE).exists()
    assert "ema_decay" not in result.metrics
    assert "rewarmup_tokens" not in result.metrics


# ── the N-seed generation mix ─────────────────────────────────────────────────


def _drain(example_generator_dir, cfg, *, seed_mix, budget=3000, seed=0):
    with open_round_stream(
        "stream_cpu", example_generator_dir, seed, cfg, token_budget=budget,
        use_sandbox=False, blocked=("socket",), allow_netns=False,
        seed_mix=seed_mix,
    ) as rs:
        points = sum(int(a.size) for a in rs.series())
        return rs.digest, rs.n_series, rs.total_points, points


def test_seed_mix_one_is_the_plain_stream(small_cfg, example_generator_dir):
    with open_round_stream(
        "stream_cpu", example_generator_dir, 0, small_cfg.generator,
        token_budget=3000, use_sandbox=False, seed_mix=1,
    ) as rs:
        assert not isinstance(rs, _SeedMixStream)


def test_seed_mix_interleaves_covers_budget_and_is_deterministic(
    small_cfg, example_generator_dir
):
    a = _drain(example_generator_dir, small_cfg.generator, seed_mix=3)
    b = _drain(example_generator_dir, small_cfg.generator, seed_mix=3)
    assert a == b                                   # deterministic replay
    assert a[3] == a[2] >= 3000                     # budget covered
    single = _drain(example_generator_dir, small_cfg.generator, seed_mix=1)
    assert a[0] != single[0]                        # a mixed corpus is its own bytes


def test_seed_mix_children_draw_distinct_seeds(small_cfg, example_generator_dir):
    # Mixing under two different base seeds must differ (the derived child
    # seeds are a pure function of the round's generation seed).
    a = _drain(example_generator_dir, small_cfg.generator, seed_mix=3, seed=0)
    b = _drain(example_generator_dir, small_cfg.generator, seed_mix=3, seed=1)
    assert a[0] != b[0]


def test_seed_mix_rejects_nonpositive(small_cfg, example_generator_dir):
    from cascade.trainer.corpus import CorpusError

    with pytest.raises(CorpusError, match="seed_mix"):
        open_round_stream(
            "stream_cpu", example_generator_dir, 0, small_cfg.generator,
            token_budget=100, use_sandbox=False, seed_mix=0,
        )


# ── contract digest ───────────────────────────────────────────────────────────


def test_bundle_fields_are_digest_bound_drop_when_default():
    from cascade.shared.manifest import contract_digest

    base = {"base_lr": 4e-3, "lr_schedule": "wsd"}
    inert = dict(base, ema_decay=0.0, gen_seed_mix=1, rewarmup_fraction=0.0)
    assert contract_digest(base) == contract_digest(inert)
    for armed in (dict(base, ema_decay=0.999),
                  dict(base, gen_seed_mix=3),
                  dict(base, rewarmup_fraction=0.02)):
        assert contract_digest(armed) != contract_digest(base)
