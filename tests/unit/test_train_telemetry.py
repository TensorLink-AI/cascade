"""Starvation + deadline telemetry — the timed corpus shim in the trainer, the
new metrics, the per-step record field, and the per-run telemetry line.

TrainResult.metrics never crosses the remote boundary (the worker's receipt is
a metrics-less TrainedEntry), so the per-run key=value line emitted from
``_train_checkpoint`` is the remote telemetry channel.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cascade.trainer import toto2_trainer as toto2_mod  # noqa: E402
from cascade.trainer.toto2_trainer import Toto2Trainer  # noqa: E402


def _contract(max_secs: int) -> SimpleNamespace:
    return SimpleNamespace(
        context_length=16, horizon=8, patch_size=4, d_model=16, num_layers=1,
        num_heads=1, head_dim=16, mlp_expansion=2, num_quantiles=9,
        batch_size=4, max_train_seconds=max_secs, base_lr=1e-3, weight_decay=0.0,
        optimizer="adamw", warmup_tokens=0, input_transform="arcsinh_causal",
    )


def _series(n: int, sleep_s: float = 0.0):
    rng = np.random.default_rng(0)
    for _ in range(n):
        if sleep_s:
            time.sleep(sleep_s)
        yield rng.normal(size=32).cumsum()


# ── trainer metrics ───────────────────────────────────────────────────────────


def test_starved_stream_reports_high_data_wait_and_deadline_hit(tmp_path: Path):
    # Every series pull blocks 50ms; the (already expired) deadline stops the
    # run after one step. The wait dominates the training wall time, so the
    # starvation must be loud in the metrics — a bare deadline_hit could as
    # well mean a slow device.
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    result = trainer.train(_series(16, sleep_s=0.05), _contract(0),
                           training_seed=1, token_budget=10_000,
                           out_dir=tmp_path / "ckpt")
    m = result.metrics
    assert m["deadline_hit"] is True
    assert m["data_wait_s"] > 0.0
    assert m["data_wait_frac"] > 0.5          # starved: waiting ≫ training
    assert 0.0 < m["tokens_frac"] < 1.0       # stopped under the token budget


def test_fast_stream_reports_negligible_data_wait(tmp_path: Path):
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    result = trainer.train(_series(16), _contract(300), training_seed=1,
                           token_budget=256, out_dir=tmp_path / "ckpt")
    m = result.metrics
    assert m["deadline_hit"] is False
    assert m["data_wait_frac"] < 0.2          # in-memory stream: ~no waiting
    assert m["tokens_frac"] >= 1.0            # budget reached, not deadline


def test_per_step_record_carries_data_wait_frac(tmp_path: Path, monkeypatch):
    # The live starvation signal rides the existing per-step S3/wandb sink.
    monkeypatch.setattr(toto2_mod, "LOG_EVERY_STEPS", 1)
    records: list[dict] = []
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    trainer.train(_series(16), _contract(300), training_seed=1, token_budget=256,
                  out_dir=tmp_path / "ckpt", logger=records.append)
    steps = [r for r in records if r.get("event") == "step"]
    assert steps and all("data_wait_frac" in r for r in steps)
    assert all(r["data_wait_frac"] >= 0.0 for r in steps)


# ── a leg emits the per-run line (the remote telemetry channel) ──────────────


def test_leg_logs_the_per_run_telemetry_line(cfg, tmp_path, monkeypatch, caplog):
    from cascade.shared.hippius import HubRef, HubUpload
    from cascade.trainer import loop as loop_mod
    from cascade.trainer.contract import RoundSeeds, TrainResult
    from cascade.trainer.loop import ResolvedGenerator, TrainerRunner

    ref_b = "bob/gen-b@sha256:" + "b" * 64
    ref_out = "cascade/ckpt-out@sha256:" + "e" * 64

    class _FakeStream:
        digest, n_series, total_points = "corpusdigest", 3, 192

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def series(self):
            yield np.ones((1, 64))

    class _Trainer:
        def train(self, stream, contract, *, training_seed, token_budget,
                  out_dir, logger=None):
            for _ in stream:
                pass
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "weights.safetensors").write_bytes(b"x")
            return TrainResult(
                local_dir=out_dir, param_count=4, train_seconds=1.0,
                metrics={"deadline_hit": True, "tokens_frac": 0.4,
                         "data_wait_s": 12.3, "data_wait_frac": 0.61},
            )

    monkeypatch.setattr(loop_mod, "fetch_from_hub", lambda ref, dest, hub=None: dest)
    monkeypatch.setattr(loop_mod, "open_round_stream", lambda *a, **k: _FakeStream())
    monkeypatch.setattr(
        loop_mod, "upload_dir_to_hub_or_hf",
        lambda local_dir, repo, hub=None, *, hf_repo=None: HubUpload(
            ref=HubRef.parse(ref_out), size_bytes=1),
    )
    runner = TrainerRunner(cfg=cfg, base_trainer=_Trainer(), work_root=tmp_path,
                           use_sandbox=False)
    with caplog.at_level("INFO", logger="cascade.trainer"):
        runner.train_one(ResolvedGenerator("b", 1, ref_b), "challenger",
                         RoundSeeds.derive(1, cfg.training), 10)
    msgs = [r.message for r in caplog.records]
    # one parseable key=value line per run — the only channel a pod run's
    # telemetry has back to the orchestrator
    assert any("telemetry: deadline_hit=True tokens_frac=0.4 data_wait_s=12.3 "
               "data_wait_frac=0.61" in m for m in msgs)
