"""Trainer stamps the king's public-benchmark scores onto the signed manifest.

When [scoring] cascade_enabled, the trainer scores the king's checkpoint on
GIFT-Eval / BOOM / TIME (via an injected bench eval fn) and stamps the six
numbers onto the king's manifest entry so validators promote off one signed set.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import cascade.trainer.loop as loop_mod
from cascade.shared.chain import Commitment
from cascade.shared.manifest import BenchScores
from cascade.trainer.contract import TrainResult
from cascade.trainer.loop import TrainerRunner

REF_A = "alice/gen-a@sha256:" + "a" * 64
REF_B = "bob/gen-b@sha256:" + "b" * 64
REF_OUT = "cascade/ckpt-out@sha256:" + "e" * 64


class _FakeStream:
    digest, n_series, total_points = "d", 3, 192

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def series(self):
        for _ in range(3):
            yield np.ones((1, 64))


class _FakeBaseTrainer:
    def train(self, stream, contract, *, training_seed, token_budget, out_dir, logger=None):
        for _ in stream:
            pass
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "weights.safetensors").write_bytes(b"x")
        return TrainResult(local_dir=out_dir, param_count=4_000_000, train_seconds=1.0, metrics={})


def _fake_upload(local_dir, repo, hub=None, **kw):
    from cascade.shared.hippius import HubRef, HubUpload

    return HubUpload(ref=HubRef.parse(REF_OUT), size_bytes=1)


def _commit(uid, hotkey, ref, block):
    return Commitment(uid=uid, hotkey=hotkey, coldkey=None,
                      payload=f"metro-v1:gen:hippius:{ref}", commit_block=block)


_BENCH = BenchScores(
    gifteval_crps=0.42, gifteval_mase=0.81, boom_crps=0.55,
    boom_mase=0.90, time_crps=0.38, time_mase=0.77,
)


@pytest.fixture()
def cascade_cfg(cfg):
    return replace(cfg, scoring=replace(cfg.scoring, cascade_enabled=True))


def _patch(monkeypatch):
    monkeypatch.setattr(loop_mod, "fetch_from_hub", lambda ref, dest, hub=None: dest)
    monkeypatch.setattr(loop_mod, "open_round_stream", lambda *a, **k: _FakeStream())
    monkeypatch.setattr(loop_mod, "upload_dir_to_hub_or_hf", _fake_upload)


def test_run_round_stamps_king_bench_scores_when_enabled(cascade_cfg, tmp_path, monkeypatch):
    _patch(monkeypatch)
    seen: list = []

    def _bench_eval(ckpt_dir):
        seen.append(ckpt_dir)
        return _BENCH

    runner = TrainerRunner(cfg=cascade_cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, bench_eval_fn=_bench_eval)
    commits = [_commit(0, "a", REF_A, 1), _commit(1, "b", REF_B, 1)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    king = manifest.entry_for_role("king")
    assert king.bench_scores == _BENCH
    # Only the king carries scores; challengers do not.
    assert manifest.entry_for_role("challenger").bench_scores is None
    # The eval ran on the king's fetched checkpoint exactly once.
    assert len(seen) == 1
    # And the numbers are in the signed body.
    assert b"bench_scores" in manifest.canonical_body()


def test_run_round_omits_scores_when_disabled(cfg, tmp_path, monkeypatch):
    _patch(monkeypatch)
    called = []
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, bench_eval_fn=lambda d: called.append(d) or _BENCH)
    manifest = runner.run_round([_commit(0, "a", REF_A, 1)], king_hotkey="a", base_seed=1, block=10)
    # cascade_enabled is False in the shipped chain.toml ⇒ no stamping, fn never called.
    assert manifest.entry_for_role("king").bench_scores is None
    assert called == []


def test_bench_eval_failure_leaves_manifest_unstamped(cascade_cfg, tmp_path, monkeypatch):
    _patch(monkeypatch)

    def _boom(_ckpt):
        raise RuntimeError("sidecar down")

    runner = TrainerRunner(cfg=cascade_cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, bench_eval_fn=_boom)
    # A benchmark failure must never fail the round — the manifest just omits scores.
    manifest = runner.run_round([_commit(0, "a", REF_A, 1)], king_hotkey="a", base_seed=1, block=10)
    assert manifest.entry_for_role("king") is not None
    assert manifest.entry_for_role("king").bench_scores is None


class _StubHost:
    name = "worker"
    workdir = "/root/cascade"


def test_remote_king_bench_dispatches_to_worker_not_local(cascade_cfg, tmp_path, monkeypatch):
    # When a cascade_bench_plan + remote_hosts are set, the king bench runs on the
    # pod (reusing the post-round-benchmark path), NOT the local bench_eval_fn.
    import cascade.eval.benchmarks as bench_mod
    import cascade.trainer.bench_hook as hook_mod

    calls: dict = {}

    def _fake_run(host, round_id, arch_preset, plan, *, work_root=None, runner=None):
        calls["dispatch"] = (host.name, round_id, arch_preset)
        return {"suites": ["stub"]}

    monkeypatch.setattr(hook_mod, "run_post_round_benchmark", _fake_run)
    monkeypatch.setattr(bench_mod, "extract_bench_scores", lambda r: {
        "gifteval_crps": 0.42, "gifteval_mase": 0.81, "boom_crps": 0.55,
        "boom_mase": 0.90, "time_crps": 0.38, "time_mase": 0.77,
    })
    local_called: list = []
    runner = TrainerRunner(
        cfg=cascade_cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path, use_sandbox=False,
        bench_eval_fn=lambda d: local_called.append(d) or _BENCH,
        remote_hosts=[_StubHost()], cascade_bench_plan=object(),
    )
    scores = runner._remote_king_bench_scores("12345", "toto2-4m")
    assert scores == _BENCH                          # parsed the six signed numbers
    assert calls["dispatch"] == ("worker", "12345", "toto2-4m")  # ran on the pod
    assert local_called == []                        # local CPU path untouched


def test_run_round_calls_stamp_when_only_cascade_bench_plan_set(cascade_cfg, tmp_path, monkeypatch):
    # Regression: the run_round guard must fire on cascade_bench_plan alone (the
    # remote-worker path leaves bench_eval_fn=None). Previously it gated on
    # bench_eval_fn, so the worker bench never ran and no scores were stamped.
    _patch(monkeypatch)
    called: list = []
    runner = TrainerRunner(cfg=cascade_cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, bench_eval_fn=None, cascade_bench_plan=object())
    monkeypatch.setattr(runner, "_stamp_king_bench_scores",
                        lambda entries, seeds: called.append(True) or entries)
    runner.run_round([_commit(0, "a", REF_A, 1)], king_hotkey="a", base_seed=1, block=10)
    assert called == [True]   # guard let it through with bench_eval_fn=None
