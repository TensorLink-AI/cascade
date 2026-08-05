"""Trainer benches BOTH duel checkpoints AFTER the manifest publishes.

When [scoring] cascade_enabled, the manifest goes out untouched (validators
start scoring immediately) and the trainer then scores the king's AND the
challenger's checkpoints on GIFT-Eval / BOOM / TIME, publishing the numbers as
a separate signed bench report (cascade.shared.bench_report). The provisioner's
teardown hold is driven by the bench_pending / bench_complete markers written
around that sweep.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

import cascade.trainer.loop as loop_mod
from cascade.shared.bench_report import load_bench_report
from cascade.shared.chain import Commitment
from cascade.shared.manifest import BenchScores
from cascade.trainer.contract import TrainResult
from cascade.trainer.loop import TrainerRunner

REF_A = "alice/gen-a@sha256:" + "a" * 64
REF_B = "bob/gen-b@sha256:" + "b" * 64


class _FakeStream:
    n_series, total_points = 3, 192

    def __init__(self, digest="d"):
        # per-generator digest (from the fetched gen dir): a constant would make
        # the king and challenger byte-identical and (rightly) trip the trainer's
        # content-clone drop, which is not what these tests probe
        self.digest = digest

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
    # Distinct pointer per checkpoint repo (as in production — the repo name
    # carries round/role/size): the bench report joins on trained_pointer, so
    # a constant ref would collapse the king's and challenger's rows.
    import hashlib

    from cascade.shared.hippius import HubRef, HubUpload

    dg = hashlib.sha256(str(repo).encode("utf-8")).hexdigest()
    return HubUpload(ref=HubRef.parse(f"cascade/ckpt-{dg[:12]}@sha256:{dg}"), size_bytes=1)


def _commit(uid, hotkey, ref, block):
    return Commitment(uid=uid, hotkey=hotkey, coldkey=None,
                      payload=f"metro-v1:gen:hippius:{ref}", commit_block=block)


_BENCH = BenchScores(
    gifteval_crps=0.42, gifteval_mase=0.81, boom_crps=0.55,
    boom_mase=0.90, time_crps=0.38, time_mase=0.77,
)


class _FakeStore:
    def __init__(self):
        self.texts: dict[str, str] = {}

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        self.texts[key] = text


class _FakeWallet:
    class _HK:
        def sign(self, body: bytes) -> bytes:
            return b"SIG:" + body[:8]

    hotkey = _HK()


@pytest.fixture()
def cascade_cfg(cfg):
    return replace(cfg, scoring=replace(cfg.scoring, cascade_enabled=True))


def _patch(monkeypatch):
    monkeypatch.setattr(loop_mod, "fetch_from_hub", lambda ref, dest, hub=None: dest)
    monkeypatch.setattr(loop_mod, "open_round_stream",
                        lambda mode, gen_dir, *a, **k: _FakeStream(digest=f"d-{gen_dir}"))
    monkeypatch.setattr(loop_mod, "upload_dir_to_hub_or_hf", _fake_upload)


def _runner(cfg, tmp_path, monkeypatch, *, bench_eval_fn, wallet=None):
    _patch(monkeypatch)
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, bench_eval_fn=bench_eval_fn, wallet=wallet)
    runner._manifest_store = _FakeStore()
    return runner


def test_run_round_never_stamps_and_never_benches(cascade_cfg, tmp_path, monkeypatch):
    # The manifest must publish untouched and immediately: even with
    # cascade_enabled, run_round neither calls the bench nor stamps
    # bench_scores — the eval is strictly post-publish now.
    called: list = []
    runner = _runner(cascade_cfg, tmp_path, monkeypatch,
                     bench_eval_fn=lambda d: called.append(d) or _BENCH)
    commits = [_commit(0, "a", REF_A, 1), _commit(1, "b", REF_B, 1)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    assert called == []
    assert manifest.entry_for_role("king").bench_scores is None
    assert manifest.entry_for_role("challenger").bench_scores is None
    assert b"bench_scores" not in manifest.canonical_body()


def test_post_publish_bench_reports_both_roles_signed(cascade_cfg, tmp_path, monkeypatch):
    seen: list = []

    def _bench_eval(ckpt_dir):
        seen.append(ckpt_dir)
        return _BENCH

    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=_bench_eval,
                     wallet=_FakeWallet())
    commits = [_commit(0, "a", REF_A, 1), _commit(1, "b", REF_B, 1)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    report = runner.run_post_publish_bench(manifest)
    assert report is not None
    assert len(seen) == 2                                    # king AND challenger benched
    assert {e.role for e in report.entries} == {"king", "challenger"}
    assert report.entry_for("king").scores == _BENCH
    assert report.signature is not None                      # trainer-hotkey signed
    # Uploaded to the manifest-bucket store under the round key, parseable.
    text = runner._manifest_store.texts["benchmarks/round-1.json"]
    assert load_bench_report(text) == report


def test_markers_bracket_the_bench(cascade_cfg, tmp_path, monkeypatch):
    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=lambda d: _BENCH)
    manifest = runner.run_round([_commit(0, "a", REF_A, 1)], king_hotkey="a",
                                base_seed=1, block=10)
    # The live loop writes the pending marker BEFORE publish (the provisioner's
    # teardown-hold arm)…
    assert runner.will_run_post_publish_bench()
    runner._mark_bench_pending(manifest.round_id)
    pending = tmp_path / "1" / "bench_pending.json"
    assert pending.is_file()
    # …and the bench releases it on completion, recording the outcome.
    runner.run_post_publish_bench(manifest)
    assert not pending.exists()
    done = json.loads((tmp_path / "1" / "bench_complete.json").read_text())
    assert done == {"round_id": "1", "uploaded": True}


def test_bench_failure_never_raises_and_releases_hold(cascade_cfg, tmp_path, monkeypatch):
    def _boom(_ckpt):
        raise RuntimeError("sidecar down")

    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=_boom)
    manifest = runner.run_round([_commit(0, "a", REF_A, 1)], king_hotkey="a",
                                base_seed=1, block=10)
    runner._mark_bench_pending(manifest.round_id)
    assert runner.run_post_publish_bench(manifest) is None   # swallowed, never raises
    assert runner._manifest_store.texts == {}                # nothing published
    # The hold is still released — a failed bench must free the pod now, not
    # ride out the provisioner's hold cap.
    assert not (tmp_path / "1" / "bench_pending.json").exists()
    done = json.loads((tmp_path / "1" / "bench_complete.json").read_text())
    assert done["uploaded"] is False


def test_one_role_miss_publishes_partial_report(cascade_cfg, tmp_path, monkeypatch):
    # extract-side miss for one checkpoint: the report ships what it has.
    results = iter([_BENCH, None])
    runner = _runner(cascade_cfg, tmp_path, monkeypatch,
                     bench_eval_fn=lambda d: next(results))
    commits = [_commit(0, "a", REF_A, 1), _commit(1, "b", REF_B, 1)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    report = runner.run_post_publish_bench(manifest)
    assert [e.role for e in report.entries] == ["king"]


def test_disabled_or_unwired_is_a_no_op(cfg, cascade_cfg, tmp_path, monkeypatch):
    # cascade_enabled off (explicit — chain.toml ships armed since 2026-08-05)
    # ⇒ no bench, no report, no marker.
    off_cfg = replace(cfg, scoring=replace(cfg.scoring, cascade_enabled=False))
    called: list = []
    runner = _runner(off_cfg, tmp_path, monkeypatch,
                     bench_eval_fn=lambda d: called.append(d) or _BENCH)
    manifest = runner.run_round([_commit(0, "a", REF_A, 1)], king_hotkey="a",
                                base_seed=1, block=10)
    assert runner.will_run_post_publish_bench() is False
    assert runner.run_post_publish_bench(manifest) is None
    assert called == [] and runner._manifest_store.texts == {}
    # And no marker lifecycle either — the hold markers exist only where the
    # bench (and therefore the provisioner hold) is armed.
    assert not (tmp_path / "1" / "bench_pending.json").exists()
    assert not (tmp_path / "1" / "bench_complete.json").exists()
    # cascade_enabled on but no eval path wired ⇒ equally a no-op predicate.
    unwired = _runner(cascade_cfg, tmp_path / "b", monkeypatch, bench_eval_fn=None)
    assert unwired.will_run_post_publish_bench() is False


def test_restart_releases_a_stale_bench_hold(cascade_cfg, tmp_path, monkeypatch):
    # A trainer restart between publish and bench completion kills the bench
    # thread but leaves bench_pending armed; the restart re-entry skip path
    # must release the hold instead of billing the pod to the cap.
    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=lambda d: _BENCH)
    runner._mark_bench_pending("42")
    runner._release_stale_bench_hold("42")
    assert not (tmp_path / "42" / "bench_pending.json").exists()
    done = json.loads((tmp_path / "42" / "bench_complete.json").read_text())
    assert done == {"round_id": "42", "uploaded": False}
    # Idempotent-safe: an already-completed round is left alone…
    before = (tmp_path / "42" / "bench_complete.json").read_text()
    runner._release_stale_bench_hold("42")
    assert (tmp_path / "42" / "bench_complete.json").read_text() == before
    # …and a round with no markers at all stays marker-free.
    runner._release_stale_bench_hold("43")
    assert not (tmp_path / "43").exists()


def test_wandb_pair_logged_per_round(cascade_cfg, tmp_path, monkeypatch):
    # open_wandb_run also serves the per-run training mirrors; the bench pair
    # opens its own run (role "bench-<size>") — assert on that sink alone.
    sinks: dict[str, list] = {}

    class _Sink:
        def __init__(self, records):
            self.records = records

        def emit(self, record):
            self.records.append(record)

        def finish(self):
            self.records.append("finished")

    def _open(wcfg, **kw):
        return _Sink(sinks.setdefault(kw["role"], []))

    monkeypatch.setattr(loop_mod, "open_wandb_run", _open)
    wandb_cfg = replace(cascade_cfg, wandb=replace(cascade_cfg.wandb, enabled=True))
    runner = _runner(wandb_cfg, tmp_path, monkeypatch, bench_eval_fn=lambda d: _BENCH)
    commits = [_commit(0, "a", REF_A, 1), _commit(1, "b", REF_B, 1)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    runner.run_post_publish_bench(manifest)
    size = cascade_cfg.throne_contracts()[0].arch_preset
    emitted = sinks[f"bench-{size}"]
    # One record per role, then closed.
    roles = [r["role"] for r in emitted if isinstance(r, dict)]
    assert sorted(roles) == ["challenger", "king"]
    assert emitted[-1] == "finished"
    assert all(r["gifteval_crps"] == _BENCH.gifteval_crps
               for r in emitted if isinstance(r, dict))
