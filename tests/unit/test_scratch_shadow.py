"""Shadow scratch control (DEC-CA-0014 Stage 1).

Every M-th warm-started round ([telemetry] scratch_shadow_every_rounds) the
trainer additionally trains the king's generator FROM SCRATCH post-publish,
benches it, and publishes a clearly-labeled telemetry report
(benchmarks/scratch/round-<id>.json + trend index) beside the round's signed
bench report. These tests pin the invariants that make it consensus-inert:
telemetry tier (never contract_digest), never a manifest or BenchReport entry,
never fed to the promotion engine, never able to fail the bench thread.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

import cascade.trainer.loop as loop_mod
from cascade.shared.bench_report import load_bench_report
from cascade.shared.chain import Commitment
from cascade.shared.config import load_chain_config
from cascade.shared.manifest import BenchScores, contract_digest
from cascade.shared.scratch_report import (
    SCRATCH_INDEX_KEY,
    ScratchBenchReport,
    bench_geomean,
    dump_scratch_report,
    load_scratch_report,
    scratch_report_key,
    sign_scratch_report,
)
from cascade.trainer.contract import TrainResult
from cascade.trainer.loop import TrainerRunner

REF_A = "alice/gen-a@sha256:" + "a" * 64
REF_B = "bob/gen-b@sha256:" + "b" * 64
WS_CKPT = "metro-v1:trained:hippius:cascade/ckpt-init@sha256:" + "e" * 64


class _FakeStream:
    n_series, total_points = 3, 192

    def __init__(self, digest="d"):
        self.digest = digest

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def series(self):
        for _ in range(3):
            yield np.ones((1, 64))


class _RecordingBaseTrainer:
    """Records each train call's kwargs so tests can assert the scratch leg ran
    from random init (no warm_start_dir forwarded)."""

    def __init__(self):
        self.calls: list[dict] = []

    def train(self, stream, contract, *, training_seed, token_budget, out_dir,
              logger=None, **kw):
        for _ in stream:
            pass
        self.calls.append({"out_dir": out_dir, "token_budget": token_budget, **kw})
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "weights.safetensors").write_bytes(b"x")
        return TrainResult(local_dir=out_dir, param_count=4_000_000,
                           train_seconds=1.0, metrics={})


def _fake_upload(local_dir, repo, hub=None, **kw):
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
_SCRATCH = BenchScores(
    gifteval_crps=0.61, gifteval_mase=0.95, boom_crps=0.72,
    boom_mase=1.02, time_crps=0.55, time_mase=0.91,
)


class _FakeStore:
    def __init__(self):
        self.texts: dict[str, str] = {}

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        self.texts[key] = text

    def get_text(self, key):
        if key not in self.texts:
            raise KeyError(key)
        return self.texts[key]


class _FakeWallet:
    class _HK:
        def sign(self, body: bytes) -> bytes:
            return b"SIG:" + body[:8]

    hotkey = _HK()


class _RecordingPromotion:
    generation = 3

    def __init__(self):
        self.recorded: list = []

    def record_bench(self, manifest, report):
        self.recorded.append(report)
        return 1

    def init_for_epoch(self, epoch_index):
        # No live member set: run_round falls through to the pointer file
        # (absent under test), i.e. random init; the tests stamp the manifest's
        # warm-start pin themselves.
        return None


def _bench_by_path(ckpt_dir):
    """Scratch checkpoints live under king-scratch/; the duel legs don't."""
    return _SCRATCH if "king-scratch" in str(ckpt_dir) else _BENCH


@pytest.fixture()
def shadow_cfg(cfg):
    return replace(
        cfg,
        scoring=replace(cfg.scoring, cascade_enabled=True),
        telemetry=replace(cfg.telemetry, scratch_shadow_every_rounds=2),
    )


def _patch(monkeypatch):
    monkeypatch.setattr(loop_mod, "fetch_from_hub", lambda ref, dest, hub=None: dest)
    monkeypatch.setattr(loop_mod, "open_round_stream",
                        lambda mode, gen_dir, *a, **k: _FakeStream(digest=f"d-{gen_dir}"))
    monkeypatch.setattr(loop_mod, "upload_dir_to_hub_or_hf", _fake_upload)


def _runner(cfg, tmp_path, monkeypatch, *, bench_eval_fn=_bench_by_path,
            wallet=None, promotion=None):
    _patch(monkeypatch)
    runner = TrainerRunner(cfg=cfg, base_trainer=_RecordingBaseTrainer(),
                           work_root=tmp_path, use_sandbox=False,
                           bench_eval_fn=bench_eval_fn, wallet=wallet,
                           promotion=promotion)
    runner._manifest_store = _FakeStore()
    return runner


def _warm_manifest(runner, *, base_seed=1, block=10, two_sided=True):
    """A published-shape manifest whose round trained warm-started (the field
    the shadow keys on); built by a real run_round, then stamped with the
    warm-start pin exactly as a promotion-consuming round records it."""
    commits = [_commit(0, "a", REF_A, 1)]
    if two_sided:
        commits.append(_commit(1, "b", REF_B, 1))
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=base_seed,
                                block=block)
    size = runner.cfg.throne_contracts()[0].arch_preset
    return replace(manifest, warm_start_ckpt=WS_CKPT, warm_start_size=size)


# ── config plumbing ─────────────────────────────────────────────────────────


def test_telemetry_key_parses_and_defaults_off(tmp_path):
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    # Mainnet template ships OFF (arming is an owner action, testnet first);
    # the testnet config carries the Stage-1 validation cadence.
    assert load_chain_config(repo_root / "chain.toml"
                             ).telemetry.scratch_shadow_every_rounds == 0
    assert load_chain_config(repo_root / "chain.testnet.toml"
                             ).telemetry.scratch_shadow_every_rounds == 2
    # And a toml without the key parses to the off default.
    stripped = tmp_path / "c.toml"
    stripped.write_text(
        (repo_root / "chain.toml").read_text(encoding="utf-8")
        .replace("scratch_shadow_every_rounds = 0\n", ""),
        encoding="utf-8")
    assert load_chain_config(stripped).telemetry.scratch_shadow_every_rounds == 0


def test_telemetry_key_never_touches_contract_digest(cfg):
    # The whole point of the [telemetry] tier: arming the shadow must not
    # change the training contract every validator gates on.
    assert contract_digest(cfg.training) == contract_digest(
        replace(cfg, telemetry=replace(cfg.telemetry, scratch_shadow_every_rounds=4)).training
    )


# ── report schema ───────────────────────────────────────────────────────────


def _report(**kw):
    defaults = dict(
        round_id="7", created_block=10, gen_ref=REF_A, miner_hotkey="a",
        miner_uid=0, trained_pointer="metro-v1:trained:hippius:cascade/x@sha256:" + "1" * 64,
        scores=_SCRATCH, warm_start_ckpt=WS_CKPT, generation=2,
        king_pointer="metro-v1:trained:hippius:cascade/k@sha256:" + "2" * 64,
        king_scores=_BENCH,
    )
    defaults.update(kw)
    return ScratchBenchReport(**defaults)


def test_scratch_report_roundtrips_and_is_labeled():
    report = sign_scratch_report(_report(), _FakeWallet())
    body = json.loads(report.canonical_body())
    assert body["kind"] == "scratch_shadow" and body["telemetry_only"] is True
    loaded = load_scratch_report(dump_scratch_report(report))
    assert loaded == report


def test_scratch_report_refuses_unlabeled_documents():
    text = dump_scratch_report(_report())
    obj = json.loads(text)
    obj["kind"] = "bench"
    with pytest.raises(ValueError, match="scratch_shadow"):
        load_scratch_report(json.dumps(obj))


def test_bench_geomean_is_the_six_number_geomean():
    assert bench_geomean(_BENCH) == pytest.approx(
        float(np.exp(np.mean(np.log([0.42, 0.81, 0.55, 0.90, 0.38, 0.77])))))


# ── the shadow leg, end to end (local path) ─────────────────────────────────


def test_shadow_publishes_labeled_report_beside_bench_report(
        shadow_cfg, tmp_path, monkeypatch):
    promotion = _RecordingPromotion()
    runner = _runner(shadow_cfg, tmp_path, monkeypatch, wallet=_FakeWallet(),
                     promotion=promotion)
    manifest = _warm_manifest(runner)
    duel = runner.run_post_publish_bench(manifest)
    assert duel is not None

    texts = runner._manifest_store.texts
    # The duel report is untouched: king + challenger only, no scratch entry.
    bench = load_bench_report(texts["benchmarks/round-1.json"])
    assert {e.role for e in bench.entries} == {"king", "challenger"}
    # The scratch report landed beside it, signed and labeled.
    report = load_scratch_report(texts[scratch_report_key("1")])
    assert report.scores == _SCRATCH
    assert report.king_scores == _BENCH          # lineage reference embedded
    assert report.warm_start_ckpt == WS_CKPT
    assert report.generation == promotion.generation
    assert report.signature is not None
    # Trend index carries the gap point.
    idx = json.loads(texts[SCRATCH_INDEX_KEY])
    assert idx["telemetry_only"] is True
    (row,) = idx["rounds"]
    assert row["round_id"] == "1"
    assert row["gap"] == pytest.approx(bench_geomean(_SCRATCH) - bench_geomean(_BENCH))
    # The promotion engine saw ONLY the duel report — never the scratch one.
    assert promotion.recorded == [duel]


def test_shadow_trains_from_random_init_at_full_budget(shadow_cfg, tmp_path, monkeypatch):
    runner = _runner(shadow_cfg, tmp_path, monkeypatch)
    manifest = _warm_manifest(runner)
    runner.run_post_publish_bench(manifest)
    scratch_calls = [c for c in runner.base_trainer.calls
                     if "king-scratch" in str(c["out_dir"])]
    assert len(scratch_calls) == 1
    # Random init: the warm-start kwarg is never forwarded on the scratch leg…
    assert "warm_start_dir" not in scratch_calls[0]
    # …at the identical full-budget token count.
    primary = shadow_cfg.throne_contracts()[0]
    assert scratch_calls[0]["token_budget"] == primary.train_tokens


def test_shadow_skips_random_init_rounds(shadow_cfg, tmp_path, monkeypatch):
    # A round with no warm_start_ckpt already IS the scratch control.
    runner = _runner(shadow_cfg, tmp_path, monkeypatch)
    manifest = runner.run_round([_commit(0, "a", REF_A, 1)], king_hotkey="a",
                                base_seed=1, block=10)
    assert manifest.warm_start_ckpt == ""
    runner.run_post_publish_bench(manifest)
    assert scratch_report_key("1") not in runner._manifest_store.texts
    assert not any("king-scratch" in str(c["out_dir"])
                   for c in runner.base_trainer.calls)


def test_shadow_respects_the_round_cadence(shadow_cfg, tmp_path, monkeypatch):
    # epoch index = created_block // effective_epoch_blocks; M=2 ⇒ odd epochs
    # skip. Pre-activation blocks resolve the 7200 grid of the shipped toml.
    from cascade.shared.config import effective_epoch_blocks

    runner = _runner(shadow_cfg, tmp_path, monkeypatch)
    eb = effective_epoch_blocks(shadow_cfg.round, 10)
    manifest = _warm_manifest(runner, base_seed=2, block=eb + 5)  # epoch 1: odd
    assert runner._scratch_shadow_due(manifest) is False
    manifest = replace(manifest, created_block=2 * eb + 5)        # epoch 2: even
    assert runner._scratch_shadow_due(manifest) is True


def test_shadow_off_by_default_and_when_unwired(cfg, shadow_cfg, tmp_path, monkeypatch):
    # M=0 (the shipped default) ⇒ never due, even on a warm round.
    off = replace(shadow_cfg,
                  telemetry=replace(shadow_cfg.telemetry, scratch_shadow_every_rounds=0))
    runner = _runner(off, tmp_path, monkeypatch)
    manifest = _warm_manifest(runner)
    assert runner._scratch_shadow_due(manifest) is False
    # Armed but no train+bench path wired ⇒ skipped, not crashed.
    unwired = _runner(shadow_cfg, tmp_path / "b", monkeypatch, bench_eval_fn=None)
    manifest2 = replace(manifest)
    assert unwired._scratch_shadow_due(manifest2) is False


def test_shadow_failure_never_sinks_the_bench_thread(shadow_cfg, tmp_path, monkeypatch):
    calls = {"n": 0}

    def _bench(ckpt_dir):
        if "king-scratch" in str(ckpt_dir):
            raise RuntimeError("scratch bench exploded")
        calls["n"] += 1
        return _BENCH

    runner = _runner(shadow_cfg, tmp_path, monkeypatch, bench_eval_fn=_bench)
    manifest = _warm_manifest(runner)
    runner._mark_bench_pending(manifest.round_id)
    duel = runner.run_post_publish_bench(manifest)
    # Duel report still published; no scratch report; hold released.
    assert duel is not None and calls["n"] == 2
    assert "benchmarks/round-1.json" in runner._manifest_store.texts
    assert scratch_report_key("1") not in runner._manifest_store.texts
    assert not (tmp_path / "1" / "bench_pending.json").exists()


def test_shadow_without_scores_publishes_nothing(shadow_cfg, tmp_path, monkeypatch):
    # Bench returns None for the scratch leg: no report, no index, no raise.
    runner = _runner(
        shadow_cfg, tmp_path, monkeypatch,
        bench_eval_fn=lambda d: None if "king-scratch" in str(d) else _BENCH)
    manifest = _warm_manifest(runner)
    runner.run_post_publish_bench(manifest)
    assert scratch_report_key("1") not in runner._manifest_store.texts
    assert SCRATCH_INDEX_KEY not in runner._manifest_store.texts


def test_scratch_pod_snapshot_taken_before_the_duel_bench(shadow_cfg, tmp_path, monkeypatch):
    # The scratch leg runs HOURS after publish; by then run_round may have
    # started the next round and repointed _final_role_hosts' identical
    # (role, size, hotkey) key at the NEW round's live final pod. The shadow
    # must only ever see the snapshot taken at bench-thread start.
    runner = _runner(shadow_cfg, tmp_path, monkeypatch)
    manifest = _warm_manifest(runner)
    size = shadow_cfg.throne_contracts()[0].arch_preset
    runner._final_role_hosts = {("king", size, "a"): "pod-round-N"}

    orig_bench = runner._post_publish_bench

    def _bench_then_next_round_starts(m):
        runner._final_role_hosts[("king", size, "a")] = "pod-round-N+1"
        return orig_bench(m)

    captured = {}

    def _capture_shadow(m, report=None, final_hosts=None):
        captured["final_hosts"] = final_hosts
        return None

    monkeypatch.setattr(runner, "_post_publish_bench", _bench_then_next_round_starts)
    monkeypatch.setattr(runner, "_run_scratch_shadow", _capture_shadow)
    runner.run_post_publish_bench(manifest)
    assert captured["final_hosts"] == {("king", size, "a"): "pod-round-N"}


def test_remote_scratch_skips_without_a_snapshotted_pod(shadow_cfg, tmp_path, monkeypatch):
    # No snapshot entry for the king's pod ⇒ skip the leg (a missed telemetry
    # point), never fall back to a live host lookup that could name the next
    # round's fleet.
    from cascade.trainer.contract import RoundSeeds
    from cascade.trainer.loop import ResolvedGenerator

    runner = _runner(shadow_cfg, tmp_path, monkeypatch)
    manifest = _warm_manifest(runner)          # round itself trains locally
    runner.remote_hosts = ["h"]                # remote wiring appears for the shadow
    runner.trainer_spec = "mod:Cls"
    runner.cascade_bench_plan = object()
    gen = ResolvedGenerator(hotkey="a", uid=0, ref=REF_A)
    seeds = RoundSeeds.derive(1, shadow_cfg.training)
    primary = shadow_cfg.throne_contracts()[0]
    assert runner._scratch_shadow_remote(manifest, gen, primary, seeds,
                                         final_hosts={}) == (None, None)
    assert runner._scratch_shadow_remote(manifest, gen, primary, seeds,
                                         final_hosts=None) == (None, None)


def test_scratch_index_survives_transient_read_failures():
    # A flaky (non-404) index read must SKIP the update — not overwrite the
    # published trend roll-up with a single-round index.
    from cascade.shared.scratch_report import update_scratch_index

    class _FlakyStore:
        def __init__(self):
            self.writes = {}

        def get_text(self, key):
            raise RuntimeError("503 service unavailable")

        def put_text(self, key, text, **kw):
            self.writes[key] = text

    store = _FlakyStore()
    with pytest.raises(RuntimeError, match="skipping index update"):
        update_scratch_index(store, _report())
    assert store.writes == {}


def test_shadow_runs_even_when_the_duel_bench_missed(shadow_cfg, tmp_path, monkeypatch):
    # A duel-bench wipeout must not cancel the scratch point — the report just
    # carries no lineage reference.
    runner = _runner(
        shadow_cfg, tmp_path, monkeypatch,
        bench_eval_fn=lambda d: _SCRATCH if "king-scratch" in str(d) else None)
    manifest = _warm_manifest(runner)
    assert runner.run_post_publish_bench(manifest) is None   # no duel report
    report = load_scratch_report(runner._manifest_store.texts[scratch_report_key("1")])
    assert report.scores == _SCRATCH and report.king_scores is None
    (row,) = json.loads(runner._manifest_store.texts[SCRATCH_INDEX_KEY])["rounds"]
    assert row["king_geomean"] is None and row["gap"] is None
