"""TrainerRunner plumbing with the GPU and Hippius boundaries faked (no torch,
no Hub, no S3): the per-leg train path, the live loop's chain-read guards and
idle-until-rollover behaviour, the lane retry, and the submission burn."""

from __future__ import annotations

import json
import queue
from dataclasses import replace

import numpy as np
import pytest

from cascade.shared.hippius import HubRef, HubUpload, StorageError
from cascade.trainer import loop as loop_mod
from cascade.trainer.contract import RoundSeeds, TrainResult
from cascade.trainer.loop import ResolvedGenerator, TrainerRunner

REF_B = "bob/gen-b@sha256:" + "b" * 64
REF_C = "carol/gen-c@sha256:" + "c" * 64
REF_OUT = "cascade/ckpt-out@sha256:" + "e" * 64


class _FakeStream:
    n_series = 3
    total_points = 192

    def __init__(self, digest="corpusdigest"):
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
        if logger:
            logger({"event": "step", "step": 1, "loss": 0.1})
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "weights.safetensors").write_bytes(b"x")
        return TrainResult(local_dir=out_dir, param_count=4_000_000, train_seconds=1.0,
                           metrics={"final_loss": 0.1})


def _fake_upload(local_dir, repo, hub=None, *, hf_repo=None, hf_token=None):
    return HubUpload(ref=HubRef.parse(REF_OUT), size_bytes=1)


def _runner(cfg, tmp_path, **kw):
    return TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                         use_sandbox=False, **kw)


# ── the per-leg train path ───────────────────────────────────────────────────


def test_train_one_end_to_end_with_faked_boundaries(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(loop_mod, "fetch_from_hub", lambda ref, dest, hub=None: dest)
    monkeypatch.setattr(loop_mod, "open_round_stream",
                        lambda mode, gen_dir, *a, **k: _FakeStream(digest="d-b"))
    monkeypatch.setattr(loop_mod, "upload_dir_to_hub_or_hf", _fake_upload)
    runner = _runner(cfg, tmp_path)
    seeds = RoundSeeds.derive(7, cfg.training)
    entry = runner.train_one(ResolvedGenerator("b", 1, REF_B), "challenger", seeds, 10,
                             repo_suffix="-u1")
    assert entry.miner_hotkey == "b" and entry.role == "challenger"
    assert entry.corpus_digest == "d-b" and entry.train_block == 10
    assert entry.size == cfg.training.primary_size.arch_preset
    # The marker makes an upload-only retry skip the retrain.
    out = tmp_path / "7" / entry.size / "challenger-u1" / "checkpoint"
    assert (out / TrainerRunner.TRAIN_COMPLETE_MARKER).exists()


def test_reduced_budget_leg_tags_telemetry_apart_from_the_full_leg(two_size_cfg, tmp_path,
                                                                   monkeypatch):
    # A reduced-budget run (the bench-anneal leg, worker --train-hours) routes
    # its S3/wandb telemetry to heat-<hotkey>, never the full leg's
    # <role>-<size> key, which would collide the two logs for one challenger.
    monkeypatch.setattr(loop_mod, "upload_dir_to_hub_or_hf", _fake_upload)
    runner = _runner(two_size_cfg, tmp_path)
    seen: list[str] = []

    def _capture(gen, seeds, contract, budget, out_dir, *, log_role, warm_start_dir=None):
        seen.append(log_role)
        out_dir.mkdir(parents=True, exist_ok=True)
        return TrainResult(local_dir=out_dir, param_count=1, train_seconds=1.0,
                           metrics={}), "digest", 1, 1

    monkeypatch.setattr(runner, "_train_checkpoint", _capture)
    seeds = RoundSeeds.derive(1, two_size_cfg.training)
    gen = ResolvedGenerator(hotkey="b", uid=1, ref=REF_B)

    runner.train_one(gen, "challenger", seeds, 10, heat=True)
    runner.train_one(gen, "challenger", seeds, 10, heat=False)
    assert seen[0] == "heat-b"
    assert seen[1].startswith("challenger-")


# ── the live loop ────────────────────────────────────────────────────────────


class _StopLoop(Exception):
    pass


def _drive(runner, monkeypatch, client, *, ticks: int) -> None:
    """Run ``run_forever`` for ``ticks`` polls (the poll sleep ends it)."""
    n = {"sleeps": 0}

    def _sleep(_s):
        n["sleeps"] += 1
        if n["sleeps"] >= ticks:
            raise _StopLoop

    monkeypatch.setattr(loop_mod.time, "sleep", _sleep)
    with pytest.raises(_StopLoop):
        runner.run_forever(client)


class _Client:
    def __init__(self, block):
        self.block = block

    def current_block(self):
        return self.block


def test_run_forever_idles_until_the_rollover(cfg, tmp_path, monkeypatch, caplog):
    runner = _runner(cfg, tmp_path)
    monkeypatch.setattr(runner, "_rolling",
                        lambda: pytest.fail("the scheduler must not tick before the rollover"))
    assert int(cfg.round.rolling_from_block or 0) == 0          # undecided in the fixture
    with caplog.at_level("INFO", logger="cascade.trainer"):
        _drive(runner, monkeypatch, _Client(5_000_000), ticks=3)
    idle = [r for r in caplog.records if "rolling intake not active yet" in r.getMessage()]
    assert len(idle) == 1                                       # once per boundary, not per poll


def test_run_forever_ticks_the_scheduler_once_rolling_is_active(cfg, tmp_path, monkeypatch):
    eb = cfg.round.epoch_blocks
    rolled = replace(cfg, round=replace(cfg.round, rolling_from_block=10 * eb))
    runner = _runner(rolled, tmp_path)
    ticks: list[int] = []

    class _Sched:
        def tick(self, client, block):
            ticks.append(block)

    monkeypatch.setattr(runner, "_rolling", lambda: _Sched())
    _drive(runner, monkeypatch, _Client(10 * eb + 5), ticks=2)
    assert ticks == [10 * eb + 5, 10 * eb + 5]


def test_run_forever_survives_a_failed_tick(cfg, tmp_path, monkeypatch):
    eb = cfg.round.epoch_blocks
    runner = _runner(replace(cfg, round=replace(cfg.round, rolling_from_block=eb)), tmp_path)
    calls = {"n": 0}

    class _Sched:
        def tick(self, client, block):
            calls["n"] += 1
            raise RuntimeError("one bad tick")

    monkeypatch.setattr(runner, "_rolling", lambda: _Sched())
    _drive(runner, monkeypatch, _Client(eb + 1), ticks=3)
    assert calls["n"] == 3                                       # retried every poll


def test_frozen_block_rebuilds_substrate_and_reads_fresh(cfg, tmp_path):
    # A quietly-dead bittensor websocket keeps answering current_block() with a
    # stale height. Once the height stops advancing past stale_block_after_s,
    # the guard rebuilds the connection and trusts the fresh, advanced read.
    class FrozenClient:
        def __init__(self):
            self.block = 1000
            self.reconnects = 0

        def current_block(self):
            return self.block

        def reconnect(self):
            self.reconnects += 1
            self.block = 5000

    client = FrozenClient()
    now = {"t": 0.0}
    runner = _runner(cfg, tmp_path, stale_block_after_s=300.0, chain_clock=lambda: now["t"])

    assert runner._block_with_freeze_guard(client) == 1000
    now["t"] = 200.0
    assert runner._block_with_freeze_guard(client) == 1000     # within window: trusted
    assert client.reconnects == 0
    now["t"] = 400.0
    assert runner._block_with_freeze_guard(client) == 5000     # rebuilt + fresh read
    assert client.reconnects == 1
    now["t"] = 410.0
    client.block = 5100
    assert runner._block_with_freeze_guard(client) == 5100
    assert client.reconnects == 1


def test_raising_or_hung_chain_read_rebuilds_and_recovers(cfg, tmp_path):
    import time as _time

    from cascade.shared.chain import ChainError

    class _Raises:
        def __init__(self):
            self.reconnects = 0

        def current_block(self):
            if self.reconnects == 0:
                raise ChainError("get_current_block_failed")
            return 7000

        def reconnect(self):
            self.reconnects += 1

    raiser = _Raises()
    runner = _runner(cfg, tmp_path)
    assert runner._block_with_freeze_guard(raiser) == 7000
    assert raiser.reconnects == 1

    class _Hangs:
        def __init__(self):
            self.reconnects = 0

        def current_block(self):
            if self.reconnects == 0:
                _time.sleep(0.4)          # exceeds chain_read_timeout_s below
                return 1
            return 8000

        def reconnect(self):
            self.reconnects += 1

    hanger = _Hangs()
    runner2 = _runner(cfg, tmp_path, chain_read_timeout_s=0.05)
    assert runner2._block_with_freeze_guard(hanger) == 8000
    assert hanger.reconnects == 1

    class _RaisesNoReconnect:
        def current_block(self):
            raise ChainError("down")

    with pytest.raises(ChainError):
        runner._block_with_freeze_guard(_RaisesNoReconnect())


# ── lane retry ───────────────────────────────────────────────────────────────


def test_storage_failure_classifier():
    from cascade.trainer.loop import _storage_failure

    assert _storage_failure(StorageError("fetch of x failed: chunk 0 failed"))
    assert _storage_failure(RuntimeError(
        "remote challenger on pod-g2 failed (rc=1): .../hippius_hub/"
        "file_download.py, line 424, in hf_hub_download"))
    assert _storage_failure(RuntimeError("CorpusError: generator_artifact_unreachable: HTTP 401"))
    assert not _storage_failure(RuntimeError("ImportError: no module named torch"))
    assert not _storage_failure(RuntimeError("remote failed (rc=255): ssh timed out"))


def _lanes(*names):
    q = queue.Queue()
    for n in names:
        q.put(n)
    return q


@pytest.mark.parametrize("error, backoff", [
    ("worker died in hippius_hub fetch: server returned 401", True),
    ("boom: OOM on device 0", False),
])
def test_lane_retry_backs_off_only_on_storage_failures(monkeypatch, error, backoff):
    naps = []
    monkeypatch.setattr(loop_mod.time, "sleep", lambda s: naps.append(s))
    hosts: list = []

    class _Disp:
        def dispatch(self, host, lane_count=None, **kw):
            hosts.append(host)
            if len(hosts) == 1:
                raise RuntimeError(error)
            return "entry"

    lanes = _lanes("h1", "h2")
    out = TrainerRunner._dispatch_on_free_lane(_Disp(), lanes, ["h1", "h2"], describe="leg x")
    assert out == "entry" and hosts == ["h1", "h2"]
    assert naps == ([loop_mod.STORAGE_RETRY_BACKOFF_SECONDS] if backoff else [])
    assert sorted(lanes.queue) == ["h1", "h2"]          # both lanes back in the rotation


# ── 1 hotkey = 1 submission ──────────────────────────────────────────────────


def test_settled_challengers_are_burned(cfg, tmp_path):
    runner = _runner(cfg, tmp_path)
    runner._burn_hotkeys([ResolvedGenerator("b", 1, REF_B)])
    runner._burn_hotkeys([ResolvedGenerator("c", 2, REF_C)])
    assert set(json.loads(runner._submissions_path().read_text())) == {"b", "c"}


def test_burn_is_off_without_one_submission_per_hotkey(cfg, tmp_path):
    runner = _runner(replace(cfg, round=replace(cfg.round, one_submission_per_hotkey=False)),
                     tmp_path)
    runner._burn_hotkeys([ResolvedGenerator("b", 1, REF_B)])
    assert not runner._submissions_path().exists()
