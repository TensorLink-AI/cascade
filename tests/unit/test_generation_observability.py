"""CPU corpus telemetry stays separate from admission, training and rewards."""
from __future__ import annotations

import json
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from cascade.generation import progress as P
from cascade.generation.stage import GenerationFault, GenerationStage, StageResult
from cascade.generation.tier import generate_corpus
from cascade.miner.submission_status import generation_view
from cascade.shared.config import load_chain_config
from cascade.trainer.runner_generation import _GenerationMixin
from tests.unit.test_generation_tier import GEN, _gcfg, fake_runner

NOW = datetime(2026, 10, 27, tzinfo=UTC)


def event(**kw):
    return dict(round_id="seed-ref", hotkey="hk", generator_ref="gen@sha256:123", attempt_id="cpu1",
                stage="generating", as_of=NOW.isoformat(), **kw)


def test_public_progress_is_strict_bounded_and_has_no_private_payload():
    doc = P.public_progress(event(points=5, points_per_second=float("nan"), stored_bytes=-1,
                                  error="secret", url="https://secret", values=[1], cpu_key="secret"))
    assert doc["points"] == 5
    assert not set(doc) & {"points_per_second", "stored_bytes", "error", "url", "values", "cpu_key"}
    assert P.public_progress({"stage": "training", "as_of": NOW.isoformat()}) is None
    assert P.public_progress({"stage": "generating", "as_of": "no timezone"}) is None
    json.dumps(doc, allow_nan=False)


def runner(tmp_path):
    cfg = load_chain_config()
    obj = _GenerationMixin()
    obj.cfg = replace(cfg, wandb=replace(cfg.wandb, enabled=False))
    obj.work_root = tmp_path
    return obj


def test_progress_survives_restart_and_ignores_late_terminal_attempt(tmp_path):
    obj = runner(tmp_path)
    obj._leg_local_obj = threading.local()
    obj._leg_local_obj.funded_progress = (None, "hk", "ref", 0, None, "queue1")
    obj._observe_generation(event(points=10))
    final = event(points=20)
    final["stage"] = "ready"
    obj._observe_generation(final)
    obj._observe_generation(event(points=11))
    report = runner(tmp_path)._generation_progress_snapshot()[0]
    assert report["stage"] == "ready" and report["points"] == 20
    assert report["submission_attempt_id"] == "queue1"
    del obj._leg_local_obj.funded_progress
    fresh = event()
    fresh["attempt_id"] = "cpu2"
    fresh["stage"] = "waiting_cpu"
    obj._observe_generation(fresh)
    report = obj._generation_progress_snapshot()[0]
    assert report["attempt_id"] == "cpu2" and "points" not in report
    assert "submission_attempt_id" not in report
    obj._observe_generation(event(points=99))  # Late reply from the previous VM.
    assert obj._generation_progress_snapshot()[0]["attempt_id"] == "cpu2"


def test_corrupt_progress_row_and_local_disk_failure_are_observability_only(tmp_path, monkeypatch):
    obj = runner(tmp_path)
    (tmp_path / "generation_progress.json").write_text(json.dumps({"seed-ref:hk": []}))
    monkeypatch.setattr(P, "atomic_json", lambda *a: (_ for _ in ()).throw(OSError("private")))
    obj._observe_generation(event(points=3))
    assert obj._generation_progress_snapshot()[0]["points"] == 3
    cpu = P.CpuProgress(tmp_path / "cpu.json")
    cpu({"stage": "generating", "points": 3})
    cpu.close()
    assert cpu.latest["points"] == 3


def test_exact_attempt_stale_and_historical_are_distinguished():
    doc = event(points=10, submission_attempt_id="queue1")
    view = dict(hotkey="hk", ref="a|gen@sha256:123", observed={"attempt_id": "queue1"})
    out = generation_view({"generation_progress": [doc]}, view, now=NOW, max_age=180)
    assert out["freshness"]["fresh"] and out["binding"] == "submission_attempt"
    assert not out["historical"]
    stale = generation_view({"generation_progress": [doc]}, view, now=NOW + timedelta(hours=1), max_age=180)
    assert not stale["freshness"]["fresh"]
    view["observed"]["attempt_id"] = "queue2"
    assert generation_view({"generation_progress": [doc]}, view, now=NOW, max_age=180) is None
    del doc["submission_attempt_id"]
    doc["stage"] = "ready"
    out = generation_view({"generation_progress": [doc]}, view, now=NOW, max_age=180)
    assert out["binding"] == "generator_ref_only" and out["historical"]


def test_generation_counters_match_corpus_and_cannot_change_digest(tmp_path):
    events = []
    cfg = _gcfg(shard_encoding="float32_bss")
    a = generate_corpus(tmp_path, 123, GEN, cfg, tmp_path / "a", runner=fake_runner(), progress=events.append)
    def broken(_):
        raise OSError("backend down")
    b = generate_corpus(tmp_path, 123, GEN, cfg, tmp_path / "b", runner=fake_runner(), progress=broken)
    assert a["corpus_digest"] == b["corpus_digest"]
    assert events[-1]["stage"] == "hashing"
    assert events[-1]["points"] == a["points"]
    assert events[-1]["stored_bytes"] == sum(p.stat().st_size for p in (tmp_path / "a").glob("*.parquet"))
    assert events[0]["compression_ratio"] == 0
    assert a["raw_bytes"] == a["points"] * 4


@pytest.mark.parametrize("failure", [False, True])
def test_parent_owns_terminal_outcome_and_does_not_claim_target_met(tmp_path, monkeypatch, failure):
    events = []
    cfg = load_chain_config()
    stage = GenerationStage(cfg, None, [], "unused", "unused", observer=events.append)
    def generate(**kw):
        if failure:
            raise GenerationFault("private details")
        return StageResult(index={"points": 50, "n_series": 1}, base="base", provider="fake", shape="fake", row={})
    monkeypatch.setattr(stage, "_generate", generate)
    kw = dict(round_id="r", hotkey="h", gen_ref="ref", uid=1)
    if failure:
        with pytest.raises(GenerationFault):
            stage.generate(**kw)
        assert events[-1]["stage"] == "generator_fault" and "private details" not in json.dumps(events)
    else:
        stage.generate(**kw)
        assert events[-1]["stage"] == "ready" and events[-1]["percent"] < 100
    assert events[0]["stage"] == "waiting_cpu"
    assert len({e["attempt_id"] for e in events}) == 1


def test_wandb_delivery_is_background_and_finishes_terminal(monkeypatch):
    from cascade.trainer import wandb_sink

    arrived, release = threading.Event(), threading.Event()
    calls, finished = [], []
    def opened(*a, **kw):
        arrived.set()
        assert release.wait(3)
        return SimpleNamespace(emit=calls.append, finish=lambda: finished.append(True))
    monkeypatch.setattr(wandb_sink, "open_generation_run", opened)
    worker = P.WandbProgress(None, {})
    assert arrived.wait(3)
    worker.emit(event(points=1))
    terminal = event(points=2)
    terminal["stage"] = "ready"
    worker.emit(terminal)
    release.set()
    worker.thread.join(timeout=3)
    assert not worker.thread.is_alive() and finished == [True]
    assert calls == [{"generation/points": 2, "generation/stage": "ready"}]


def test_cli_outputs_one_json_document_and_keeps_local_progress(tmp_path, monkeypatch, capsys):
    from pathlib import Path

    from cascade.generation.cli import main

    cfg = load_chain_config()
    cfg = replace(cfg, generation=_gcfg(vcpus=1, target_points=5000, max_length=4096, shard_encoding="float32_bss"))
    monkeypatch.setattr("cascade.shared.config.load_chain_config", lambda *a: cfg)
    repo = Path(__file__).resolve().parents[2] / "cascade/miner/templates/generator"
    output = tmp_path / "out"
    rc = main([str(repo), str(output), "--seed", "3", "--allow-disabled", "--no-sandbox",
               "--json", "--progress-id", "nonce"])
    summary = json.loads(capsys.readouterr().out)
    assert rc == 0 and summary["ok"] and summary["shard_encoding"] == "float32_bss"
    report = json.loads((output / "progress.json").read_text())
    assert report["attempt_id"] == "nonce" and report["stage"] == "uploaded"
    assert report["points"] == summary["points"]


def test_upload_observer_failure_cannot_fail_successful_upload(tmp_path, monkeypatch):
    from cascade.generation import transfer
    from cascade.generation.cli import upload_corpus_dir
    from tests.unit.test_generation_stage import _corpus

    idx = _corpus(tmp_path / "out")
    names = [s["name"] for c in idx["children"] for s in c["shards"]] + ["corpus.json"]
    urls = tmp_path / "urls.json"
    urls.write_text(json.dumps(dict.fromkeys(names, "https://scoped-put")))
    monkeypatch.setattr(transfer, "put_file", lambda *a: None)
    def broken(_):
        raise OSError("down")
    assert upload_corpus_dir(tmp_path / "out", idx, urls, progress=broken) == len(names)
    assert not list((tmp_path / "out").glob("*.parquet"))


def test_two_generation_runs_and_trainer_event_do_not_close_each_other(monkeypatch):
    from cascade.shared.config import WandbConfig
    from cascade.trainer import wandb_sink as W
    from tests.unit.test_wandb_sink import _FakeRun

    active = []
    def init(**kw):
        if kw["reinit"] is True:
            for run in active:
                run.finish()
        assert kw["reinit"] == "create_new"
        run = _FakeRun()
        active.append(run)
        return run
    monkeypatch.setattr(W, "_wandb_module", lambda cfg: SimpleNamespace(init=init))
    cfg = WandbConfig(enabled=True, mode="offline")
    a = W.open_generation_run(cfg, round_id="r", hotkey="a", uid=1, generator_ref="ref")
    b = W.open_generation_run(cfg, round_id="r", hotkey="b", uid=2, generator_ref="ref")
    assert W.log_wandb_event(cfg, kind="promotion", round_id="r", record={"generation": 1})
    assert not a.run.finished and not b.run.finished and active[2].finished
    a.emit({"generation/points": 3})
    b.emit({"generation/points": 4})
    assert a.run.logged[-1]["generation/points"] == 3
    assert b.run.logged[-1]["generation/points"] == 4


def test_stage_polls_scoped_snapshot_and_rejects_an_old_vm_nonce(tmp_path, monkeypatch):
    import subprocess

    from cascade.trainer.remote import RemoteHost
    from tests.unit.test_generation_stage import FakeStore

    cfg = load_chain_config()
    cfg = replace(cfg, generation=replace(cfg.generation, progress_seconds=1))
    store, observed = FakeStore(), []
    arrived = threading.Event()
    def observer(doc):
        observed.append(doc)
        if doc["stage"] == "generating":
            arrived.set()
    stage = GenerationStage(cfg, store, [], "unused", "unused", observer=observer)
    identity = dict(round_id="r", hotkey="hk", generator_ref="ref", uid=1, attempt_id="current")
    stage._events.identity = identity
    monkeypatch.setattr(stage, "_copy", lambda *a, **kw: None)
    stale = event(points=999)
    store.put_text("corpora/round-r/hk/progress.json", json.dumps(stale))
    original_get = store.get_text
    reads = []
    def snapshot(key):
        reads.append(key)
        return json.dumps(stale) if len(reads) == 1 else original_get(key)
    monkeypatch.setattr(store, "get_text", snapshot)
    def run(argv, timeout):
        if "cascade-generate" in argv[-1]:
            # The old snapshot is on the same key; only this CPU nonce may
            # advance the job. No account credentials appear in argv.
            current = event(points=42)
            current["attempt_id"] = "current"
            store.put_text("corpora/round-r/hk/progress.json", json.dumps(current))
            assert arrived.wait(4)
            return subprocess.CompletedProcess(argv, 0, '{"ok":true}', "")
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(stage, "runner", run)
    host = RemoteHost(name="fake", host="127.0.0.1", user="fake", key_path="unused", workdir="/tmp/fake")
    assert stage._run(host, tmp_path, seed=1, gen_ref="ref", urls={})["ok"]
    assert len(reads) >= 2
    assert len(observed) == 1 and observed[0]["points"] == 42
    assert observed[0]["generator_ref"] == "ref" and observed[0]["attempt_id"] == "current"


def test_malformed_completion_metrics_cannot_change_a_successful_result(monkeypatch):
    cfg = load_chain_config()
    stage = GenerationStage(cfg, None, [], "unused", "unused", observer=lambda _: None)
    result = StageResult(index={"points": "unreadable counter"}, base="b", provider="p", shape="s", row={})
    monkeypatch.setattr(stage, "_generate", lambda **kw: result)
    assert stage.generate(round_id="r", hotkey="h", gen_ref="ref", uid=1) is result
