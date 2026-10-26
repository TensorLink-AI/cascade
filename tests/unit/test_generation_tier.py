"""DEC-CA-0053 generation tier: fan-out, shards, stop reasons, faults, CLI."""

from __future__ import annotations

import contextlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from cascade.generation import cli, shards  # noqa: E402
from cascade.generation.tier import generate_corpus  # noqa: E402
from cascade.shared.config import GenerationConfig, load_chain_config  # noqa: E402
from cascade.trainer.corpus import CorpusError  # noqa: E402

# These deterministic infrastructure fixtures model historical 1–2-channel corpora.
GEN = replace(load_chain_config().generator, min_channels=1)


def _gcfg(**kw) -> GenerationConfig:
    base = dict(vcpus=3, target_points=3000, shard_points=400, window_seconds=600)
    base.update(kw)
    return GenerationConfig(**base)


def fake_runner(*, fail_child_seed=None, exc=None, length=100, limit=None):
    """Deterministic per-seed series; optionally one seed faults midway."""
    exc = exc if exc is not None else CorpusError("boom")
    def runner(repo, seed, cfg, budget, max_wall):
        def it():
            rng = np.random.default_rng(seed)
            i = 0
            while limit is None or i < limit:
                if seed == fail_child_seed and i == 3:
                    raise exc
                c = 1 + (i % 2)
                yield rng.standard_normal((c, length))
                i += 1
        return contextlib.nullcontext(it())
    return runner


def test_generate_splits_target_and_reproduces(tmp_path: Path) -> None:
    g = _gcfg()
    a = generate_corpus(tmp_path, 11, GEN, g, tmp_path / "a", runner=fake_runner())
    b = generate_corpus(tmp_path, 11, GEN, g, tmp_path / "b", runner=fake_runner())
    assert a["stop_reason"] == "target" and not a["faults"]
    assert a["corpus_digest"] == b["corpus_digest"]
    assert len(a["children"]) == 3
    # each child reaches its share (1000 points); series are 100 or 200 points
    for c in a["children"]:
        assert 1000 <= c["points"] < 1200
        assert c["seed"] == shards.child_seed(11, c["index"])
        assert all(sh["name"].startswith(f"c{c['index']}-") for sh in c["shards"])
    assert len(a["children"][0]["shards"]) >= 2   # shard_points=400 rolls shards
    # the index digest is the round-robin stream digest of the stored values
    dg, n, pts = shards.stream_digest(shards.iter_corpus(tmp_path / "a"))
    assert (dg, n, pts) == (a["corpus_digest"], a["n_series"], a["points"])
    on_disk = json.loads((tmp_path / "a" / shards.INDEX_NAME).read_text())
    assert on_disk["corpus_digest"] == a["corpus_digest"]
    assert on_disk["vcpus"] == 3 and on_disk["cascade_version"]


def test_different_seed_changes_corpus(tmp_path: Path) -> None:
    g = _gcfg()
    a = generate_corpus(tmp_path, 1, GEN, g, tmp_path / "a", runner=fake_runner())
    b = generate_corpus(tmp_path, 2, GEN, g, tmp_path / "b", runner=fake_runner())
    assert a["corpus_digest"] != b["corpus_digest"]


def test_window_stops_children(tmp_path: Path) -> None:
    clock = iter(range(0, 10**6, 50))
    idx = generate_corpus(tmp_path, 5, GEN, _gcfg(vcpus=1), tmp_path / "o",
                          deadline_s=120, runner=fake_runner(), now=lambda: next(clock))
    assert idx["stop_reason"] == "window"
    assert 0 < idx["n_series"] < 10


def test_exhausted_generator(tmp_path: Path) -> None:
    idx = generate_corpus(tmp_path, 5, GEN, _gcfg(vcpus=2), tmp_path / "o",
                          runner=fake_runner(limit=2))
    assert idx["stop_reason"] == "exhausted"
    assert idx["n_series"] == 4


def test_generator_fault_is_classified_and_stops(tmp_path: Path) -> None:
    g = _gcfg(vcpus=2, target_points=10**6)
    bad = shards.child_seed(9, 1)
    idx = generate_corpus(tmp_path, 9, GEN, g, tmp_path / "o",
                          runner=fake_runner(fail_child_seed=bad))
    assert idx["stop_reason"] == "generator_fault"
    assert idx["faults"][0]["child"] == 1 and idx["faults"][0]["kind"] == "generator"
    assert "boom" in idx["faults"][0]["error"]


def test_infra_fault_is_classified(tmp_path: Path) -> None:
    g = _gcfg(vcpus=1, target_points=10**6)
    idx = generate_corpus(tmp_path, 9, GEN, g, tmp_path / "o",
                          runner=fake_runner(fail_child_seed=shards.child_seed(9, 0),
                                             exc=OSError("disk gone")))
    assert idx["stop_reason"] == "infra_fault"
    assert idx["faults"][0]["kind"] == "infra"


def test_record_with_extra_fields_is_a_generator_fault(tmp_path: Path) -> None:
    def runner(repo, seed, cfg, budget, max_wall):
        return contextlib.nullcontext(iter([{"values": np.ones(64), "mask": np.ones(64)}]))
    idx = generate_corpus(tmp_path, 1, GEN, _gcfg(vcpus=1), tmp_path / "o", runner=runner)
    assert idx["stop_reason"] == "generator_fault"


def test_validation_rejection_is_a_generator_fault(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2] / "scripts" / "example_generator"
    gen = replace(GEN, max_length=128)    # the example yields longer series
    idx = generate_corpus(repo, 3, gen, _gcfg(vcpus=1, target_points=20_000, max_length=128),
                          tmp_path / "o", use_sandbox=False)
    assert idx["stop_reason"] == "generator_fault"
    assert "outside" in idx["faults"][0]["error"]


def test_child_cfg_carries_memory_and_length(tmp_path: Path) -> None:
    seen = []

    def runner(repo, seed, cfg, budget, max_wall):
        seen.append((cfg.max_memory_mb, cfg.max_length, budget, max_wall))
        return contextlib.nullcontext(iter([np.zeros((1, 64))]))
    generate_corpus(tmp_path, 1, GEN, _gcfg(vcpus=2, child_memory_mb=1536,
                    max_length=8192, target_points=1001), tmp_path / "o",
                    deadline_s=77, runner=runner)
    assert sorted(seen) == [(1536, 8192, 500, 77), (1536, 8192, 501, 77)]


def test_inprocess_runner_with_example_generator(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2] / "scripts" / "example_generator"
    idx = generate_corpus(repo, 3, GEN, _gcfg(vcpus=2, target_points=20_000, max_length=4096),
                          tmp_path / "o", use_sandbox=False)
    assert not idx["faults"], idx["faults"]
    assert idx["points"] >= 20_000


def test_cli_refuses_when_disabled(tmp_path: Path, capsys, monkeypatch) -> None:
    cfg = load_chain_config()
    off = replace(cfg, generation=replace(cfg.generation, enabled=False))
    monkeypatch.setattr("cascade.shared.config.load_chain_config", lambda *a: off)
    rc = cli.main([str(tmp_path), str(tmp_path / "o"), "--seed", "1", "--json"])
    assert rc == cli.EXIT_REFUSED
    assert "enabled = false" in capsys.readouterr().out


def test_cli_runs_locally(tmp_path: Path, capsys, monkeypatch) -> None:
    repo = Path(__file__).resolve().parents[2] / "cascade/miner/templates/generator"
    cfg = load_chain_config()
    small = replace(cfg, generation=_gcfg(vcpus=1, target_points=5000, max_length=4096))
    monkeypatch.setattr("cascade.shared.config.load_chain_config", lambda *a: small)
    rc = cli.main([str(repo), str(tmp_path / "o"), "--seed", "4", "--json",
                   "--no-sandbox", "--allow-disabled", "--generator-ref", "x/y@sha256:" + "0" * 64])
    out = json.loads(capsys.readouterr().out)
    assert rc == cli.EXIT_OK and out["ok"] and out["points"] >= 5000
    assert shards.read_index(tmp_path / "o")["generator_ref"].startswith("x/y@")
