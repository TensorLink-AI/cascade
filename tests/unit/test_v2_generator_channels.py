from __future__ import annotations

import contextlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cascade.generation.shards import ShardError, finalize_index, write_shard
from cascade.generation.tier import generate_corpus
from cascade.interface.generator import SeriesValidator, check_series
from cascade.miner.verify import verify_repo
from cascade.shared.config import GenerationConfig, load_chain_config
from cascade.trainer.corpus import CorpusError
from cascade.trainer.stream import open_round_stream

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("network", ["chain.toml", "chain.testnet.toml"])
def test_release_width_policy_and_historical_replay(network):
    cfg = load_chain_config(ROOT / network)
    assert (cfg.generator.min_channels, cfg.generator.max_channels) == (32, 32)
    assert cfg.generator_for_contract(cfg.training).min_channels == 1
    assert cfg.generator_for_contract(cfg.training.v2_contract).min_channels == 32


@pytest.mark.parametrize("channels", [None, 1, 7, 8, 31, 32, 33])
def test_every_array_and_record_yield_obeys_v2_bounds(channels):
    cfg = load_chain_config().generator
    shape = (64,) if channels is None else (channels, 64)
    values = np.random.default_rng(0).normal(size=shape)
    for item in (values, {"values": values}):
        validator = SeriesValidator.from_config(cfg)
        if channels == 32:
            validator.process(item, 0)
        else:
            with pytest.raises(ValueError, match=r"outside \[32, 32\]"):
                validator.process(item, 0)
    # Old direct callers/configs retain the original one-channel interface.
    check_series(np.ones(64), min_length=64, max_length=64)


def test_parent_refuses_underwidth_frames_even_if_child_skips_validation(tmp_path, monkeypatch):
    cfg = load_chain_config().generator
    monkeypatch.setattr("cascade.trainer.sandbox.stream_series", lambda *a, **k:
                        contextlib.nullcontext(iter([np.ones((7, 64))])))
    with open_round_stream("stream_cpu", tmp_path, 0, cfg, token_budget=1000,
                           use_sandbox=True) as stream:
        with pytest.raises(CorpusError, match=r"outside \[32, 32\]"):
            next(stream.series())
        assert stream.n_series == 0


def test_cpu_parent_rejects_nonfinite_output_from_a_bypassed_child(tmp_path):
    cfg = load_chain_config().generator
    index = generate_corpus(tmp_path, 1, cfg,
                            GenerationConfig(vcpus=1, target_points=64, shard_points=64),
                            tmp_path / "nan", runner=lambda *_:
                            contextlib.nullcontext(iter([np.full((8, 64), np.nan)])))
    assert index["stop_reason"] == "generator_fault" and index["n_series"] == 0


@pytest.mark.parametrize("channels", [7, 8, 31, 32, 33])
def test_cpu_tier_rejects_invalid_width_before_gpu_rental(tmp_path, channels):
    cfg = load_chain_config().generator
    def child(*_):
        return contextlib.nullcontext(iter([np.ones((channels, 64))]))
    index = generate_corpus(tmp_path, 1, cfg,
                            GenerationConfig(vcpus=1, target_points=64, shard_points=64),
                            tmp_path / "corpus", runner=child)
    if channels == 32:
        assert not index["faults"] and index["channels_histogram"] == {str(channels): 1}
        assert index["min_channels"] == 32 and index["max_channels"] == 32
    else:
        assert index["stop_reason"] == "generator_fault" and index["n_series"] == 0


def test_shard_reader_cannot_bypass_width_policy(tmp_path):
    shard = write_shard(tmp_path / "bad.parquet", [np.ones((7, 64))])
    finalize_index(tmp_path, seed=1, children=[{"index": 0, "seed": 1, "shards": [shard]}])
    with open_round_stream("shards", tmp_path, 1, load_chain_config().generator,
                           token_budget=1000, corpus_dir=tmp_path) as stream, \
            pytest.raises(ShardError, match=r"outside \[32, 32\]"):
        next(stream.series())


def test_real_starter_passes_current_verification_and_custom_widths(tmp_path):
    source = ROOT / "cascade/miner/templates/generator"
    cfg = load_chain_config()
    cfg = replace(cfg, generator=replace(cfg.generator, sandbox_strict=False))
    from cascade.miner.training_context import with_contract

    cfg = with_contract(cfg, cfg.training.v2_contract)
    report = verify_repo(source, cfg, quick=True)
    assert report.ok and not report.runtime_skipped, report.render()
    spec = importlib.util.spec_from_file_location("channel_starter", source / "generator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for channels in (32,):
        (tmp_path / "config.json").write_text(json.dumps({"channels": channels}))
        a = list(module.Generator(str(tmp_path), seed=2).generate(2))
        b = list(module.Generator(str(tmp_path), seed=2).generate(2))
        assert all(x.shape[0] == channels for x in a)
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_invalid_channel_configuration_is_rejected():
    cfg = load_chain_config().generator
    for lo, hi in ((0, 32), (33, 32), (8, 7)):
        with pytest.raises(ValueError, match="min_channels"):
            replace(cfg, min_channels=lo, max_channels=hi)


def test_empty_generator_cannot_queue_a_gpu_leg(tmp_path):
    cfg = load_chain_config().generator
    index = generate_corpus(tmp_path, 0, cfg,
                            GenerationConfig(vcpus=1, target_points=64, shard_points=64),
                            tmp_path / "empty", runner=lambda *_: contextlib.nullcontext(iter(())))
    assert index["stop_reason"] == "generator_fault" and index["n_series"] == 0
    assert index["faults"][0]["kind"] == "generator"


def test_static_verification_does_not_claim_channel_bounds_were_checked():
    report = verify_repo(ROOT / "cascade/miner/templates/generator", load_chain_config(), skip_runtime=True)
    assert report.ok and report.scope == "static" and report.runtime_skipped
    assert "remain unchecked" in report.render()


def test_old_width_cached_corpora_are_not_reused_for_new_gpu_legs(monkeypatch):
    from cascade.generation.stage import GenerationStage

    cfg = load_chain_config()
    stage = GenerationStage(cfg, object(), [], "", "")
    index = {"generator_ref": "source@sha256:one", "seed": 3,
             "corpus_digest": "known", "n_series": 2, "channels_histogram": {"7": 2}}
    monkeypatch.setattr("cascade.generation.stage.read_dataset_index", lambda *_:
                        [{"round_id": "1", "hotkey": "H", "corpus_digest": "known"}])
    monkeypatch.setattr(stage, "_uploaded_index", lambda *_: index)
    args = {"round_id": "1", "hotkey": "H", "gen_ref": index["generator_ref"], "seed": 3}
    assert stage.existing(**args) is None
    index["channels_histogram"] = {"8": 1, "32": 1}
    assert stage.existing(**args) is None
    index["channels_histogram"] = {"32": 2}
    assert stage.existing(**args).index is index


_WIDTH_GENERATOR = """\
import json
from pathlib import Path

import numpy as np

from cascade.interface import DataGenerator


class Generator(DataGenerator):
    def __init__(self, config_dir, *, seed):
        self._seed = int(seed)
        self._c = json.loads((Path(config_dir) / "config.json").read_text())["channels"]

    @property
    def name(self):
        return "width-fixture"

    def generate(self, n_series):
        rng = np.random.default_rng(self._seed)
        for _ in range(n_series):
            yield np.cumsum(rng.normal(size=(self._c, 256)), axis=1)
"""


@pytest.mark.parametrize("network", ["chain.toml", "chain.testnet.toml"])
def test_unbound_verification_holds_on_both_sides_of_the_launch(tmp_path, network):
    # Without --live/--at-block the CLI cannot know which contract trains the
    # entry; the pre-v2 one-channel floor must not admit a v2-invalid source.
    cfg = load_chain_config(ROOT / network)
    cfg = replace(cfg, generator=replace(cfg.generator, sandbox_strict=False))
    (tmp_path / "generator.py").write_text(_WIDTH_GENERATOR)
    (tmp_path / "requirements.txt").write_text("")
    for channels, ok in ((32, True), (8, False)):
        (tmp_path / "config.json").write_text(json.dumps({"channels": channels}))
        assert verify_repo(tmp_path, cfg, quick=True).ok  # the base contract alone
        report = verify_repo(tmp_path, cfg, quick=True, across_launch=True)
        assert report.ok is ok, report.render()
    assert "channels outside [32, 32]" in report.render()


@pytest.mark.parametrize('channels', [7, 8, 31, 32, 33])
def test_grouped_tier_enforces_width_and_reproduces_group_draw(tmp_path, channels):
    base = load_chain_config()
    cfg = base.generator_for_contract(base.training.v2_contract)
    def child(_repo, seed, *_):
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 3, size=channels)
        mapping = {}
        groups = np.array([mapping.setdefault(int(label), len(mapping)) for label in labels], dtype=np.uint8)
        return contextlib.nullcontext(iter([{'values': rng.normal(size=(channels, 64)), 'groups': groups}]))
    g = GenerationConfig(vcpus=2, target_points=1024, shard_points=512)
    a = generate_corpus(tmp_path, 4, cfg, g, tmp_path / 'a', runner=child)
    b = generate_corpus(tmp_path, 4, cfg, g, tmp_path / 'b', runner=child)
    if channels == 32:
        assert not a['faults'] and a['schema'] == 2
        assert a['corpus_digest'] == b['corpus_digest']
    else:
        assert a['stop_reason'] == 'generator_fault' and a['n_series'] == 0


def test_packing_template_is_optional_seeded_and_verifiable(tmp_path):
    import shutil

    from cascade.interface.generator import check_groups
    from cascade.miner.training_context import with_contract

    source = ROOT / 'cascade/miner/templates/generator'
    for name in ('generator.py', 'config.json', 'requirements.txt'):
        shutil.copy2(source / name, tmp_path / name)
    config = json.loads((tmp_path / 'config.json').read_text())
    config['channels'] = 32
    config['packing']['enabled'] = True
    config['packing']['shares'] = [30, 25, 25, 15, 5]
    (tmp_path / 'config.json').write_text(json.dumps(config))
    spec = importlib.util.spec_from_file_location('packed_template', tmp_path / 'generator.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a = list(module.Generator(str(tmp_path), seed=7).generate(4))
    b = list(module.Generator(str(tmp_path), seed=7).generate(4))
    for left, right in zip(a, b, strict=True):
        assert left['values'].dtype == np.float32
        check_groups(left['groups'], 32)
        assert np.array_equal(left['values'], right['values'])
        assert np.array_equal(left['groups'], right['groups'])
    cfg = load_chain_config()
    cfg = with_contract(cfg, cfg.training.v2_contract)
    cfg = replace(cfg, generator=replace(cfg.generator, sandbox_strict=False))
    report = verify_repo(tmp_path, cfg, quick=True)
    assert report.ok, report.render()


def test_king_min_channels_relaxes_only_king_legs(tmp_path, monkeypatch):
    base = load_chain_config()
    genesis = "kingrepo@sha256:" + "a" * 64
    base = replace(base, generator=replace(base.generator, king_min_channels=0,
                                           king_min_channels_refs=(genesis,)))
    assert base.for_king(genesis) is base  # unset: kings obey the release floor
    cfg = replace(base, generator=replace(base.generator, king_min_channels=1))
    king = cfg.for_king(genesis).generator_for_contract(cfg.training.v2_contract)
    assert (king.min_channels, king.max_channels) == (1, 32)
    # Genesis-only: a king that re-commits any other ref trains at the floor.
    recommit = "kingrepo@sha256:" + "b" * 64
    assert cfg.for_king(recommit) is cfg
    assert cfg.generator_for_contract(cfg.training.v2_contract).min_channels == 32
    monkeypatch.setattr("cascade.trainer.sandbox.stream_series", lambda *a, **k:
                        contextlib.nullcontext(iter([np.ones((3, 64))])))
    with open_round_stream("stream_cpu", tmp_path, 0, king, token_budget=1000,
                           use_sandbox=True) as stream:
        assert next(stream.series()).shape[0] == 3
    with pytest.raises(ValueError, match="king_min_channels"):
        replace(base.generator, king_min_channels=33)
