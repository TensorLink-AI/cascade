"""Shard admission uses a child's CPU budget without changing legacy sandboxes."""
from dataclasses import replace

from cascade.miner import verify
from cascade.miner.training_context import with_contract
from cascade.trainer.sandbox import _child_env, run_in_sandbox


def test_shard_profile_uses_tier_geometry_and_local_cpuset(cfg, monkeypatch):
    monkeypatch.setattr('os.sched_getaffinity', lambda _: {8, 9, 10, 11})
    assert verify._verification_cores(cfg) is None
    assert verify._verification_cores(cfg, across_launch=True) == (8,)
    active = with_contract(cfg, cfg.training.v2_contract)
    assert verify._verification_cores(active) == (8,)
    active = replace(active, generation=replace(active.generation, children=8))
    assert verify._verification_cores(active) == (8, 9, 10, 11)


def test_explicit_cores_override_inherited_geometry_and_scrub_env(monkeypatch):
    monkeypatch.setenv('CASCADE_LANE_INDEX', '0')
    monkeypatch.setenv('CASCADE_LANE_COUNT', '2')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '64')
    monkeypatch.setenv('MINER_API_KEY', 'do-not-copy')
    env = _child_env(cpu_cores=(8,))
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        assert env[name] == '1'
    assert 'MINER_API_KEY' not in env


def test_full_and_quick_verification_pin_real_children(tmp_path, small_cfg, monkeypatch):
    import os

    if not hasattr(os, 'sched_getaffinity'):
        return
    core = min(os.sched_getaffinity(0))
    repo = tmp_path / 'gen'
    repo.mkdir()
    (repo / 'config.json').write_text('{}')
    (repo / 'requirements.txt').write_text('')
    (repo / 'generator.py').write_text('''import os
import numpy as np
from cascade.interface.generator import DataGenerator
class Generator(DataGenerator):
    def __init__(self, config_dir, *, seed): self.seed = seed
    @property
    def name(self): return "cpu-profile"
    def generate(self, n_series):
        assert len(os.sched_getaffinity(0)) == 1
        for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert os.environ[key] == "1"
        rng = np.random.default_rng(self.seed)
        for _ in range(n_series): yield rng.normal(size=64)
''')
    # Trusted isolated fixture; production's AST policy still applies elsewhere.
    cfg = replace(small_cfg, static_guard=replace(small_cfg.static_guard, blocked=()))
    monkeypatch.delenv('CASCADE_LANE_INDEX', raising=False)
    monkeypatch.delenv('CASCADE_LANE_COUNT', raising=False)
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '64')
    first, err = verify._determinism_check(repo, cfg, cpu_cores=(core,))
    assert err is None
    second, err = verify._quick_determinism_check(repo, cfg, cpu_cores=(core,))
    assert err is None
    # Quick changes corpus size, so compare its digest with the same size directly.
    quick_cfg = replace(cfg.generator, corpus_target_points=65_536, corpus_n_series=128)
    assert second == run_in_sandbox(repo, 0, quick_cfg, cpu_cores=(core,)).digest
    assert first
    assert os.environ['OPENBLAS_NUM_THREADS'] == '64'
    assert 'CASCADE_LANE_COUNT' not in os.environ
