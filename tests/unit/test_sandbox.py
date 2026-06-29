"""The generation sandbox runs untrusted generator code out-of-process.

allow_netns=False keeps these tests off the unshare path (which may be
unavailable in CI); the subprocess, rlimits, env-scrub, and digest round-trip
are still exercised.
"""

from __future__ import annotations

import pytest

from cascade.trainer.corpus import CorpusError, build_corpus
from cascade.trainer.sandbox import run_in_sandbox


def test_sandbox_matches_in_process_digest(small_cfg, example_generator_dir):
    # Same generator + seed in the sandbox yields the byte-identical corpus the
    # in-process path does — the digest survives the subprocess round-trip.
    in_proc = build_corpus(example_generator_dir, 0, small_cfg.generator)
    boxed = run_in_sandbox(
        example_generator_dir, 0, small_cfg.generator,
        blocked=small_cfg.static_guard.blocked, allow_netns=False,
    )
    assert boxed.digest == in_proc.digest
    assert boxed.n_series == in_proc.n_series == 6
    assert boxed.total_points == in_proc.total_points


def test_sandbox_preflight_rejects_blocked_import(tmp_path, small_cfg):
    # The static guard runs before any miner code is imported/executed.
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "requirements.txt").write_text("")
    (tmp_path / "generator.py").write_text("import socket\n")
    with pytest.raises(CorpusError):
        run_in_sandbox(
            tmp_path, 0, small_cfg.generator,
            blocked=small_cfg.static_guard.blocked, allow_netns=False,
        )


def test_sandbox_reports_generator_runtime_error(tmp_path, small_cfg):
    # A generator that blows up at runtime fails the round cleanly (CorpusError),
    # not by crashing the parent.
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "requirements.txt").write_text("")
    (tmp_path / "generator.py").write_text(
        "from cascade.interface import DataGenerator\n"
        "class Generator(DataGenerator):\n"
        "    def __init__(self, config_dir, *, seed): pass\n"
        "    @property\n"
        "    def name(self): return 'boom'\n"
        "    def generate(self, n_series):\n"
        "        raise RuntimeError('boom')\n"
        "        yield\n"
    )
    with pytest.raises(CorpusError):
        run_in_sandbox(
            tmp_path, 0, small_cfg.generator,
            blocked=small_cfg.static_guard.blocked, allow_netns=False,
        )


def test_sandbox_rejects_oversize_repo(tmp_path, small_cfg):
    from dataclasses import replace

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "requirements.txt").write_text("")
    (tmp_path / "generator.py").write_text("x = 1\n")
    (tmp_path / "weights.safetensors").write_bytes(b"z" * 4096)
    tiny = replace(small_cfg.generator, max_repo_mb=0)  # nothing fits
    with pytest.raises(CorpusError):
        run_in_sandbox(tmp_path, 0, tiny, blocked=small_cfg.static_guard.blocked, allow_netns=False)
