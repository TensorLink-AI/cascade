"""Round corpus streaming: stream_cpu (fresh, no reuse) and cache_reuse.

Budgets are tiny so the example generator yields only a handful of series.
"""

from __future__ import annotations

import pytest

from cascade.trainer.corpus import CorpusError, build_corpus
from cascade.trainer.stream import open_round_stream

BUDGET = 3000


def _drain(mode, example_generator_dir, cfg, *, use_sandbox, seed=0):
    with open_round_stream(
        mode, example_generator_dir, seed, cfg, token_budget=BUDGET,
        use_sandbox=use_sandbox, blocked=("socket",), allow_netns=False,
    ) as rs:
        points = sum(int(a.size) for a in rs.series())
        return rs.digest, rs.n_series, rs.total_points, points


def test_stream_cpu_in_process(small_cfg, example_generator_dir):
    digest, n, total, points = _drain(
        "stream_cpu", example_generator_dir, small_cfg.generator, use_sandbox=False
    )
    assert points == total >= BUDGET  # stops once the budget is covered
    assert n >= 1
    assert len(digest) == 64


def test_stream_cpu_is_deterministic(small_cfg, example_generator_dir):
    a = _drain("stream_cpu", example_generator_dir, small_cfg.generator, use_sandbox=False)
    b = _drain("stream_cpu", example_generator_dir, small_cfg.generator, use_sandbox=False)
    assert a == b
    # A different seed draws a different fresh stream.
    c = _drain("stream_cpu", example_generator_dir, small_cfg.generator, use_sandbox=False, seed=1)
    assert c[0] != a[0]


def test_stream_cpu_sandbox_matches_in_process(small_cfg, example_generator_dir):
    # The sandboxed pipe round-trip yields byte-identical series to the in-process
    # path: same digest, same count, same points.
    in_proc = _drain("stream_cpu", example_generator_dir, small_cfg.generator, use_sandbox=False)
    boxed = _drain("stream_cpu", example_generator_dir, small_cfg.generator, use_sandbox=True)
    assert boxed == in_proc


def test_cache_reuse_digest_is_unique_corpus(small_cfg, example_generator_dir):
    corpus = build_corpus(example_generator_dir, 0, small_cfg.generator)
    digest, n, total, points = _drain(
        "cache_reuse", example_generator_dir, small_cfg.generator, use_sandbox=False
    )
    assert digest == corpus.digest          # unique-corpus digest, not the repeats
    assert n == 6                           # corpus_n_series unique series
    assert points >= BUDGET                 # cycled to fill the budget (reuse)
    assert total >= BUDGET


def test_stream_gpu_is_wired(small_cfg, example_generator_dir):
    # stream_gpu shares the fresh-series path; on a CPU (numpy) generator under
    # the GPU sandbox profile it streams just like stream_cpu (the profile only
    # relaxes the address-space rlimit + passes CUDA env, both no-ops here).
    digest, n, total, points = _drain(
        "stream_gpu", example_generator_dir, small_cfg.generator, use_sandbox=False
    )
    assert points == total >= BUDGET
    assert n >= 1 and len(digest) == 64


def test_unknown_corpus_mode_rejected(small_cfg, example_generator_dir):
    with pytest.raises(CorpusError):
        open_round_stream(
            "stream_tpu", example_generator_dir, 0, small_cfg.generator, token_budget=BUDGET
        )
