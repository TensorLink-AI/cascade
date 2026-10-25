"""DEC-CA-0050 generation surface: per-child read-ahead keeps the seed-mix
interleave (and so the corpus digest) byte-identical; cache_reuse children
build concurrently; corpus_epochs sizes the unique corpus to budget / epochs."""

from __future__ import annotations

import math

import pytest

from cascade.shared.config import validate_corpus_epochs
from cascade.shared.manifest import contract_payload
from cascade.trainer.contract import _mix
from cascade.trainer.corpus import CorpusError
from cascade.trainer.stream import _Prefetch, _StreamDigest, open_round_stream


def _child_series(gen_dir, cfg, seed, budget):
    with open_round_stream("stream_cpu", gen_dir, seed, cfg, token_budget=budget,
                           use_sandbox=False) as rs:
        return list(rs.series())


def test_prefetched_interleave_matches_manual_round_robin(small_cfg, example_generator_dir):
    n, budget = 3, 3000
    share = budget // n
    budgets = [budget - share * (n - 1)] + [share] * (n - 1)
    kids = [_child_series(example_generator_dir, small_cfg.generator, _mix(0, "seed-mix", i), b)
            for i, b in enumerate(budgets)]
    dig = _StreamDigest()
    longest = max(len(k) for k in kids)
    for j in range(longest):
        for k in kids:
            if j < len(k):
                dig.update(k[j])
    with open_round_stream("stream_cpu", example_generator_dir, 0, small_cfg.generator,
                           token_budget=budget, use_sandbox=False, seed_mix=n) as rs:
        for _ in rs.series():
            pass
        assert rs.digest == dig.hexdigest()


def test_prefetch_propagates_errors_in_order():
    def gen():
        yield 1
        yield 2
        raise RuntimeError("boom")

    it = _Prefetch(gen(), maxsize=1)
    assert next(it) == 1 and next(it) == 2
    with pytest.raises(RuntimeError, match="boom"):
        next(it)


def test_corpus_epochs_sizes_unique_corpus(small_cfg, example_generator_dir):
    budget, epochs = 6000, 3
    with open_round_stream("cache_reuse", example_generator_dir, 0, small_cfg.generator,
                           token_budget=budget, use_sandbox=False,
                           corpus_epochs=epochs) as rs:
        consumed = sum(int(a.size) for a in rs.series())
        unique = rs._corpus.total_points
    assert unique >= math.ceil(budget / epochs)
    assert consumed >= budget
    assert consumed / unique <= epochs + 1     # at most `epochs` full passes (+ the tail)


def test_corpus_epochs_guards():
    assert validate_corpus_epochs(0, "stream_cpu") == 0
    assert validate_corpus_epochs(4, "cache_reuse") == 4
    with pytest.raises(ValueError, match="cache_reuse"):
        validate_corpus_epochs(4, "stream_cpu")
    with pytest.raises(ValueError):
        validate_corpus_epochs(17, "cache_reuse")


def test_corpus_epochs_refused_outside_cache_mode(small_cfg, example_generator_dir):
    with pytest.raises(CorpusError, match="cache_reuse"):
        open_round_stream("stream_cpu", example_generator_dir, 0, small_cfg.generator,
                          token_budget=100, use_sandbox=False, corpus_epochs=2)


def test_corpus_epochs_is_drop_when_default():
    assert "corpus_epochs" not in contract_payload({"corpus_epochs": 0, "x": 1})
    assert contract_payload({"corpus_epochs": 4})["corpus_epochs"] == 4
