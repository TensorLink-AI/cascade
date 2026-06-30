"""The reference generator runs, is deterministic, and passes verify."""

from __future__ import annotations

import pytest

from cascade.miner.verify import verify_repo
from cascade.trainer.corpus import assert_corpus_reproducible, build_corpus


def test_example_generator_builds_corpus(small_cfg, example_generator_dir):
    res = build_corpus(example_generator_dir, generation_seed=0, cfg=small_cfg.generator)
    assert res.n_series == 6
    assert res.total_points > 0
    assert len(res.digest) == 64


def test_example_generator_is_deterministic(small_cfg, example_generator_dir):
    d = assert_corpus_reproducible(example_generator_dir, 0, small_cfg.generator)
    assert len(d) == 64
    # Different seed → different corpus.
    other = build_corpus(example_generator_dir, generation_seed=1, cfg=small_cfg.generator)
    assert other.digest != d


def test_verify_accepts_example_generator(small_cfg, example_generator_dir):
    report = verify_repo(example_generator_dir, small_cfg, skip_runtime=False)
    assert report.ok, report.render()
    assert report.corpus_digest is not None


def test_verify_static_path_only(small_cfg, example_generator_dir):
    report = verify_repo(example_generator_dir, small_cfg, skip_runtime=True)
    assert report.ok
    assert report.runtime_skipped


def test_verify_ignores_judge_when_disabled(small_cfg, example_generator_dir):
    # [judge] enabled = false in the shipped config ⇒ the client is never called
    # even if one is passed.
    class BoomJudge:
        def complete(self, system, user):
            raise AssertionError("judge must not run when [judge] is disabled")

    report = verify_repo(example_generator_dir, small_cfg, skip_runtime=True, judge=BoomJudge())
    assert report.ok


def test_verify_runs_judge_when_enabled(small_cfg, example_generator_dir):
    import json
    from dataclasses import replace

    class FakeJudge:
        def complete(self, system, user):
            return json.dumps({"distillation": "pass", "benchmark_targeting": "warn",
                               "benchmark_targeting_reason": "weekly period"})

    cfg = replace(small_cfg, judge=replace(small_cfg.judge, enabled=True))
    report = verify_repo(example_generator_dir, cfg, skip_runtime=True, judge=FakeJudge())
    assert report.ok                                   # a warn does not reject
    assert ("benchmark_targeting", "weekly period") in report.warnings
    assert "warning [benchmark_targeting]" in report.render()


def test_verify_judge_distillation_fail_rejects(small_cfg, example_generator_dir):
    import json
    from dataclasses import replace

    class FakeJudge:
        def complete(self, system, user):
            return json.dumps({"distillation": "fail",
                               "distillation_reason": "replays fitted weights",
                               "benchmark_targeting": "pass"})

    cfg = replace(small_cfg, judge=replace(small_cfg.judge, enabled=True))
    report = verify_repo(example_generator_dir, cfg, skip_runtime=True, judge=FakeJudge())
    assert not report.ok
    assert any(step == "distillation" for step, _ in report.failures)


def test_build_round_corpus_cache_reuse(small_cfg, example_generator_dir):
    from cascade.trainer.corpus import build_round_corpus

    # use_sandbox=False keeps this a fast in-process unit test; the sandbox path
    # is exercised in test_sandbox.py.
    res = build_round_corpus(
        example_generator_dir, 0, small_cfg.generator, "cache_reuse", use_sandbox=False
    )
    assert res.n_series == 6
    assert len(res.digest) == 64


def test_build_round_corpus_rejects_stream_modes(small_cfg, example_generator_dir):
    # build_round_corpus is the materialised helper; streaming goes through
    # stream.open_round_stream. It rejects stream modes so a miswired caller fails.
    from cascade.trainer.corpus import CorpusError, build_round_corpus

    for mode in ("stream_cpu", "stream_gpu"):
        with pytest.raises(CorpusError):
            build_round_corpus(example_generator_dir, 0, small_cfg.generator, mode)
