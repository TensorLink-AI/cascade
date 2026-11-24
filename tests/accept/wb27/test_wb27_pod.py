"""WB27 on-pod surface, on the box (no torch): ``pyramid_trainer.py``'s pure
lever defs read from source, the curriculum shard stream (stage corpora read
in order, each to its share), the merged pod index, the generator keyword,
and the miner CLI helpers. The torch legs run in CI only."""

from __future__ import annotations

import json

import numpy as np
import pytest

from . import helpers as h

SEL = {0: [1]}


class _C:
    def __init__(self, **kw):
        self.lane_b_lr_scale = 1.0
        self.power_ema_eval = -1
        self.power_ema_sigma_rels = ()
        self.__dict__.update(kw)


def test_lever_terms_are_lane_b_only():
    f = h.trainer_namespace()["lane_b_lever_terms"]
    sig = (0.05, 0.10)
    assert f(_C(), None, False, sig, 1) == (1.0, sig, 1)            # every lane A leg
    assert f(_C(lane_b_lr_scale=1.5), SEL, False, sig, 1) == (1.5, sig, 1)
    one = _C(power_ema_sigma_rels=(0.15,), power_ema_eval=0)
    assert f(one, SEL, True, sig, 1) == (1.0, (0.15,), 0)           # ONLY the chosen EMA
    for c, sel, nc in ((_C(lane_b_lr_scale=1.5), None, False), (one, None, True),
                       (one, SEL, False),
                       (_C(power_ema_sigma_rels=(0.15,), power_ema_eval=1), SEL, True),
                       (_C(power_ema_sigma_rels=(0.5,), power_ema_eval=0), SEL, True),
                       (_C(power_ema_sigma_rels=(0.05, 0.1), power_ema_eval=0), SEL, True),
                       (_C(lane_b_lr_scale=9.0), SEL, False)):
        with pytest.raises(ValueError):
            f(c, sel, nc, sig, 1)


def test_context_window_is_lane_b_only():
    f = h.trainer_namespace()["lane_b_context_patches"]
    assert f(4096, None, 32) == 128 and f(4096, SEL, 32) == 128
    assert f(16384, SEL, 32) == 512 and f(32768, SEL, 32) == 1024
    for c, sel in ((8192, None), (5000, SEL), (65536, SEL)):
        with pytest.raises(ValueError):
            f(c, sel, 32)
    src = h.trainer_path().read_text(encoding="utf-8")
    assert "max_ctx_patches=ctx_patches" in src and "max_ctx_patches=128" not in src
    assert "replace(contract, context_length=4096)" in src     # model config stays 4096


def test_leg_record_names_the_limit_that_stopped_it():
    f = h.trainer_namespace()["lane_b_leg_record"]
    assert f(10, 10, 99, 5.0, 100) == {"steps": 10, "tokens": 99, "stop": "steps"}
    assert f(4, 10, 99, 100.0, 100) == {"steps": 4, "tokens": 99, "stop": "hours"}
    assert f(4, 10, 99, 5.0, 100, 99) == {"steps": 4, "tokens": 99, "stop": "tokens"}
    assert f(4, 10, 50, 5.0, 100, 99) == {"steps": 4, "tokens": 50, "stop": "corpus"}


def test_train_pyramid_applies_the_levers():
    src = h.trainer_path().read_text(encoding="utf-8")
    assert "lane_b_lever_terms(contract, expert_selection, no_cooldown" in src
    assert 'group["base_lr"] *= lever_lr_scale' in src
    assert "lane_b_leg_record(step, steps, tokens, duration" in src


def _stage_corpus(d, n, value, seed):
    from cascade.generation.shards import finalize_index, write_shard

    d.mkdir(parents=True)
    series = [np.full((2, 8), float(value), dtype=np.float32) for _ in range(n)]
    sh = write_shard(d / "child-0.parquet", series)
    return finalize_index(d, seed=seed, children=[{"index": 0, "seed": seed, "shards": [sh]}])


def test_curriculum_stream_reads_stages_in_order(tmp_path):
    import shutil

    pytest.importorskip("pyarrow")       # the [generation] extra (CI installs it)

    from cascade.generation.shards import ShardError, merge_curriculum_index, write_index
    from cascade.trainer.stream import open_round_stream

    cur = (("easy", 25), ("hard", 75))
    a = _stage_corpus(tmp_path / "a", 10, 1.0, 1)
    b = _stage_corpus(tmp_path / "b", 10, 2.0, 2)
    merged, renames = merge_curriculum_index([a, b], cur)
    assert [c["stage"] for c in merged["children"]] == [0, 1]
    assert merged["curriculum"] == [["easy", 25], ["hard", 75]]
    pod = tmp_path / "pod"
    pod.mkdir()
    for new, old in renames.items():
        src = tmp_path / ("a" if new.startswith("s0-") else "b") / old
        shutil.copy(src, pod / new)
    write_index(pod, merged)
    with open_round_stream("shards", tmp_path, 0, None, token_budget=16 * 8,
                           corpus_dir=pod, curriculum=cur) as rs:
        vals = [float(np.asarray(x)[0, 0]) for x in rs.series()]
    # 128 points: stage 0 to 32 points (2 series), then stage 1 to the rest
    assert vals == [1.0, 1.0] + [2.0] * 6
    with pytest.raises(ShardError):
        open_round_stream("shards", tmp_path, 0, None, token_budget=64, corpus_dir=pod)
    with pytest.raises(ShardError):
        open_round_stream("shards", tmp_path, 0, None, token_budget=64, corpus_dir=pod,
                          curriculum=(("easy", 50), ("hard", 50)))


def test_curriculum_needs_a_shards_leg(tmp_path):
    from cascade.trainer.corpus import CorpusError
    from cascade.trainer.stream import open_round_stream

    with pytest.raises(CorpusError):
        open_round_stream("stream_cpu", tmp_path, 0, None, token_budget=1,
                          curriculum=(("a", 50), ("b", 50)))


def test_generator_keyword_is_opt_in():
    from cascade.trainer import corpus

    class Old:
        def __init__(self, config_dir, *, seed):
            pass

    class New:
        def __init__(self, config_dir, *, seed, curriculum_stage=""):
            pass

    assert not corpus._ctor_accepts_kwarg(Old, "curriculum_stage")
    assert corpus._ctor_accepts_kwarg(New, "curriculum_stage")
    src = (h.REPO / "cascade" / "trainer" / "corpus.py").read_text(encoding="utf-8")
    assert "generator_curriculum_unsupported" in src
    from cascade.shared.config.sections import GeneratorConfig

    assert GeneratorConfig.__dataclass_fields__["curriculum_stage"].default == ""


def test_generation_cli_takes_the_stage():
    from cascade.generation.cli import build_parser

    args = build_parser().parse_args(["r", "o", "--seed", "1", "--curriculum-stage", "easy"])
    assert args.curriculum_stage == "easy"
    assert build_parser().parse_args(["r", "o", "--seed", "1"]).curriculum_stage == ""


def test_cli_levers_and_limits(tmp_path):
    from cascade.miner.lane_b import generator_declares_curriculum_stage, lane_b_levers_json
    from cascade.shared.lane_b_levers import LeversError

    cfg, _ = h.armed_cfg()
    got = lane_b_levers_json(cfg, lr_scale="1.5", curriculum="easy:0.3,hard:0.7", ema=0.05)
    assert json.loads(got) == {"lr_scale": 1.5, "curriculum": [["easy", "0.3"], ["hard", "0.7"]],
                               "ema": 0.05}
    assert lane_b_levers_json(cfg) == ""
    assert json.loads(lane_b_levers_json(cfg, context_length=8192)) == {"context_length": 8192}
    with pytest.raises(LeversError) as e:
        lane_b_levers_json(cfg, context_length=65536)
    assert e.value.code == "context_length_out_of_range"
    with pytest.raises(LeversError) as e:
        lane_b_levers_json(cfg, ema=0.25)
    assert e.value.code == "ema_out_of_range"
    (tmp_path / "generator.py").write_text(
        "class Generator:\n    def __init__(self, d, *, seed, curriculum_stage=''):\n        pass\n")
    assert generator_declares_curriculum_stage(tmp_path)
    (tmp_path / "generator.py").write_text(
        "class Generator:\n    def __init__(self, d, *, seed):\n        pass\n")
    assert not generator_declares_curriculum_stage(tmp_path)
