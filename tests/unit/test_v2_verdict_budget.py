"""A bigger private pool must actually produce more evidence in v2 verdicts."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.eval.window import EvalWindow
from cascade.validator.expert_lane import verdict_windows
from cascade.validator.pool import BucketWindowSource
from cascade.validator.windows import MixParams, RotatingWindowSource


def _source(count=120):
    return RotatingWindowSource(tuple(EvalWindow(str(i), np.arange(64.), np.arange(8.),
        metadata={'domain': f'd{i%3}', 'source': f'family{i%20}'}) for i in range(count)),
        mix=MixParams(from_block=1, target_windows=20, series_bag_frac=0.25))


def test_v2_budget_keeps_the_launch_boundary_historical(cfg):
    era = cfg.round.epoch_blocks * max(1, cfg.round.era_settlements)
    gate = era * 3
    cfg = replace(cfg, v2=replace(cfg.v2, from_block=gate, eval_n_windows=80),
                  eval=replace(cfg.eval, n_windows=100, scored_from_block=0))
    source = _source()
    assert cfg.eval_windows_at(gate) == 100  # Last settlement of the old era.
    assert len(verdict_windows(cfg, source, 7, block=gate)) == 20
    actual = verdict_windows(cfg, source, 7, block=gate + cfg.round.epoch_blocks)
    assert cfg.eval_windows_at(gate + cfg.round.epoch_blocks) == 80
    assert len(actual) == 80 and len({w.series_id for w in actual}) == 80
    assert source.mix.target_windows == 20 and source.mix.series_bag_frac == 0.25
    assert [w.series_id for w in actual] == [w.series_id for w in verdict_windows(cfg, source, 7, block=gate + cfg.round.epoch_blocks)]
    assert len(source.windows_for_verdict(7, 200, block=gate)) == 120  # Capacity, no fabricated/duplicate windows.


def test_bucket_draw_forwards_the_authoritative_budget_without_mutating_legacy_source():
    source = _source()
    bucket = object.__new__(BucketWindowSource)
    bucket._select = lambda block: SimpleNamespace(sha256='fixture')
    bucket._ensure_snapshot = lambda meta: source
    assert len(bucket.windows_for_verdict(8, 80, block=100)) == 80
    assert len(bucket.windows_for_round(8, 100, block=100)) == 20


def test_v2_ladder_uses_the_new_budget(cfg, monkeypatch):
    era = cfg.round.epoch_blocks * max(1, cfg.round.era_settlements)
    cfg = replace(cfg, v2=replace(cfg.v2, from_block=era*3, eval_n_windows=4000),
                  eval=replace(cfg.eval, scored_horizons=(64, 256), scored_from_block=1))
    seen = []
    monkeypatch.setattr('cascade.validator.windows.ladder_windows_for_round',
                        lambda *args, **kw: seen.append(kw) or [])
    verdict_windows(cfg, object(), 8, block=era*4)
    assert seen[0]['n_windows'] == 4000


@pytest.mark.parametrize('value', [-1, True, 2.5, '4000'])
def test_bad_v2_window_budgets_are_rejected(value):
    from cascade.shared.config.v2 import parse_v2_economics

    with pytest.raises(ValueError, match='eval_n_windows'):
        parse_v2_economics({'eval_n_windows': value})
