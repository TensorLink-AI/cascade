"""Correlated feeds are evidence families, not independent bootstrap votes."""
from dataclasses import replace

import numpy as np
import pytest

from cascade.eval.koth import _window_clusters
from cascade.eval.scoring import score_forecaster_on_windows
from cascade.pool.builder import PoolBuildConfig, collect_records, prepare_series
from cascade.pool.families import catalog_families
from cascade.pool.source import HarvestContext, HarvestedSeries
from cascade.validator.windows import build_windows_from_series


def test_crypto_fx_sensor_and_declared_pairs_share_conservative_families():
    catalog = [{'id': f'venue{i}_btc_trade_rate_1m', 'dgp_class': 'market_microstructure',
                'endpoint': {'url': f'https://venue{i}.example/trades'}} for i in range(13)]
    catalog += [
        {'id': 'ecb_fx', 'dgp_class': 'fx_rate', 'endpoint': {'url': 'https://ecb.example/fx'}},
        {'id': 'other_fx', 'dgp_class': 'fx_rate', 'endpoint': {'url': 'https://bank.example/fx'}},
        {'id': 'sensor_1', 'endpoint': {'url': 'https://sensors.example/station/1'}},
        {'id': 'sensor_2', 'endpoint': {'url': 'https://sensors.example/station/2'}},
        {'id': 'clean', 'source_family': 'clean-noisy', 'paired_with': 'noisy'},
        {'id': 'noisy', 'endpoint': {'url': 'https://different.example/data'}},
    ]
    mapping = catalog_families(catalog)
    assert {mapping[r['id']] for r in catalog[:13]} == {'market:crypto'}
    assert mapping['ecb_fx'] == mapping['other_fx'] == 'market:fx'
    assert mapping['sensor_1'] == mapping['sensor_2']
    assert mapping['clean'] == mapping['noisy']
    assert catalog_families(list(reversed(catalog))) == mapping


def test_family_metadata_reaches_the_actual_scoring_bootstrap():
    rows = []
    cfg = PoolBuildConfig(context_length=64, horizon=8, min_context=64,
                          cluster_by='source_family', require_source=True)
    for i in range(13):
        hs = HarvestedSeries(f'feed{i}', np.arange(72.0) + i, 'H', 'econ_fin',
                             source=f'venue{i}', attrs={'source_family': 'market:crypto'})
        record, error = prepare_series(hs, cfg)
        assert error is None
        assert record.metadata['catalog_source'] == f'venue{i}'
        rows.append(record)
    windows = build_windows_from_series([r.values for r in rows], context_length=64, horizon=8,
                                        metadata=[r.metadata for r in rows])
    def forecast(history, horizon, samples):
        return np.full((1, samples, horizon), history[-1])
    scores = score_forecaster_on_windows(forecast, windows, 4)
    labels, count = _window_clusters(scores)
    assert len(scores) == 13 and count == 1
    assert set(labels) == {'market:crypto'}


def test_legacy_metadata_unchanged_and_primary_rejects_missing_families():
    hs = HarvestedSeries('s', np.arange(72.0), 'H', 'energy', source='old-feed',
                         attrs={'source_family': 'origin:provider.example'})
    cfg = PoolBuildConfig(context_length=64, horizon=8, min_context=64)
    legacy, _ = prepare_series(hs, cfg)
    previous, _ = prepare_series(replace(hs, attrs={}), cfg)
    assert legacy.metadata == previous.metadata
    assert legacy.metadata['source'] == 'old-feed'
    for attrs in ({}, {'source_family': 'unresolved:missing'}):
        record, reason = prepare_series(replace(hs, attrs=attrs), replace(cfg, cluster_by='source_family'))
        assert record is None and reason == 'missing_source_family'
    with pytest.raises(ValueError, match='source_family'):
        catalog_families([{'id': 'bad', 'source_family': ''}])


def test_both_pool_caps_must_increase_to_use_the_extra_domain_capacity():
    import datetime as dt

    class Source:
        def harvest(self, _fetch, _ctx):
            for d in range(2):
                for f in ('H', '15min'):
                    for i in range(500):
                        yield HarvestedSeries(f'{d}-{f}-{i}', np.arange(72.0) + d*10000 + (f == 'H')*1000+i,
                                              f, f'domain{d}', source=f'feed{d}-{f}')
    ctx = HarvestContext(as_of=dt.date(2026, 10, 27))
    cfg = PoolBuildConfig(context_length=64, horizon=8, min_context=64,
                          max_series_per_domain=600, max_series_per_domain_freq=300)
    old, _ = collect_records([Source()], ctx, cfg, fetch=None)
    domain_only, _ = collect_records([Source()], ctx, replace(cfg, max_series_per_domain=800), fetch=None)
    both, _ = collect_records([Source()], ctx, replace(cfg, max_series_per_domain=800,
                                                    max_series_per_domain_freq=400), fetch=None)
    assert len(old) == len(domain_only) == 1200
    assert len(both) == 1600
