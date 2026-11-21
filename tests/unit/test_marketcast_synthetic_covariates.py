"""DEC-CA-0064 open item 5: a MarketCast generator names its rows
(``synthetic_channels``) so its samples carry their own covariates — past
rows from the generator, known rows (calendar / session / events) from its
``time`` row — instead of NaN."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np
import pytest

from cascade.eval.marketcast import PAST_COVARIATES
from cascade.pool import marketcast as P
from cascade.shared.config.marketcast import parse_marketcast
from cascade.shared.marketcast_recipe import parse_marketcast_recipe

T0 = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())
MC = parse_marketcast({
    "suites": ["returns", "vol"], "assets": {"crypto": ["BTC", "ETH"]},
    "frequencies": {"returns": ["1m", "5m"], "vol": ["5m"]},
    "horizons": {"returns": ["1m", "5m"], "vol": ["5m"]}, "max_steps": 1000,
})
N = 600


def _pure(**over):
    r = {"target": "mkt:returns",
         "data": {"assets": {}, "class_weights": {}, "frequencies": ["1m", "5m"],
                  "start": "2025-01-01", "end": "2025-01-02"},
         "context_length": 128, "horizon_mix": {"1m": 1.0, "5m": 1.0},
         "covariates": ["volume", "funding", "calendar"], "augmentations": [],
         "objective": {"suite_weights": {"returns": 1.0}},
         "optimizer": {"lr": 1e-3, "schedule": "cosine", "warmup_fraction": 0.1, "steps": 6,
                       "batch": 4, "weight_decay": 0.0, "clip": 1.0},
         "trainable": {"scope": "lora", "lora_rank": 2}, "synthetic_fraction": 1.0,
         "synthetic_channels": ["target", "volume", "time"]}
    r.update(over)
    return r


def _parse(**over):
    return parse_marketcast_recipe(_pure(**over), MC, has_generator=True)


def _snap(tmp_path):
    P.write_snapshot(tmp_path / "t", t0=T0, minutes=60, bars={}, sources={},
                     anchor_block=0, anchor_time=float(T0))
    return P.load_snapshot(tmp_path / "t", verify_signature=False)


def _arr(time=True):
    target = np.cumsum(np.ones(N))
    volume = 1000.0 + np.arange(N)                       # a recognisable row
    rows = [target, volume]
    if time:
        rows.append(T0 + 60.0 * np.arange(N))
    return np.stack(rows)


# ── recipe ───────────────────────────────────────────────────────────────────


def test_channels_parse_and_default_to_none():
    assert _parse().synthetic_channels == ("target", "volume", "time")
    r = dict(_pure())
    r.pop("synthetic_channels")
    assert parse_marketcast_recipe(r, MC, has_generator=True).synthetic_channels == ()


@pytest.mark.parametrize("chans, why", [
    (["volume", "target"], 'start with "target"'),
    (["target", "calendar"], "not allowed"),          # known rows come from "time"
    (["target", "spread"], "not allowed"),            # not in the recipe's covariates
    (["target", "volume", "volume"], "unique"),
    ([], 'start with "target"'),
])
def test_bad_channels_refused(chans, why):
    with pytest.raises(ValueError, match=why):
        _parse(synthetic_channels=chans)


def test_channels_need_a_generator_mix():
    over = {"synthetic_fraction": 0.0,
            "data": {"assets": {"crypto": ["BTC"]}, "class_weights": {"crypto": 1.0},
                     "frequencies": ["1m"], "start": "2025-01-01", "end": "2025-01-02"}}
    with pytest.raises(ValueError, match="needs synthetic_fraction > 0"):
        parse_marketcast_recipe(_pure(**over), MC)


def test_recipe_digest_is_unchanged_without_channels():
    L = pytest.importorskip("cascade.trainer.marketcast_leg")
    r = dict(_pure())
    r.pop("synthetic_channels")
    rec = parse_marketcast_recipe(r, MC, has_generator=True)
    d = asdict(rec)
    d.pop("synthetic_channels")
    old = hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert L.recipe_digest(rec) == old
    assert L.recipe_digest(_parse()) != old


# ── sampler ──────────────────────────────────────────────────────────────────


def _batch(tmp_path, arrs, recipe=None, n=8):
    L = pytest.importorskip("cascade.trainer.marketcast_leg")
    s = L.MktSampler(recipe or _parse(), MC, _snap(tmp_path), seed=3, synthetic=iter(arrs))
    return s.batch(n)


def test_named_rows_become_the_samples_covariates(tmp_path):
    b = _batch(tmp_path, [_arr() for _ in range(64)])
    assert b.synthetic == 8 and b.past is not None and b.known is not None
    order = [c for c in PAST_COVARIATES if c in ("volume", "funding")]
    vol, fund = b.past[:, order.index("volume")], b.past[:, order.index("funding")]
    for i in range(8):
        n = int(np.isfinite(b.contexts[i]).sum())
        # the volume row is the generator's own, aligned with the context
        assert np.all(vol[i, -n:] >= 1000.0) and np.all(np.diff(vol[i, -n:]) == 1.0)
        assert np.isnan(fund[i]).all()                # a row it does not carry: missing
    assert np.isfinite(b.known[:, :, -1]).all()       # calendar from its clock


def test_no_time_row_leaves_known_rows_missing(tmp_path):
    r = _parse(synthetic_channels=["target", "volume"])
    b = _batch(tmp_path, [_arr(time=False) for _ in range(64)], recipe=r)
    assert b.past is not None and np.isfinite(b.past).any()
    assert b.known is None or np.isnan(b.known).all()


def test_a_broken_clock_is_dropped_not_trusted(tmp_path):
    arrs = []
    for _ in range(64):
        a = _arr()
        a[2] = T0                                     # not increasing
        arrs.append(a)
    b = _batch(tmp_path, arrs)
    assert b.known is None or np.isnan(b.known).all()


def test_non_finite_rows_are_missing_not_inf(tmp_path):
    arrs = []
    for _ in range(64):
        a = _arr()
        a[1, :] = np.inf
        arrs.append(a)
    b = _batch(tmp_path, arrs)
    assert not np.isinf(b.past).any()


def test_wrong_row_count_is_the_miners_fault(tmp_path):
    L = pytest.importorskip("cascade.trainer.marketcast_leg")
    with pytest.raises(L.MktLegError, match="synthetic_channels names 3 rows"):
        _batch(tmp_path, [_arr()[:2] for _ in range(64)])


def test_without_channels_rows_stay_separate_series(tmp_path):
    r = dict(_pure())
    r.pop("synthetic_channels")
    rec = parse_marketcast_recipe(r, MC, has_generator=True)
    b = _batch(tmp_path, [_arr() for _ in range(64)], recipe=rec)
    assert b.synthetic == 8
    assert b.past is None or np.isnan(b.past).all()


# ── bar size: a generator array's clock tags it ──────────────────────────────


def _tagged(minutes: int, mark: float):
    return np.stack([np.cumsum(np.ones(N)), mark + np.arange(N),
                     T0 + 60.0 * minutes * np.arange(N)])


def test_arrays_are_used_only_at_their_clocks_bar_size(tmp_path):
    L = pytest.importorskip("cascade.trainer.marketcast_leg")
    arrs = [_tagged(1, 1000.0) if i % 2 else _tagged(5, 5000.0) for i in range(4000)]
    s = L.MktSampler(_parse(), MC, _snap(tmp_path), seed=5, synthetic=iter(arrs))
    order = [c for c in PAST_COVARIATES if c in ("volume", "funding")]
    seen = set()
    for _ in range(12):
        b = s.batch(4)
        vol = b.past[:, order.index("volume")]
        fin = vol[np.isfinite(vol)]
        seen.add(b.freq_minutes)
        assert (fin >= 5000.0).all() if b.freq_minutes == 5 else (fin < 5000.0).all()
    assert seen == {1, 5}


def test_a_bar_size_the_generator_never_serves_is_dropped_not_fatal(tmp_path):
    L = pytest.importorskip("cascade.trainer.marketcast_leg")
    s = L.MktSampler(_parse(), MC, _snap(tmp_path), seed=5,
                     synthetic=iter([_tagged(1, 1000.0) for _ in range(20000)]))
    for _ in range(8):
        assert s.batch(4).freq_minutes == 1
    assert all(f == 1 for f, _st in s.geom.values())


def test_untagged_arrays_serve_every_bar_size(tmp_path):
    r = _parse(synthetic_channels=["target", "volume"])
    L = pytest.importorskip("cascade.trainer.marketcast_leg")
    s = L.MktSampler(r, MC, _snap(tmp_path), seed=5,
                     synthetic=iter([_arr(time=False) for _ in range(4000)]))
    assert {s.batch(4).freq_minutes for _ in range(12)} == {1, 5}


def test_a_32_row_yield_is_cut_into_channel_groups(tmp_path):
    """The sandbox fixes 32 rows per yield on mainnet: 3 names ⇒ 10 samples
    from one yield, its last 2 rows ignored."""
    L = pytest.importorskip("cascade.trainer.marketcast_leg")
    one = np.concatenate([_tagged(1, 1000.0 + 100 * g) for g in range(10)]
                         + [np.full((2, N), 9e9)])
    assert one.shape == (32, N)
    s = L.MktSampler(_parse(), MC, _snap(tmp_path), seed=5, synthetic=iter([one] * 50))
    got = [s._syn_take(1) for _ in range(10)]
    assert all(g is not None for g in got)
    assert sorted(int(g["volume"][0]) for g in got) == [1000 + 100 * g for g in range(10)]
    assert all(np.all(g["volume"] < 9e9) for g in got)   # the 2 spare rows never used
