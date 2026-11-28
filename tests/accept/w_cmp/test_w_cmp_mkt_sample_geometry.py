"""MarketCast sample geometry: refuse at fund time a recipe whose leg cannot
cut its training samples (2026-11-27 23:24).

A funded leg (5Cu2sxndK8Dw, Alemaj-ft-ai-7/mkt-7c39bee2: mkt:market,
``data.start`` 2024-01-01 .. ``data.end`` 2026-10-25, context 4096, batch 6,
``synthetic_fraction`` 0) ended as a miner fault "could not cut a
high/15mx96 training sample" after ~32 min of CPU prep on a paid pod. Why:
the 15m bank (the recorder's native 15m candles) starts 2026-09-25, so each
15m series over the recipe's range is ~98,900 blocks with ~2,900 finite;
``MktSampler._real`` draws the sample end uniformly over the whole span (16
tries per sample), so a batch of 6 misses with ~6 % and the leg's thousands
of 15m batches certainly hit one. The context length is not the cause (the
context is cut to whatever precedes the sample); the date range is.

These pin the fund-time check (``cascade.funding.mkt_geom`` over the public
post-embargo bar index): this recipe is refused naming the 15m history start
and the earliest data.start that fits; the same recipe with a shorter
context is STILL refused; with that data.start, or without 15m, it is
accepted. Fixtures: the leg's recipe (from its failed-leg log) and the
public index of 2026-11-27 (first/last bar per class/symbol/timeframe).
Torch-free: no worker, no pod, no network.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

FIX = Path(__file__).resolve().parent / "fixtures"
RECIPE = json.loads((FIX / "mkt_geom_recipe.json").read_text())
INDEX = json.loads((FIX / "mkt_public_index_20261127.json").read_text())
ROOT = Path(__file__).resolve().parents[3]
GPU = "1xA100"          # fits the recipe's memory: only the geometry is under test


def _intake(tmp_path, recipe=RECIPE, history=lambda: INDEX):
    from cascade.shared.config import load_chain_config
    from tests.unit.test_mkt_gpus import _mkt_intake

    intake = _mkt_intake(tmp_path)
    intake.fetch_mkt_recipe = lambda ref: recipe
    intake.mkt_cfg = load_chain_config(ROOT / "chain.toml").marketcast
    intake.fetch_mkt_history = history
    return intake


def _fund(intake):
    from tests.unit.test_mkt_gpus import _headers

    return intake.lane_c_fund(_headers(GPU))


def _with(**kw):
    r = copy.deepcopy(RECIPE)
    for k, v in kw.items():
        if k in ("start", "frequencies"):
            r["data"][k] = v
        else:
            r[k] = v
    return r


def test_intake_refuses_the_recipe_whose_15m_samples_cannot_be_cut(tmp_path):
    from tests.unit.test_miner_chosen_gpu import HK

    intake = _intake(tmp_path)
    status, body = _fund(intake)
    assert status == 400 and body.get("code") == "recipe_data_range_uncuttable", (status, body)
    assert body["horizon"] == "15mx96" and body["timeframe"] == "15m"
    assert body["history_start"] == "2026-09-25" and body["risk"] >= 0.99
    assert body["fits_from_start"] and "2026-01-01" < body["fits_from_start"] <= "2026-09-26"
    msg = body["message"]
    assert "could not cut a" in msg and "context_length is not the limit" in msg
    assert f"set data.start to {body['fits_from_start']} or later" in msg
    assert intake.lane_c_queue.get(HK) is None


def test_a_shorter_context_does_not_make_it_cuttable(tmp_path):
    status, body = _fund(_intake(tmp_path, _with(context_length=512)))
    assert status == 400 and body.get("code") == "recipe_data_range_uncuttable", (status, body)


def test_the_fitting_start_or_no_15m_is_accepted(tmp_path):
    from tests.unit.test_miner_chosen_gpu import HK

    status, body = _fund(_intake(tmp_path / "x"))
    start = body.get("fits_from_start")
    assert start, body
    intake = _intake(tmp_path / "start", _with(start=start))
    status, body = _fund(intake)
    assert status == 202, body
    assert intake.lane_c_queue.get(HK) is not None
    no15 = [f for f in RECIPE["data"]["frequencies"] if f != "15m"]
    status, body = _fund(_intake(tmp_path / "no15m", _with(frequencies=no15)))
    assert status == 202, body


def test_unreadable_history_skips_the_check(tmp_path, caplog):
    import logging

    consulted = []

    def down():
        consulted.append(1)
        raise OSError("no source served index.json")

    with caplog.at_level(logging.WARNING, logger="cascade.funding.intake"):
        status, body = _fund(_intake(tmp_path, history=down))
    assert status == 202, body
    # the check DID run and was skipped for the stated reason (logged)
    assert consulted, "the intake never consulted the bar history"
    skipped = [r.getMessage() for r in caplog.records
               if "sample-geometry check" in r.getMessage()]
    assert skipped and "no source served index.json" in skipped[0], skipped
    assert "not checked" in skipped[0]


def test_the_estimate_matches_the_sampler_arithmetic():
    spec = importlib.util.find_spec("cascade.funding.mkt_geom")
    assert spec is not None, "no fund-time sample-geometry check (cascade.funding.mkt_geom)"
    from cascade.funding.mkt_geom import _cells, _recipe, history_spans
    from cascade.shared.config import load_chain_config

    mc = load_chain_config(ROOT / "chain.toml").marketcast
    r = _recipe(RECIPE)
    cells = {(c.suite, c.horizon): c for c in _cells(r, mc, history_spans(INDEX), r.start)}
    hi = cells[("high", "15mx96")]
    # ~2,900 of ~98,900 15m blocks finite: a sample misses (1 - 0.029)^16 ~ 0.62,
    # a batch of 6 ~ 0.06; ~76 high/15m batches over 3,600 steps
    assert hi.bar == "15m" and 0.04 <= hi.batch_fail <= 0.09, hi
    assert 50 <= hi.draws <= 100, hi
    # no 1m / 5m bars recorded in the range: those horizons are dropped, not fatal
    assert not any(h in ("1mx60", "5mx36") for _s, h in cells)


def test_history_fetcher_caches_and_falls_back():
    spec = importlib.util.find_spec("cascade.funding.main")
    assert spec is not None
    from types import SimpleNamespace

    from cascade.funding import main

    assert hasattr(main, "mkt_history_fetcher"), "no bar-history fetcher for the intake"
    calls, t = [], [0.0]

    def get(url):
        calls.append(url)
        if "huggingface" in url:
            raise OSError("hf down")
        return json.dumps(INDEX).encode()

    st = SimpleNamespace(s3_endpoint="https://s3.example", manifest_bucket="b")
    fetch = main.mkt_history_fetcher(st, http_get=get, clock=lambda: t[0])
    assert fetch()["files"] and len(calls) == 2            # HF failed, Hippius served
    assert calls[1] == "https://s3.example/b/marketcast/public-bars/index.json"
    fetch()
    assert len(calls) == 2                                 # cached
    t[0] = 7200.0
    fetch()
    assert len(calls) == 4                                 # refreshed after the TTL
