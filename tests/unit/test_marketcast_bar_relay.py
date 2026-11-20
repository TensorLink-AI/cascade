"""The bar-recording relay to MarketCast training pods (2026-11-18): the
training snapshot reads the 1m, native (5m / 15m) and hourly banks from the
trainer-relayed recording — the API keeps only ~17 days of 5m, so an
embargoed range has no sub-hour candles there — and the same recording
gives the same ``data_digest``. Plus the feeds' 429 backoff."""
from __future__ import annotations

import json
import shutil
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from cascade.pool import marketcast as P
from cascade.pool import marketcast_feeds as F
from cascade.pool import marketcast_relay as R
from cascade.pool.marketcast_store import MktBarStore
from cascade.shared.config.marketcast import parse_marketcast
from cascade.shared.marketcast_recipe import parse_marketcast_recipe
from cascade.trainer import marketcast_leg as L

T0 = int(datetime(2026, 9, 1, tzinfo=UTC).timestamp())
DAY = 86400
MC = parse_marketcast({
    "assets": {"crypto": ["BTC", "ETH"]}, "data_sources": {"crypto": "hyperliquid"},
    "suites": ["returns", "vol", "liquidity"],
    "frequencies": {"returns": ["5m"], "vol": ["15m"], "liquidity": ["1h"]},
    "horizons": {"returns": ["5m"], "vol": ["15m"], "liquidity": ["1h"]},
    "native_intervals": ["5m", "15m"], "embargo_days": 1,
})


def _recipe():
    return parse_marketcast_recipe({
        "target": "mkt:returns",
        "data": {"assets": {"crypto": ["BTC", "ETH"]}, "class_weights": {"crypto": 1.0},
                 "frequencies": ["5m", "15m", "1h"], "start": "2026-09-01",
                 "end": "2026-09-03"},
        "context_length": 64, "horizon_mix": {"5m": 1.0},
        "covariates": [], "augmentations": [],
        "objective": {"suite_weights": {"returns": 1.0}},
        "optimizer": {"lr": 1e-3, "schedule": "cosine", "warmup_fraction": 0.1, "steps": 2,
                      "batch": 2, "weight_decay": 0.0, "clip": 1.0},
        "trainable": {"scope": "lora", "lora_rank": 2}, "synthetic_fraction": 0.0}, MC)


def _record(root: Path, days: int = 3) -> MktBarStore:
    store = MktBarStore(root)
    rng = np.random.default_rng(0)
    for a in ("BTC", "ETH"):
        for iv, m in (("1m", 1), ("5m", 5), ("15m", 15), ("1h", 60)):
            n = days * DAY // (60 * m)
            c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
            rows = [((T0 + 60 * m * i) * 1000, c[i], c[i] * 1.001, c[i] * 0.999, c[i], 1.0 + i % 3)
                    for i in range(n)]
            store.write_bars("crypto", a, rows, now=T0 + days * DAY, interval=iv)
        store.write_funding("crypto", a, [((T0 + 3600 * h) * 1000, 1e-5, 0.0) for h in range(72)])
    return store


def _no_api(*a, **k):
    raise AssertionError("the API must not be called when the recording is relayed")


def test_training_snapshot_reads_every_bank_from_the_recording(tmp_path):
    store = _record(tmp_path / "rec")
    rc = _recipe()
    meta = L.build_training_snapshot(tmp_path / "a", MC, rc, post=_no_api, get=_no_api,
                                     store=store)
    assert set(meta["native"]) == {"5m", "15m"} and meta["coarse"]["steps"] == 72
    snap = P.load_snapshot(tmp_path / "a", verify_signature=False)
    assert np.isfinite(snap.bars("crypto", "BTC", bank="n5m")["close"]).all()
    assert snap.bank("crypto", "BTC", "returns", 5)[2] == 5      # cut from the 5m bank
    # the same recording → the same bytes → the same data digest
    L.build_training_snapshot(tmp_path / "b", MC, rc, post=_no_api, get=_no_api, store=store)
    assert P.load_snapshot(tmp_path / "b", verify_signature=False).digest == snap.digest
    # and the sampler cuts the 5m horizon from real data
    s = L.MktSampler(rc, MC, snap, seed=1)
    assert s.batch(2).contexts.shape[0] == 2


def test_relay_rels_cover_the_span_and_intervals():
    rels = R.relay_rels([("crypto", ["BTC"])], ("1m", "5m", "1h"), T0, T0 + 2 * DAY)
    assert "crypto/BTC/funding.jsonl" in rels
    assert "crypto/BTC/2026-09-02/close.npy" in rels
    assert "crypto/BTC/5m/2026-09-01/volume.npy" in rels
    assert "crypto/BTC/1h/2026-09-03/close.npy" not in rels               # t1 exclusive
    assert len(rels) == 2 + 3 * 2 * len(F.FIELDS)                          # + funding, ctx
    rc = _recipe()
    assert L.training_intervals(MC) == ("1m", "5m", "15m", "1h")
    assert L.training_span(rc) == (T0, T0 + 3 * DAY)
    assert L.training_span(rc, not_after=T0 + DAY + 1800)[1] == T0 + DAY   # clipped to the hour


class _S3:
    """A local-directory stand-in for the trainer's S3Store."""

    def __init__(self, root: Path):
        self.root, self.cfg = root, type("C", (), {"bucket": "b"})()

    def client(self):
        root = self.root

        class C:
            def get_paginator(self, _op):
                class Pg:
                    def paginate(self, Bucket, Prefix):
                        keys = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]
                        yield {"Contents": [{"Key": k} for k in sorted(keys)
                                            if k.startswith(Prefix)]}
                return Pg()

            def generate_presigned_url(self, _m, Params, ExpiresIn):
                return f"file://{root / Params['Key']}?exp={ExpiresIn}"
        return C()


def test_relay_round_trip_to_the_pod(tmp_path):
    _record(tmp_path / "bucket" / "pre" / "bars")
    s3 = _S3(tmp_path / "bucket")
    plan = L.training_relay_plan(s3, "pre/bars/", MC, _recipe())
    links = plan["links"]
    assert plan["version"] == R.RELAY_VERSION and links
    assert all(not r.startswith("pre/") for r in links)                   # relative to the prefix
    assert "crypto/BTC/2026-09-04/close.npy" not in links                 # outside the recipe
    got = []

    def get_file(url, p):
        got.append(url)
        shutil.copyfile(url[len("file://"):].split("?")[0], p)

    store = R.fetch_relay(json.loads(json.dumps(plan)), tmp_path / "pod", get_file=get_file)
    assert len(got) == len(links)
    a = L.build_training_snapshot(tmp_path / "s1", MC, _recipe(), post=_no_api, get=_no_api,
                                  store=store)
    b = L.build_training_snapshot(tmp_path / "s2", MC, _recipe(), post=_no_api, get=_no_api,
                                  store=MktBarStore(tmp_path / "bucket" / "pre" / "bars"))
    assert a["files"] == b["files"]                                     # the pod sees the recording


@pytest.mark.parametrize("bad", ["../etc/passwd", "crypto/BTC/2026-09-01/close.py",
                                 "/abs/crypto/BTC/funding.jsonl", "crypto/BTC/../../x.npy"])
def test_fetch_relay_refuses_paths_outside_the_layout(tmp_path, bad):
    with pytest.raises(ValueError, match="layout"):
        R.fetch_relay({"version": 1, "links": {bad: "http://x"}}, tmp_path,
                      get_file=lambda u, p: None)


def test_fetch_relay_refuses_an_unknown_plan(tmp_path):
    with pytest.raises(ValueError, match="relay plan"):
        R.fetch_relay({"version": 99, "links": {}}, tmp_path, get_file=lambda u, p: None)
    assert R.read_relay_file(None) is None and R.read_relay_file(tmp_path) is None


def test_relay_corpus_is_off_by_default_and_signs_when_armed(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace

    from cascade.trainer.marketcast_runner import mkt_relay_corpus

    wrap = json.dumps({"marketcast": json.loads(json.dumps(_raw_recipe())),
                       "has_generator": False})
    assert mkt_relay_corpus(SimpleNamespace(marketcast=MC), wrap) == {}
    _record(tmp_path / "bucket" / "pre" / "bars")
    cfg = SimpleNamespace(marketcast=replace(MC, bar_relay="b/pre/bars/"))
    out = mkt_relay_corpus(cfg, wrap, store=_S3(tmp_path / "bucket"))
    plan = json.loads(out["corpus_files"][R.RELAY_FILE])
    assert plan["links"] and all(u.startswith("file://") for u in plan["links"].values())


def _raw_recipe():
    return {"target": "mkt:returns",
            "data": {"assets": {"crypto": ["BTC", "ETH"]}, "class_weights": {"crypto": 1.0},
                     "frequencies": ["5m", "15m", "1h"], "start": "2026-09-01",
                     "end": "2026-09-03"},
            "context_length": 64, "horizon_mix": {"5m": 1.0}, "covariates": [],
            "augmentations": [], "objective": {"suite_weights": {"returns": 1.0}},
            "optimizer": {"lr": 1e-3, "schedule": "cosine", "warmup_fraction": 0.1, "steps": 2,
                          "batch": 2, "weight_decay": 0.0, "clip": 1.0},
            "trainable": {"scope": "lora", "lora_rank": 2}, "synthetic_fraction": 0.0}


def test_config_bar_relay_and_micro_batch_bounds():
    assert parse_marketcast({}).bar_relay == "" and parse_marketcast({}).micro_batch_tokens > 0
    assert parse_marketcast({"bar_relay": "bkt/marketcast/bars/"}).bar_relay
    for bad in ("bkt", "bkt/prefix"):
        with pytest.raises(ValueError, match="bar_relay"):
            parse_marketcast({"bar_relay": bad})
    with pytest.raises(ValueError, match="micro_batch_tokens"):
        parse_marketcast({"micro_batch_tokens": 0})


def _http(code, retry_after=None):
    hdrs = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("u", code, "x", hdrs, None)


def test_feeds_back_off_on_429_and_5xx_only():
    calls, slept = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _http(429, "2") if len(calls) == 1 else _http(503)
        return "ok"

    assert F.with_backoff(flaky, sleep=slept.append) == "ok"
    assert slept == [2.0, 2.0]                         # Retry-After, then 2^1
    with pytest.raises(urllib.error.HTTPError):
        F.with_backoff(lambda: (_ for _ in ()).throw(_http(404)), sleep=slept.append)
    n = []
    with pytest.raises(urllib.error.HTTPError):
        F.with_backoff(lambda: (n.append(1), (_ for _ in ()).throw(_http(429)))[1],
                       attempts=3, sleep=lambda s: None)
    assert len(n) == 3


def test_bar_relay_reads_its_own_creds_when_set(monkeypatch):
    """The recorder's bucket need not share the trainer's manifest creds:
    ``MKT_BARS_S3_*`` when set, else ``HIPPIUS_S3_*``."""
    from types import SimpleNamespace

    from cascade.trainer.marketcast_runner import mkt_bars_store

    st = SimpleNamespace(s3_endpoint="https://s3.example", s3_region="r")
    monkeypatch.delenv("MKT_BARS_S3_ACCESS_KEY", raising=False)
    c = mkt_bars_store(st, "bars-bucket").cfg
    assert (c.access_key_env, c.secret_key_env) == ("HIPPIUS_S3_ACCESS_KEY", "HIPPIUS_S3_SECRET_KEY")
    monkeypatch.setenv("MKT_BARS_S3_ACCESS_KEY", "x")
    c = mkt_bars_store(st, "bars-bucket").cfg
    assert (c.access_key_env, c.secret_key_env) == ("MKT_BARS_S3_ACCESS_KEY", "MKT_BARS_S3_SECRET_KEY")
    assert (c.endpoint, c.region, c.bucket) == ("https://s3.example", "r", "bars-bucket")
