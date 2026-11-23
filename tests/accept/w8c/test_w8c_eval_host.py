"""W8c, eval-host side: ``forecast_entry`` computes one digest's forecasts on
the boundary's windows with the W5d scorer over the W15 prefetch cache (a
miss with a grant fills through W4's ingest), with every failure typed so the
box can tell a miner's checkpoint fault from an infrastructure failure. The
scorer is stubbed (no torch, no GPU). Contract: README.md."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.accept.w15 import helpers as w15

from . import helpers as h

H = 8
CTX = 4096
CONTEXTS = (4096, 8192, 16384)


def _windows(n: int = 3) -> list[dict]:
    rng = np.random.default_rng(3)
    return [{"window_id": f"s{i}:0:h{H}", "history": rng.normal(10, 1, 64 + i)}
            for i in range(n)]


class Scorer:
    """A stand-in for ``lane_d_scorer.forecast_quantiles``."""

    def __init__(self, out=None, exc: BaseException | None = None):
        self.calls: list[dict] = []
        self.out, self.exc = out, exc

    def __call__(self, checkpoint_dir, histories, *, context, horizon, contexts=None,
                 device="cpu"):
        histories = [np.asarray(x) for x in histories]
        self.calls.append({"dir": Path(checkpoint_dir), "histories": histories,
                           "context": context, "horizon": horizon,
                           "contexts": None if contexts is None else tuple(contexts)})
        if self.exc is not None:
            raise self.exc
        if self.out is not None:
            return self.out
        return np.stack([np.full((horizon, 9), float(len(x))) for x in histories])


def _cache(tmp_path, *entries):
    from cascade.eval.lane_d_cache import PrefetchCache

    c = PrefetchCache(tmp_path / "cache", budget_bytes=1 << 20)
    for e in entries:
        c.put(e.digest, e.fill)
    return c


def _assert_code(exc: BaseException, code: str) -> None:
    m = h.forecasts_mod()
    assert isinstance(exc, m.LaneDForecastError), f"expected LaneDForecastError, got {exc!r}"
    assert exc.code in m.FORECAST_ERROR_CODES and exc.code == code, (exc.code, code)


def test_codes_are_exported():
    m = h.forecasts_mod()
    assert set(h.FORECAST_CODES) <= set(m.FORECAST_ERROR_CODES)
    assert tuple(sorted(m.MINER_FAULT_CODES)) == tuple(sorted(h.MINER_FAULT_CODES))
    with pytest.raises(ValueError):
        m.LaneDForecastError("nope", "x")
    assert isinstance(m.LaneDForecastError("forecast_failed", "x"), RuntimeError)


def test_cache_hit_runs_the_scorer_on_the_cached_checkpoint(tmp_path):
    e = w15.Entry(1)
    cache = _cache(tmp_path, e)
    scorer = Scorer()
    wins = _windows()
    out = h.forecasts_mod().forecast_entry(cache, e.digest, wins, context=CTX, horizon=H,
                                           contexts=CONTEXTS, forecast=scorer)
    assert len(scorer.calls) == 1
    call = scorer.calls[0]
    assert call["dir"] == cache.get(e.digest), "the verified cache entry is scored"
    assert call["context"] == CTX and call["horizon"] == H and call["contexts"] == CONTEXTS
    for got, w in zip(call["histories"], wins, strict=True):
        assert np.array_equal(got, w["history"]), "histories in the windows' order"
    assert out.shape == (len(wins), H, 9)
    assert np.array_equal(out[:, 0, 0], [64.0, 65.0, 66.0])


def test_default_scorer_is_the_w5d_forecast_quantiles(tmp_path, monkeypatch):
    """With no ``forecast=``, the W5d scorer is resolved at call time as
    ``cascade.eval.lane_d_scorer.forecast_quantiles``."""
    import cascade.eval.lane_d_scorer as scorer_mod

    e = w15.Entry(2)
    cache = _cache(tmp_path, e)
    scorer = Scorer()
    monkeypatch.setattr(scorer_mod, "forecast_quantiles", scorer)
    h.forecasts_mod().forecast_entry(cache, e.digest, _windows(), context=CTX, horizon=H,
                                     contexts=CONTEXTS)
    assert len(scorer.calls) == 1


def test_miss_without_grant_is_checkpoint_unavailable(tmp_path):
    cache = _cache(tmp_path)
    scorer = Scorer()
    with pytest.raises(Exception) as ei:
        h.forecasts_mod().forecast_entry(cache, w15.Entry(3).digest, _windows(), context=CTX,
                                         horizon=H, contexts=CONTEXTS, forecast=scorer)
    _assert_code(ei.value, "checkpoint_unavailable")
    assert scorer.calls == []


def _fake_ingest(monkeypatch, entry, exc: BaseException | None = None):
    import cascade.eval.lane_d_ingest as ingest

    calls = []

    def ingest_presigned(grant, dest_dir, *, contract=None, expect=None):
        calls.append(dict(grant))
        if exc is not None:
            raise exc
        out = Path(dest_dir) / grant["digest"]
        out.mkdir(parents=True, exist_ok=True)
        for name, data in entry.files.items():
            (out / name).write_bytes(data)
        return out

    monkeypatch.setattr(ingest, "ingest_presigned", ingest_presigned)
    return calls


def _grant(digest: str) -> dict:
    return {"hotkey": "w8cMinerHotkey", "digest": digest,
            "manifest_url": "https://vault.invalid/manifest.json", "files": {}}


def test_miss_with_grant_prefetches_through_w4_ingest(tmp_path, monkeypatch):
    e = w15.Entry(4)
    cache = _cache(tmp_path)
    calls = _fake_ingest(monkeypatch, e)
    scorer = Scorer()
    h.forecasts_mod().forecast_entry(cache, e.digest, _windows(), context=CTX, horizon=H,
                                     contexts=CONTEXTS, grant=_grant(e.digest),
                                     forecast=scorer)
    assert len(calls) == 1, "a miss fills through ingest_presigned (W15 prefetch_grant)"
    assert e.digest in cache.entries(), "the entry is admitted to the cache"
    assert scorer.calls[0]["dir"] == cache.get(e.digest)
    h.forecasts_mod().forecast_entry(cache, e.digest, _windows(), context=CTX, horizon=H,
                                     contexts=CONTEXTS, grant=_grant(e.digest),
                                     forecast=scorer)
    assert len(calls) == 1, "a verified hit makes no vault request"


@pytest.mark.parametrize("ingest_code,code", [("shape_mismatch", "checkpoint_refused"),
                                              ("non_finite", "checkpoint_refused"),
                                              ("digest_mismatch", "checkpoint_refused"),
                                              ("fetch_failed", "checkpoint_unavailable")])
def test_ingest_refusal_is_typed(tmp_path, monkeypatch, ingest_code, code):
    """A W4 refusal of the miner's bytes is the miner's fault; a failed fetch
    is the vault's."""
    from cascade.eval.lane_d_ingest import IngestError

    e = w15.Entry(5)
    cache = _cache(tmp_path)
    _fake_ingest(monkeypatch, e, exc=IngestError(ingest_code, "w8c"))
    scorer = Scorer()
    with pytest.raises(Exception) as ei:
        h.forecasts_mod().forecast_entry(cache, e.digest, _windows(), context=CTX, horizon=H,
                                         contexts=CONTEXTS, grant=_grant(e.digest),
                                         forecast=scorer)
    _assert_code(ei.value, code)
    assert isinstance(ei.value.__cause__, IngestError)
    assert scorer.calls == [] and e.digest not in cache.entries()


@pytest.mark.parametrize("case,code", [("value_error", "forecast_malformed"),
                                       ("wrong_shape", "forecast_malformed"),
                                       ("non_finite", "forecast_malformed"),
                                       ("runtime_error", "forecast_failed")])
def test_scorer_failures_are_typed(tmp_path, case, code):
    """The scorer refusing the checkpoint (ValueError: a context it cannot
    build, a non-finite output) or returning a wrong-shape / non-finite array
    is the checkpoint's fault; any other error is the host's."""
    e = w15.Entry(6)
    cache = _cache(tmp_path, e)
    n = 3
    scorer = {"value_error": Scorer(exc=ValueError("non-finite lane D quantiles")),
              "wrong_shape": Scorer(out=np.zeros((n, H, 7))),
              "non_finite": Scorer(out=np.full((n, H, 9), np.inf)),
              "runtime_error": Scorer(exc=RuntimeError("CUDA out of memory"))}[case]
    with pytest.raises(Exception) as ei:
        h.forecasts_mod().forecast_entry(cache, e.digest, _windows(n), context=CTX, horizon=H,
                                         contexts=CONTEXTS, forecast=scorer)
    _assert_code(ei.value, code)


def test_eval_host_module_is_torch_free_at_import():
    """The scorer imports torch lazily; forecast_entry's module must too (the
    box-side source imports its error class)."""
    import subprocess
    import sys

    code = ("import sys; import cascade.eval.lane_d_forecasts; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    proc = subprocess.run([sys.executable, "-P", "-c", code], cwd=str(h.w7a.REPO),
                          capture_output=True, text=True, timeout=300,
                          env={**os.environ, "PYTHONPATH": str(h.w7a.REPO)})
    assert proc.returncode == 0, proc.stderr[-2000:] or "torch imported at module import"
