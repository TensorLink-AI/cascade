"""W3b6a, bench-host side: ``serve_sweep`` fetches each digest through the
W15 cache (W4 ingest on a miss with a grant), materialises it in lane A's
sidecar layout at its effective context, runs ONE paired sidecar pass and
answers ``item_rows_from_report`` rows per digest. Infrastructure faults fail
the whole sweep; the miner's bytes fail only that digest. Contract:
README.md."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from . import helpers as h


def _serve(tmp_path, req, cache, sidecar, **kw):
    return h.sweep_mod().serve_sweep(req, cache=cache, work_dir=tmp_path / "work",
                                     sidecar=sidecar, fingerprint=dict(h.FINGERPRINT), **kw)


def test_schemas_are_exported():
    m = h.sweep_mod()
    assert m.REQUEST_SCHEMA == h.REQUEST_SCHEMA and m.REPLY_SCHEMA == h.REPLY_SCHEMA
    e = m.LaneDBenchSweepError("checkpoint_unavailable", "x")
    assert isinstance(e, RuntimeError) and e.code == "checkpoint_unavailable"


def test_one_paired_pass_returns_item_rows_per_digest(tmp_path):
    from cascade.eval.benchmarks import item_rows_from_report

    a, b = h.Entry(1), h.Entry(2)
    cache = h.cache(tmp_path, a, b)
    sidecar = h.Sidecar({a.digest: h.report_for(1), b.digest: h.report_for(2)})
    got = _serve(tmp_path, h.request([a.digest, b.digest]), cache, sidecar)
    assert got["schema"] == h.REPLY_SCHEMA
    assert got["fingerprint"] == h.FINGERPRINT, "the host's hardware fingerprint is echoed"
    assert set(got["rows"]) == {a.digest, b.digest}
    assert got["rows"][a.digest] == item_rows_from_report(h.report_for(1))
    assert got["rows"][b.digest] == item_rows_from_report(h.report_for(2))
    assert len(sidecar.calls) == 1, "every digest in ONE paired sidecar pass (one session)"
    assert sidecar.calls[0]["suites"] == ("gift-eval", "boom"), "sidecar suite names"
    json.dumps(got, allow_nan=False)


@pytest.mark.parametrize("context", [4096, 16384, 131072])
def test_materialised_dir_is_the_entry_at_its_effective_context(tmp_path, context):
    from cascade.eval.checkpoint_guard import verify_checkpoint_code

    e = h.Entry(3)
    cache = h.cache(tmp_path, e)
    sidecar = h.Sidecar({e.digest: h.report_for(3)})
    _serve(tmp_path, h.request([e.digest], contexts={e.digest: context}), cache, sidecar)
    (d,) = sidecar.calls[0]["dirs"]
    assert d.name == f"{e.digest}-c{context}", "one sidecar dir per (digest, context)"
    assert verify_checkpoint_code(d) == "cascade_pyramid", "the release's own pinned code"
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    want = json.loads(e.files["config.json"])
    assert cfg == dict(want, context_length=context), "only context_length changes"
    for name in ("forecast_wrapper.py", "model.py", "weights.safetensors"):
        assert (d / name).read_bytes() == e.files[name], f"{name} is the cached entry's"
    assert sorted(p.name for p in d.iterdir() if not p.name.startswith(".")) == \
        ["config.json", "forecast_wrapper.py", "model.py", "weights.safetensors"]
    cached = cache.get(e.digest)
    assert json.loads((Path(cached) / "config.json").read_text()) == want, \
        "the cache entry itself is never modified"


def test_miss_without_grant_fails_the_sweep(tmp_path):
    e, missing = h.Entry(4), h.Entry(5)
    cache = h.cache(tmp_path, e)
    sidecar = h.Sidecar({e.digest: h.report_for(4)})
    with pytest.raises(Exception) as ei:
        _serve(tmp_path, h.request([e.digest, missing.digest]), cache, sidecar)
    m = h.sweep_mod()
    assert isinstance(ei.value, m.LaneDBenchSweepError), repr(ei.value)
    assert ei.value.code == "checkpoint_unavailable"
    assert sidecar.calls == [], "nothing is swept on an infrastructure fault"


def test_miss_with_grant_prefetches_through_w4_ingest(tmp_path, monkeypatch):
    e = h.Entry(6)
    cache = h.cache(tmp_path)
    calls = h.fake_ingest(monkeypatch, e)
    sidecar = h.Sidecar({e.digest: h.report_for(6)})
    got = _serve(tmp_path, h.request([e.digest], grants={e.digest: h.grant(e.digest)}),
                 cache, sidecar)
    assert len(calls) == 1 and e.digest in cache.entries()
    assert got["rows"][e.digest]["gift"]["status"] == "ok"


def test_fetch_failure_fails_the_sweep(tmp_path, monkeypatch):
    from cascade.eval.lane_d_ingest import IngestError

    e = h.Entry(7)
    cache = h.cache(tmp_path)
    h.fake_ingest(monkeypatch, e, exc=IngestError("fetch_failed", "w3b6a vault down"))
    sidecar = h.Sidecar()
    with pytest.raises(Exception) as ei:
        _serve(tmp_path, h.request([e.digest], grants={e.digest: h.grant(e.digest)}),
               cache, sidecar)
    assert getattr(ei.value, "code", None) == "checkpoint_unavailable", repr(ei.value)
    assert sidecar.calls == []


@pytest.mark.parametrize("ingest_code", ["shape_mismatch", "non_finite", "digest_mismatch"])
def test_w4_refusal_is_that_digests_error_only(tmp_path, monkeypatch, ingest_code):
    from cascade.eval.lane_d_ingest import IngestError

    good, bad = h.Entry(8), h.Entry(9)
    cache = h.cache(tmp_path, good)
    h.fake_ingest(monkeypatch, bad, exc=IngestError(ingest_code, "w3b6a"))
    sidecar = h.Sidecar({good.digest: h.report_for(8)})
    got = _serve(tmp_path, h.request([good.digest, bad.digest],
                                     grants={good.digest: None, bad.digest: h.grant(bad.digest)}),
                 cache, sidecar)
    assert got["rows"][bad.digest] == {s: {"status": "error", "rows": []} for s in h.SUITES}, \
        "the miner's refused bytes: error on every suite (checkpoint_fault on a candidate)"
    assert got["rows"][good.digest]["gift"]["status"] == "ok"
    assert [Path(d).name.split("-c")[0] for d in sidecar.calls[0]["dirs"]] == [good.digest]


def test_no_report_for_one_digest_is_its_error(tmp_path):
    a, b = h.Entry(10), h.Entry(11)
    cache = h.cache(tmp_path, a, b)
    sidecar = h.Sidecar({a.digest: h.report_for(10)})
    got = _serve(tmp_path, h.request([a.digest, b.digest]), cache, sidecar)
    assert got["rows"][b.digest] == {s: {"status": "error", "rows": []} for s in h.SUITES}
    assert got["rows"][a.digest]["gift"]["status"] == "ok"


def test_failed_pass_fails_the_sweep(tmp_path):
    a = h.Entry(12)
    cache = h.cache(tmp_path, a)
    with pytest.raises(Exception) as ei:
        _serve(tmp_path, h.request([a.digest]), cache, h.Sidecar(fail=True))
    assert getattr(ei.value, "code", None) == "sweep_failed", repr(ei.value)


@pytest.mark.parametrize("damage", ["schema", "digests", "suites", "contexts", "grants",
                                    "unknown_suite", "context_type"])
def test_malformed_request_is_refused(tmp_path, damage):
    a = h.Entry(13)
    cache = h.cache(tmp_path, a)
    req = h.request([a.digest])
    if damage == "schema":
        req["schema"] = "other/1"
    elif damage == "digests":
        req["digests"] = "not-a-list"
    elif damage == "suites":
        req["suites"] = []
    elif damage == "contexts":
        req["contexts"] = {}
    elif damage == "grants":
        req["grants"] = []
    elif damage == "unknown_suite":
        req["suites"] = ["nope"]
    else:
        req["contexts"] = {a.digest: "4096"}
    sidecar = h.Sidecar({a.digest: h.report_for(13)})
    with pytest.raises(ValueError):
        _serve(tmp_path, req, cache, sidecar)
    assert sidecar.calls == []


def test_reply_order_is_deterministic(tmp_path):
    a, b = h.Entry(14), h.Entry(15)
    sidecar = h.Sidecar({a.digest: h.report_for(14), b.digest: h.report_for(15)})
    x = _serve(tmp_path / "x", h.request([b.digest, a.digest]), h.cache(tmp_path / "x", a, b),
               sidecar)
    y = _serve(tmp_path / "y", h.request([b.digest, a.digest]), h.cache(tmp_path / "y", a, b),
               sidecar)
    assert json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)
    assert [Path(d).name.split("-c")[0] for d in sidecar.calls[0]["dirs"]] == [b.digest, a.digest], \
        "the sidecar sweeps the request's digest order"


def test_bench_sweep_module_is_torch_free_at_import():
    code = ("import sys; import cascade.eval.lane_d_bench_sweep; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    proc = subprocess.run([sys.executable, "-P", "-c", code], cwd=str(h.REPO),
                          capture_output=True, text=True, timeout=300,
                          env={**os.environ, "PYTHONPATH": str(h.REPO)})
    assert proc.returncode == 0, proc.stderr[-2000:] or "torch imported at module import"


def test_each_entry_is_leased_from_fetch_through_materialise(tmp_path, monkeypatch):
    """W15 shared-root lease: while ``serve_sweep`` copies an entry into its
    sidecar dir, the entry is under a live lease (no process sharing the cache
    root may evict it mid-copy); the lease is released once the copy is made,
    also when the copy fails."""
    m = h.sweep_mod()
    a, b = h.Entry(6), h.Entry(7)
    cache = h.cache(tmp_path, a, b)
    assert hasattr(cache, "leased"), "W15: PrefetchCache.leased() lists the live leases"
    real, seen = m.materialise, []

    def spy(entry, work_dir, *, digest, context):
        seen.append((digest, cache.leased()))
        return real(entry, work_dir, digest=digest, context=context)

    monkeypatch.setattr(m, "materialise", spy)
    sidecar = h.Sidecar({a.digest: h.report_for(6), b.digest: h.report_for(7)})
    _serve(tmp_path, h.request([a.digest, b.digest]), cache, sidecar)
    assert seen == [(a.digest, frozenset({a.digest})), (b.digest, frozenset({b.digest}))]
    assert cache.leased() == frozenset(), "the leases are released after the copy"

    def broken(entry, work_dir, *, digest, context):
        raise OSError("disk full")

    monkeypatch.setattr(m, "materialise", broken)
    with pytest.raises(m.LaneDBenchSweepError):
        _serve(tmp_path, h.request([a.digest]), cache, sidecar)
    assert cache.leased() == frozenset()
