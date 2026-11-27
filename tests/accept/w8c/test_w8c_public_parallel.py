"""Lane D public half: the drawn series are read on a bounded worker pool
(owner 2026-11-27). Serial reads (31-45 blobs/min measured on the 4090 host)
put one settlement's 10,000 public windows far past the 3600 s eval timeout.

``read_public_windows(..., workers=N)`` must return EXACTLY the serial
result (the windows, the drops and their order depend on the data only; the
draw is untouched), read up to N series at once and never more, and fail on
the first reader error. ``public_scores_entry`` passes the mirror's pool size.
Index-only fixture (W7a), a fake reader: no pyarrow, no torch.
"""

from __future__ import annotations

import inspect
import threading
import time

import numpy as np
import pytest

BH = "0x" + "cd" * 32


def _draws(tmp_path):
    from cascade.validator.lane_d_public import draw_public
    from tests.accept.w7a import pub

    idx = pub.index_only(tmp_path / "public")
    return idx, [draw_public(idx[s], source=s, boundary_block_hash=BH, horizon=8,
                             n_windows=40) for s in sorted(idx)]


class Reader:
    """``row -> (1, L)``: deterministic per series, slow, concurrency counted."""

    def __init__(self, delay=0.02, fail_on=None):
        self.delay, self.fail_on = delay, fail_on
        self.now = self.peak = 0
        self.lock = threading.Lock()

    def __call__(self, row):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        try:
            time.sleep(self.delay)
            if row["series_id"] == self.fail_on:
                raise OSError("mirror unavailable")
            seed = sum(row["series_id"].encode())
            x = np.random.default_rng(seed).random(int(row["length"])) + 1.0
            if row["series_id"].endswith("/3"):
                x[:] = np.nan               # its windows drop (non-finite target)
            return x.reshape(1, -1)
        finally:
            with self.lock:
                self.now -= 1


def _read(d, idx, reader, **kw):
    from cascade.validator.lane_d_public import read_public_windows

    return read_public_windows(d, index_dir=idx[d["source"]], reader=reader, block_hash=BH,
                               max_channels=4, **kw)


def _same(a, b):
    (wa, da), (wb, db) = a, b
    assert da == db
    assert [w["window_id"] for w in wa] == [w["window_id"] for w in wb]
    for x, y in zip(wa, wb, strict=True):
        assert np.array_equal(x["history"], y["history"]) and np.array_equal(x["target"],
                                                                              y["target"])


def test_parallel_reads_return_the_serial_result(tmp_path):
    from cascade.validator import lane_d_public

    assert "workers" in inspect.signature(lane_d_public.read_public_windows).parameters, (
        "2026-11-27: read_public_windows has no bounded worker pool (serial reads cannot "
        "meet the eval timeout)")
    idx, draws = _draws(tmp_path)
    for d in draws:
        serial = _read(d, idx, Reader(delay=0.0))
        r = Reader(delay=0.01)
        par = _read(d, idx, r, workers=6)
        _same(serial, par)
        assert 1 < r.peak <= 6, f"peak concurrency {r.peak}: the pool is bounded at 6"
    assert any(x for d in draws for x in _read(d, idx, Reader(delay=0.0))[1]), \
        "the fixture exercises drops"


def test_parallel_reads_are_faster_than_serial(tmp_path):
    from cascade.validator import lane_d_public

    assert "workers" in inspect.signature(lane_d_public.read_public_windows).parameters
    idx, draws = _draws(tmp_path)
    d = draws[0]
    n = len({w["series_id"] for w in d["windows"]})
    t0 = time.monotonic()
    _read(d, idx, Reader(delay=0.03), workers=8)
    took = time.monotonic() - t0
    assert took < 0.03 * n / 2, f"{n} series at 30 ms each took {took:.2f}s on 8 workers"


def test_a_reader_error_fails_the_read(tmp_path):
    from cascade.validator import lane_d_public

    assert "workers" in inspect.signature(lane_d_public.read_public_windows).parameters
    idx, draws = _draws(tmp_path)
    d = draws[0]
    sid = sorted({w["series_id"] for w in d["windows"]})[2]
    with pytest.raises(OSError, match="mirror unavailable"):
        _read(d, idx, Reader(delay=0.0, fail_on=sid), workers=4)


def test_the_eval_host_reads_with_the_mirrors_pool(tmp_path, monkeypatch):
    from cascade.eval import lane_d_forecasts as fc
    from cascade.validator import lane_d_public

    seen = {}

    def fake_read(draw, *, index_dir, block_hash, max_channels, data_root=None, reader=None,
                  workers=1):
        seen["workers"] = workers
        return [], []

    monkeypatch.setattr(lane_d_public, "read_public_windows", fake_read)

    class Mirror:
        workers = 12

        def reader(self, draw):
            return tmp_path, (lambda row: None)

    fc.public_scores_entry(object(), "d" * 64, [{"source": "rmisc", "windows": []}],
                           public_root=None, context=8, horizon=8, contexts=(8,),
                           block_hash=None, max_channels=4, mirror=Mirror())
    assert seen["workers"] == 12, "public_scores_entry reads on the mirror's pool"
    from cascade.eval.lane_d_forecasts import _build_parser

    a = _build_parser().parse_args(["public", "--cache-dir", "c", "--budget-bytes", "1",
                                    "--public-workers", "24"])
    assert a.public_workers == 24
