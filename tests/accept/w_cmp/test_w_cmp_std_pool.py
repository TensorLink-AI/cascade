"""W-CMP (DEC-CA-0075): every lane-C competition reads ONE standard pool
schema through ONE reader; the MarketCast snapshot is a special case and the
generic reader gives the MarketCast judge byte-identical arrays and windows."""

from __future__ import annotations

import json

import numpy as np
import pytest

from .helpers import T0, mkt_snapshot, small_mc


def test_w_cmp_std_pool_roundtrip_and_tamper(tmp_path):
    from cascade.shared import std_pool as S

    ts = T0 + 3600 * np.arange(5, dtype=np.int64)
    body = S.write_std_pool(
        tmp_path / "p", pool_id="energy-demo",
        series={"grid/de/load": {"timestamps": "grid/de/ts.npy",
                                 "channels": {"load": "grid/de/load.npy"}, "freq": "1h",
                                 "tags": {"domain": "energy", "source": "entsoe"},
                                 "first_seen": T0, "embargo_until": 0}},
        arrays={"grid/de/ts.npy": ts, "grid/de/load.npy": np.arange(5.0)})
    assert body["schema"] == S.SCHEMA
    pool = S.load_std_pool(tmp_path / "p", verify=False)
    assert pool.select(domain="energy") == ["grid/de/load"] and not pool.select(domain="x")
    np.testing.assert_array_equal(pool.timestamps("grid/de/load"), ts)
    assert pool.values("grid/de/load").shape == (1, 5)
    with pytest.raises(S.StdPoolError, match="signature"):
        S.load_std_pool(tmp_path / "p", signer_hotkey="5Signer")
    with pytest.raises(S.StdPoolError, match="schema"):
        S.load_std_pool(tmp_path / "p", verify=False, schema="cascade-std-pool/9")
    (tmp_path / "p/grid/de/extra.npy").write_bytes(b"x")
    with pytest.raises(S.StdPoolError, match="unlisted"):
        S.load_std_pool(tmp_path / "p", verify=False)
    (tmp_path / "p/grid/de/extra.npy").unlink()
    np.save(tmp_path / "p/grid/de/load.npy", np.arange(5.0) + 1)
    with pytest.raises(S.StdPoolError, match="sha256"):
        S.load_std_pool(tmp_path / "p", verify=False)


def test_w_cmp_std_pool_refuses_malformed_bodies():
    from cascade.shared import std_pool as S

    good = {"schema": S.SCHEMA, "pool_id": "x", "files": {}, "signer": "", "series": {}}
    S.check_body(good)
    for bad in ({**good, "schema": "cascade-std-pool/2"}, {**good, "extra": 1},
                {**good, "files": {"../x.npy": "0" * 64}},
                {**good, "series": {"a": {"channels": {}, "freq": "1h", "tags": {}}}}):
        with pytest.raises(S.StdPoolError):
            S.check_body(bad)


def test_w_cmp_marketcast_reads_the_standard_pool_byte_identically(tmp_path):
    """The MarketCast builder re-emits its snapshot as a standard pool; the
    judge's reader gives the same bars, assets and windows."""
    from cascade.pool import marketcast as P
    from cascade.shared import std_pool as S

    mkt_snapshot(tmp_path / "legacy")
    body = S.write_std_pool_from_mkt(tmp_path / "legacy", tmp_path / "std")
    assert not (tmp_path / "std" / P.SNAPSHOT_FILE).exists()
    assert {r["tags"]["domain"] for r in body["series"].values()} == {"market"}
    a = P.load_snapshot(tmp_path / "legacy", verify_signature=False)
    b = P.load_snapshot(tmp_path / "std", verify_signature=False)
    assert a.digest != b.digest                      # two artefacts, two identities
    assert json.dumps({k: v for k, v in a.meta.items()}, sort_keys=True) == json.dumps(
        {k: v for k, v in b.meta.items()}, sort_keys=True)
    for cls in ("crypto", "fx"):
        for asset in a.assets(cls):
            for f, arr in a.bars(cls, asset).items():
                assert arr.tobytes() == b.bars(cls, asset)[f].tobytes()
    mc = small_mc()
    for suite in mc.suites:
        wa = P.snapshot_windows(a, mc, suite, after=0.0)
        wb = P.snapshot_windows(b, mc, suite, after=0.0)
        assert [w.wid for w in wa] == [w.wid for w in wb] and wa
        for x, y in zip(wa, wb, strict=True):
            assert np.asarray(x.target).tobytes() == np.asarray(y.target).tobytes()


def test_w_cmp_spec_schema_and_select_reach_the_reader(tmp_path):
    from cascade.pool import marketcast as P
    from cascade.shared import std_pool as S

    mkt_snapshot(tmp_path / "legacy")
    S.write_std_pool_from_mkt(tmp_path / "legacy", tmp_path / "std")
    with pytest.raises(P.SnapshotError, match="legacy"):
        P.load_snapshot(tmp_path / "legacy", verify_signature=False, schema=S.SCHEMA)
    snap = P.load_snapshot(tmp_path / "std", verify_signature=False, schema=S.SCHEMA,
                           select={"class": "crypto"})
    assert snap.assets("crypto") == ["BTC", "ETH"] and snap.assets("fx") == []
    assert not any(k.startswith("bars/fx/") for k in snap.meta["files"])
    leg = P.load_snapshot(tmp_path / "legacy", verify_signature=False,
                          select={"class": "fx"})
    assert leg.assets("fx") == ["EUR"] and leg.assets("crypto") == []
