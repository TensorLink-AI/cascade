"""W-CMP (owner 2026-11-27 "set up a new HF dataset"): the lane C MarketCast
bars OLDER than the training embargo are published as a public dataset (HF
tensorlink-dev/cascade-marketcast-bars + the public Hippius manifest bucket),
and `cascade lane-c data` downloads them. Torch-free: numpy, pyarrow, argparse.

Pins: the embargo cut (strictly before today - [marketcast] embargo_days, read
from chain.toml; no staged file, index entry or upload reaches it), the
Parquet layout and schema, checksum idempotence, and the CLI arguments."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from argparse import Namespace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "publish_marketcast_public.py"
WORKFLOW = ROOT / ".github" / "workflows" / "publish-mkt-public.yml"
TODAY = date(2026, 11, 27)


def _pub():
    spec = importlib.util.find_spec("cascade.shared.marketcast_public")
    assert spec is not None, "cascade.shared.marketcast_public (public bar layout + cut) is missing"
    return importlib.import_module("cascade.shared.marketcast_public")


def _script():
    assert SCRIPT.is_file(), "scripts/publish_marketcast_public.py is missing"
    spec = importlib.util.spec_from_file_location("publish_marketcast_public", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg():
    from cascade.shared.config import load_chain_config

    return load_chain_config(ROOT / "chain.toml")


def _day_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())


def _store(root: Path, days: list[date], *, cls="crypto", sym="BTC", tf="1m"):
    """A MktBarStore-layout bank with deterministic bars on ``days``."""
    from cascade.pool.marketcast_store import MktBarStore

    step = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}[tf]
    s = MktBarStore(root)
    for d in days:
        t0 = _day_ts(d)
        candles = [((t0 + k * step) * 1000, 100.0 + k, 101.0 + k, 99.0 + k, 100.5 + k, 2.0 + k)
                   for k in range(0, 86400 // step, 7)]       # sparse: gaps are absent rows
        s.write_bars(cls, sym, candles, now=t0 + 2 * 86400, interval=tf)
    s.write_funding(cls, sym, [((_day_ts(d) + 3600) * 1000, 1e-5, float("nan")) for d in days])
    return s


# ── the embargo cut ──────────────────────────────────────────────────────────


def test_w_cmp_public_cut_is_strictly_before_today_minus_embargo_days_from_chain_toml():
    P = _pub()
    emb = _cfg().marketcast.embargo_days
    cut = P.embargo_cutoff(TODAY, emb)
    assert (TODAY - cut).days == emb
    assert P.is_public_day(date.fromordinal(cut.toordinal() - 1), cut)
    assert not P.is_public_day(cut, cut), "the cut day itself is embargoed"
    with pytest.raises(P.EmbargoViolation):
        P.assert_before_cutoff([P.cutoff_ms(cut)], cut)
    P.assert_before_cutoff([P.cutoff_ms(cut) - 1], cut)


def test_w_cmp_publisher_never_stages_an_embargoed_day(tmp_path):
    P, S = _pub(), _script()
    cfg = _cfg()
    cut = P.embargo_cutoff(TODAY, cfg.marketcast.embargo_days)
    days = [date.fromordinal(cut.toordinal() + k) for k in range(-3, 3)]   # 3 public, 3 embargoed
    _store(tmp_path / "bars", days)
    entries = S.build(tmp_path / "bars", tmp_path / "stage", [("crypto", "BTC")], cut)
    assert entries, "nothing staged"
    S.verify_staged(tmp_path / "stage", entries, cut)
    for rel, e in entries.items():
        ts = P.read_parquet_timestamps_ms(tmp_path / "stage" / rel)
        assert int(ts.max()) < P.cutoff_ms(cut), f"{rel} holds a bar inside the embargo"
        assert date.fromisoformat(e["last"][:10]) < cut
    bars = [e for e in entries.values() if e["timeframe"] == "1m"]
    assert {e["last"][:10] for e in bars} == {date.fromordinal(cut.toordinal() - 1).isoformat()}
    fund = [e for e in entries.values() if e["timeframe"] == "funding"]
    assert sum(e["rows"] for e in fund) == 3, "only the 3 public funding samples"


def test_w_cmp_a_tampered_staged_file_aborts_before_upload(tmp_path):
    P, S = _pub(), _script()
    cut = P.embargo_cutoff(TODAY, 30)
    _store(tmp_path / "bars", [date.fromordinal(cut.toordinal() - 1)])
    entries = S.build(tmp_path / "bars", tmp_path / "stage", [("crypto", "BTC")], cut)
    rel = next(p for p, e in entries.items() if e["timeframe"] == "1m")
    late = {"timestamp": np.array([P.cutoff_ms(cut) + 60_000], dtype=np.int64),
            **{f: np.array([1.0]) for f in P.BAR_FIELDS}}
    with pytest.raises(P.EmbargoViolation):
        P.write_parquet(tmp_path / "stage" / rel, late, cut)       # the writer refuses
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"timestamp": pa.array(late["timestamp"]).cast(pa.timestamp("ms", tz="UTC")),
                             **{f: late[f] for f in P.BAR_FIELDS}}), tmp_path / "stage" / rel)
    with pytest.raises(P.EmbargoViolation):
        S.verify_staged(tmp_path / "stage", entries, cut)
    bad = {rel: {**entries[rel], "last": cut.isoformat() + "T00:01:00+00:00"}}
    with pytest.raises(P.EmbargoViolation):
        S.merged_index(bad, P.empty_index(), cutoff=cut, embargo_days=30,
                       now=datetime(2026, 11, 27, tzinfo=UTC))


def test_w_cmp_publisher_reads_embargo_days_from_the_chain_config_not_a_literal():
    src = SCRIPT.read_text() if SCRIPT.is_file() else ""
    assert "mc.embargo_days" in src, "the publisher must read [marketcast] embargo_days"
    assert WORKFLOW.is_file(), ".github/workflows/publish-mkt-public.yml is missing"
    wf = WORKFLOW.read_text()
    assert "schedule:" in wf and "publish_marketcast_public.py" in wf
    assert "secrets.HF_TOKEN" in wf


# ── layout + schema ──────────────────────────────────────────────────────────


def test_w_cmp_parquet_layout_and_schema(tmp_path):
    P, S = _pub(), _script()
    import pyarrow as pa
    import pyarrow.parquet as pq

    cut = P.embargo_cutoff(TODAY, 30)
    days = [date(2026, 9, 29), date(2026, 9, 30), date(2026, 10, 1)]        # spans two months
    _store(tmp_path / "bars", days)
    _store(tmp_path / "bars", days, tf="1h")
    entries = S.build(tmp_path / "bars", tmp_path / "stage", [("crypto", "BTC")], cut)
    assert set(entries) == {
        "data/crypto/BTC/1m/2026-09.parquet", "data/crypto/BTC/1m/2026-10.parquet",
        "data/crypto/BTC/1h/2026-09.parquet", "data/crypto/BTC/1h/2026-10.parquet",
        "data/crypto/BTC/funding/2026-09.parquet", "data/crypto/BTC/funding/2026-10.parquet"}
    f = tmp_path / "stage" / "data/crypto/BTC/1m/2026-09.parquet"
    meta = pq.ParquetFile(f).metadata
    assert meta.row_group(0).column(0).compression == "ZSTD"
    t = pq.read_table(f)
    assert t.schema.names == ["timestamp", *P.BAR_FIELDS]
    assert t.schema.field("timestamp").type == pa.timestamp("ms", tz="UTC")
    assert all(t.schema.field(c).type == pa.float64() for c in P.BAR_FIELDS)
    ts = t.column("timestamp").cast("int64").to_numpy()
    assert np.all(np.diff(ts) > 0) and np.all(ts % 60_000 == 0)
    assert not np.isnan(t.column("close").to_numpy()).any(), "missing bars are absent rows, not NaN"
    assert t.num_rows == 2 * len(range(0, 1440, 7))
    assert t.column("close").to_numpy()[0] == 100.5
    ft = pq.read_table(tmp_path / "stage" / "data/crypto/BTC/funding/2026-09.parquet")
    assert ft.schema.names == ["timestamp", "rate", "premium"]
    assert P.parse_file_path("data/crypto/BTC/1h/2026-10.parquet") == ("crypto", "BTC", "1h", "2026-10")


def test_w_cmp_public_universe_is_the_hyperliquid_recorded_classes():
    S = _script()
    uni = S.public_universe(_cfg().marketcast)
    classes = {c for c, _ in uni}
    assert "crypto" in classes and "equity_etf" in classes
    assert "alpha" not in classes, "alpha is not in the recorder's bank"


def test_w_cmp_dataset_card_states_the_lag_schema_sources_and_load_example(tmp_path):
    P, S = _pub(), _script()
    cfg = _cfg()
    cut = P.embargo_cutoff(TODAY, cfg.marketcast.embargo_days)
    _store(tmp_path / "bars", [date.fromordinal(cut.toordinal() - 1)])
    entries = S.build(tmp_path / "bars", tmp_path / "stage", [("crypto", "BTC")], cut)
    idx = S.merged_index(entries, P.empty_index(), cutoff=cut,
                         embargo_days=cfg.marketcast.embargo_days, now=datetime.now(UTC))
    card = S.render_card(idx, cfg.marketcast, hf_repo=P.HF_REPO,
                         hippius_url="https://s3.hippius.com/cascade-manifests/marketcast/public-bars")
    for needle in (f"embargo_days = {cfg.marketcast.embargo_days}", cut.isoformat(), "timestamp[ms, UTC]",
                   "api.hyperliquid.xyz", "load_dataset", "read_parquet", "cascade lane-c data",
                   "license:", "MarketCast", "public-bars/index.json"):
        assert needle in card, needle


# ── idempotence ──────────────────────────────────────────────────────────────


def test_w_cmp_reupload_skips_files_with_the_same_checksum(tmp_path):
    P, S = _pub(), _script()
    cut = P.embargo_cutoff(TODAY, 30)
    days = [date(2026, 10, 20), date(2026, 10, 21)]
    _store(tmp_path / "bars", days)
    e1 = S.build(tmp_path / "bars", tmp_path / "s1", [("crypto", "BTC")], cut)
    remote = S.merged_index(e1, P.empty_index(), cutoff=cut, embargo_days=30,
                            now=datetime.now(UTC))
    assert S.plan_uploads(e1, P.empty_index()) == sorted(e1), "first run uploads everything"
    e2 = S.build(tmp_path / "bars", tmp_path / "s2", [("crypto", "BTC")], cut)
    assert S.plan_uploads(e2, remote) == [], "an unchanged rebuild uploads nothing"
    # the next day leaves the embargo: only the month file it lands in changes
    _store(tmp_path / "bars", [date(2026, 10, 22)])
    cut2 = P.embargo_cutoff(date(2026, 11, 28), 30)
    e3 = S.build(tmp_path / "bars", tmp_path / "s3", [("crypto", "BTC")], cut2)
    assert S.plan_uploads(e3, remote) == ["data/crypto/BTC/1m/2026-10.parquet",
                                          "data/crypto/BTC/funding/2026-10.parquet"]


def test_w_cmp_upload_sends_only_planned_files_then_index(tmp_path):
    P, S = _pub(), _script()
    cut = P.embargo_cutoff(TODAY, 30)
    _store(tmp_path / "bars", [date(2026, 10, 20)])
    entries = S.build(tmp_path / "bars", tmp_path / "stage", [("crypto", "BTC")], cut)

    class Store:
        def __init__(self):
            self.objs = {}

        def put_bytes(self, key, data, *, content_type="", acl=None):
            self.objs[key] = data

        def get_bytes(self, key):
            from cascade.shared.hippius import ObjectNotFound

            if key not in self.objs:
                raise ObjectNotFound(key)
            return self.objs[key]

    st = Store()
    remote = S.hippius_remote_index(st, P.HIPPIUS_PREFIX)
    todo = S.plan_uploads(entries, remote)
    idx = S.merged_index(entries, remote, cutoff=cut, embargo_days=30, now=datetime.now(UTC))
    S.upload_hippius(st, P.HIPPIUS_PREFIX, tmp_path / "stage", todo, idx, "card")
    assert set(st.objs) == {f"{P.HIPPIUS_PREFIX}/{p}" for p in entries} | {
        f"{P.HIPPIUS_PREFIX}/index.json", f"{P.HIPPIUS_PREFIX}/README.md"}
    again = S.hippius_remote_index(st, P.HIPPIUS_PREFIX)
    assert S.plan_uploads(entries, again) == []
    for p, e in again["files"].items():
        assert hashlib.sha256(st.objs[f"{P.HIPPIUS_PREFIX}/{p}"]).hexdigest() == e["sha256"]


# ── the miner command ────────────────────────────────────────────────────────


def _parse(argv):
    from cascade.miner.cli import build_parser

    return build_parser().parse_args(argv)


def test_w_cmp_cli_lane_c_data_arguments():
    _pub()
    a = _parse(["lane-c", "data", "--symbols", "BTC,ETH", "--timeframes", "1m,1h,funding",
                "--start", "2026-09-01", "--end", "2026-09-30", "--out", "/tmp/x"])
    assert a.command == "lane-c data"
    assert a.symbols == ["BTC", "ETH"] and a.timeframes == ["1m", "1h", "funding"]
    assert a.start == date(2026, 9, 1) and a.end == date(2026, 9, 30)
    assert a.source == "hf" and a.out == Path("/tmp/x")
    assert _parse(["lane-c", "data", "--symbols", "SP500", "--out", "d", "--source", "hippius"]).source == "hippius"
    for bad in (["--start", "2026/09/01"], ["--source", "s3"]):
        with pytest.raises(SystemExit):
            _parse(["lane-c", "data", "--symbols", "BTC", "--out", "d", *bad])
    with pytest.raises(SystemExit):
        _parse(["lane-c", "data", "--out", "d"])                      # --symbols required


def test_w_cmp_cli_download_selects_verifies_and_falls_back(tmp_path):
    _pub()
    from cascade.miner import marketcast_data as M

    blobs = {"data/crypto/BTC/1m/2026-09.parquet": b"a", "data/crypto/BTC/1m/2026-10.parquet": b"b",
             "data/crypto/ETH/1m/2026-09.parquet": b"c", "data/crypto/BTC/1h/2026-09.parquet": b"d"}
    index = {"schema": "cascade-marketcast-public-bars/1", "public_before": "2026-10-28", "embargo_days": 30,
             "files": {p: {"sha256": hashlib.sha256(b).hexdigest()} for p, b in blobs.items()}}

    def broken(*_a, **_k):
        raise OSError("hf down")

    served = {"index.json": json.dumps(index).encode(), **blobs}
    hf = M.PublicBars("hf", hf_repo="x/y", hippius_base="https://h", hf_download=broken)
    hip = M.PublicBars("hippius", hf_repo="x/y", hippius_base="https://h",
                       http_get=lambda url: served[url.removeprefix("https://h/")])
    args = Namespace(symbols=["btc"], timeframes=["1m"], start=date(2026, 9, 15), end=date(2026, 9, 20),
                     out=tmp_path / "out")
    res = M.fetch_public_bars(args, sources=[hf, hip])
    assert res["files"] == ["data/crypto/BTC/1m/2026-09.parquet"]
    assert (tmp_path / "out/data/crypto/BTC/1m/2026-09.parquet").read_bytes() == b"a"
    assert M.fetch_public_bars(args, sources=[hf, hip])["already_present"] == res["files"]
    served["data/crypto/BTC/1m/2026-09.parquet"] = b"tampered"
    (tmp_path / "out/data/crypto/BTC/1m/2026-09.parquet").unlink()
    with pytest.raises(RuntimeError, match="sha256"):
        M.fetch_public_bars(args, sources=[hf, hip])
