"""W6p (plan rev6.18 A1, A6; F3): the pinned public sources, the per-dataset
series index, its hashes, the series reader and the allowed-list format.
Contract: README.md. Fixtures: `fixtures.py` (tiny synthetic arrow /
parquet mirrors, nothing downloaded).
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import re

import numpy as np
import pytest

from . import fixtures as fx


def _idx():
    return importlib.import_module("cascade.eval.lane_d_public_index")


def test_each_public_source_is_pinned_at_one_revision():
    idx = _idx()
    src = idx.PUBLIC_SOURCES
    assert set(src) == {fx.GIFT, fx.RMISC}
    assert src[fx.GIFT].repo == "Salesforce/GiftEvalPretrain"
    assert src[fx.RMISC].repo == "nju-zhangsq/RMISC"
    for s in src.values():
        assert re.fullmatch(r"[0-9a-f]{40}", s.revision), s
    assert src[fx.GIFT].multi_window is False
    assert src[fx.RMISC].multi_window is True


@pytest.mark.parametrize("folder,family", [
    ("cmip6_1850", "cmip6"), ("cmip6_2010", "cmip6"), ("CMIP6-2000-PartI", "cmip6"),
    ("era5_1989", "era5"), ("ERA5-2001", "era5"), ("largest_2017", "largest"),
    ("LargeST", "largest"), ("cmip6", "cmip6"), ("m4_daily", "m4_daily"),
    ("era5hourly", "era5hourly")])
def test_year_split_folders_merge_into_one_family(folder, family):
    assert _idx().family_of(folder) == family


def test_gift_index_has_one_row_per_series(tmp_path):
    idx = _idx()
    b = fx.build(tmp_path, fx.GIFT)
    rows = list(idx.iter_index(b["index"]))
    assert [r["ordinal"] for r in rows] == list(range(len(rows)))
    assert len(rows) == 6 + 6 + 5 + 2 + 12 + 4 + 2
    assert all(set(r) == set(idx.INDEX_FIELDS) for r in rows)
    by = {r["series_id"]: r for r in rows}
    r = by["multi_traffic/data-00000-of-00001#2"]
    assert (r["n_channels"], r["length"], r["freq"]) == (3, 300, "H")
    assert r["file"] == "multi_traffic/data-00000-of-00001.arrow" and r["rows"] == [[2, 3]]
    assert {r["family"] for r in rows if r["dataset"].startswith("cmip6_")} == {"cmip6"}
    d = by["daily_sales/data-00000-of-00001#3"]
    assert (d["length"], d["n_channels"], d["family"]) == (290, 1, "daily_sales")
    # index order: (dataset, file, first row)
    keys = [(r["dataset"], r["file"], r["rows"][0][0]) for r in rows]
    assert keys == sorted(keys)


def test_rmisc_index_groups_rows_by_original_filename(tmp_path):
    idx = _idx()
    b = fx.build(tmp_path, fx.RMISC)
    by = {r["series_id"]: r for r in idx.iter_index(b["index"])}
    assert set(by) == {"FewLong/s0", "FewLong/s1", *(f"WideSensors/w{i}" for i in range(3)),
                       *(f"TwoPart/p{i}" for i in range(6))}
    assert (by["FewLong/s0"]["length"], by["FewLong/s0"]["n_channels"]) == (2000, 2)
    # the timestamp column is not a channel; empty columns are not counted
    assert by["WideSensors/w0"]["n_channels"] == 40
    assert by["WideSensors/w1"]["n_channels"] == 38
    assert by["FewLong/s1"]["freq"] == "h" and by["WideSensors/w2"]["freq"] == "10min"
    files = {by[f"TwoPart/p{i}"]["file"] for i in range(6)}
    assert files == {"TwoPart/part0.parquet", "TwoPart/part1.parquet"}
    for i in range(6):
        r = by[f"TwoPart/p{i}"]
        assert sum(b - a for a, b in r["rows"]) == r["length"] == 300 + 20 * i


def test_index_hashes_are_canonical_and_reproducible(tmp_path):
    idx = _idx()
    a = fx.build(tmp_path / "a", fx.GIFT)
    b = fx.build(tmp_path / "b", fx.GIFT)
    ma, mb = a["manifest"], b["manifest"]
    assert ma["index_sha256"] == mb["index_sha256"]
    assert ma["files_sha256"] == mb["files_sha256"]
    raw = gzip.decompress((a["index"] / idx.INDEX_FILE).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == ma["index_sha256"]
    assert ma["revision"] == idx.PUBLIC_SOURCES[fx.GIFT].revision
    assert ma["repo"] == "Salesforce/GiftEvalPretrain" and ma["schema"] == idx.INDEX_SCHEMA
    files = ma["files"]
    assert [f["path"] for f in files] == sorted(f["path"] for f in files)
    want = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()
    assert ma["files_sha256"] == want
    for f in files:
        p = a["root"] / f["path"]
        assert p.stat().st_size == f["size"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"]
    line = raw.splitlines()[0]
    assert line == json.dumps(json.loads(line), sort_keys=True, separators=(",", ":")).encode()


def test_a_tampered_index_or_file_list_is_refused(tmp_path):
    idx = _idx()
    b = fx.build(tmp_path, fx.GIFT)
    p = b["index"] / idx.INDEX_FILE
    raw = gzip.decompress(p.read_bytes()).replace(b'"length":400', b'"length":401', 1)
    p.write_bytes(gzip.compress(raw))
    with pytest.raises(ValueError):
        list(idx.iter_index(b["index"]))
    c = fx.build(tmp_path / "c", fx.GIFT)
    m = json.loads((c["index"] / idx.MANIFEST_FILE).read_text())
    m["files"][0]["size"] += 1
    (c["index"] / idx.MANIFEST_FILE).write_text(json.dumps(m))
    with pytest.raises(ValueError):
        idx.load_manifest(c["index"])
    m["files"][0]["size"] -= 1
    m["revision"] = "0" * 40
    (c["index"] / idx.MANIFEST_FILE).write_text(json.dumps(m))
    with pytest.raises(ValueError):
        idx.load_manifest(c["index"])


def test_read_series_returns_the_stored_series(tmp_path):
    idx = _idx()
    g = fx.build(tmp_path, fx.GIFT)
    rows = {r["series_id"]: r for r in idx.iter_index(g["index"])}
    a = idx.read_series(g["root"], rows["daily_sales/data-00000-of-00001#5"], source=fx.GIFT)
    want = fx.series_values("daily_sales", 5, 310).astype(np.float32).astype(np.float64)
    assert a.shape == (1, 310) and np.array_equal(a, want)
    m = idx.read_series(g["root"], rows["multi_traffic/data-00000-of-00001#1"], source=fx.GIFT)
    want = fx.series_values("multi_traffic", 1, 300, 3).astype(np.float32).astype(np.float64)
    assert m.shape == (3, 300) and np.array_equal(m, want)
    r = fx.build(tmp_path / "r", fx.RMISC)
    rows = {x["series_id"]: x for x in idx.iter_index(r["index"])}
    s = idx.read_series(r["root"], rows["WideSensors/w1"], source=fx.RMISC)
    assert s.shape == (40, 400)
    assert np.isnan(s[5]).all() and np.isnan(s[6]).all()
    names = sorted(f"w{i}" for i in range(3))
    want = fx.series_values("WideSensors", names.index("w1"), 400, 40)
    keep = [c for c in range(40) if c not in (5, 6)]
    assert np.array_equal(s[keep], want[keep])


def test_read_series_reads_only_the_row_groups_of_that_series(tmp_path, monkeypatch):
    idx = _idx()
    import pyarrow.parquet as pq

    r = fx.build(tmp_path, fx.RMISC)
    rows = {x["series_id"]: x for x in idx.iter_index(r["index"])}
    row = rows["TwoPart/p3"]
    seen = []
    orig = pq.ParquetFile.read_row_groups

    def spy(self, groups, *a, **k):
        seen.append(list(groups))
        return orig(self, groups, *a, **k)

    monkeypatch.setattr(pq.ParquetFile, "read_row_groups", spy)
    out = idx.read_series(r["root"], row, source=fx.RMISC)
    n_groups = pq.ParquetFile(str(r["root"] / row["file"])).metadata.num_row_groups
    assert seen and all(len(g) < n_groups for g in seen)
    names = [f"p{i}" for i in range(6)]
    assert np.array_equal(out, fx.series_values("TwoPart", names.index("p3"), 360, 3))


def test_the_repo_allowed_list_is_a_valid_pending():
    idx = _idx()
    assert idx.ALLOWED_LIST_PATH.as_posix().endswith("data/lane_d/rmisc_allowed.toml")
    doc = idx.load_allowed_list()
    assert doc["status"] in ("pending", "final")
    assert doc["revision"] == idx.PUBLIC_SOURCES[fx.RMISC].revision
    if doc["status"] == "pending":
        assert doc["names"] == []


def _folder(name, **kw):
    return {"name": name, "reason": "kept", "licence": "CC BY 4.0", "domain": "Energy",
            "freq": "h", "data_check": "pass", **kw}


def test_the_allowed_list_format_round_trips_and_refuses_defects():
    idx = _idx()
    text = idx.render_allowed_list(status="final", folders=[_folder("B"), _folder("A")],
                                   generated_by="test")
    doc = idx.parse_allowed_list(text)
    assert doc["names"] == ["A", "B"] and doc["status"] == "final"
    bad = [
        idx.render_allowed_list(status="final", folders=[_folder("XKnownOpen")], generated_by="t"),
        idx.render_allowed_list(status="final", folders=[_folder("A", data_check="duplicate")],
                                generated_by="t"),
        idx.render_allowed_list(status="final", folders=[], generated_by="t"),
        idx.render_allowed_list(status="pending", folders=[_folder("A")], generated_by="t"),
        idx.render_allowed_list(status="final", folders=[_folder("A")], generated_by="t",
                                revision="0" * 40),
        text.replace('status = "final"', 'status = "draft"'),
        text.replace('licence = "CC BY 4.0"', 'licence = ""', 1),
        text + '\nextra = "x"\n',
    ]
    for t in bad:
        with pytest.raises(ValueError):
            idx.parse_allowed_list(t)
