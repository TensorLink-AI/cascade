"""W6p (plan rev6.18 A5, A6; F3): the owner tools. `scripts/laned_public_mirror.py`
(size check, mirror, index builder) and `scripts/laned_rmisc_allowlist.py`
(candidates from metadata, the data compare the owner runs, finalize,
validate). Everything here runs offline on the fixtures; the HF API and the
buckets are never reached. Contract: README.md "Owner tools".
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from . import fixtures as fx

REPO = Path(__file__).resolve().parents[3]
MIRROR = REPO / "scripts" / "laned_public_mirror.py"
ALLOW = REPO / "scripts" / "laned_rmisc_allowlist.py"


def _load(path: Path, name: str):
    assert path.is_file(), f"F3 (rev6.18 A5/A6): {path.relative_to(REPO)} is missing"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _listing(tmp_path: Path, root: Path) -> Path:
    files = [{"path": p.relative_to(root).as_posix(), "size": p.stat().st_size}
             for p in sorted(root.rglob("*")) if p.is_file()]
    out = tmp_path / "listing.json"
    out.write_text(json.dumps(files))
    return out


def test_size_check_prints_the_bytes_from_the_metadata(tmp_path, capsys):
    m = _load(MIRROR, "laned_public_mirror_accept")
    root, _ = fx.gift_mirror(tmp_path)
    listing = _listing(tmp_path, root)
    total = sum(f["size"] for f in json.loads(listing.read_text()))
    assert m.main(["size", "--source", fx.GIFT, "--listing", str(listing)]) == 0
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if x.startswith("SIZE "))
    assert f"bytes={total} " in line and "datasets=7" in line
    assert "Salesforce/GiftEvalPretrain@6830b624de7ed2b3d3e5b85bb6959d81dcc5d874" in line


def test_rmisc_size_and_mirror_need_a_final_allowed_list(tmp_path):
    m = _load(MIRROR, "laned_public_mirror_accept2")
    root, _ = fx.rmisc_mirror(tmp_path)
    listing = _listing(tmp_path, root)
    with pytest.raises(SystemExit, match="pending"):
        m.main(["size", "--source", fx.RMISC, "--listing", str(listing)])
    idx = importlib.import_module("cascade.eval.lane_d_public_index")
    allowed = tmp_path / "allowed.toml"
    allowed.write_text(idx.render_allowed_list(status="final", generated_by="t", folders=[
        {"name": "FewLong", "reason": "k", "licence": "MIT", "domain": "d", "freq": "h",
         "data_check": "pass"}]))
    ds, files = m.selected_files(fx.RMISC, json.loads(listing.read_text()), str(allowed))
    assert ds == ["FewLong"] and files and all(f["path"].startswith("FewLong/") for f in files)


def test_mirror_refuses_to_start_without_disk_headroom(tmp_path):
    m = _load(MIRROR, "laned_public_mirror_accept3")
    with pytest.raises(SystemExit, match="disk"):
        m.check_disk(tmp_path, 1 << 62)
    m.check_disk(tmp_path, 1)


def test_mirror_reads_store_keys_from_the_environment_by_name(monkeypatch):
    pytest.importorskip("boto3")
    m = _load(MIRROR, "laned_public_mirror_accept4")
    for v in ("HIPPIUS_S3_ACCESS_KEY", "HIPPIUS_S3_SECRET_KEY", "BACKUP_S3_ACCESS_KEY",
              "BACKUP_S3_SECRET_KEY"):
        monkeypatch.delenv(v, raising=False)
    from cascade.shared.hippius import StorageError

    with pytest.raises(StorageError, match="HIPPIUS_S3_ACCESS_KEY"):
        m._store(m.DEFAULT_HIPPIUS_ENDPOINT, "bucket", "HIPPIUS_S3")
    with pytest.raises(StorageError, match="BACKUP_S3_ACCESS_KEY"):
        m._store(m.DEFAULT_R2_ENDPOINT, "bucket", "BACKUP_S3")
    assert m.DEFAULT_PREFIX == "lane_d_public"


def test_index_command_builds_the_index_from_a_local_copy(tmp_path, capsys):
    m = _load(MIRROR, "laned_public_mirror_accept5")
    root, _ = fx.gift_mirror(tmp_path)
    out = tmp_path / "idx"
    assert m.main(["index", "--source", fx.GIFT, "--data-root", str(root),
                   "--index-dir", str(out)]) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("INDEX "))
    man = json.loads((out / "manifest.json").read_text())
    assert f"index_sha256={man['index_sha256']}" in line and man["n_series"] == 37


def _listings(rmisc_meta: dict) -> dict:
    a = _load(ALLOW, "laned_rmisc_allowlist_accept0")
    sources = {k: {"repo": r, "revision": v, "names": []}
               for k, (r, v) in a.EXCLUSION_SOURCES.items()}
    sources["gift_eval"]["names"] = ["m4_daily", "electricity"]
    sources["gift_eval_pretrain"]["names"] = ["bdg-2_bear", "cmip6_1850"]
    sources["time"]["names"] = ["Crypto"]
    return {"schema": a.LISTINGS_SCHEMA, "sources": sources,
            "rmisc": {"repo": a.RMISC_REPO, "revision": a.RMISC_REVISION,
                      "folders": sorted(rmisc_meta), "meta": rmisc_meta}}


def _meta(**kw):
    return {"name": "x", "source": "https://example.org", "desp": "Hourly load readings.",
            "domain": "Energy", "license": "CC BY 4.0", "freq": ["h"], **kw}


def test_candidates_apply_the_a5_rules(tmp_path):
    a = _load(ALLOW, "laned_rmisc_allowlist_accept1")
    assert set(a.EXCLUSION_SOURCES) == {"gift_eval", "gift_eval_pretrain", "fev_datasets",
                                        "chronos_datasets", "time", "boom"}
    assert a.EXCLUSION_SOURCES["time"] == ("Real-TSF/TIME",
                                           "83e3d0b3be28d11c7182bffcc1892d19b36c4da1")
    assert a.EXCLUSION_SOURCES["boom"] == ("Datadog/BOOM",
                                           "69325b544c45ff0d6c43c7a99c49a6601a01725b")
    meta = {
        "M4": _meta(), "BDG2-Bear": _meta(), "CMIP6-2000-PartI": _meta(),
        "Cryptocurrency": _meta(), "SP500KnownOpen": _meta(), "FavoritaKnownOil": _meta(),
        "BCI_Competetion_IV_1": _meta(desp="EEG motor imagery recordings."),
        "HARTH": _meta(desp="Human activity recognition from accelerometers."),
        "ACSF1": _meta(source="https://www.timeseriesclassification.com/x"),
        "NAB": _meta(desp="Numenta anomaly benchmark."),
        "Locked": _meta(license="Proprietary: no redistribution"),
        "FreshLoad": _meta(), "MysteryLicence": _meta(license="Unknown"),
        "NoMeta": None,
    }
    doc = a.candidates(_listings(meta))
    assert doc["schema"] == a.CANDIDATES_SCHEMA and doc["n_folders"] == len(meta)
    reasons = {e["name"]: e["reason"] for e in doc["excluded"]}
    assert all(r in a.EXCLUDE_REASONS for r in reasons.values())
    for n in ("M4", "BDG2-Bear", "CMIP6-2000-PartI", "Cryptocurrency"):
        assert reasons[n] == "name_match", n
    assert reasons["SP500KnownOpen"] == reasons["FavoritaKnownOil"] == "duplicate_folder"
    for n in ("BCI_Competetion_IV_1", "HARTH", "ACSF1", "NAB"):
        assert reasons[n] == "non_forecasting", n
    assert reasons["Locked"] == "licence_forbids" and reasons["NoMeta"] == "no_meta"
    kept = {c["name"]: c for c in doc["candidates"]}
    assert set(kept) == {"FreshLoad", "MysteryLicence"}
    assert kept["FreshLoad"]["licence_class"] == "ok"
    assert kept["MysteryLicence"]["licence_class"] == "unknown"
    assert set(kept["FreshLoad"]) == {"name", "domain", "freq", "licence", "licence_class",
                                      "source_url", "near"}


def test_candidates_refuse_unpinned_listings():
    a = _load(ALLOW, "laned_rmisc_allowlist_accept2")
    doc = _listings({"A": _meta()})
    doc["sources"]["boom"]["revision"] = "f" * 40
    with pytest.raises(SystemExit):
        a.candidates(doc)


def _long_parquet(path: Path, values: np.ndarray, name: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"timestamp": pa.array(range(values.size), type=pa.int64()),
                             "v": pa.array(values.astype(np.float64)),
                             "_original_filename": pa.array([name] * values.size)}), str(path))


def test_datacompare_compares_data_not_names(tmp_path, capsys):
    a = _load(ALLOW, "laned_rmisc_allowlist_accept3")
    gift_root, _ = fx.gift_mirror(tmp_path)
    rm = tmp_path / "rm"
    copied = fx.series_values("daily_sales", 4, 300).astype(np.float32)[0]
    _long_parquet(rm / "Renamed" / "part0.parquet", copied, "a")
    _long_parquet(rm / "Fresh" / "part0.parquet",
                  np.cumsum(np.random.default_rng(3).normal(size=400)), "b")
    cands = {"schema": a.CANDIDATES_SCHEMA, "candidates": [
        {"name": "Fresh"}, {"name": "Renamed"}]}
    cpath = tmp_path / "c.json"
    cpath.write_text(json.dumps(cands))
    empty = tmp_path / "empty"
    empty.mkdir()
    roots = [f"{k}={gift_root if k == 'gift_eval_pretrain' else empty}"
             for k in a.EXCLUSION_SOURCES]
    out = tmp_path / "dc.json"
    with pytest.raises(SystemExit):        # every exclusion source must be given
        a.main(["datacompare", "--candidates", str(cpath), "--rmisc-root", str(rm),
                "--exclude-root", roots[0], "--out", str(out)])
    args = ["datacompare", "--candidates", str(cpath), "--rmisc-root", str(rm), "--out", str(out)]
    for r in roots:
        args += ["--exclude-root", r]
    assert a.main(args) == 0
    doc = json.loads(out.read_text())
    assert doc["schema"] == a.DATACOMPARE_SCHEMA
    # W-LAUNCH: the process pools and the candidate cache change nothing
    out2 = tmp_path / "dc2.json"
    fast = [x if x != str(out) else str(out2) for x in args] + [
        "--workers", "2", "--candidate-workers", "2", "--candidate-cache", str(tmp_path / "cc")]
    assert a.main(fast) == 0 and a.main(fast) == 0          # second run: cache hits
    assert json.loads(out2.read_text())["folders"] == doc["folders"]
    v = {f["name"]: f for f in doc["folders"]}
    assert v["Renamed"]["verdict"] == "duplicate" and v["Renamed"]["hits"]["gift_eval_pretrain"]
    assert v["Fresh"]["verdict"] == "pass"
    assert "DATACOMPARE" in capsys.readouterr().out


def test_finalize_writes_a_valid_final_list(tmp_path):
    a = _load(ALLOW, "laned_rmisc_allowlist_accept4")
    idx = importlib.import_module("cascade.eval.lane_d_public_index")
    cands = {"schema": a.CANDIDATES_SCHEMA, "candidates": [
        {"name": n, "domain": "Energy", "freq": "h", "licence": lic, "licence_class": lc,
         "source_url": "", "near": []}
        for n, lic, lc in (("Dup", "MIT", "ok"), ("Good", "CC BY 4.0", "ok"),
                           ("Mystery", "Unknown", "unknown"))]}
    dc = {"schema": a.DATACOMPARE_SCHEMA, "params": {}, "folders": [
        {"name": "Dup", "verdict": "duplicate", "max_hit_fraction": 0.9},
        {"name": "Good", "verdict": "pass", "max_hit_fraction": 0.0},
        {"name": "Mystery", "verdict": "pass", "max_hit_fraction": 0.0}]}
    text = a.finalize(cands, dc, allow_unknown_licence=False, generated_by="t")
    assert idx.parse_allowed_list(text)["names"] == ["Good"]
    text = a.finalize(cands, dc, allow_unknown_licence=True, generated_by="t")
    assert idx.parse_allowed_list(text)["names"] == ["Good", "Mystery"]
    p = tmp_path / "allowed.toml"
    p.write_text(text)
    assert a.main(["validate", str(p)]) == 0


def test_datacompare_vectorised_scan_equals_the_reference_channels(tmp_path):
    """W-LAUNCH: `file_hits` (Arrow buffers, batch by batch, the --workers /
    --stream path) hits exactly the hashes the reference `iter_channels` +
    `kgram_hashes` loop hits: flat and nested list columns, nulls, empty rows,
    channels past max_points, scalar numeric columns, arrow and parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    a = _load(ALLOW, "laned_rmisc_allowlist_accept5")
    rng = np.random.default_rng(7)
    k, mod, maxp = 4, 2, 50
    flat = [list(np.round(rng.normal(size=n), 2)) for n in (0, 3, 30, 80, 200)]
    flat[2][5] = None
    nested = [[list(np.round(rng.normal(size=n), 1)) for n in (10, 60)], [], None,
              [list(np.round(rng.normal(size=70), 1))]]
    scalar = np.round(rng.normal(size=4000), 1)
    t = pa.table({"target": pa.array(flat + [None], type=pa.list_(pa.float64())),
                  "multi": pa.array(nested + [None, [[1.0, 2.0, 3.0, 4.0, 5.0]]],
                                    type=pa.list_(pa.list_(pa.float32()))),
                  "start": pa.array([f"s{i}" for i in range(6)])})
    pq.write_table(t, str(tmp_path / "x.parquet"), row_group_size=2)
    with pa.OSFile(str(tmp_path / "x.arrow"), "wb") as sink, \
            pa.ipc.new_stream(sink, t.schema) as w:
        for b in t.to_batches(max_chunksize=2):
            w.write_batch(b)
    pq.write_table(pa.table({"v": pa.array(scalar), "i": pa.array(np.arange(4000))}),
                   str(tmp_path / "long.parquet"), row_group_size=333)
    for name in ("x.parquet", "x.arrow", "long.parquet"):
        p = tmp_path / name
        ref = [a.kgram_hashes(ch, k, mod) for ch in a.iter_channels(p, maxp)]
        ref = np.unique(np.concatenate(ref)) if ref else np.empty(0, dtype=np.uint64)
        assert ref.size > 10, name
        half = np.sort(ref[::2])                           # a candidate set: half of them
        decoy = np.sort(np.concatenate([half, np.asarray([1, 3, 2**63 + 1], dtype=np.uint64)]))
        got = a.file_hits(p, decoy, k, mod, maxp)
        assert got.tolist() == half.tolist(), name


def test_datacompare_candidate_hashing_equals_the_reference(tmp_path):
    """W-LAUNCH: the Arrow candidate path (`index_in`, run on --workers)
    samples the same series and hashes the same values as the reference
    per-row path: string and dictionary<string> names with nulls, int names,
    no name column."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cascade.eval.lane_d_public_index import RMISC_SERIES_COLUMN as SC

    a = _load(ALLOW, "laned_rmisc_allowlist_accept6")
    rng = np.random.default_rng(11)
    n = 3000
    names = [None if i % 97 == 0 else f"s{i % 41}" for i in range(n)]
    cases = {
        "str.parquet": {SC: pa.array(names), "v": pa.array(np.round(rng.normal(size=n), 1)),
                        "t": pa.array(np.arange(n))},
        "dict.parquet": {SC: pa.array(names).dictionary_encode(),
                         "v": pa.array(np.round(rng.normal(size=n), 1))},
        "int.parquet": {SC: pa.array(np.arange(n) % 13), "v": pa.array(rng.normal(size=n))},
        "none.parquet": {"v": pa.array(np.round(rng.normal(size=n), 2))},
    }
    for fn, cols in cases.items():
        p = tmp_path / fn
        pq.write_table(pa.table(cols), str(p), row_group_size=700)
        for sample in (5, 64):
            ref = a.cand_file_hashes(p, sample, 50, 4, 2, reference=True)
            got = a.cand_file_hashes(p, sample, 50, 4, 2)
            assert got[0] == ref[0] and got[1].tolist() == ref[1].tolist(), (fn, sample)
            assert ref[1].size > 0
