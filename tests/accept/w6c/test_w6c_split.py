"""W6c: `pool_d_hist` / `pool_d_tgt` split of one `pool_d/1` snapshot
(plan §3.0 W6c row, §1.7 "Artifacts", invariant 8).

On main each test fails at its first import of `cascade.pool.pool_d_split`
(in PATHS).
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import time

import numpy as np
import pytest

from .helpers import AS_OF, HORIZON, SERIES, sha, snapshot_sha, split_mod, write_pool_d


def _split(tmp_path, name="pool"):
    mod = split_mod()
    pool = write_pool_d(tmp_path / name)
    return mod, pool, mod.split_pool_d(pool, tmp_path / f"{name}_out")


def _members(tar_path):
    with tarfile.open(tar_path) as tf:
        return {m.name: tf.extractfile(m).read() for m in tf.getmembers()}


def test_split_writes_two_tars_with_exact_members(tmp_path):
    mod, pool, res = _split(tmp_path)
    assert mod.HIST_MEMBERS == ("hist.npz", "hist_metadata.json")
    assert mod.TGT_MEMBERS == ("metadata.json", "tgt.npz")
    snap = f"pool_d-{AS_OF}"
    assert res.snapshot == snap
    assert res.hist_tar.name == f"{snap}.hist.tar" and res.tgt_tar.name == f"{snap}.tgt.tar"
    assert res.snapshot_sha256 == snapshot_sha(pool)
    assert res.hist_sha256 == sha(res.hist_tar) and res.tgt_sha256 == sha(res.tgt_tar)
    with tarfile.open(res.hist_tar) as tf:
        assert [m.name for m in tf.getmembers()] == list(mod.HIST_MEMBERS)
    with tarfile.open(res.tgt_tar) as tf:
        assert [m.name for m in tf.getmembers()] == list(mod.TGT_MEMBERS)


def test_hist_tar_holds_no_target_data(tmp_path):
    """Invariant 8: nothing in the eval host's artefact carries a target."""
    _mod, pool, res = _split(tmp_path)
    raw = res.hist_tar.read_bytes()
    with np.load(pool / "tgt.npz") as tgt:
        for sid in tgt.files:
            assert tgt[sid].tobytes() not in raw, sid
            assert tgt[sid][:8].tobytes() not in raw, sid
    members = _members(res.hist_tar)
    assert members["hist.npz"] == (pool / "hist.npz").read_bytes()
    text = members["hist_metadata.json"].decode()
    for forbidden in ('"target_ts"', '"first_seen_ts"', '"end_ts"', '"frontier"', '"feeds"',
                      '"embargo_hours"', '"panel"', "tgt.npz"):
        assert forbidden not in text, forbidden


def test_hist_metadata_fields_exact(tmp_path):
    mod, pool, res = _split(tmp_path)
    hm = json.loads(_members(res.hist_tar)["hist_metadata.json"])
    assert hm["schema"] == mod.HIST_META_SCHEMA == "pool_d_hist/1"
    assert set(hm) == {"schema", "snapshot", "snapshot_sha256", "as_of_ts", "horizon",
                       "history_length", "series"}
    assert hm["snapshot_sha256"] == res.snapshot_sha256
    assert hm["horizon"] == HORIZON and hm["as_of_ts"] == AS_OF
    meta = json.loads((pool / "metadata.json").read_text())
    want = [{k: s[k] for k in mod.HIST_SERIES_FIELDS} for s in meta["series"]]
    assert hm["series"] == want
    for s in hm["series"]:
        assert set(s) == set(mod.HIST_SERIES_FIELDS)
    assert mod.hist_metadata(meta, res.snapshot_sha256) == hm


def test_tgt_tar_holds_targets_and_timestamps(tmp_path):
    _mod, pool, res = _split(tmp_path)
    members = _members(res.tgt_tar)
    assert members["metadata.json"] == (pool / "metadata.json").read_bytes()
    assert members["tgt.npz"] == (pool / "tgt.npz").read_bytes()
    assert "hist.npz" not in members
    assert res.tgt_tar.stat().st_size < res.hist_tar.stat().st_size   # targets are small


def test_join_restores_loadable_snapshot(tmp_path):
    mod, pool, res = _split(tmp_path)
    out = mod.join_split(res.hist_tar, res.tgt_tar, tmp_path / "joined")
    for n in ("metadata.json", "hist.npz", "tgt.npz"):
        assert (out / n).read_bytes() == (pool / n).read_bytes()
    from cascade.validator.lane_d import load_pool

    loaded = load_pool(out, HORIZON)
    assert loaded["sha256"] == res.snapshot_sha256 == load_pool(pool, HORIZON)["sha256"]
    assert {w["series_id"] for w in loaded["windows"]} == {s[0] for s in SERIES}


def test_read_hist_from_hist_tar_alone(tmp_path):
    mod, pool, res = _split(tmp_path)
    os.remove(res.tgt_tar)                              # the eval host never has it
    hm, hist = mod.read_hist(res.hist_tar)
    assert hm["snapshot"] == res.snapshot
    with np.load(pool / "hist.npz") as want:
        assert set(hist) == set(want.files)
        for sid in want.files:
            np.testing.assert_array_equal(hist[sid], want[sid])
            assert hist[sid].shape == (dict((s[0], s[2]) for s in SERIES)[sid],)


def test_split_is_deterministic(tmp_path, monkeypatch):
    mod, pool, first = _split(tmp_path)
    for n in ("metadata.json", "hist.npz", "tgt.npz"):
        os.utime(pool / n, (1_000_000_000, 1_000_000_000))
    monkeypatch.setattr(time, "time", lambda: 2_000_000_000.0)
    again = mod.split_pool_d(pool, tmp_path / "again")
    assert first.hist_tar.read_bytes() == again.hist_tar.read_bytes()
    assert first.tgt_tar.read_bytes() == again.tgt_tar.read_bytes()
    for tar_path in (first.hist_tar, first.tgt_tar):
        with tarfile.open(tar_path) as tf:
            names = [m.name for m in tf.getmembers()]
            assert names == sorted(names)
            for m in tf.getmembers():
                assert m.isreg() and m.mtime == 0 and m.uid == 0 and m.gid == 0
                assert m.uname == "" and m.gname == "" and m.mode == 0o644
    other = write_pool_d(tmp_path / "other", shift=1.0)
    changed = mod.split_pool_d(other, tmp_path / "other_out")
    assert changed.hist_sha256 != first.hist_sha256


def _tar(path, members: dict, *, link: str | None = None):
    with tarfile.open(path, "w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        if link:
            info = tarfile.TarInfo(link)
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)
    return path


def test_join_refusals(tmp_path):
    mod, pool, res = _split(tmp_path)

    def code(hist, tgt):
        with pytest.raises(mod.SplitError) as err:
            mod.join_split(hist, tgt, tmp_path / f"j{len(os.listdir(tmp_path))}")
        return err.value.code

    assert issubclass(mod.SplitError, ValueError)
    assert code(res.tgt_tar, res.hist_tar) == "bad_members"
    other = mod.split_pool_d(write_pool_d(tmp_path / "p2", as_of=AS_OF + 10800),
                             tmp_path / "p2_out")
    assert code(res.hist_tar, other.tgt_tar) == "snapshot_mismatch"
    tgt = _members(res.tgt_tar)
    bad = dict(tgt)
    bad["tgt.npz"] = tgt["tgt.npz"][:-8] + b"\x00" * 8
    assert code(res.hist_tar, _tar(tmp_path / "bad.tgt.tar", bad)) == "digest_mismatch"
    evil = {"../metadata.json": tgt["metadata.json"], "tgt.npz": tgt["tgt.npz"]}
    assert code(res.hist_tar, _tar(tmp_path / "evil.tgt.tar", evil)) == "unsafe_member"
    hist = _members(res.hist_tar)
    linked = _tar(tmp_path / "link.hist.tar", {"hist_metadata.json": hist["hist_metadata.json"]},
                  link="hist.npz")
    assert code(linked, res.tgt_tar) == "unsafe_member"
    with pytest.raises(mod.SplitError):
        mod.read_hist(res.tgt_tar)


def test_split_refuses_invalid_pool_d(tmp_path):
    mod = split_mod()
    pool = write_pool_d(tmp_path / "pool")
    meta = json.loads((pool / "metadata.json").read_text())
    meta["series"][0].pop("frontier")                 # RV3 BLOCKING 1: invalid
    (pool / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        mod.split_pool_d(pool, tmp_path / "out")
    assert not list(tmp_path.glob("out/*.tar"))


def test_cli_split_and_join(tmp_path):
    mod = split_mod()
    pool = write_pool_d(tmp_path / "pool")
    assert mod.main(["split", "--pool-d", str(pool), "--out", str(tmp_path / "tars")]) == 0
    snap = f"pool_d-{AS_OF}"
    hist, tgt = tmp_path / "tars" / f"{snap}.hist.tar", tmp_path / "tars" / f"{snap}.tgt.tar"
    assert hist.is_file() and tgt.is_file()
    assert mod.main(["join", "--hist", str(hist), "--tgt", str(tgt),
                     "--out", str(tmp_path / "joined")]) == 0
    assert snapshot_sha(tmp_path / "joined") == snapshot_sha(pool)
    assert mod.main(["join", "--hist", str(tgt), "--tgt", str(hist),
                     "--out", str(tmp_path / "bad")]) != 0
