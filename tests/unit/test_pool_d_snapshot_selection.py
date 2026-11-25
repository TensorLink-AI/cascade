"""pool_d snapshot-mode selection (W-LAUNCH, owner 2026-11-25 option a).

Torch-free and pandas-free: a hand-written lane A snapshot directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cascade.pool.pool_d_build import verify_lane_a_snapshot
from cascade.pool.pool_d_select import select_from_snapshot
from cascade.shared.hippius import pack_dir_to_tar, tar_cid_digest


def _md(feed, **kw):
    return {"freq": "H", "seasonal_period": 24, "domain": "energy", "source": f"fam:{feed}",
            "source_family": f"fam:{feed}", "catalog_source": feed, **kw}


SNAPSHOT = {
    "tsforge__grid__REGION_A": (_md("grid", dgp_class="load"), (64,)),
    "tsforge__grid__REGION_B": (_md("grid"), (64,)),
    # collect_records' collision suffix: both members are key_collision
    "tsforge__bikes__id_7": (_md("bikes"), (64,)),
    "tsforge__bikes__id_7-2": (_md("bikes"), (64,)),
    # an id that merely ends in -3 with no base sibling is its own base
    "tsforge__air__station_K-3": (_md("air"), (64,)),
    # a packed 2-channel record: one selected series per channel
    "tsforge__mv": (_md("mv"), (2, 64)),
    # no catalog source
    "tsforge__orphan": ({"freq": "H", "seasonal_period": 24, "domain": "web",
                         "source": "fam:orphan", "source_family": "fam:orphan"}, (64,)),
}


def _write(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for sid, (_m, shape) in SNAPSHOT.items():
        np.save(d / f"{sid}.npy", np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
                allow_pickle=False)
    (d / "metadata.json").write_text(json.dumps({k: m for k, (m, _s) in SNAPSHOT.items()},
                                                indent=2, sort_keys=True))
    (d / "provenance.json").write_text(json.dumps(
        {"builder_version": "cascade-pool/1", "config": {"cluster_by": "source_family"},
         "summary": {"as_of": "2026-11-25"}}))
    return d


def _ref(tmp_path: Path, d: Path, sha: str | None = None) -> Path:
    data = pack_dir_to_tar(d)
    (tmp_path / "lane_a.tar").write_bytes(data)
    ref = tmp_path / "lane_a.ref.json"
    ref.write_text(json.dumps({"effective_block": 9587700, "sha256": sha or tar_cid_digest(data),
                               "key": "pool/snapshots/block-9587700.tar", "tar": "lane_a.tar"}))
    return ref


def test_selects_exactly_the_snapshot_series(tmp_path):
    sel = select_from_snapshot(_write(tmp_path / "lane_a"))
    assert sorted(s.key for s in sel.series) == sorted([
        ("air", "tsforge__air__station_K-3", 0),
        ("grid", "tsforge__grid__REGION_A", 0),
        ("grid", "tsforge__grid__REGION_B", 0),
        ("mv", "tsforge__mv", 0),
        ("mv", "tsforge__mv", 1),
    ])
    assert dict(sel.refused) == {"tsforge__bikes__id_7": "key_collision",
                                 "tsforge__bikes__id_7-2": "key_collision",
                                 "tsforge__orphan": "no_source"}
    by = sel.by_key()
    a = by[("grid", "tsforge__grid__REGION_A", 0)]
    assert (a.pool_series_id, a.feed, a.cluster, a.domain, a.freq, a.n_channels) == (
        "tsforge__grid__REGION_A", "grid", "fam:grid", "energy", "H", 1)
    assert a.pool_d_series_id == "tsforge__grid__REGION_A~grid.c0"
    assert a.attrs == {"dgp_class": "load", "source_family": "fam:grid"}
    assert by[("mv", "tsforge__mv", 1)].n_channels == 2
    # every snapshot record is selected or refused, never both, never dropped
    picked = {s.pool_series_id for s in sel.series}
    assert picked | {sid for sid, _r in sel.refused} == set(SNAPSHOT)
    assert not picked & {sid for sid, _r in sel.refused}


def test_verified_snapshot_returns_its_index_entry(tmp_path):
    d = _write(tmp_path / "lane_a")
    ref = _ref(tmp_path, d)
    sha = json.loads(ref.read_text())["sha256"]
    assert verify_lane_a_snapshot(d) == {"mode": "snapshot", "effective_block": 9587700,
                                         "sha256": sha}


def test_index_sha_mismatch_is_refused(tmp_path):
    d = _write(tmp_path / "lane_a")
    _ref(tmp_path, d, sha="f" * 64)
    with pytest.raises(ValueError, match="lane_a_snapshot_refused: tar sha256"):
        verify_lane_a_snapshot(d)


def test_dir_that_differs_from_the_tar_is_refused(tmp_path):
    d = _write(tmp_path / "lane_a")
    _ref(tmp_path, d)
    np.save(d / "tsforge__mv.npy", np.zeros((2, 64), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="differs from the verified tar"):
        verify_lane_a_snapshot(d)


def test_missing_ref_is_refused(tmp_path):
    with pytest.raises(ValueError, match="lane_a_snapshot_refused: ref"):
        verify_lane_a_snapshot(_write(tmp_path / "lane_a"))


def test_lane_a_pool_keeps_the_verified_tar_and_ref(tmp_path, monkeypatch):
    from cascade.laned import elastic
    from cascade.shared import hippius
    from cascade.shared.hippius import PoolSnapshotMeta

    src = _write(tmp_path / "src")
    data = pack_dir_to_tar(src)

    def meta(block, sha):
        return PoolSnapshotMeta(effective_block=block, key=f"pool/snapshots/block-{block}.tar",
                                sha256=sha, size_bytes=len(data), as_of="2026-11-25",
                                n_series=7, context_length=4096, horizon=64)

    good = meta(9587700, tar_cid_digest(data))

    class Store:
        def get_bytes(self, key):
            assert key == good.key
            return data

    monkeypatch.setattr(hippius, "pool_s3_store", lambda storage: Store())
    monkeypatch.setattr(hippius, "read_pool_index",
                        lambda store: [meta(9586800, "0" * 64), good,
                                       meta(999_999_999_999, "1" * 64)])
    out = tmp_path / "work" / "lane_a_pool"
    doc = elastic.lane_a_pool(Path("chain.toml"), out)
    assert doc["effective_block"] == 9587700
    assert verify_lane_a_snapshot(out)["sha256"] == good.sha256
