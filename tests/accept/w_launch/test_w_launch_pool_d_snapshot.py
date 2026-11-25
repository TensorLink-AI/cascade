"""W-LAUNCH: pool_d takes lane A's selection from the sha-verified snapshot.

Owner 2026-11-25 22:40 UTC (option a): hourly lane A snapshots are built from
a same-day forge that keeps growing, so the W6b byte-identical replay refused
every real snapshot (2902 mismatching files on 9587700). ``pool_d_build``'s
default is now ``--lane-a-selection snapshot``: it re-verifies the fetched tar
against the index sha256 and selects exactly the snapshot's series. On a base
without this change the drifted-snapshot build exits 3
(``lane_a_replay_mismatch``) and the test fails.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.accept.w6b.helpers import AS2, CTX_A, LANE_A_CFG, build_world, forge_source, no_fetch


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("wl_pool_d"))


def _snapshot(world, tmp_path, *, drift: bool, block: int = 9587700, bad_sha: bool = False):
    """A lane A snapshot as the fetch step leaves it: dir + ``.tar`` + ``.ref.json``."""
    from cascade.pool import builder
    from cascade.shared.hippius import pack_dir_to_tar, tar_cid_digest

    out = tmp_path / "lane_a"
    builder.build_pool([forge_source(world["root"])], out, CTX_A, LANE_A_CFG, fetch=no_fetch)
    if drift:   # lane A was built from an older forge: its tail values differ
        victim = out / "tsforge__air_temp.npy"
        arr = np.load(victim)
        arr[-1] += 1.0
        np.save(victim, arr, allow_pickle=False)
    data = pack_dir_to_tar(out)
    (tmp_path / "lane_a.tar").write_bytes(data)
    sha = "0" * 64 if bad_sha else tar_cid_digest(data)
    (tmp_path / "lane_a.ref.json").write_text(json.dumps(
        {"effective_block": block, "sha256": sha, "key": f"pool/snapshots/block-{block}.tar",
         "tar": "lane_a.tar"}))
    return out, sha


def _main(world, tmp_path, lane_a, out_name="out", extra=()):
    from cascade.pool import pool_d_build

    return pool_d_build.main([
        "--forge-dir", str(world["root"]), "--first-seen-dir", str(world["root"] / "first_seen"),
        "--lane-a-pool", str(lane_a), "--out", str(tmp_path / out_name),
        "--as-of-ts", str(AS2), "--prior-dir", str(world["d1"]), *extra])


def _strip(meta):
    return {k: v for k, v in meta.items() if k not in ("selection_sha256", "lane_a_selection")}


def test_drifted_snapshot_builds_the_replay_series(world, tmp_path):
    lane_a, sha = _snapshot(world, tmp_path, drift=True)
    assert _main(world, tmp_path, lane_a) == 0
    got = json.loads((tmp_path / "out" / "metadata.json").read_text())
    want = json.loads((world["d2"] / "metadata.json").read_text())
    # identical series, refusals, arrays and feeds to the replay-selection build
    assert _strip(got) == _strip(want)
    for name in ("hist.npz", "tgt.npz"):
        with np.load(tmp_path / "out" / name) as g, np.load(world["d2"] / name) as w:
            assert sorted(g.files) == sorted(w.files)
            for k in g.files:
                np.testing.assert_array_equal(g[k], w[k])
    assert got["lane_a_selection"] == {"mode": "snapshot", "effective_block": 9587700,
                                       "sha256": sha}


def test_index_sha_mismatch_is_refused(world, tmp_path, capsys):
    lane_a, _sha = _snapshot(world, tmp_path, drift=False, bad_sha=True)
    assert _main(world, tmp_path, lane_a, out_name="refused") == 3
    assert "lane_a_snapshot_refused" in capsys.readouterr().err
    assert not (tmp_path / "refused" / "metadata.json").exists()


def test_replay_mode_stays_opt_in_and_strict(world, tmp_path, capsys):
    lane_a, _sha = _snapshot(world, tmp_path, drift=True)
    try:
        rc = _main(world, tmp_path, lane_a, out_name="replay",
                   extra=("--lane-a-selection", "replay", "--max-panel-series-per-feed", "5"))
    except SystemExit as e:     # a base without --lane-a-selection: argparse exits 2
        rc = e.code
    assert rc == 3
    assert "lane_a_replay_mismatch" in capsys.readouterr().err
