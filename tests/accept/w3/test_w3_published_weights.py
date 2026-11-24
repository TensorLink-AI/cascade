"""W3 follow-up: the producer of ``<state-dir>/published_weights.json``
(``lane-d-published-weights/1``), the file the intake's dedup reads
(README "Dedup"; ``503 dedup_unavailable`` until it exists).

The producer is ``cascade.lane_d.published_weights``. Every import of it is
inside a test body, so on a base without it the failures are a missing
module in W3's PATHS.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from .helpers import checkpoint, digest_of, key, manifest, ss58, weights_sha

SCHEMA = "lane-d-published-weights/1"


def producer():
    return importlib.import_module("cascade.lane_d.published_weights")


def _king(hotkey: str, digest: str, crown_block: int) -> dict:
    return {"hotkey": hotkey, "context": 8192, "digest": digest, "crown_block": crown_block,
            "lcb": 0.01, "gate": {"outcome": "pass"}, "sentinel": {"suite": "time"},
            "license": "MIT"}


def _mirror_with(root: Path, files: dict[str, bytes]) -> str:
    """Put one king's manifest (and files) on a local mirror copy; its digest."""
    man = manifest(files)
    digest = digest_of(man)
    d = root / "lane-d-kings" / digest
    d.mkdir(parents=True)
    (d / "manifest.json").write_bytes(json.dumps(man, sort_keys=True,
                                                 separators=(",", ":")).encode())
    for name, data in files.items():
        (d / name).write_bytes(data)
    return digest


def _signed_index(tmp_path: Path, *kings_files: dict[str, bytes], signer: str = "index"):
    from cascade.lane_d.publish import build_index

    mirror = tmp_path / "mirror"
    kings = [_king(ss58(f"king{i}"), _mirror_with(mirror, files), 9_000_000 + i)
             for i, files in enumerate(kings_files)]
    index = build_index(kings, key(signer))
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index))
    return path, mirror


def _checkpoint_dir(root: Path, files: dict[str, bytes]) -> Path:
    root.mkdir(parents=True)
    for name, data in files.items():
        (root / name).write_bytes(data)
    return root


def test_written_registry_is_what_the_intake_reads(tmp_path):
    from cascade.lane_d.chain_reads import PublishedWeightsFile

    a, b = checkpoint("pa"), checkpoint("pb")
    out = tmp_path / "state" / "published_weights.json"
    doc = producer().write_registry(out, [weights_sha(b), weights_sha(a), weights_sha(a)])
    assert doc["schema"] == SCHEMA
    assert doc["weights_sha256"] == sorted({weights_sha(a), weights_sha(b)})
    assert PublishedWeightsFile(out).read() == {weights_sha(a), weights_sha(b)}
    assert [p.name for p in out.parent.iterdir()] == ["published_weights.json"]  # no temp left


def test_non_hex_values_are_refused_and_nothing_is_written(tmp_path):
    out = tmp_path / "published_weights.json"
    with pytest.raises(producer().PublishedWeightsSourceError):
        producer().write_registry(out, ["not-a-sha"])
    assert not out.exists()


def test_lane_d_kings_from_the_signed_index_and_a_mirror(tmp_path):
    p = producer()
    k1, k2 = checkpoint("k1"), checkpoint("k2")
    index_path, mirror = _signed_index(tmp_path, k1, k2)
    index = json.loads(index_path.read_text())
    got = p.lane_d_king_weights(index, ss58("index"), p.DirMirror(mirror))
    assert sorted(got) == sorted([weights_sha(k1), weights_sha(k2)])


def test_an_index_from_another_signer_is_refused(tmp_path):
    p = producer()
    index_path, mirror = _signed_index(tmp_path, checkpoint("k3"), signer="intruder")
    with pytest.raises(p.PublishedWeightsSourceError):
        p.lane_d_king_weights(json.loads(index_path.read_text()), ss58("index"),
                              p.DirMirror(mirror))


def test_a_mirror_manifest_that_does_not_hash_to_the_signed_digest_is_refused(tmp_path):
    p = producer()
    files = checkpoint("k4")
    index_path, mirror = _signed_index(tmp_path, files)
    index = json.loads(index_path.read_text())
    digest = index["kings"][0]["digest"]
    forged = manifest({**files, "weights.safetensors": b"FORGED" * 8})
    (mirror / "lane-d-kings" / digest / "manifest.json").write_text(json.dumps(forged))
    with pytest.raises(p.PublishedWeightsSourceError):
        p.lane_d_king_weights(index, ss58("index"), p.DirMirror(mirror))
    (mirror / "lane-d-kings" / digest / "manifest.json").unlink()
    with pytest.raises(p.PublishedWeightsSourceError):
        p.lane_d_king_weights(index, ss58("index"), p.DirMirror(mirror))


def test_cli_merges_kings_checkpoints_and_the_previous_file(tmp_path, capsys):
    from cascade.lane_d.chain_reads import PublishedWeightsFile

    p = producer()
    king, leg, house, old = (checkpoint("ck"), checkpoint("leg"), checkpoint("house"),
                             checkpoint("old"))
    index_path, mirror = _signed_index(tmp_path, king)
    out = tmp_path / "state" / "published_weights.json"
    p.write_registry(out, [weights_sha(old)])
    rc = p.main(["--out", str(out), "--lane-d-index", str(index_path),
                 "--index-signer", ss58("index"), "--mirror-dir", str(mirror),
                 "--checkpoint-dir", str(_checkpoint_dir(tmp_path / "leg", leg)),
                 "--checkpoint-dir", str(_checkpoint_dir(tmp_path / "house", house))])
    assert rc == 0, capsys.readouterr()
    assert PublishedWeightsFile(out).read() == {weights_sha(x) for x in (king, leg, house, old)}
    assert p.main(["--out", str(out), "--replace",
                   "--checkpoint-dir", str(tmp_path / "leg")]) == 0
    assert PublishedWeightsFile(out).read() == {weights_sha(leg)}


def test_cli_fails_closed_and_leaves_the_previous_file(tmp_path, capsys):
    p = producer()
    keep = checkpoint("keep")
    out = tmp_path / "published_weights.json"
    p.write_registry(out, [weights_sha(keep)])
    before = out.read_bytes()
    index_path, mirror = _signed_index(tmp_path, checkpoint("k5"), signer="intruder")
    assert p.main(["--out", str(out), "--lane-d-index", str(index_path),
                   "--index-signer", ss58("index"), "--mirror-dir", str(mirror)]) == 1
    assert p.main(["--out", str(out), "--checkpoint-dir", str(tmp_path / "absent")]) == 1
    assert p.main(["--out", str(out), "--lane-d-index", str(index_path)]) == 2
    assert out.read_bytes() == before
    out.write_text('{"schema": "something-else/1", "weights_sha256": []}')
    assert p.main(["--out", str(out), "--checkpoint-dir", str(tmp_path / "absent")]) == 1
    assert json.loads(out.read_text())["schema"] == "something-else/1"
