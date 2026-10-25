"""DEC-CA-0053: the tagged corpus dataset on Hippius S3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cascade.generation import dataset as D
from cascade.generation import shards as S
from cascade.generation.cli import EXIT_REFUSED
from cascade.generation.cli import main as cli_main
from cascade.shared.config import GenerationConfig
from cascade.shared.hippius import ObjectNotFound

HK = "5Fhotkey"


class FakeStore:
    def __init__(self):
        self.objs: dict[str, bytes] = {}
        self.acl: dict[str, str] = {}

    def put_bytes(self, key, data, content_type="", acl=None):
        self.objs[key] = bytes(data)
        if acl:
            self.acl[key] = acl

    def put_text(self, key, text, content_type="", acl=None):
        self.put_bytes(key, text.encode(), content_type, acl)

    def get_bytes(self, key):
        if key not in self.objs:
            raise ObjectNotFound(key)
        return self.objs[key]

    def get_text(self, key):
        return self.get_bytes(key).decode()

    def list_keys(self, prefix):
        return [k for k in self.objs if k.startswith(prefix)]

    def delete(self, key):
        self.objs.pop(key, None)


def _corpus(d: Path, seed: int = 7, kids: int = 2, shards_per: int = 2) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    children = []
    for i in range(kids):
        sh = [S.write_shard(d / f"c{i}-{n}.parquet",
                            [rng.standard_normal((1 + j % 2, 64 * (j + 1))) for j in range(4)])
              for n in range(shards_per)]
        children.append({"index": i, "seed": S.child_seed(seed, i), "shards": sh})
    return S.finalize_index(d, seed=seed, children=children,
                            meta={"generator_ref": "r@sha256:" + "a" * 64, "cpu_seconds": 12.5})


def _upload(store, tmp_path, round_id="100", hotkey=HK, now=1000.0):
    d = tmp_path / f"{round_id}-{hotkey}"
    _corpus(d)
    return D.upload_corpus(store, d, round_id=round_id, hotkey=hotkey, uid=12, era=3,
                           provider="targon", shape="cpu-xlarge-amd", now=now)


def test_upload_tags_and_layout(tmp_path):
    store = FakeStore()
    row = _upload(store, tmp_path)
    base = "corpora/round-100/5Fhotkey/"
    assert {base + n for n in ("c0-0.parquet", "c1-1.parquet", "meta.json", "corpus.json")} \
        <= set(store.objs)
    meta = json.loads(store.get_text(base + "meta.json"))
    assert (meta["uid"], meta["era"], meta["provider"]) == (12, 3, "targon")
    # lengths 64/128/192/256 per shard, four shards → pow2 buckets 64, 128, 256 (×2)
    assert meta["n_series"] == 16
    assert meta["length_histogram"] == {"64": 4, "128": 4, "256": 8}
    assert sum(meta["channels_histogram"].values()) == 16
    assert row["crowned"] is False and row["crps_geomean"] is None
    assert D.read_dataset_index(store) == [row]
    assert not store.acl, "every corpus starts operator-private"


def test_reupload_replaces_row_and_scores_fill_later(tmp_path):
    store = FakeStore()
    _upload(store, tmp_path)
    _upload(store, tmp_path)
    assert len(D.read_dataset_index(store)) == 1
    D.record_scores(store, "100", HK, crps_geomean=0.41, verdict="dethroned", crowned=True)
    row = D.read_dataset_index(store)[0]
    assert (row["crps_geomean"], row["verdict"], row["crowned"]) == (0.41, "dethroned", True)
    with pytest.raises(ValueError):
        D.record_scores(store, "100", HK, made_up=1)
    with pytest.raises(KeyError):
        D.record_scores(store, "999", HK, verdict="x")


def test_bad_shard_is_never_uploaded(tmp_path):
    store = FakeStore()
    d = tmp_path / "c"
    _corpus(d)
    (d / "c0-0.parquet").write_bytes(b"tampered")
    with pytest.raises(S.ShardError):
        D.upload_corpus(store, d, round_id="1", hotkey=HK, uid=1, era=0,
                        provider="targon", shape="x")
    assert not store.objs


def test_hotkey_and_round_are_path_safe():
    with pytest.raises(ValueError):
        D.leg_prefix("1", "../evil")
    with pytest.raises(ValueError):
        D.leg_prefix("a/b", HK)


@pytest.mark.parametrize("mode,expect", [("full", 4), ("recipe", 2)])
def test_publish_king_corpus(tmp_path, mode, expect):
    src, pub = FakeStore(), FakeStore()
    _upload(src, tmp_path)
    out = D.publish_king_corpus(src, pub, "100", HK,
                                GenerationConfig(king_publish=mode, sample_shards=2))
    assert len(out["shards"]) == expect
    shards = [k for k in pub.objs if k.endswith(".parquet")]
    assert len(shards) == expect and all(pub.acl[k] == "public-read" for k in pub.objs)
    if mode == "recipe":
        assert sorted(out["shards"]) == ["c0-0.parquet", "c1-0.parquet"]  # first in training order
    row = D.read_dataset_index(src)[0]
    assert row["crowned"] is True and row["published"] == mode


def test_expire_keeps_kings_recent_and_kept(tmp_path):
    store = FakeStore()
    day = 86400.0
    _upload(store, tmp_path, round_id="1", hotkey="old", now=0.0)
    _upload(store, tmp_path, round_id="2", hotkey="king", now=0.0)
    _upload(store, tmp_path, round_id="3", hotkey="kept", now=0.0)
    _upload(store, tmp_path, round_id="4", hotkey="new", now=20 * day)
    D.record_scores(store, "2", "king", crowned=True)
    gone = D.expire_corpora(store, 21 * day, GenerationConfig(retention_days=14),
                            keep={("3", "kept")})
    assert gone == [("1", "old")]
    assert not store.list_keys("corpora/round-1/")
    assert store.list_keys("corpora/round-2/") and store.list_keys("corpora/round-4/")
    rows = {r["hotkey"]: r for r in D.read_dataset_index(store)}
    assert rows["old"]["expired"] is True and rows["king"]["expired"] is False
    assert D.expire_corpora(store, 21 * day, GenerationConfig(retention_days=14)) == [
        ("3", "kept")]


def test_cli_refuses_when_tier_disabled(tmp_path, capsys, monkeypatch):
    from dataclasses import replace

    from cascade.shared.config import load_chain_config

    cfg = load_chain_config()
    off = replace(cfg, generation=replace(cfg.generation, enabled=False))
    monkeypatch.setattr("cascade.shared.config.load_chain_config", lambda *a: off)
    rc = cli_main([str(tmp_path), str(tmp_path / "out"), "--seed", "1", "--json"])
    assert rc == EXIT_REFUSED
    assert json.loads(capsys.readouterr().out)["ok"] is False
