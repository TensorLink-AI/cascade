"""Private R2 routing, lossless shards and receipt-bound corpus lifecycle."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.generation import dataset as D
from cascade.generation import lifecycle as L
from cascade.generation import shards as S
from cascade.shared.config import GenerationConfig, load_chain_config
from cascade.shared.config.generation import parse_generation
from cascade.trainer.runner_generation import _GenerationMixin
from tests.unit.test_generation_dataset import HK, FakeStore, _upload


def test_bss_is_lossless_including_signed_zero_and_group_labels(tmp_path):
    _, pq = S._pa()
    values = np.random.default_rng(1).normal(size=(32, 65)).astype(np.float32)
    values[0, :4] = [0.0, -0.0, np.nextafter(np.float32(0), np.float32(1)), np.finfo(np.float32).max]
    groups = np.repeat(np.arange(8, dtype=np.uint8), 4)
    entries = []
    for name, encoding in [("old", "legacy"), ("new", "float32_bss")]:
        path = tmp_path / f"{name}.parquet"
        entry = S.write_shard(path, [{"values": values, "groups": groups}, values], encoding=encoding)
        got = list(S.iter_shard(path))
        assert got[0]["values"].astype("<f4").tobytes() == values.astype("<f4").tobytes()
        assert got[1].astype("<f4").tobytes() == values.astype("<f4").tobytes()
        assert np.array_equal(got[0]["groups"], groups)
        entries.append(S.stream_digest(got))
        if name == "new":
            assert entry["schema"] == 3
            assert "BYTE_STREAM_SPLIT" in pq.ParquetFile(path).metadata.row_group(0).column(2).encodings
    assert entries[0] == entries[1]


def test_legacy_bytes_equal_original_writer_even_with_explicit_none(tmp_path):
    pa, pq = S._pa()
    values = np.arange(32 * 128, dtype=np.float32).reshape(32, 128)
    expected = tmp_path / "expected.parquet"
    pq.write_table(pa.table({"channels": pa.array([32], type=pa.int32()),
                             "length": pa.array([128], type=pa.int32()),
                             "values": pa.array([values.astype("<f4").tobytes()], type=pa.large_binary())}),
                   expected, compression="zstd")
    for record in (values, {"values": values, "groups": None}):
        S.write_shard(tmp_path / "actual.parquet", [record])
        assert (tmp_path / "actual.parquet").read_bytes() == expected.read_bytes()


def test_schema3_multiple_record_batches_and_index(tmp_path):
    values = np.arange(32 * 64, dtype=np.float32).reshape(32, 64)
    entry = S.write_shard(tmp_path / "c0-0.parquet", [values] * 1030, encoding="float32_bss")
    idx = S.finalize_index(tmp_path, seed=3, children=[dict(index=0, seed=3, shards=[entry])])
    assert idx["schema"] == 3 and idx["n_series"] == 1030
    assert all(a.astype("<f4").tobytes() == values.tobytes() for a in S.iter_corpus(tmp_path))


def test_r2_route_is_explicit_and_never_uses_primary_credentials():
    cfg = parse_generation(dict(storage_backend="r2", s3_endpoint="https://account.r2.cloudflarestorage.com",
                                bucket="private-corpora"))
    obj = _GenerationMixin()
    obj.cfg = replace(load_chain_config(), generation=cfg)
    store = obj._corpus_store()
    assert store.cfg.endpoint == cfg.s3_endpoint and store.cfg.bucket == "private-corpora"
    assert store.cfg.access_key_env == "CORPUS_S3_ACCESS_KEY"
    assert store.cfg.secret_key_env == "CORPUS_S3_SECRET_KEY"
    assert obj._corpus_store() is store


@pytest.mark.parametrize("overrides", [dict(bucket=""), dict(s3_endpoint="http://bad"),
    dict(s3_endpoint="https://user:pass@example.com"), dict(s3_endpoint="https://host/?key=secret"),
    dict(access_key_env="actual-secret-value"), dict(secret_key_env="")])
def test_bad_r2_configuration_refuses_instead_of_falling_back(overrides):
    raw = dict(storage_backend="r2", s3_endpoint="https://account.r2.cloudflarestorage.com", bucket="private")
    raw.update(overrides)
    with pytest.raises(ValueError):
        parse_generation(raw)


def receipt(*, hotkey=HK, role="challenger", winner=True, seed=7, ref="r@sha256:" + "a" * 64, block=100):
    return SimpleNamespace(status="scored", generation_seed=seed, round_id="settlement", epoch_start_block=block,
        manifest={"entries": [dict(miner_hotkey=hotkey, gen_ref=ref, role=role)]},
        verdict=SimpleNamespace(inconclusive=False, dethroned=winner, king_hotkey=hotkey if winner else "other"))


def test_verified_winner_publishes_recipe_private_loser_expires_unknown_held(tmp_path):
    src, public = FakeStore(), FakeStore()
    for r, h in [("king", HK), ("loser", "loser"), ("unknown", "unknown"), ("active", "active")]:
        _upload(src, tmp_path, round_id=r, hotkey=h)
    g = GenerationConfig(retention_days=1, sample_shards=1)
    L.apply_verified_receipt(src, public, receipt(), g, now=2000)
    for h in ("loser", "active"):
        L.apply_verified_receipt(src, public, receipt(hotkey=h, winner=False), g, now=2000)
    gone = L.maintain(src, public, receipt(), g, now=1000 + 2 * 86400, keep={("active", "active")})
    assert gone == [("loser", "loser")]
    rows = {r["hotkey"]: r for r in D.read_dataset_index(src)}
    assert rows[HK]["crowned"] and rows[HK]["published"] == "recipe"
    assert rows["unknown"]["expired"] is False and rows["active"]["expired"] is False
    assert len([k for k in public.objs if k.endswith("parquet")]) == 1
    assert set(public.acl.values()) == {"public-read"} and not src.acl


def test_failed_public_copy_retains_crown_and_retries(tmp_path, monkeypatch):
    src, public = FakeStore(), FakeStore()
    _upload(src, tmp_path)
    g = GenerationConfig(retention_days=0, sample_shards=1)
    original = public.put_bytes
    monkeypatch.setattr(public, "put_bytes", lambda *a, **kw: (_ for _ in ()).throw(OSError("down")))
    with pytest.raises(OSError):
        L.apply_verified_receipt(src, public, receipt(), g)
    row = D.read_dataset_index(src)[0]
    assert row["crowned"] and not row["published"]
    assert D.expire_corpora(src, 1e10, g) == []
    monkeypatch.setattr(public, "put_bytes", original)
    L.maintain(src, public, receipt(), g, now=1e10)
    assert D.read_dataset_index(src)[0]["published"] == "recipe"


@pytest.mark.parametrize("kwargs", [dict(seed=999), dict(ref="wrong"), dict(hotkey="wrong")])
def test_receipt_must_match_hotkey_source_and_generation_seed(tmp_path, kwargs):
    src, public = FakeStore(), FakeStore()
    _upload(src, tmp_path)
    assert L.apply_verified_receipt(src, public, receipt(**kwargs), GenerationConfig()) == []
    assert not public.objs and not D.read_dataset_index(src)[0]["crowned"]


def test_old_receipt_and_reregister_cannot_erase_crown_or_settlement(tmp_path):
    src, public = FakeStore(), FakeStore()
    _upload(src, tmp_path)
    L.apply_verified_receipt(src, public, receipt(block=200), GenerationConfig())
    assert L.apply_verified_receipt(src, public, receipt(block=100, winner=False), GenerationConfig()) == []
    _upload(src, tmp_path, now=9000)
    row = D.read_dataset_index(src)[0]
    assert row["crowned"] and row["settlement_block"] == 200 and row["uploaded_at"] == 1000


def test_runtime_invalid_signature_cannot_publish_or_expire(tmp_path, monkeypatch):
    import cascade.shared.receipt as R

    obj = _GenerationMixin()
    cfg = load_chain_config()
    obj.cfg = replace(cfg, generation=replace(cfg.generation, enabled=True))
    obj.manifest_store = lambda: SimpleNamespace(get_text=lambda key: "invalid")
    monkeypatch.setattr(R, "load_receipt", lambda text: receipt())
    monkeypatch.setattr(R, "verify_receipt_signature", lambda rec, anchor: False)
    monkeypatch.setattr(obj, "_corpus_store", lambda: pytest.fail("storage must not be touched"))
    assert obj._maintain_generation_storage() == []


def test_slow_storage_upkeep_does_not_overlap_after_outer_timeout(tmp_path, monkeypatch):
    import threading

    obj = _GenerationMixin()
    cfg = load_chain_config()
    obj.cfg = replace(cfg, generation=replace(cfg.generation, enabled=True))
    entered, release = threading.Event(), threading.Event()
    calls = []
    def slow():
        calls.append(True)
        entered.set()
        assert release.wait(3)
        return []
    monkeypatch.setattr(obj, "_maintain_generation_storage_locked", slow)
    thread = threading.Thread(target=obj._maintain_generation_storage)
    thread.start()
    try:
        assert entered.wait(3)
        assert obj._maintain_generation_storage() == []
        assert calls == [True]
    finally:
        release.set()
        thread.join(timeout=3)
    assert not thread.is_alive()
