"""DEC-CA-0053 ``shards`` corpus mode: a materialised corpus read under the
token budget, digested like the fresh streams, never running miner code."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from cascade.generation import shards as S  # noqa: E402
from cascade.shared.config import load_chain_config  # noqa: E402
from cascade.trainer.stream import (  # noqa: E402
    CORPUS_DIR_ENV,
    element_points,
    open_round_stream,
)


def _corpus(d, n_children=3, per_child=6, seed=7):
    rng = np.random.default_rng(0)
    kids = []
    for i in range(n_children):
        ser = [rng.standard_normal((1 + (j % 2), 64 + 8 * j)) for j in range(per_child + i)]
        kids.append({"index": i, "seed": S.child_seed(seed, i),
                     "shards": [S.write_shard(d / f"c{i}-0.parquet", ser)]})
    return S.finalize_index(d, seed=seed, children=kids)


def _gen_cfg():
    # Archived shard fixtures intentionally exercise the old 1–2 channel rule.
    return dataclasses.replace(load_chain_config().generator, min_channels=1)


def _read(rs):
    return list(rs.series())


def test_full_pass_digest_matches_index(tmp_path):
    idx = _corpus(tmp_path)
    with open_round_stream("shards", tmp_path, 0, _gen_cfg(), token_budget=10**12,
                           corpus_dir=tmp_path) as rs:
        got = _read(rs)
    assert rs.n_series == idx["n_series"] == len(got)
    assert rs.total_points == idx["points"]
    assert rs.digest == idx["corpus_digest"]


@pytest.mark.parametrize("denom", ["points", "series_points"])
def test_budget_stop_digests_the_consumed_prefix(tmp_path, denom):
    _corpus(tmp_path)
    budget = 700
    with open_round_stream("shards", tmp_path, 0, _gen_cfg(), token_budget=budget,
                           corpus_dir=tmp_path, budget_denomination=denom) as rs:
        got = _read(rs)
    pts = [element_points(a, denom) for a in got]
    assert sum(pts) >= budget and sum(pts[:-1]) < budget
    prefix = list(S.iter_corpus(tmp_path))[: len(got)]
    assert rs.digest == S.stream_digest(prefix)[0]
    assert all(np.array_equal(a, b) for a, b in zip(got, prefix, strict=True))


def test_epochs_cycle_at_most_n_passes(tmp_path):
    idx = _corpus(tmp_path)
    big = 10**12
    with open_round_stream("shards", tmp_path, 0, _gen_cfg(), token_budget=big,
                           corpus_dir=tmp_path, corpus_epochs=3) as rs:
        got = _read(rs)
    assert len(got) == 3 * idx["n_series"]
    with open_round_stream("shards", tmp_path, 0, _gen_cfg(),
                           token_budget=idx["points"] + 1, corpus_dir=tmp_path,
                           corpus_epochs=3) as rs:
        got = _read(rs)
    assert idx["n_series"] < len(got) < 2 * idx["n_series"]


def test_corpus_dir_from_env(tmp_path, monkeypatch):
    idx = _corpus(tmp_path)
    monkeypatch.setenv(CORPUS_DIR_ENV, str(tmp_path))
    with open_round_stream("shards", "unused", 0, _gen_cfg(), token_budget=10**12) as rs:
        _read(rs)
    assert rs.digest == idx["corpus_digest"]


def test_missing_corpus_dir_is_not_a_miner_fault(monkeypatch):
    from cascade.trainer.corpus import CorpusError

    monkeypatch.delenv(CORPUS_DIR_ENV, raising=False)
    with pytest.raises(ValueError) as ei:
        open_round_stream("shards", "unused", 0, _gen_cfg(), token_budget=10)
    assert not isinstance(ei.value, CorpusError)


def test_tampered_shard_refused_as_infra(tmp_path):
    from cascade.trainer.corpus import CorpusError

    _corpus(tmp_path)
    p = tmp_path / "c1-0.parquet"
    p.write_bytes(p.read_bytes()[:-8] + b"\x00" * 8)
    with pytest.raises(S.ShardError) as ei:
        open_round_stream("shards", tmp_path, 0, _gen_cfg(), token_budget=10,
                          corpus_dir=tmp_path)
    assert not isinstance(ei.value, CorpusError)


def test_epochs_still_refused_for_live_streams(tmp_path):
    from cascade.trainer.corpus import CorpusError

    with pytest.raises(CorpusError):
        open_round_stream("stream_cpu", tmp_path, 0, _gen_cfg(), token_budget=10,
                          corpus_epochs=2)


def test_context_schedule_applies_at_block():
    t = load_chain_config().training
    assert t.at_block(10**9).context_length == t.context_length       # no schedule
    sched = dataclasses.replace(t, context_length_after=8192, context_length_after_block=100)
    assert sched.at_block(99).context_length == t.context_length
    on = sched.at_block(100)
    assert on.context_length == 8192
    assert on.context_length_after == 0 and on.context_length_after_block == 0
    assert sched.at_block(None).context_length == t.context_length


def test_rederive_shards_entry(tmp_path):
    from cascade.audit.rederive import _rederive_digest

    idx = _corpus(tmp_path)
    cfg = load_chain_config()
    cfg = dataclasses.replace(cfg, generator=_gen_cfg())
    got = _rederive_digest(tmp_path, 0, cfg, mode="shards", token_budget=10**12,
                           corpus_dir=tmp_path)
    assert got == idx["corpus_digest"]


def test_continuation_protocol_accounts_for_last_yield_without_an_extra_read(tmp_path):
    from cascade.trainer.stream import _StreamDigest
    _corpus(tmp_path)
    with open_round_stream('shards', tmp_path, 7, _gen_cfg(), token_budget=100000,
        corpus_dir=tmp_path, continuation_protocol='cascade-continuation/1') as stream:
        iterator = stream.series()
        first = next(iterator)
        expected = _StreamDigest()
        expected.update(first)
        assert stream.n_series == 1 and stream.total_points == element_points(first)
        assert stream.digest == expected.hexdigest()
        iterator.close()


def test_historical_shard_accounting_still_replays_its_original_digest(tmp_path):
    index = _corpus(tmp_path)
    with open_round_stream('shards', tmp_path, 7, _gen_cfg(), token_budget=100000,
                           corpus_dir=tmp_path) as stream:
        iterator = stream.series()
        next(iterator)
        assert stream.n_series == 0
        list(iterator)
        assert stream.digest == index['corpus_digest']


def test_schema1_bytes_match_pre_group_writer_and_none(tmp_path):
    pa, pq = S._pa()
    values = np.arange(8 * 64, dtype=float).reshape(8, 64) / 7
    arrays = [values, values * 2]
    # Frozen pre-change schema-1 writer, same Arrow/runtime/compression.
    table = pa.table({
        'channels': pa.array([8, 8], type=pa.int32()),
        'length': pa.array([64, 64], type=pa.int32()),
        'values': pa.array([np.ascontiguousarray(a.astype('<f4')).tobytes() for a in arrays], type=pa.large_binary()),
    })
    old = tmp_path / 'old.parquet'
    pq.write_table(table, old, compression='zstd')
    bare = S.write_shard(tmp_path / 'bare.parquet', arrays)
    none = S.write_shard(tmp_path / 'none.parquet', [{'values': a, 'groups': None} for a in arrays])
    assert bare['sha256'] == none['sha256'] == S.file_sha256(old)
    assert 'schema' not in bare and 'schema' not in none
    assert old.read_bytes() == (tmp_path / 'bare.parquet').read_bytes() == (tmp_path / 'none.parquet').read_bytes()
    for actual, expected in zip(S.iter_shard(old), arrays, strict=True):
        assert np.array_equal(actual, expected.astype(np.float32).astype(np.float64))


def test_schema2_mixed_rows_roundtrip_and_contract_gating(tmp_path):
    groups = np.array([0, 1, 0, 1, 2, 2, 3, 3], dtype=np.uint8)
    values = np.arange(8 * 64, dtype=np.float32).reshape(8, 64)
    rows = [{'values': values, 'groups': groups}, values + 1, {'values': values + 2, 'groups': None}]
    shard = S.write_shard(tmp_path / 'mixed.parquet', rows)
    assert shard['schema'] == 2
    idx = S.finalize_index(tmp_path, seed=4, children=[{'index': 0, 'seed': 4, 'shards': [shard]}])
    assert S.read_index(tmp_path)['schema'] == idx['schema'] == 2
    read = list(S.iter_corpus(tmp_path))
    assert read[0]['groups'].dtype == np.uint8 and np.array_equal(read[0]['groups'], groups)
    assert np.array_equal(read[0]['values'], values)
    assert isinstance(read[1], np.ndarray) and isinstance(read[2], np.ndarray)
    cfg = dataclasses.replace(_gen_cfg(), min_channels=8, accepted_fields=('groups',))
    with open_round_stream('shards', tmp_path, 4, cfg, token_budget=10**8, corpus_dir=tmp_path) as stream:
        assert len(list(stream.series())) == 3
    assert stream.digest == idx['corpus_digest']
    with open_round_stream('shards', tmp_path, 4, dataclasses.replace(cfg, accepted_fields=()),
                           token_budget=10**8, corpus_dir=tmp_path) as stream, \
            pytest.raises(S.ShardError, match='not accepted'):
        list(stream.series())


def test_shard_writer_and_reader_reject_noncanonical_groups(tmp_path):
    pa, pq = S._pa()
    values = np.zeros((8, 64), dtype=float)
    with pytest.raises(ValueError, match='canonical'):
        S.write_shard(tmp_path / 'bad.parquet', [{'values': values, 'groups': [1]*8}])
    table = pa.table({'channels': [8], 'length': [64],
        'values': [values.astype('<f4').tobytes()], 'groups': [bytes([1]*8)]})
    pq.write_table(table, tmp_path / 'bad.parquet')
    with pytest.raises(S.ShardError, match='canonical'):
        list(S.iter_shard(tmp_path / 'bad.parquet'))
