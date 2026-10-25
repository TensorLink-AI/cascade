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
