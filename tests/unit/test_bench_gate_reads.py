"""DEC-CA-0054: a bench-row READ failure retries; only absence is a verdict."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cascade.shared.hippius import ObjectNotFound, StorageError
from cascade.validator.evaluation import EvaluationMixin

ENTRY = SimpleNamespace(trained_pointer="P", miner_hotkey="5Fchal")


def _me(get_text):
    e = SimpleNamespace(rows_key="rows/P.json", rows_sha256="0" * 64)
    report = SimpleNamespace(entry_for_pointer=lambda p: e)
    calls = []

    def fetch(rid, *, require=None, raise_unavailable=False):
        calls.append(raise_unavailable)
        return report

    return SimpleNamespace(_fetch_bench_report=fetch,
                           _bench_store=lambda: SimpleNamespace(get_text=get_text)), calls


def test_a_store_outage_raises_for_a_retry():
    def down(key):
        raise ConnectionError("503")

    me, calls = _me(down)
    with pytest.raises(StorageError, match="bench_rows_unavailable"):
        EvaluationMixin._bench_gate_side(me, "1", ENTRY)
    assert calls == [True]                       # the report read raises too


def test_absent_rows_are_uncomputable():
    def gone(key):
        raise ObjectNotFound(key)

    me, _ = _me(gone)
    assert EvaluationMixin._bench_gate_side(me, "1", ENTRY) == (None, None)
