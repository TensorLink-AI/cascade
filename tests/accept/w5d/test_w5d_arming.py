"""W5d (plan rev6.6, owner 2026-11-23; §1.1 context bullet, §1.7 readiness,
T7 row): per-variant arming and the context an entry is scored at.
Contract: README.md "Arming (rev6.6)".

Every variant in ``[lane_d] contexts`` is declarable from launch. A variant
is ARMED at a settlement only when both gates pass:

1. readiness: at least ``min_long_series`` series from at least
   ``GATE_MIN_SOURCES`` (20) sources can use it (the W1/R16 per-rung counts,
   a ``laned_readiness/1`` document);
2. T7: its wall time and memory fit the settlement budget on the eval host
   (a ``laned_t7/1`` pod document, ``fits`` recomputed, never trusted).

``armed_record`` is a pure function of the pinned document BYTES (their
sha256s are the evidence the signed report carries); ``effective_context``
scores an entry at its declared context when armed, else at the highest
armed variant. Numpy-free and torch-free: nothing here imports torch.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

ALL_CONTEXTS = (4096, 8192, 16384, 32768, 65536, 131072)
MIN_LONG_SERIES = 300
MIN_SOURCES = 20
BUDGET = {"wall_s": 5400.0, "mem_gb": 24.0}


def _s():
    import cascade.eval.lane_d_scorer as s

    missing = [n for n in ("armed_record", "effective_context", "GATE_MIN_SOURCES",
                           "READINESS_SCHEMA") if not hasattr(s, n)]
    assert not missing, f"rev6.6: cascade.eval.lane_d_scorer lacks {missing}"
    return s


def _bytes(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _readiness(counts: dict[int, tuple[int, int]]) -> dict:
    return {"schema": "laned_readiness/1",
            "rungs": [{"context": c, "series": n, "sources": k}
                      for c, (n, k) in sorted(counts.items())]}


READY_ALL = {c: (4633, 42) for c in ALL_CONTEXTS}


def _row(variant: int, *, wall_s: float = 1200.0, mem: float = 10.0,
         source: str = "cuda_max_allocated") -> dict:
    fits = wall_s <= BUDGET["wall_s"] and mem <= BUDGET["mem_gb"]
    return {"host": "valeval-laned" if variant > 16384 else "valpriv-4090",
            "gpu": "RTX 4090", "variant": variant, "input_len": variant,
            "history_min": variant, "history_max": variant + 64, "wall_s": wall_s,
            "peak_mem_gb": mem, "peak_mem_source": source, "budget": dict(BUDGET),
            "fits": fits, "n_models": 10, "n_windows": 40}


def _t7(rows: list[dict], *, source: str = "pod") -> dict:
    armed = sorted(r["variant"] for r in rows if r["fits"])
    return {"schema": "laned_t7/1", "wp": "W5d", "source": source, "torch": "2.4.1+cu124",
            "tool": {"git_sha": "1" * 39 + "2", "dirty": False},
            "measured_at": "2026-11-23T00:00:00Z", "rows": rows,
            "short_window": {"base_context": 4096, "declared_context": 131072,
                             "n_windows": 12, "max_abs_diff": 0.0, "identical": True},
            "armed": armed, "passed": 4096 in armed}


T7_ALL = _t7([_row(c) for c in ALL_CONTEXTS])


def _record(readiness=READY_ALL, t7=T7_ALL, contexts=ALL_CONTEXTS, mls=MIN_LONG_SERIES):
    s = _s()
    rb = None if readiness is None else _bytes(_readiness(readiness))
    tb = None if t7 is None else _bytes(t7)
    return s.armed_record(contexts, readiness=rb, t7=tb, min_long_series=mls)


# ── constants and the record's shape ─────────────────────────────────────────


def test_gate_constants():
    s = _s()
    assert s.GATE_MIN_SOURCES == MIN_SOURCES
    assert s.READINESS_SCHEMA == "laned_readiness/1"


def test_record_shape_and_evidence_sha256s():
    rb, tb = _bytes(_readiness(READY_ALL)), _bytes(T7_ALL)
    rec = _s().armed_record(ALL_CONTEXTS, readiness=rb, t7=tb, min_long_series=MIN_LONG_SERIES)
    assert set(rec) == {"armed_contexts", "evidence"}
    ev = rec["evidence"]
    assert set(ev) == {"min_long_series", "min_sources", "readiness_sha256", "t7_sha256",
                       "variants"}
    assert ev["min_long_series"] == MIN_LONG_SERIES and ev["min_sources"] == MIN_SOURCES
    assert ev["readiness_sha256"] == hashlib.sha256(rb).hexdigest()
    assert ev["t7_sha256"] == hashlib.sha256(tb).hexdigest()
    assert [v["context"] for v in ev["variants"]] == list(ALL_CONTEXTS)
    for v in ev["variants"]:
        assert set(v) == {"context", "series", "sources", "t7_fits", "armed"}
        assert (v["series"], v["sources"]) == READY_ALL[v["context"]]
        assert v["t7_fits"] is True and v["armed"] is True
    assert rec["armed_contexts"] == list(ALL_CONTEXTS)
    json.dumps(rec, allow_nan=False)


def test_record_is_deterministic_and_sorted():
    a = _record()
    b = _record(contexts=tuple(reversed(ALL_CONTEXTS)))
    assert a == b
    assert _bytes(a) == _bytes(_record())


# ── the two gates ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("ready", "fits", "armed"), [
    (True, True, True), (True, False, False), (False, True, False), (False, False, False)])
def test_a_variant_arms_only_when_both_gates_pass(ready, fits, armed):
    counts = dict(READY_ALL)
    counts[32768] = (4633, 42) if ready else (12, 3)
    rows = [_row(c) for c in ALL_CONTEXTS if c != 32768]
    rows.append(_row(32768, wall_s=1200.0 if fits else 9000.0))
    rec = _record(readiness=counts, t7=_t7(rows))
    v = {x["context"]: x for x in rec["evidence"]["variants"]}[32768]
    assert v["t7_fits"] is fits and v["armed"] is armed
    assert (32768 in rec["armed_contexts"]) is armed


@pytest.mark.parametrize(("series", "sources", "armed"), [
    (MIN_LONG_SERIES, MIN_SOURCES, True),
    (MIN_LONG_SERIES - 1, MIN_SOURCES, False),
    (MIN_LONG_SERIES, MIN_SOURCES - 1, False),
    (10_000, MIN_SOURCES - 1, False)])
def test_readiness_gate_edges(series, sources, armed):
    counts = dict(READY_ALL)
    counts[65536] = (series, sources)
    rec = _record(readiness=counts)
    assert (65536 in rec["armed_contexts"]) is armed


def test_min_long_series_is_the_configured_floor():
    counts = dict(READY_ALL)
    counts[131072] = (400, 25)
    assert 131072 in _record(readiness=counts, mls=300)["armed_contexts"]
    rec = _record(readiness=counts, mls=500)
    assert 131072 not in rec["armed_contexts"]
    assert rec["evidence"]["min_long_series"] == 500


def test_t7_memory_over_budget_does_not_arm():
    rows = [_row(c) for c in ALL_CONTEXTS if c != 16384] + [_row(16384, mem=30.0)]
    rec = _record(t7=_t7(rows))
    assert 16384 not in rec["armed_contexts"]


def test_a_variant_without_a_rung_or_a_t7_row_is_not_armed():
    counts = {c: v for c, v in READY_ALL.items() if c != 65536}
    rows = [_row(c) for c in ALL_CONTEXTS if c != 131072]
    rec = _record(readiness=counts, t7=_t7(rows))
    v = {x["context"]: x for x in rec["evidence"]["variants"]}
    assert v[65536]["series"] == 0 and v[65536]["sources"] == 0 and not v[65536]["armed"]
    assert v[131072]["t7_fits"] is False and not v[131072]["armed"]
    assert rec["armed_contexts"] == [4096, 8192, 16384, 32768]


def test_t7_from_ci_or_process_rss_never_arms():
    """Only pod evidence with each variant's own CUDA peak is a budget input."""
    rec = _record(t7=_t7([_row(c) for c in ALL_CONTEXTS], source="ci"))
    assert rec["armed_contexts"] == []
    rows = [_row(c) for c in ALL_CONTEXTS if c != 8192]
    rows.append(_row(8192, source="process_peak_rss"))
    assert 8192 not in _record(t7=_t7(rows))["armed_contexts"]


def test_absent_documents_arm_nothing():
    rec = _record(readiness=None, t7=None)
    assert rec["armed_contexts"] == []
    assert rec["evidence"]["readiness_sha256"] is None and rec["evidence"]["t7_sha256"] is None
    assert all(not v["armed"] for v in rec["evidence"]["variants"])
    assert _record(t7=None)["armed_contexts"] == []
    assert _record(readiness=None)["armed_contexts"] == []


def test_a_stated_fits_that_disagrees_with_its_numbers_is_refused():
    t7 = copy.deepcopy(T7_ALL)
    t7["rows"][3]["wall_s"] = 9000.0          # fits stays True: a hand edit
    with pytest.raises(ValueError):
        _record(t7=t7)


@pytest.mark.parametrize("bad", [
    {"schema": "laned_readiness/2", "rungs": []},
    {"schema": "laned_readiness/1"},
    {"schema": "laned_readiness/1", "rungs": [{"context": 4096, "series": -1, "sources": 2}]},
    {"schema": "laned_readiness/1", "rungs": [{"context": 4096, "series": True, "sources": 2}]},
    {"schema": "laned_readiness/1", "rungs": [{"context": 4000, "series": 1, "sources": 2}]},
    {"schema": "laned_readiness/1", "rungs": [{"context": 4096, "series": 1, "sources": 2},
                                              {"context": 4096, "series": 5, "sources": 2}]},
])
def test_malformed_readiness_is_refused(bad):
    with pytest.raises(ValueError):
        _s().armed_record(ALL_CONTEXTS, readiness=_bytes(bad), t7=_bytes(T7_ALL),
                          min_long_series=MIN_LONG_SERIES)


def test_malformed_documents_are_refused():
    s = _s()
    for rb, tb in ((b"{not json", _bytes(T7_ALL)),
                   (_bytes(_readiness(READY_ALL)), b"[]"),
                   (_bytes(_readiness(READY_ALL)), _bytes({**T7_ALL, "schema": "laned_t7/0"}))):
        with pytest.raises(ValueError):
            s.armed_record(ALL_CONTEXTS, readiness=rb, t7=tb, min_long_series=MIN_LONG_SERIES)


def test_contexts_outside_the_scorer_set_are_refused():
    s = _s()
    for bad in ((4096, 262144), (4000,), ()):
        with pytest.raises(ValueError):
            s.armed_record(bad, readiness=_bytes(_readiness(READY_ALL)), t7=_bytes(T7_ALL),
                           min_long_series=MIN_LONG_SERIES)


# ── the effective (scored) context ───────────────────────────────────────────


def test_effective_context_is_the_declared_one_when_armed():
    ec = _s().effective_context
    assert ec(8192, (4096, 8192, 16384)) == 8192
    assert ec(4096, [4096]) == 4096


def test_an_unarmed_declaration_is_scored_at_the_highest_armed_variant():
    ec = _s().effective_context
    assert ec(131072, (4096, 8192, 16384)) == 16384
    assert ec(32768, [4096]) == 4096


def test_never_scored_above_the_declared_context():
    """The highest armed variant is taken among those not above the declared
    context (a model declared at 32k is never run at 64k); only when every
    armed variant is above it does the lowest armed one score it."""
    ec = _s().effective_context
    assert ec(32768, (4096, 8192, 16384, 65536, 131072)) == 16384
    assert ec(65536, (4096, 131072)) == 4096
    assert ec(4096, (8192, 16384)) == 8192


def test_the_scored_context_switches_when_its_variant_arms():
    counts = {c: (4633, 42) for c in (4096, 8192, 16384)}
    counts[32768] = (250, 30)                 # below min_long_series
    day1 = _record(readiness=counts)
    counts[32768] = (310, 30)                 # crossed it at the next settlement
    day2 = _record(readiness=counts)
    ec = _s().effective_context
    assert 32768 not in day1["armed_contexts"] and 32768 in day2["armed_contexts"]
    assert ec(32768, day1["armed_contexts"]) == 16384
    assert ec(32768, day2["armed_contexts"]) == 32768


def test_effective_context_refusals():
    ec = _s().effective_context
    with pytest.raises(ValueError):
        ec(4096, ())                          # nothing armed: nothing can be scored
    for bad in (5000, 262144, True, "4096"):
        with pytest.raises(ValueError):
            ec(bad, (4096, 8192))
