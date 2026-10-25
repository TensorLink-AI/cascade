"""DEC-CA-0063 paired sweeps ([eval] bench_paired): the checkpoints one
decision needs — a clearer and an unswept king, or the king and the init at
the king check — go through ONE sidecar pass, charged in equal shares to each
member's own allowance and once in full to the settlement cap."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cascade.shared.bench_report import load_bench_report, validator_bench_report_key
from cascade.shared.config.scoring import parse_bench_workers

from .test_validator_bench import (
    CHAL_PTR,
    INIT_PTR,
    KING_PTR,
    _manifest,
    _record,
    _runner,
    _settle,
    _sidecar_report,
    _Store,
    _vcfg,
)


def _cfg(cfg, *, cap=6.0, allowance=2.0, paired=True):
    v = _vcfg(cfg)
    return replace(v, v2=replace(v.v2, bench_budget_hours=cap, bench_sweep_hours=allowance),
                   eval=replace(v.eval, bench_paired=paired))


def _paired(runner, *, hours=None, reports=None):
    """Give ``runner`` a paired sweep; each pass advances the clock by the
    SUM of its members' hours (one pass, sequential cost model)."""
    clock = {"t": 0.0}
    runner.vbench_clock = lambda: clock["t"]
    passes: list[list[str]] = []
    single = runner.bench_sweep_fn
    reports = reports or {}

    def many(entries):
        passes.append([e.trained_pointer for e in entries])
        clock["t"] += sum((hours or {}).get(e.trained_pointer, 0.0) for e in entries) * 3600.0
        return [reports.get(e.trained_pointer, single(e)) for e in entries]

    def solo(entry):
        clock["t"] += (hours or {}).get(entry.trained_pointer, 0.0) * 3600.0
        return single(entry)

    runner.bench_sweep_many_fn = many
    runner.bench_sweep_fn = solo
    return passes, clock


def test_clearer_and_unswept_king_share_one_pass(cfg):
    v = _cfg(cfg)
    store = _Store()
    main, calls = _runner(v, store=store)
    passes, _ = _paired(main)
    out = _settle(main, _manifest(v))
    assert passes == [[CHAL_PTR, KING_PTR]]
    assert sorted(calls) == sorted([CHAL_PTR, KING_PTR])   # once each, in the pass
    assert out.result.challenger_wins_round
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert {e.trained_pointer for e in rep.entries} == {CHAL_PTR, KING_PTR}


def test_paired_verdict_and_report_equal_the_unpaired_ones(cfg):
    outs = []
    for paired in (False, True):
        v = _cfg(cfg, paired=paired)
        store = _Store()
        main, _calls = _runner(v, store=store)
        passes, _ = _paired(main)
        out = _settle(main, _manifest(v))
        assert bool(passes) is paired
        rep = load_bench_report(store.objects[validator_bench_report_key("101")])
        outs.append((out.result.challenger_wins_round, out.transition.note,
                     sorted((e.trained_pointer, e.rows_sha256, e.scores) for e in rep.entries)))
    assert outs[0] == outs[1]


def test_a_paired_pass_is_charged_in_equal_shares(cfg):
    # 3 h pass over two members: 1.5 h of each one's 2 h allowance, 3 h of the cap.
    # The clearer's report fails in the pass; it retries alone inside its
    # remaining 0.5 h (and succeeds) — the king never re-runs.
    v = _cfg(cfg, cap=6.0, allowance=2.0)
    main, calls = _runner(v, store=_Store())
    passes, _ = _paired(main, hours={CHAL_PTR: 1.5, KING_PTR: 1.5},
                        reports={CHAL_PTR: None})
    timeouts = []
    inner = main._vbench_sweep

    def spy(entry, *, require_fev, timeout_s=None):
        timeouts.append(timeout_s)
        return inner(entry, require_fev=require_fev, timeout_s=timeout_s)

    main._vbench_sweep = spy
    main._vbench_begin(_manifest(v))
    out = main.process_round(_manifest(v), windows=[], base_seed=7)
    assert passes == [[CHAL_PTR, KING_PTR]]
    assert timeouts == [int(0.5 * 3600)]                    # the clearer's leftover
    assert calls.count(CHAL_PTR) == 2 and calls.count(KING_PTR) == 1
    assert main._vbench["spent"] == pytest.approx(3.0 * 3600 + 1.5 * 3600)
    assert out.result.challenger_wins_round


def test_pass_timeout_is_the_members_combined_allowance(cfg):
    v = _cfg(cfg, cap=6.0, allowance=2.0)
    main, _calls = _runner(v, store=_Store())
    _paired(main)
    seen = {}
    main.bench_sweep_many_fn = None

    def run_many(entries, *, require_fev, timeout_s=None):
        seen["n"], seen["timeout"] = len(entries), timeout_s
        return [_sidecar_report(1.0 if e.trained_pointer == KING_PTR else 0.9) for e in entries]

    main.bench_sweep_fn = None
    main._vbench_run_sidecar_many = run_many
    _settle(main, _manifest(v))
    assert seen == {"n": 2, "timeout": 4 * 3600}


def test_no_pass_when_the_cap_cannot_hold_both_allowances_or_the_king_is_cached(cfg):
    v = _cfg(cfg, cap=3.0, allowance=2.0)                  # 2 + 2 > 3: the clearer defers
    main, calls = _runner(v, store=_Store())
    passes, _ = _paired(main)
    out = _settle(main, _manifest(v))
    assert passes == [] and calls == [] and out.result.inconclusive
    # king cached from an earlier settlement: the clearer is swept alone
    v2 = _cfg(cfg)
    main2, calls2 = _runner(v2, store=_Store())
    passes2, _ = _paired(main2)
    main2._vbench_cache_put(KING_PTR, {k: 1.0 for k in (
        "gifteval_crps", "gifteval_mase", "boom_crps", "boom_mase", "time_crps", "time_mase")},
        {"gift": {"status": "ok", "rows": []}})
    _settle(main2, _manifest(v2))
    assert passes2 == [] and calls2 == [CHAL_PTR]


def test_king_check_pairs_king_and_init(cfg):
    v = _cfg(cfg)
    store = _Store()
    main, calls = _runner(v, store=store)
    passes, _ = _paired(main)
    m = _manifest(v)
    main._vbench_begin(m)
    rec = {"reasons": [], "bench_round": ""}
    king = m.entry_for_role("king")
    got = main._v2_king_check_validator_bench(m, king, KING_PTR, rec)
    assert passes == [[KING_PTR, INIT_PTR]]
    assert got is not None and got[2] == INIT_PTR
    main._vbench_publish(m)
    # every other validator reads the same numbers from the signed report — no sweep
    other, other_calls = _runner(v, store=store, signer="5Other")
    _record(other, store)
    other._vbench_begin(m)
    assert other._v2_king_check_validator_bench(m, king, KING_PTR,
                                                {"reasons": [], "bench_round": ""}) == got
    assert other_calls == []


def test_parse_bench_workers():
    assert parse_bench_workers("auto") == "auto" and parse_bench_workers(" Auto") == "auto"
    assert parse_bench_workers(4) == "4" and parse_bench_workers("2") == "2"
    for bad in (0, "0", -2, "many", True, 1.5):
        with pytest.raises(ValueError):
            parse_bench_workers(bad)
