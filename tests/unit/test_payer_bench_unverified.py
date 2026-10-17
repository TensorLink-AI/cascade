"""[telemetry] funded_bench_unverified (owner 2026-10-17): payer-pod bench
numbers publish as reported, without the operator's king-pod re-bench, and the
published report flags them OUTSIDE the signed body."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

from cascade.shared.bench_report import (
    BenchEntry,
    BenchReport,
    dump_bench_report,
    load_bench_report,
)
from cascade.shared.config import load_chain_config
from cascade.trainer.loop import TrainerRunner
from tests.unit.test_verify_deferral import ERA, SCORES, _entry, _fake

REPO = Path(__file__).resolve().parents[2]


def test_loader_parses_the_knob_default_off(tmp_path):
    text = (REPO / "chain.toml").read_text(encoding="utf-8")
    assert load_chain_config(REPO / "chain.toml").telemetry.funded_bench_unverified is False
    armed = text.replace("funded_bench_verify_tolerance = 0.02",
                         "funded_bench_verify_tolerance = 0.02\nfunded_bench_unverified = true", 1)
    assert armed != text
    p = tmp_path / "chain.toml"
    p.write_text(armed, encoding="utf-8")
    assert load_chain_config(p).telemetry.funded_bench_unverified is True


def _unverified_fake(tmp_path):
    f = _fake()
    f.cfg.telemetry.funded_bench_unverified = True
    f.work_root = tmp_path
    f.noted = []
    f._note_payer_reported = lambda p: f.noted.append(p)
    return f


def test_rolling_payer_bench_publishes_as_reported_without_a_rebench(tmp_path):
    f = _unverified_fake(tmp_path)
    alfa = _entry("ALFA", "challenger")
    out = TrainerRunner._rolling_bench_challenger(f, alfa, _entry("KING", "king"), ERA)
    assert out == dataclasses.asdict(SCORES)          # the payer's own numbers, no VERIFY_PENDING
    assert f.verify_calls == [] and f.noted == [alfa.trained_pointer]
    # also with no era king leg yet: nothing waits for one
    assert TrainerRunner._rolling_bench_challenger(f, alfa, None, ERA) == dataclasses.asdict(SCORES)


def _runner_ns(tmp_path):
    ns = SimpleNamespace(work_root=tmp_path)
    for name in ("_payer_reported_path", "_payer_reported_pointers", "_note_payer_reported"):
        setattr(ns, name, getattr(TrainerRunner, name).__get__(ns))
    return ns


def _report(*pointers):
    return BenchReport(round_id="9", created_block=100, entries=tuple(
        BenchEntry(role="challenger", size="toto2-4m", miner_hotkey=f"hk{i}", miner_uid=i,
                   trained_pointer=p, scores=SCORES) for i, p in enumerate(pointers)),
        signature="ab" * 32)


def test_provenance_rides_outside_the_signed_body(tmp_path):
    ns = _runner_ns(tmp_path)
    rep = _report("ptr-a", "ptr-b")
    # nothing payer-reported ⇒ byte-identical to the plain dump
    assert TrainerRunner._dump_bench_with_provenance(ns, rep) == dump_bench_report(rep)
    ns._note_payer_reported("ptr-b")
    ns._note_payer_reported("ptr-b")                  # idempotent
    ns._note_payer_reported("ptr-z")                  # not in this report
    text = TrainerRunner._dump_bench_with_provenance(ns, rep)
    obj = json.loads(text)
    assert obj["payer_reported_pointers"] == ["ptr-b"]
    # every loader still parses it, and the signed payload is unchanged
    back = load_bench_report(text)
    assert back.canonical_body() == rep.canonical_body()
    assert back.signature == rep.signature


def test_legacy_sweep_publishes_payer_numbers_as_reported(tmp_path):
    ns = SimpleNamespace(cfg=SimpleNamespace(telemetry=SimpleNamespace(
        funded_bench_unverified=True, funded_bench_verify_top=1,
        funded_bench_verify_tolerance=0.02)), _payer_benched={"ptr-a"}, noted=[])
    ns._note_payer_reported = lambda p: ns.noted.append(p)
    scored = {"ptr-k": SCORES, "ptr-a": SCORES}
    duel = [SimpleNamespace(trained_pointer="ptr-k", role="king"),
            SimpleNamespace(trained_pointer="ptr-a", role="challenger")]
    out = TrainerRunner._verify_payer_benches(ns, duel, scored, "9", "toto2-4m")
    assert out == scored and ns.noted == ["ptr-a"]
