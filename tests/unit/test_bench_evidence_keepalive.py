"""Bench-sweep evidence + the weight keep-alive during a long sweep.

2026-11-16: a pool clearer was held INFRA twice on ``gift-eval:partial`` and
the skipped configs were lost (the pod run dir is deleted after the report is
read); and every multi-hour bench sweep starved the weight loop (no weight
set for 2–3 h). Both fixes are operator-local — no consensus effect.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import pytest

from cascade.validator import eval_offload as eo
from cascade.validator.loop import ValidatorRunner
from cascade.validator.state import genesis

KING = "5KingHotkey"


# ── evidence ────────────────────────────────────────────────────────────────

def _report(gift_skipped=(), gift_status="ok", fev_scope=False):
    fev_skips = [{"full": "m/x", "stage": "scope", "reason": "multivariate"}] if fev_scope else []
    return {"suites": [
        {"suite": "gift-eval", "status": gift_status, "skipped": list(gift_skipped),
         "detail": "boom\nsecond line" if gift_status != "ok" else ""},
        {"suite": "fev", "status": "ok", "skipped": fev_skips},
    ]}


def test_problems_clean_report_and_fev_scope_ignored():
    assert eo.bench_report_problems(_report(fev_scope=True)) == []


def test_problems_name_skips_and_status():
    skip = {"full": "solar/10T/long", "stage": "score", "reason": "CUDA OOM"}
    p = eo.bench_report_problems(_report(gift_skipped=[skip]))
    assert p == ["gift-eval: skipped solar/10T/long at score: CUDA OOM"]
    p = eo.bench_report_problems(_report(gift_status="error"))
    assert p == ["gift-eval: status=error (boom)"]
    assert eo.bench_report_problems(None) == ["no report"]
    assert eo.bench_report_problems({"suites": "garbage"})  # never raises


def test_record_evidence_writes_and_prunes(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(eo, "BENCH_EVIDENCE_KEEP", 3)
    skip = {"full": "a/D/short", "stage": "load", "reason": "download failed"}
    proc = subprocess.CompletedProcess([], 0, "", "gift-eval: a/D/short not loaded\n")
    with caplog.at_level("WARNING", logger="cascade.validator.eval_offload"):
        path = eo.record_bench_evidence("challenger-5G6j", reports=[_report([skip]), None],
                                        proc=proc, root=tmp_path)
    data = json.loads(path.read_text())
    assert data["returncode"] == 0 and "not loaded" in data["stderr"]
    assert data["problems"][0].startswith("report 0: gift-eval: skipped a/D/short at load")
    assert data["problems"][1] == "report 1: no report"
    assert "a/D/short" in caplog.text and "stderr tail" in caplog.text
    for i in range(5):
        eo.record_bench_evidence(f"x{i}", reports=[_report()], root=tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_record_evidence_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert eo.record_bench_evidence("x", reports=[None], root=blocker / "sub") is None


# ── run_ssh_ticking ─────────────────────────────────────────────────────────

def test_ticking_runs_and_ticks():
    ticks = []
    proc = eo.run_ssh_ticking(
        [sys.executable, "-c", "import time,sys; time.sleep(0.5); print('out'); "
         "print('err', file=sys.stderr)"], 30, lambda: ticks.append(1), every=0.05)
    assert proc.returncode == 0 and proc.stdout.strip() == "out" and "err" in proc.stderr
    assert len(ticks) >= 3


def test_ticking_tick_error_is_ignored():
    def bad():
        raise RuntimeError("chain down")
    proc = eo.run_ssh_ticking([sys.executable, "-c", "import time; time.sleep(0.2)"],
                              30, bad, every=0.05)
    assert proc.returncode == 0


def test_ticking_timeout_kills():
    with pytest.raises(subprocess.TimeoutExpired):
        eo.run_ssh_ticking([sys.executable, "-c", "import time; time.sleep(30)"],
                           0.3, lambda: None, every=0.05)


# ── keep-alive ──────────────────────────────────────────────────────────────

class _Client:
    def __init__(self, block=1000, hotkeys=None):
        self.block = block
        self.hotkeys = hotkeys or [f"hk{i}" for i in range(256)]
        self.equal_calls: list[list[int]] = []
        self.vec_calls: list[list[float]] = []
        self.accept = True

    def current_block(self):
        return self.block

    def n_uids(self):
        return len(self.hotkeys)

    def registered_hotkeys(self):
        return list(self.hotkeys)

    def uid_for_hotkey(self, hotkey):
        return self.hotkeys.index(hotkey) if hotkey in self.hotkeys else None

    def set_equal_share_weights(self, reward_uids, n_uids, *, decay, burn_uid):
        self.equal_calls.append(list(reward_uids))

    def set_weight_vector(self, weights):
        self.vec_calls.append(list(weights))
        return self.accept


def _runner(cfg, **validator):
    v = replace(cfg.validator, weight_set_interval_blocks=180, **validator)
    return ValidatorRunner(cfg=replace(cfg, validator=v), state=genesis(KING, 7))


def _pushed(cfg):
    client = _Client()
    r = _runner(cfg)
    vec = r._apply_weights(client, "round-x", [7])
    assert vec and r._last_weight_block == 1000
    return r, client, vec


def test_keepalive_noop_before_first_push(cfg):
    r = _runner(cfg)
    r._weights_keepalive_tick()  # no client, no vector: nothing happens


def test_keepalive_waits_for_interval_then_repushes_same_vector(cfg):
    r, client, vec = _pushed(cfg)
    client.block = 1100
    r._weights_keepalive_tick()
    assert client.vec_calls == []
    client.block = 1180
    r._weights_keepalive_tick()
    assert client.vec_calls == [list(vec)]
    assert r._last_weight_block == 1180
    client.block = 1200
    r._weights_keepalive_tick()  # fresh again: no second push
    assert len(client.vec_calls) == 1


def test_keepalive_skips_when_paid_uid_changed_hands(cfg):
    r, client, _ = _pushed(cfg)
    client.hotkeys[7] = "someone-new"
    client.block = 1200
    r._weights_keepalive_tick()
    assert client.vec_calls == [] and r._last_pushed is None


def test_keepalive_skips_when_uid_count_changed(cfg):
    r, client, _ = _pushed(cfg)
    client.hotkeys.append("extra")
    client.block = 1200
    r._weights_keepalive_tick()
    assert client.vec_calls == []


def test_keepalive_rejected_retries_later(cfg):
    r, client, _ = _pushed(cfg)
    client.accept = False
    client.block = 1200
    r._weights_keepalive_tick()
    assert len(client.vec_calls) == 1 and r._last_weight_block == 1000
    client.block = 1210
    r._weights_keepalive_tick()  # inside the retry gap
    assert len(client.vec_calls) == 1
    client.accept = True
    client.block = 1200 + 25
    r._weights_keepalive_tick()
    assert len(client.vec_calls) == 2 and r._last_weight_block == 1225


def test_keepalive_off_under_force_burn(cfg):
    r, client, _ = _pushed(cfg)
    r.cfg = replace(r.cfg, validator=replace(r.cfg.validator, force_burn=True))
    client.block = 1300
    r._weights_keepalive_tick()
    assert client.vec_calls == []


def test_failed_push_records_nothing(cfg):
    class _Fails(_Client):
        def set_equal_share_weights(self, *a, **k):
            return False
    client = _Fails()
    r = _runner(cfg)
    assert not r._apply_weights(client, "round-x", [7])
    assert getattr(r, "_last_pushed", None) is None
