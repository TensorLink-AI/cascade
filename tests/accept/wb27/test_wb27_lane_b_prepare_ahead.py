"""2026-11-27 (owner): "have the lane B verdict prepared as soon as a leg
finishes, instead of waiting for settlements". A leg uploaded 13:47 waited
through the 16:27 (eval-pod disk) and 19:27 (pod reply not JSON) settlements,
and the 22:27 one would only START the candidate bench.

* prepare-ahead + settle == settle-only (published expert manifest bytes and
  the committed driver state), and the settlement then sweeps NOTHING;
* a failed prepare step retries with backoff within minutes, bounded; its
  first failure is an ERROR and ``lane_b.prepare`` in status/v2.json;
* the bench method (#973) is part of the row-cache key, and a mixed-method
  pair never pairs.

Torch-free: the eval pod is tests/unit/test_lane_b_offload.py's FakePod (the
worker step runs pod_main in-process; settle_leg / merge_check_dirs stubbed)."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit import test_lane_b_offload as _offload_tests
from tests.unit.test_lane_b_offload import KING, _push, _settle_driver


@pytest.fixture
def pod_env(tmp_path, monkeypatch):
    """tests/unit/test_lane_b_offload.py's fake eval pod + Hub registry."""
    return _offload_tests.pod_env.__wrapped__(tmp_path, monkeypatch)


def _prep_mod():
    try:
        return importlib.import_module("cascade.trainer.lane_b_prepare")
    except ImportError:
        pytest.fail("cascade.trainer.lane_b_prepare is missing: the lane B verdict is only "
                    "prepared inside the settlement")


SUITES = ("gift-eval", "boom", "time", "fev")


def _report_of(weights: bytes, method: str | None = None) -> dict:
    h = int.from_bytes(hashlib.sha256(weights).digest()[:4], "little")
    rep = {"suites": [{"suite": s, "status": "ok",
                       "rows": [{"full": f"{s}/i{i}", "CRPS": 0.3 + ((h + i) % 97) / 1000,
                                 "MASE": 1.0 + ((h * 3 + i) % 89) / 1000}
                                for i in range(12)]} for s in SUITES],
           "data_revisions": {s: "r1" for s in SUITES}}
    if method:
        rep["bench_method"] = method
    return rep


class Sweeps:
    """The pinned sidecar sweep: rows are a function of the pinned bytes."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, host, pin, *, settings, workers, timeout_s, label):
        self.calls.append(label)
        return _report_of((Path(pin) / "weights.safetensors").read_bytes())


class Ops:
    def __init__(self, model, bench, prep):
        self.model, self.bench, self.prep, self.published = model, bench, prep, []

    def lane_b_model(self):
        return self.model

    def prebench(self, pointer):
        self.bench.prebench(pointer)

    def prepare_ahead(self, specs, paused=None):
        if self.prep is not None:
            self.prep.submit(specs, paused=paused)

    def prepare_status(self):
        return self.prep.status() if self.prep is not None else {}

    def publish_expert_manifest(self, m):
        self.published.append(m.to_json())

    def judge(self, m, round_id, block, **kw):
        return []

    def commitments(self):
        return []


def _world(env, monkeypatch, sub: str, *, prepare: bool):
    """One trainer (driver + bench + pod model) over the shared fake pod."""
    from cascade.shared.manifest import format_trained_pointer
    from cascade.trainer import lane_b_offload as off
    from cascade.trainer.lane_b_bench import LaneBBench
    from cascade.validator import expert_lane as vel

    monkeypatch.setattr(vel, "merge_check_dirs", lambda kd, cd, sel, strict: None)
    root = env.tmp / sub
    root.mkdir()
    runner = env.pod.runner()
    sweeps = Sweeps()
    bench = LaneBBench(hosts_path=root / "targon-eval.toml", cache_dir=root / "rows",
                       eval_cfg=SimpleNamespace(cascade_bench_max_series=0,
                                                gift_gate_data_dir="/data/bench",
                                                bench_workers="1"),
                       start_worker=False, pod=runner, pinned_bench_fn=sweeps)
    prep = None
    if prepare:
        tick = iter(range(0, 10**9, 30))           # the worker's wall clock
        prep = _prep_mod().LaneBPrepare(runner, bench, start_worker=False,
                                        clock=lambda: float(next(tick)))
    model = off.PodLaneBModel(runner, bench, _push)
    ops = Ops(model, bench, prep)
    king = format_trained_pointer(KING)
    d = root / "drv"
    d.mkdir()
    drv = _settle_driver(d, ops, king=king)
    return SimpleNamespace(drv=drv, ops=ops, bench=bench, prep=prep, sweeps=sweeps,
                           king=king, root=root)


def _drain(bench) -> int:
    n = 0
    while not bench._q.empty():
        tag, item = bench._q.get()
        run = getattr(bench, "run_tagged", None)
        if run is not None:
            run(tag, item)
        else:                                        # the pre-prepare worker body
            bench.run_one(item)
            bench._pending.discard(tag)
        n += 1
    return n


def _settle(w, rid="7301", block=20):
    return w.drv.settle(round_id=rid, block=block, era_index=0, king_pointer=w.king,
                        work_dir=w.root / "work")


def _committed(w) -> dict:
    doc = json.loads(Path(w.drv.path).read_text(encoding="utf-8"))
    doc.pop("heartbeat", None)
    return doc


def test_prepare_ahead_then_settle_is_byte_identical_to_settle_only(pod_env, monkeypatch):
    env = pod_env
    # ── today: settle-only (each missing bench costs a settlement) ──────────
    a = _world(env, monkeypatch, "settle-only", prepare=False)
    tries = 0
    while not a.ops.published:
        tries += 1
        assert tries <= 4
        with contextlib.suppress(OSError):           # rows queued: fails closed
            _settle(a)
        _drain(a.bench)
    assert tries == 3                                # king, then candidate, then publish
    # ── prepare-ahead: the leg's steps run before the settlement ────────────
    b = _world(env, monkeypatch, "prepare-ahead", prepare=True)
    b.ops.prepare_ahead(b.drv.prepare_specs(20, 0, b.king), paused=b.drv._settling.is_set)
    for _ in range(12):
        b.prep.run_due()
        _drain(b.bench)
    st = b.prep.status()
    by_leg = {x["leg"]: x for x in st["legs"]}
    assert by_leg["alice/energy"]["state"] == "done", st
    assert all(by_leg["alice/energy"]["steps"][s]["state"] == "done"
               for s in ("king_rows", "rebase", "candidate_rows"))
    assert by_leg["bob/web"]["steps"]["rebase"]["state"] == "refused"   # the merge guard
    swept_before = len(b.sweeps.calls)
    _settle(b)                                       # ONE settlement publishes
    assert b.ops.published, "the prepared settlement did not publish at once"
    assert len(b.sweeps.calls) == swept_before, "the settlement re-benched a prepared row"
    assert b.bench.pending() == 0 and b.bench._q.empty()
    # ── the verdict is byte-identical ───────────────────────────────────────
    assert b.ops.published[-1] == a.ops.published[-1]
    assert _committed(b) == _committed(a)


def test_specs_are_the_settlement_rebase_inputs(pod_env, monkeypatch):
    w = _world(pod_env, monkeypatch, "specs", prepare=True)
    specs = w.drv.prepare_specs(20, 0, w.king)
    assert [s["label"] for s in specs] == ["alice/energy", "bob/web"]
    for f, s in zip(w.drv.st.finished, specs, strict=True):
        rin = w.drv.rebase_inputs(f, 20, w.king)
        assert {k: s[k] for k in rin} == rin
        assert s["merge_strict"] is True                 # merge_guard_from_block = 1


def test_the_settlement_pauses_the_worker(pod_env, monkeypatch):
    w = _world(pod_env, monkeypatch, "pause", prepare=True)
    seen = []
    w.ops.prepare_ahead(w.drv.prepare_specs(20, 0, w.king), paused=w.drv._settling.is_set)
    real_judge = w.ops.judge

    def judge(m, rid, block, **kw):
        seen.append((w.drv._settling.is_set(), w.prep.run_due()))
        return real_judge(m, rid, block, **kw)

    w.ops.judge = judge
    for _ in range(12):
        w.prep.run_due()
        _drain(w.bench)
    _settle(w)
    assert seen == [(True, 0)]                       # paused: no prepare step ran
    assert not w.drv._settling.is_set()


# ── retry fast + alert early ─────────────────────────────────────────────────


class FlakyPod:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0
        self.unpinned = []

    def rebase_stage(self, **kw):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return {"host": "pod", "actuals": {}, "content_key": "k" * 64, "pin": "/p/pinned/x",
                "merge_error": None}

    def unpin(self, host, pin):
        self.unpinned.append(pin)


class StubBench:
    settings = {"suites": list(SUITES)}

    def __init__(self):
        self.rows, self.q, self.fail = {}, set(), {}
        self.pinned = []

    ptr_tag = staticmethod(lambda p: f"ptr:{p}")
    key_tag = staticmethod(lambda k: f"key:{k}")

    def pointer_rows(self, p):
        return self.rows.get(p)

    def cached(self, k):
        return self.rows.get(k)

    def queued(self, tag):
        return tag in self.q

    def failure(self, tag):
        return self.fail.get(tag)

    def prebench(self, p):
        self.q.add(f"ptr:{p}")

    def bench_pin(self, host, key, pin, label):
        self.pinned.append(pin)
        self.q.add(f"key:{key}")
        return True


def _spec():
    return {"king_pointer": "K", "base_pointer": "K", "leg_pointer": "L",
            "selection": {0: [1, 2]}, "refuse_nonfinite": True, "limits": False,
            "merge_strict": True, "label": "alice/energy"}


def test_a_failed_step_retries_within_minutes_and_alerts_once(caplog):
    from cascade.trainer.lane_b_offload import LaneBOffloadUnavailable

    m = _prep_mod()
    now = [1000.0]
    pod = FlakyPod([LaneBOffloadUnavailable("lane B rebase_stage: pod reply is not JSON"),
                    LaneBOffloadUnavailable("lane B rebase_stage: disk short on the pod")])
    bench = StubBench()
    bench.rows["K"] = {"king": 1}
    prep = m.LaneBPrepare(pod, bench, start_worker=False, clock=lambda: now[0])
    prep.submit([_spec()])
    caplog.set_level(logging.WARNING, logger="cascade.trainer.lane_b_prepare")
    prep.run_due()                                   # king_rows done
    prep.run_due()                                   # rebase: fails
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "rebase" in errors[0].getMessage()
    leg = prep.status()["legs"][0]
    assert leg["steps"]["rebase"]["state"] == "retrying"
    assert leg["steps"]["rebase"]["attempts"] == 1
    assert "not JSON" in leg["steps"]["rebase"]["last_error"]
    prep.run_due()
    assert pod.calls == 1                            # not before its backoff
    now[0] += m.BACKOFF_S[0]
    prep.run_due()                                   # second failure: a warning only
    assert pod.calls == 2
    assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 1
    now[0] += m.BACKOFF_S[1]
    prep.run_due()
    assert pod.calls == 3
    prep.run_due()                                   # the staged candidate is queued
    leg = prep.status()["legs"][0]
    assert leg["steps"]["rebase"]["state"] == "done" and bench.pinned == ["/p/pinned/x"]
    assert sum(m.BACKOFF_S[:2]) <= 600               # minutes, not a 3 h settlement


def test_retries_are_bounded_then_rearmed():
    from cascade.trainer.lane_b_offload import LaneBOffloadUnavailable

    m = _prep_mod()
    now = [0.0]
    pod = FlakyPod([LaneBOffloadUnavailable("flock: heavy lock busy")] * 50)
    bench = StubBench()
    bench.rows["K"] = {"king": 1}
    prep = m.LaneBPrepare(pod, bench, start_worker=False, clock=lambda: now[0],
                          max_attempts=3, backoff_s=(10,))
    prep.submit([_spec()])
    for _ in range(20):
        prep.run_due()
        now[0] += 10
    assert pod.calls == 3
    assert prep.status()["legs"][0]["state"] == "failed"
    now[0] += m.REARM_S
    prep.submit([_spec()])
    prep.run_due()                                   # king rows (cached)
    prep.run_due()
    assert pod.calls == 4


def test_a_failed_candidate_bench_restages_and_retries():
    m = _prep_mod()
    now = [0.0]
    pod = FlakyPod([])
    bench = StubBench()
    bench.rows["K"] = {"king": 1}
    prep = m.LaneBPrepare(pod, bench, start_worker=False, clock=lambda: now[0])
    prep.submit([_spec()])
    for _ in range(3):
        prep.run_due()
    assert bench.pinned and "key:" + "k" * 64 in bench.q
    bench.q.clear()                                  # the sweep ended without rows
    bench.fail["key:" + "k" * 64] = "no complete four-suite rows"
    now[0] += 30
    prep.run_due()
    leg = prep.status()["legs"][0]
    assert leg["steps"]["candidate_rows"]["state"] == "retrying"
    assert leg["steps"]["rebase"]["state"] == "pending"          # re-staged next
    now[0] += m.BACKOFF_S[0]
    prep.run_due()
    assert pod.calls == 2


def test_status_v2_carries_lane_b_prepare(pod_env, monkeypatch):
    from cascade.shared import v2_status

    from .helpers import chain_cfg

    w = _world(pod_env, monkeypatch, "status", prepare=True)
    w.ops.prepare_ahead(w.drv.prepare_specs(20, 0, w.king))
    w.prep.run_due()
    assert w.drv.status()["prepare"]["legs"]
    published = []
    sched = SimpleNamespace(cfg=chain_cfg(), lane_b=w.drv, lane_c=None,
                            state=SimpleNamespace(current=None),
                            r=SimpleNamespace(manifest_store=lambda: "store"))
    monkeypatch.setattr(v2_status, "publish_v2_status", lambda store, doc: published.append(doc))
    assert v2_status.publish_trainer_v2_status(sched, 20000, min_interval_s=0)
    prep = published[0]["lane_b"]["prepare"]
    legs = {x["leg"]: x for x in prep["legs"]}
    assert set(legs) == {"alice/energy", "bob/web"}
    assert set(legs["alice/energy"]["steps"]) == {"king_rows", "rebase", "candidate_rows"}
    assert {"state", "attempts", "last_error"} <= set(legs["alice/energy"]["steps"]["rebase"])


def test_judge_reuses_only_a_prepared_mergeable_check():
    m = _prep_mod()
    pod = FlakyPod([])
    bench = StubBench()
    bench.rows["K"] = {"king": 1}
    prep = m.LaneBPrepare(pod, bench, start_worker=False, clock=lambda: 0.0)
    prep.submit([_spec()])
    prep.run_due()
    prep.run_due()
    keys = {"cascade/ckpt-r9@sha256:1": "k" * 64}
    bench._ptr_key = keys.get
    live = []

    def live_check(k, c, sel, strict):
        live.append(c)
        return "MergeError: rows differ"

    check = m.merge_check_with(prep, bench, live_check)
    assert check("K", "cascade/ckpt-r9@sha256:1", {0: [1, 2]}, True) is None
    assert live == []
    # another strictness / selection / king / unknown pointer: the live check
    assert check("K", "cascade/ckpt-r9@sha256:1", {0: [1, 2]}, False) == "MergeError: rows differ"
    assert check("K", "cascade/ckpt-r9@sha256:1", {0: [1]}, True) is not None
    assert check("K2", "cascade/ckpt-r9@sha256:1", {0: [1, 2]}, True) is not None
    assert check("K", "cascade/other@sha256:2", {0: [1, 2]}, True) is not None
    assert len(live) == 4


# ── bench method (#973) in the cache key and the pairing ─────────────────────


def _lb(tmp_path):
    from cascade.trainer.lane_b_bench import LaneBBench

    return LaneBBench(hosts_path=tmp_path / "h.toml", cache_dir=tmp_path / "rows",
                      eval_cfg=SimpleNamespace(cascade_bench_max_series=0,
                                               gift_gate_data_dir="", bench_workers="1"),
                      start_worker=False)


def test_bench_method_is_in_the_row_cache_key(tmp_path):
    from cascade.eval.benchmarks import BENCH_METHOD, LEGACY_BENCH_METHOD, item_rows_from_report
    from cascade.trainer import lane_b_bench

    rows_key = getattr(lane_b_bench, "rows_key", None)
    assert rows_key is not None, "lane_b_bench has no rows_key: bench_method is not in the key"
    assert rows_key("a" * 64, LEGACY_BENCH_METHOD) == "a" * 64        # pre-stamp files stay
    assert rows_key("a" * 64, BENCH_METHOD) != "a" * 64
    lb = _lb(tmp_path)
    legacy = item_rows_from_report(_report_of(b"king"))               # unstamped = impute/0
    lb._store("a" * 64, legacy, "king")
    assert lb.cached("a" * 64) == legacy
    # the pod's sidecar switches method: a stamped sweep moves the cache over
    lb._store("b" * 64, item_rows_from_report(_report_of(b"cand", BENCH_METHOD)), "cand")
    assert lb.method == BENCH_METHOD
    assert lb.cached("a" * 64) is None, "rows swept with the old method were served"
    assert lb.cached("b" * 64) is not None
    # persisted across a restart
    assert _lb(tmp_path).method == BENCH_METHOD and _lb(tmp_path).cached("a" * 64) is None


def test_mixed_bench_methods_never_pair():
    from cascade.eval.benchmarks import BENCH_METHOD, LEGACY_BENCH_METHOD, item_rows_from_report
    from cascade.eval.public_guards import MissingPublicGuard, pair_public_rows

    old = item_rows_from_report(_report_of(b"king"))
    new = item_rows_from_report(_report_of(b"cand", BENCH_METHOD))
    assert set(pair_public_rows(old, new).missing) == {"gift", "boom", "time", "fev"}
    with pytest.raises(MissingPublicGuard):
        pair_public_rows(old, new, require=("gift",))
    stamped_legacy = item_rows_from_report(_report_of(b"cand", LEGACY_BENCH_METHOD))
    assert pair_public_rows(old, stamped_legacy, require=("gift",)).missing == []
    assert pair_public_rows(new, item_rows_from_report(_report_of(b"x", BENCH_METHOD)),
                            require=("gift",)).missing == []
