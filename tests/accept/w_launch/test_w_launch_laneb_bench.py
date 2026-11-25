"""W-LAUNCH / cascade#909: lane B's public rows (king, every rebased candidate,
the anchor) are benched on an EVAL POD (``--lane-b-bench-hosts``, e.g.
targon-eval.toml), never on the CPU-only trainer box; a settlement never
benches inline (a full four-suite sweep is ~1 h, past the scheduler's lane-B
hold); and every failure fails closed (legs kept, nothing published).

Torch-free: fakes for the host, the offload bench and the driver's ops."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

SUITES = ("gift-eval", "boom", "time", "fev")


def _ckpt(root: Path, name: str, payload: bytes = b"w") -> Path:
    d = root / name
    d.mkdir(parents=True)
    for f in ("config.json", "forecast_wrapper.py", "model.py"):
        (d / f).write_text(f"{f}\n", encoding="utf-8")
    (d / "weights.safetensors").write_bytes(payload)
    return d


def _report(status: str = "ok", rev: str = "r1") -> dict:
    return {"suites": [{"suite": s, "status": status,
                        "rows": [{"full": f"{s}/i{i}", "CRPS": 0.4 + i / 100, "MASE": 1.0}
                                 for i in range(12)]} for s in SUITES],
            "data_revisions": {s: rev for s in SUITES}}


def _eval_cfg(**kw):
    base = {"cascade_bench_max_series": 0, "gift_gate_data_dir": "/data/bench",
            "bench_workers": "auto", "benchmark_project_dir": "benchmarks"}
    return SimpleNamespace(**{**base, **kw})


class _Bench:
    def __init__(self, report=None):
        self.calls, self.report = [], report if report is not None else _report()

    def __call__(self, host, dirs, **kw):
        self.calls.append((host, [Path(d) for d in dirs], kw))
        return [self.report for _ in dirs]


def _bench(tmp_path, *, host="pod", report=None, hosts=True, fetch=None):
    from cascade.trainer.lane_b_bench import LaneBBench

    fake = _Bench(report)
    lb = LaneBBench(hosts_path=(tmp_path / "targon-eval.toml") if hosts else None,
                    cache_dir=tmp_path / "rows", eval_cfg=_eval_cfg(),
                    fetch=fetch, host_fn=(lambda: host), bench_fn=fake, start_worker=False)
    return lb, fake


@pytest.fixture(autouse=True)
def _no_local_sidecar(monkeypatch):
    """Any local sidecar run is a test failure (the trainer box never benches)."""
    from cascade.eval import benchmarks

    def boom(*a, **k):
        raise AssertionError("lane B benched on the trainer box")

    for name in ("run_item_rows", "run_bench_report", "run_bench_reports"):
        monkeypatch.setattr(benchmarks, name, boom)


def _drain(lb):
    while not lb._q.empty():
        tag, item = lb._q.get()
        lb.run_one(item)
        lb._pending.discard(tag)


def test_rows_never_bench_inline_and_run_on_the_eval_pod(tmp_path):
    lb, fake = _bench(tmp_path)
    king = _ckpt(tmp_path, "king")
    assert lb.rows(king) is None                 # queued, never benched on this thread
    assert fake.calls == [] and lb.pending() == 1
    assert lb.rows(king) is None and lb.pending() == 1      # queued once
    _drain(lb)
    host, dirs, kw = fake.calls[0]
    assert host == "pod" and dirs == [king]
    assert kw["device"] == "cuda" and kw["suites"] == ",".join(SUITES)
    assert kw["data_dir"] == "/data/bench" and kw["max_series"] == 0
    rows = lb.rows(king)
    assert set(rows) == {"gift", "boom", "time", "fev"}
    assert all(rows[k]["status"] == "ok" and rows[k]["data_revision"] == "r1" for k in rows)
    assert len(fake.calls) == 1                  # cached


def test_rows_are_cached_by_content_across_restarts_and_dir_names(tmp_path):
    lb, fake = _bench(tmp_path)
    a = _ckpt(tmp_path, "rebased-r1-alice-energy", b"same")
    lb.rows(a)
    _drain(lb)
    # the next settlement rebases onto the same king again: new dir, same bytes
    b = _ckpt(tmp_path, "rebased-r2-alice-energy", b"same")
    lb2, fake2 = _bench(tmp_path)                 # a trainer restart
    assert lb2.rows(b) == lb.rows(a) and fake2.calls == []
    c = _ckpt(tmp_path, "rebased-r2-bob-energy", b"other")
    assert lb2.rows(c) is None                    # different bytes ⇒ its own bench


@pytest.mark.parametrize("why", ["no_hosts_flag", "no_host", "incomplete", "no_report"])
def test_every_failure_fails_closed_and_caches_nothing(tmp_path, why):
    report = {"no_report": {}, "incomplete": _report(status="error")}.get(why)
    lb, fake = _bench(tmp_path, host=None if why == "no_host" else "pod",
                      hosts=why != "no_hosts_flag", report=report)
    if why == "no_report":
        fake.report = None
    king = _ckpt(tmp_path, "king")
    assert lb.rows(king) is None
    _drain(lb)
    assert lb.rows(king) is None
    assert not list((tmp_path / "rows").glob("*.json")) if (tmp_path / "rows").exists() else True
    if why in ("no_hosts_flag", "no_host"):
        assert fake.calls == []


def test_prebench_fetches_on_the_worker_and_stops_once_cached(tmp_path):
    fetched = []
    root = tmp_path / "hub"

    def fetch(p):
        fetched.append(p)
        d = root / p
        return d if d.exists() else _ckpt(root, p)

    lb, fake = _bench(tmp_path, fetch=fetch)
    lb.prebench("kingptr")
    assert fetched == []                          # never fetched on the tick thread
    _drain(lb)
    assert fetched == ["kingptr"] and len(fake.calls) == 1
    lb.prebench("kingptr")                        # cached: nothing queued
    assert lb.pending() == 0 and lb._q.empty()
    assert lb.rows(root / "kingptr") is not None


def test_runner_ops_without_bench_host_never_run_locally(caplog):
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    ops = RunnerLaneBOps(SimpleNamespace(cfg=SimpleNamespace(eval=_eval_cfg())), None, None)
    with caplog.at_level("ERROR"):
        assert ops.bench_rows(Path("/nonexistent")) is None
    assert "never benched on the trainer box" in caplog.text
    ops.prebench("p")                             # no bench: a no-op


def test_trainer_flag_wires_the_eval_pod_bench(tmp_path):
    from cascade.trainer import main
    from cascade.trainer.lane_b_bench import LaneBBench

    args = main._build_parser().parse_args(
        ["--lane-b-bench-hosts", str(tmp_path / "targon-eval.toml"),
         "--work-root", str(tmp_path)])
    cfg = SimpleNamespace(eval=_eval_cfg(), lane_b=SimpleNamespace(require_public=("gift",)))
    got = main._lane_b_bench(args, cfg, SimpleNamespace(_fetch_checkpoint_dir=str))
    assert isinstance(got, LaneBBench) and got.hosts_path == tmp_path / "targon-eval.toml"
    assert got.cache_dir == tmp_path / "lane_b_bench_rows"
    none = main._build_parser().parse_args(["--work-root", str(tmp_path)])
    assert main._lane_b_bench(none, cfg, SimpleNamespace()) is None


# ── the driver: candidates are all queued in one settlement, nothing published ──


GEN = "alice/metro-gen@sha256:" + "a" * 64
KING = "metro-v1:trained:hippius:ns/king@sha256:" + "b" * 64
KING_DIR = Path("/nonexistent", KING).name


def _rows():
    from cascade.eval.benchmarks import item_rows_from_report

    return item_rows_from_report(_report())


class _Ops:
    def __init__(self, ready: set[str]):
        self.ready, self.asked, self.published, self.prebenched = ready, [], [], []

    def fetch(self, p):
        return Path("/nonexistent") / p

    def upload(self, d, repo):
        import hashlib

        return "metro-v1:trained:hippius:ns/rb@sha256:" + hashlib.sha256(
            Path(d).name.encode()).hexdigest()

    def bench_rows(self, d):
        self.asked.append(Path(d).name)
        return _rows() if Path(d).name in self.ready else None

    def prebench(self, pointer):
        self.prebenched.append(pointer)

    def publish_expert_manifest(self, m):
        self.published.append(m)

    def judge(self, m, round_id, block, **kw):
        return []

    def commitments(self):
        return []

    def contexts(self, domain, block):
        return []

    def dispatch_leg(self, *a, **k):
        pass


def _driver(tmp_path, ready, monkeypatch):
    from cascade.shared.config import parse_lane_b
    from cascade.trainer import lane_b_driver
    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg
    from cascade.trainer.lane_b_driver import FinishedExpertLeg, LaneBDriver

    def fake_settle_leg(king, base, leg, sel, out, **k):
        Path(out).mkdir(parents=True, exist_ok=True)
        return Path(out)

    monkeypatch.setattr(lane_b_driver, "settle_leg", fake_settle_leg)
    # settle() imports moe_model.MergeError (a torch module): a torch-free stand-in
    import sys
    import types

    moe = types.ModuleType("cascade.trainer.moe_model")
    moe.MergeError = type("MergeError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "cascade.trainer.moe_model", moe)
    cfg = SimpleNamespace(lane_b=parse_lane_b({
        "from_block": 1, "domains": ["energy", "web"], "open_tokens": 0,
        "require_public": ["gift", "boom", "time", "fev"], "min_public_items": 10}))
    ops = _Ops(ready)
    drv = LaneBDriver(cfg, ops, tmp_path / "lane_b.json")
    fins = []
    for hk, dom in (("alice", "energy"), ("bob", "web")):
        leg = OpenLeg(sub=LaneBSubmission(hk, 7, GEN, dom, 5), selection={0: [1]},
                      base_king_ref=KING, base_era=0)
        fins.append(asdict(FinishedExpertLeg(leg=asdict(leg), leg_pointer=f"leg-{hk}",
                                             corpus_digest="c", train_block=3)))
    drv.st.finished = fins
    drv.save()
    return drv, ops


def test_settlement_queues_every_candidate_then_fails_closed_then_publishes(tmp_path,
                                                                          monkeypatch):
    drv, ops = _driver(tmp_path, {KING_DIR}, monkeypatch)
    with pytest.raises(OSError, match="candidate public benchmarks unavailable"):
        drv.settle(round_id="r1", block=20, era_index=0, king_pointer=KING,
                   work_dir=tmp_path / "w")
    # both candidates asked (queued) in the SAME settlement, not one per settlement
    assert [a for a in ops.asked if a.startswith("rebased-")] == [
        "rebased-r1-alice-energy", "rebased-r1-bob-web"]
    assert ops.published == []
    assert [f["leg_pointer"] for f in drv.st.finished] == ["leg-alice", "leg-bob"]
    assert "r1" not in drv.st.settled
    # the worker benched both: the next settlement publishes them
    ops.ready |= {"rebased-r2-alice-energy", "rebased-r2-bob-web"}
    drv.settle(round_id="r2", block=40, era_index=0, king_pointer=KING,
               work_dir=tmp_path / "w")
    assert len(ops.published) == 1
    entries = ops.published[0].entries
    assert [e.public_rows["candidate"] == e.public_rows["king"] for e in entries] == [True, True]


def test_tick_prebenches_the_king_and_a_settlement_without_it_fails_fast(tmp_path,
                                                                        monkeypatch):
    drv, ops = _driver(tmp_path, set(), monkeypatch)
    drv.tick(20, 0, KING)
    assert ops.prebenched == [KING]
    with pytest.raises(OSError, match="king public benchmarks unavailable"):
        drv.settle(round_id="r1", block=20, era_index=0, king_pointer=KING,
                   work_dir=tmp_path / "w")
    assert ops.asked == [KING_DIR] and ops.published == []      # no candidate benched
    drv.st.finished = []
    drv.tick(21, 0, KING)
    assert ops.prebenched == [KING]          # nothing finished: no prebench
