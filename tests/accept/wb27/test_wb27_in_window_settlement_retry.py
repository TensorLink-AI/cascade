"""In-window settlement retries (owner 2026-11-28, ~02:00: "a failed lane
settlement retries quickly within the same window, so a mistake costs minutes
instead of 3 hours ... It must NOT require a satellite validator update").

Tonight four consecutive lane B settlements failed for one-off reasons (pod
disk, a pod reply parse, a queued bench, a CUDA arch) and each cost a whole
3-hourly settlement: the lane-A manifest went out without lane B and the
finished legs waited for the next boundary.

The rule these tests pin:

* a failed lane B (or lane C) settlement step is retried after 2, 5, then
  every 10 min for THE SAME settlement (same round id, boundary, king and
  the first attempt's leg set); the lane-A settlement waits for it;
* the retried publish is byte-identical to an on-time success (the expert /
  lane-C manifest text, the lane-A binding), published exactly once;
* the retries stop inside the lane-A hold bound the trainer already allows a
  slow lane B (``LANE_B_SETTLE_MAX_SECONDS`` from the first attempt) and
  before the next boundary, so lane A is never later than today's worst case;
* the main validator's MarketCast report publish run retries in-window too
  (no validator reads a report before a trainer-signed lane-C manifest
  records its sha256, so nothing a satellite reads changes).

No torch: the trainer's fakes from ``tests/unit/test_rolling.py``.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from cascade.shared.config import parse_lane_b
from cascade.shared.manifest import dump_manifest


def _cfg(tmp_path):
    """The mainnet template with its enforcing pins neutralised (the
    ``tests/conftest.py`` ``cfg`` neutralisation; runs under --noconftest)."""
    from cascade.shared.config import load_chain_config
    from tests.unit.predecision_main import predecision_path

    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    c = load_chain_config(predecision_path(tmp_path))
    return replace(
        c, training=replace(c.training, expected_gpu="", train_image_digest=""),
        netconfig=replace(c.netconfig, owner_hotkey=""),
        v2=replace(c.v2, launch_at_utc="", commit_open_at_utc="", commit_from_block=0,
                   prelaunch_burn_readiness="", image_decoupled_from_block=0),
        generator=replace(c.generator, corpus_target_points=0, min_channels=1,
                          sandbox_strict=False),
        round=replace(c.round, commit_floor_block=0, dedup_mode="off", epoch_blocks_prev=0,
                      epoch_activation_block=0, funded_mode="off", funded_pods="off",
                      funded_king_rent=False),
        scoring=replace(c.scoring, win_margin_start_prev=0.0, margin_activation_block=0,
                        forfeit_from_block=0, margin_v2_from_block=0),
        manifest=replace(c.manifest, validator_hotkey=""))


class _LaneB:
    """A lane-B driver whose settlement fails ``fails`` times, then publishes
    an expert manifest derived ONLY from the settlement's inputs."""

    def __init__(self, fails=0):
        self.fails, self.calls, self.published, self.settled = fails, [], [], {}
        self.abandoned = set()

    def tick(self, block, era_index, king_pointer):
        return []

    def add_lineage_tokens(self, n):
        pass

    def init_for(self, era_index, init_pointer, work_dir, era_start=None):
        return init_pointer, []

    def settle(self, *, round_id, block, era_index, king_pointer, work_dir, **kw):
        args = {"round_id": round_id, "block": block, "era_index": era_index,
                "king_pointer": king_pointer, **kw}
        self.calls.append(args)
        if round_id in self.settled or round_id in self.abandoned:
            return []
        if len(self.calls) <= self.fails:
            raise OSError(f"pod disk short (attempt {len(self.calls)})")
        text = json.dumps(args, sort_keys=True)
        self.published.append(text)
        self.settled[round_id] = hashlib.sha256(text.encode()).hexdigest()
        return []

    def expert_manifest_digest(self, round_id):
        return self.settled.get(round_id, "")

    def abandon(self, round_id):
        if round_id not in self.settled:
            self.abandoned.add(round_id)
        return self.settled.get(round_id, "")


class _LaneC:
    """A lane-C driver with the real driver's write-ahead: the text is built
    once per round, a publish that fails ``fails`` times keeps it."""

    def __init__(self, fails=0):
        self.fails, self.puts, self.pending, self.settled, self.out = fails, 0, None, {}, []

    def tick(self, *a, **k):
        return None

    def live_hotkeys(self):
        return []

    def status(self):
        return {}

    def settle(self, *, round_id, block, era_index, base_pointer, created_block,
               base_round_id, pool_block):
        if round_id in self.settled:
            return self.settled[round_id]
        if self.pending is None or self.pending["rid"] != round_id:
            text = json.dumps({"round_id": round_id, "block": block,
                               "created_block": created_block, "base": base_pointer})
            self.pending = {"rid": round_id, "text": text}
        self.puts += 1
        if self.puts <= self.fails:
            raise OSError("hippius 503")
        self.out.append(self.pending["text"])
        self.settled[round_id] = hashlib.sha256(self.pending["text"].encode()).hexdigest()
        return self.settled[round_id]

    def published(self, round_id):
        return ({"key": f"lane-c/{round_id}.json", "text": self.pending["text"]}
                if round_id in self.settled else None)


class _Run:
    def __init__(self, tmp_path, *, lane_b=None, lane_c=None):
        from tests.unit.test_rolling import (
            EB,
            REF,
            Clock,
            FakeClient,
            _armed,
            _commit,
            _join,
            _sched,
        )

        armed = _armed(_cfg(tmp_path / "cfg"))
        armed = replace(armed, lane_b=parse_lane_b({"from_block": 1, "domains": ["energy"]}))
        if lane_c is not None:
            armed = replace(armed, lane_c=replace(armed.lane_c, from_block=1))
        self.cfg, self.EB, self._join = armed, EB, _join
        self.clock = Clock()
        self.sched, self.ops = _sched(armed, tmp_path, self.clock)
        self.events = []
        self.lane_b = lane_b or _LaneB()
        self.sched.lane_b = self.lane_b
        if lane_c is not None:
            self.sched.lane_c = lane_c
        orig_settle = self.lane_b.settle

        def settle(**kw):
            self.events.append(("lane_b", self.clock.t))
            return orig_settle(**kw)

        self.lane_b.settle = settle
        orig = self.ops.publish_manifest
        self.ops.publish_manifest = lambda m: (self.events.append(("lane_a", self.clock.t)),
                                               orig(m))
        self.client = FakeClient()
        era_start = armed.round.rolling_from_block
        self.boundary = era_start + EB
        b0 = era_start + 5
        self.ops.commits += [_commit("ALFA", REF["ALFA"], b0 - 100)]
        self.ops.queue().add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
        self.sched.tick(self.client, b0)
        _join(self.sched)
        # the boundary tick: the clock follows the block grid
        self.block = self.boundary + 3
        self.clock.t += (self.block - b0) * 12.0
        self.t0 = self.clock.t
        self.sched.tick(self.client, self.block)
        _join(self.sched)

    def step(self, dt=60.0, blocks=5):
        self.clock.t += dt
        self.block += blocks
        self.sched.tick(self.client, self.block)
        self._join(self.sched)

    def until_published(self, limit=4 * 3600.0):
        while not self.ops.manifests and self.clock.t - self.t0 < limit:
            self.step()
        return self.ops.manifests


def _body(m):
    """A lane-A manifest's bytes with its publish-tick stamp normalised."""
    return dump_manifest(replace(m, created_block=0, signature=getattr(m, "signature", "")))


def test_lane_b_failure_then_success_publishes_the_on_time_bytes(tmp_path):
    on_time = _Run(tmp_path / "a")
    (m_ok,) = on_time.ops.manifests                   # same tick as the boundary
    retried = _Run(tmp_path / "b", lane_b=_LaneB(fails=1))
    assert retried.ops.manifests == [], "lane A must wait for the in-window lane-B retry"
    (m_rt,) = retried.until_published()
    # the retry ran minutes later, not at the next 3-hourly boundary
    assert retried.clock.t - retried.t0 <= 180.0
    # the same settlement: identical driver inputs on every attempt
    a0, a1 = retried.lane_b.calls
    assert a0 == a1 == on_time.lane_b.calls[0]
    # byte-identical expert manifest, published exactly once
    assert retried.lane_b.published == on_time.lane_b.published
    assert len(retried.lane_b.published) == 1
    # the lane-A manifest binds the same digest; only its publish-tick stamp
    # moved, still inside the boundary's grid slot (the validators' only clock)
    assert m_rt.expert_manifest_sha256 == m_ok.expert_manifest_sha256 != ""
    assert m_rt.round_id == m_ok.round_id
    assert _body(m_rt) == _body(m_ok)
    eb = retried.EB
    assert retried.boundary <= m_rt.created_block < retried.boundary + eb
    # ordering kept (DEC-CA-0050): every lane-B attempt before the one lane-A publish
    kinds = [k for k, _ in retried.events]
    assert kinds.count("lane_a") == 1 and kinds[-1] == "lane_a"
    # a later tick publishes nothing more
    for _ in range(5):
        retried.step()
    assert len(retried.ops.manifests) == 1 and len(retried.lane_b.published) == 1


def test_lane_b_retries_back_off_and_stop_at_the_lane_a_bound(tmp_path):
    run = _Run(tmp_path, lane_b=_LaneB(fails=10**6))
    (m,) = run.until_published()
    sched = run.sched
    bound = float(sched.LANE_B_SETTLE_MAX_SECONDS)
    times = [t - run.t0 for k, t in run.events if k == "lane_b"]
    # 2 min, 5 min, then every 10 min
    assert times[:4] == [0.0, 120.0, 420.0, 1020.0]
    assert all(b - a == 600.0 for a, b in zip(times[3:], times[4:], strict=False))
    # every attempt STARTED inside the bound; lane A never waits past it
    assert times[-1] < bound
    lane_a = [t - run.t0 for k, t in run.events if k == "lane_a"]
    assert len(lane_a) == 1 and lane_a[0] <= bound
    # the window spent: lane A publishes without lane B, as before
    assert m.expert_manifest_sha256 == ""
    assert run.lane_b.published == []
    n = len(times)
    for _ in range(10):
        run.step()
    assert len([k for k, _ in run.events if k == "lane_b"]) == n   # no retry after


def test_retry_window_ends_at_the_next_boundary(tmp_path):
    run = _Run(tmp_path / "w")
    ok = getattr(run.sched, "_settle_retry_ok", None)
    assert callable(ok), "the in-window retry bound is missing"
    b, eb, t = run.boundary, run.EB, 1000.0
    assert ok(b, b + 3, t, t + 120.0)
    assert ok(b, b + eb - 1, t, t + 120.0)
    assert not ok(b, b + eb, t, t + 120.0)           # the next boundary: never
    assert not ok(b, b + 3, t, t + run.sched.LANE_B_SETTLE_MAX_SECONDS)
    assert not ok(b, None, t, t + 120.0)             # no head: no retry


def test_lane_c_publish_failure_retries_the_same_bytes(tmp_path):
    on_time = _Run(tmp_path / "a", lane_c=_LaneC())
    (m_ok,) = on_time.ops.manifests
    retried = _Run(tmp_path / "b", lane_c=_LaneC(fails=1))
    assert retried.ops.manifests == [], "lane A must wait for the in-window lane-C retry"
    (m_rt,) = retried.until_published()
    assert retried.clock.t - retried.t0 <= 180.0
    lc_ok, lc_rt = on_time.sched.lane_c, retried.sched.lane_c
    assert lc_rt.out == lc_ok.out and len(lc_rt.out) == 1
    assert m_rt.lane_c_digest == m_ok.lane_c_digest != ""
    assert m_rt.expert_manifest_sha256 == m_ok.expert_manifest_sha256
    # lane B ran once for the boundary — the lane-C wait never re-runs it
    assert len(retried.lane_b.published) == 1
    assert len([k for k, _ in retried.events if k == "lane_b"]) == 1


def test_driver_retry_settles_the_first_attempts_legs_only(tmp_path):
    """A leg that finishes during the retry wait settles with the NEXT round,
    exactly as it would after an on-time success."""
    from cascade.trainer.lane_b_driver import LaneBDriver, LaneBDriverState

    drv = LaneBDriver.__new__(LaneBDriver)
    lb = SimpleNamespace(active=lambda b: True, max_base_age_eras=0, require_public=(),
                         acceptance_at_settlement=lambda b: False)
    drv.cfg = SimpleNamespace(lane_b=lb)
    drv._lock, drv._settling, drv._abandoned = threading.Lock(), threading.Event(), set()
    drv._settle_pins = {}
    drv.st = LaneBDriverState()
    drv.ops = SimpleNamespace(lane_b_model=lambda: object())
    drv.save = lambda: None

    def leg(name):
        sub = {"hotkey": name, "uid": 1, "domain": "energy", "gen_ref": f"g-{name}",
               "commit_block": 1}
        return {"leg": {"sub": sub, "base_era": 0, "base_king_ref": "K"},
                "leg_pointer": f"L-{name}", "corpus_digest": "c", "train_block": 5}

    drv.st.finished = [leg("a")]
    calls = []

    def sentinel(block):
        calls.append(block)
        if len(calls) == 1:
            raise OSError("block hash unreadable")
        return None

    drv._settlement_sentinel = sentinel
    try:
        drv.settle(round_id="R", block=900, era_index=5, king_pointer="K", work_dir=tmp_path)
    except OSError:
        pass
    else:
        raise AssertionError("the first attempt must fail")
    drv.st.finished.append(leg("b"))      # finishes during the retry wait
    drv.settle(round_id="R", block=900, era_index=5, king_pointer="K", work_dir=tmp_path)
    assert "R" in drv.st.settled
    assert [f["leg_pointer"] for f in drv.st.finished] == ["L-b"], \
        "the retry must settle only the first attempt's legs"
    assert [o["leg_pointer"] for o in drv.st.outcomes] == ["L-a"]


class _Store:
    def __init__(self):
        self.objs = {}

    def get_text(self, key):
        from cascade.shared.hippius import ObjectNotFound

        if key not in self.objs:
            raise ObjectNotFound(key)
        return self.objs[key]


def test_marketcast_report_publish_retries_in_window(monkeypatch):
    """The main validator's offloaded MarketCast report run that failed (pod
    not up, a session error) retries before the next boundary instead of at
    the next settlement; the report bytes are whatever the run publishes."""
    import cascade.validator.marketcast as mk
    from cascade.validator.marketcast import MarketCastMixin, current_ctx

    now = [1000.0]
    monkeypatch.setattr(mk.time, "time", lambda: now[0], raising=False)

    class V(MarketCastMixin):
        pass

    v = V()
    v.cfg = None
    store = _Store()
    v._mkt_store = lambda: store
    runs = []

    def remote(todo, holders):
        runs.append([i["round_id"] for i in todo])
        if len(runs) >= 2:                 # the pod came up on the retry
            for i in todo:
                store.objs[current_ctx(None).report_key(i["round_id"])] = "signed-report"
        return 0

    v._mkt_publish_remote = remote
    todo = [{"round_id": "R1", "block": 900}]
    retry_tick = getattr(v, "_mkt_retry_tick", None)
    assert callable(retry_tick), "the in-window MarketCast retry is missing"
    assert v._mkt_publish_async(todo, {}, window_end_block=1800)
    v.__dict__["_mkt_pub_thread"].join(5)
    assert runs == [["R1"]]
    assert not retry_tick(1000)            # backoff: not yet
    now[0] += 120.0
    assert retry_tick(1000)
    v.__dict__["_mkt_pub_thread"].join(5)
    assert runs == [["R1"], ["R1"]]
    assert store.objs == {current_ctx(None).report_key("R1"): "signed-report"}
    now[0] += 3600.0
    assert not retry_tick(1001)            # published: nothing left to retry
    assert runs == [["R1"], ["R1"]]


def test_marketcast_retry_stops_at_the_next_boundary(monkeypatch):
    import cascade.validator.marketcast as mk
    from cascade.validator.marketcast import MarketCastMixin

    now = [1000.0]
    monkeypatch.setattr(mk.time, "time", lambda: now[0], raising=False)

    class V(MarketCastMixin):
        pass

    v = V()
    v.cfg = None
    store = _Store()
    v._mkt_store = lambda: store
    runs = []
    v._mkt_publish_remote = lambda todo, holders: runs.append(1) or 0
    retry_tick = getattr(v, "_mkt_retry_tick", None)
    assert callable(retry_tick), "the in-window MarketCast retry is missing"
    assert v._mkt_publish_async([{"round_id": "R1", "block": 900}], {},
                                window_end_block=1800)
    v.__dict__["_mkt_pub_thread"].join(5)
    now[0] += 10_000.0
    assert not retry_tick(1800)            # the next boundary: the settlement retries it
    assert runs == [1] and not v.__dict__.get("_mkt_pub_retry")
