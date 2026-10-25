"""DEC-CA-0055 on the trainer: seed epochs (one king leg per warm start), the
carried king entry, the dethroner's entry as the king leg, lane-B merges only
at a warm start, the init bench entry, and the king's own key funding."""
from __future__ import annotations

from dataclasses import replace

from cascade.shared.era import era_for_block, era_length_blocks
from tests.unit.test_rolling import (
    EB,
    REF,
    Clock,
    FakeClient,
    FakeOps,
    _advance,
    _armed,
    _commit,
    _ptr,
    _sched,
)


def _join(sched, timeout=10.0):
    import time

    end = time.time() + timeout
    while (sched._threads or sched._king_threads or sched._init_threads) and time.time() < end:
        time.sleep(0.01)
    assert not (sched._threads or sched._king_threads or sched._init_threads)


INIT_PTR = "metro-v1:trained:hippius:cascade/ckpt-random-init@sha256:" + "e" * 64
INIT_SCORES = {"gifteval_crps": 0.9, "gifteval_mase": 0.9, "boom_crps": 0.9,
               "boom_mase": 0.9, "time_crps": 0.9, "time_mase": 0.9}


class EpochOps(FakeOps):
    """FakeOps plus the DEC-CA-0055 hooks."""

    def __init__(self, tmp_path, clock):
        super().__init__(tmp_path, clock)
        self.init_benches: list[int] = []
        self.carried: list[tuple[int, int]] = []
        self.refreshed: list[str] = []
        self.funding = ""
        self.bench_kw: list[dict] = []

    def bench_init(self, era):
        self.init_benches.append(era.index)
        return {**INIT_SCORES, "_pointer": "" if era.warm_start_ckpt else INIT_PTR}

    def carry_king_pod(self, old, new):
        self.carried.append((old.index, new.index))

    def refresh_king_key(self, hotkey):
        self.refreshed.append(hotkey)
        return True

    def king_funding(self, era, gen):
        return self.funding

    def publish_bench(self, round_id, created_block, entries, **kw):
        self.bench_kw.append(dict(kw))
        return super().publish_bench(round_id, created_block, entries, **kw)


def _v2(cfg, *, ladder=True, window=3):
    armed = _armed(cfg)
    L = era_length_blocks(armed.round, armed.round.rolling_from_block)
    v2_block = armed.round.rolling_from_block + 2 * L
    return replace(armed, v2=replace(armed.v2, from_block=v2_block, random_init_eras=window,
                                     king_ladder=ladder)), v2_block, L


def _step(clock, ops, sched, client, *, from_block, to_block):
    _advance(clock, ops, sched, client, from_block=from_block, to_block=to_block)
    _join(sched)


def _tick_era(clock, ops, sched, client, start, L, *, last):
    """Tick through one era: early, then inside the pre-train window."""
    _step(clock, ops, sched, client, from_block=last, to_block=start + 5)
    _step(clock, ops, sched, client, from_block=start + 5, to_block=start + 10)
    _step(clock, ops, sched, client, from_block=start + 10, to_block=start + L - 900)
    return start + L - 900


def test_king_trains_once_per_seed_epoch_and_is_carried(cfg, tmp_path):
    v2cfg, v2_block, L = _v2(cfg)
    clock = Clock()
    ops = EpochOps(tmp_path, clock)
    sched, _ = _sched(v2cfg, tmp_path, clock, ops=ops)
    client = FakeClient()
    first = era_for_block(v2cfg.round, v2_block).index
    sched.tick(client, v2_block + 5)
    _join(sched)
    _step(clock, ops, sched, client, from_block=v2_block + 5, to_block=v2_block + 6)
    cur = sched.state.current
    assert cur.index == first and cur.epoch_start == first
    assert [c[1] for c in ops.king_calls] == [first]
    seed_block, base_seed = cur.seed_block, cur.base_seed
    last = v2_block + 6
    for k in (1, 2):
        last = _tick_era(clock, ops, sched, client, v2_block + k * L, L, last=last)
        cur = sched.state.current
        assert cur.index == first + k and cur.epoch_start == first
        # one seed set and one init for the whole epoch
        assert (cur.seed_block, cur.base_seed) == (seed_block, base_seed)
        assert cur.king_entry is not None
        assert cur.king().trained_pointer == _ptr(f"KKING-{first}")
    assert [c[1] for c in ops.king_calls] == [first]          # never retrained
    assert (first, first + 1) in ops.carried and (first + 1, first + 2) in ops.carried
    assert first not in ops.retired and first + 1 not in ops.retired
    # the init is benched ONCE for the epoch, then carried
    assert ops.init_benches == [first]
    assert sched.state.current.init_bench == INIT_SCORES
    assert "KING" in ops.refreshed                             # the king's key TTL


def test_a_warm_start_opens_a_new_epoch_with_one_new_king_leg(cfg, tmp_path):
    v2cfg, v2_block, L = _v2(cfg)
    clock = Clock()
    ops = EpochOps(tmp_path, clock)
    sched, _ = _sched(v2cfg, tmp_path, clock, ops=ops)
    client = FakeClient()
    first = era_for_block(v2cfg.round, v2_block).index
    sched.tick(client, v2_block + 5)
    _join(sched)
    _step(clock, ops, sched, client, from_block=v2_block + 5, to_block=v2_block + 6)
    base0 = sched.state.current.base_seed
    # a v2 king-only promotion effective at first + 3 (the window's end)
    ops.gens = (1, [(_ptr("KKING-first"), "")], first + 3)
    last = v2_block + 6
    for k in (1, 2, 3):
        last = _tick_era(clock, ops, sched, client, v2_block + k * L, L, last=last)
    cur = sched.state.current
    assert cur.index == first + 3 and cur.epoch_start == first + 3
    assert cur.warm_start_ckpt == _ptr("KKING-first")
    assert cur.base_seed != base0
    assert cur.seed_block == era_for_block(v2cfg.round, cur.start_block).seed_block
    assert [c[1] for c in ops.king_calls] == [first, first + 3]
    assert ops.init_benches == [first, first + 3]


def test_dethroner_entry_is_the_king_leg_for_the_rest_of_the_epoch(cfg, tmp_path):
    v2cfg, v2_block, L = _v2(cfg)
    clock = Clock()
    ops = EpochOps(tmp_path, clock)
    sched, _ = _sched(v2cfg, tmp_path, clock, ops=ops)
    client = FakeClient()
    first = era_for_block(v2cfg.round, v2_block).index
    q = ops.queue()
    ops.commits.append(_commit("ALFA", REF["ALFA"], v2_block - 100))
    q.add("ALFA", REF["ALFA"], reveal_block=v2_block - 100)
    sched.tick(client, v2_block + 5)
    _join(sched)
    _step(clock, ops, sched, client, from_block=v2_block + 5, to_block=v2_block + 6)
    _advance(clock, ops, sched, client, from_block=v2_block + 6, to_block=v2_block + EB + 3)
    assert len(ops.manifests) == 1
    m = ops.manifests[0]
    # the era stamp's seed block is the epoch's (train window [seed_block, end))
    assert m.era["seed_block"] == sched.state.current.seed_block
    # the init entry rides the v2 report
    assert ops.bench_kw[-1]["init"]["scores"] == INIT_SCORES
    # random init: the published random-init checkpoint
    assert ops.bench_kw[-1]["init"]["pointer"] == INIT_PTR
    # every ladder-era king entry names its payer (the operator trained it)
    # funding unknown to the fake (no completed-leg record) is never a slash
    assert m.entries[0].funded_by == "operator_fault" and m.entries[1].funded_by == ""
    # validators crown ALFA; the winner's checkpoint is adopted, no king leg
    ops.king_hk = ops.chain_king = "ALFA"
    _advance(clock, ops, sched, client, from_block=v2_block + EB + 3,
             to_block=v2_block + EB + 10)
    cur = sched.state.current
    assert cur.king_hotkey == "ALFA" and cur.king().trained_pointer == _ptr(f"ALFA-{first}")
    assert cur.king_funded_by == "king"
    _tick_era(clock, ops, sched, client, v2_block + L, L, last=v2_block + EB + 10)
    cur = sched.state.current
    assert cur.index == first + 1 and cur.king_hotkey == "ALFA"
    assert cur.king().trained_pointer == _ptr(f"ALFA-{first}")
    assert [c[0] for c in ops.king_calls] == ["KING"]             # ALFA never retrained


def test_dethrone_at_the_eras_last_settlement_adopts_after_the_roll(cfg, tmp_path):
    v2cfg, v2_block, L = _v2(cfg)
    clock = Clock()
    ops = EpochOps(tmp_path, clock)
    sched, _ = _sched(v2cfg, tmp_path, clock, ops=ops)
    client = FakeClient()
    first = era_for_block(v2cfg.round, v2_block).index
    q = ops.queue()
    ops.commits.append(_commit("ALFA", REF["ALFA"], v2_block - 100))
    q.add("ALFA", REF["ALFA"], reveal_block=v2_block - 100)
    sched.tick(client, v2_block + 5)
    _join(sched)
    _step(clock, ops, sched, client, from_block=v2_block + 5, to_block=v2_block + 6)
    _advance(clock, ops, sched, client, from_block=v2_block + 6, to_block=v2_block + EB + 3)
    # the receipt crowning ALFA is read only after the roll into first + 1
    _advance(clock, ops, sched, client, from_block=v2_block + EB + 3, to_block=v2_block + L + 2)
    assert sched.state.current.index == first + 1
    ops.king_hk = ops.chain_king = "ALFA"
    _advance(clock, ops, sched, client, from_block=v2_block + L + 2, to_block=v2_block + L + 9)
    cur = sched.state.current
    assert cur.king_hotkey == "ALFA" and cur.king().trained_pointer == _ptr(f"ALFA-{first}")
    assert [c[0] for c in ops.king_calls] == ["KING"]


def test_pre_v2_eras_are_untouched(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    ops = EpochOps(tmp_path, clock)
    sched, _ = _sched(armed, tmp_path, clock, ops=ops)
    client = FakeClient()
    start = armed.round.rolling_from_block
    L = era_length_blocks(armed.round, start)
    sched.tick(client, start + 5)
    _join(sched)
    _tick_era(clock, ops, sched, client, start + L, L, last=start + 5)
    cur = sched.state.current
    assert cur.epoch_start == -1 and cur.init_bench is None
    assert cur.seed_block == era_for_block(armed.round, cur.start_block).seed_block
    assert ops.carried == [] and ops.init_benches == [] and ops.refreshed == []
    # a king leg per era (incl. the next era's pre-train), as before
    first = era_for_block(armed.round, start).index
    assert [c[1] for c in ops.king_calls] == [first, first + 1, first + 2]


def test_without_the_ladder_no_init_bench_and_no_funding_stamp(cfg, tmp_path):
    v2cfg, v2_block, L = _v2(cfg, ladder=False)
    clock = Clock()
    ops = EpochOps(tmp_path, clock)
    ops.funding = "king"
    sched, _ = _sched(v2cfg, tmp_path, clock, ops=ops)
    client = FakeClient()
    sched.tick(client, v2_block + 5)
    _join(sched)
    _step(clock, ops, sched, client, from_block=v2_block + 5, to_block=v2_block + 6)
    assert ops.init_benches == [] and ops.refreshed == []


def test_lane_b_merges_apply_only_at_a_warm_start(cfg, tmp_path, monkeypatch):
    v2cfg, v2_block, L = _v2(cfg)
    clock = Clock()

    class Drv:
        def __init__(self):
            self.calls = []

        def init_for(self, era_index, ckpt, work_dir, apply_pending=True, era_start=None):
            self.calls.append((era_index, apply_pending))
            return (f"merged-{era_index}", [{"m": 1}]) if apply_pending else (ckpt, [])

    ops = EpochOps(tmp_path, clock)
    sched, _ = _sched(v2cfg, tmp_path, clock, ops=ops)
    sched.lane_b = Drv()
    monkeypatch.setattr(type(v2cfg.lane_b), "active", lambda self, b: True)
    first = era_for_block(v2cfg.round, v2_block).index
    sched.state.generations = {"1": {"members": [["M", "s"]], "effective_era": first + 3}}
    later = sched._new_era_state(FakeClient(), era_for_block(v2cfg.round, v2_block + 4 * L))
    start = sched._new_era_state(FakeClient(), era_for_block(v2cfg.round, v2_block + 3 * L))
    assert start.warm_start_ckpt == f"merged-{first + 3}" and start.warm_start_merges
    assert later.warm_start_ckpt == "M" and later.warm_start_merges == []
    assert later.base_seed == start.base_seed                 # one epoch
    assert sched.lane_b.calls == [(first + 3, False), (first + 3, True)]
