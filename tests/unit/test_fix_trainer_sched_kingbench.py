"""Under the three-suite bench gate (DEC-CA-0054) the era king's pod is kept
until the king's bench lands, and a king with no bench host (a restart, a
released pod) is benched on its era's re-rented operator king pod (G2)."""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.shared.manifest import BenchScores
from cascade.trainer import runner_rolling as RR
from cascade.trainer.rolling import EraState

from .test_rolling import EB, Clock, FakeClient, _armed, _join, _sched

SIX = {f: 0.5 for f in BenchScores.__dataclass_fields__}


def _gated(cfg):
    armed = _armed(cfg)
    return replace(armed, scoring=replace(armed.scoring, bench_gate_mode="shadow"))


def test_king_pod_is_kept_until_the_king_bench_lands(cfg, tmp_path):
    armed = _gated(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    numbers = {"bench": None}
    ops.bench_king = lambda entry, era: numbers["bench"]
    client = FakeClient()
    b0 = armed.round.rolling_from_block + 5
    sched.tick(client, b0)
    _join(sched)
    cur = sched.state.current
    assert cur.king_entry is not None and cur.king_bench is None
    sched.tick(client, b0 + 1)                      # inside the re-bench gap: no king thread
    _join(sched)
    assert cur.index not in ops.released            # the bench is still owed
    numbers["bench"] = dict(SIX)
    clock.t += sched.KING_REBENCH_SECONDS + 1
    sched.tick(client, b0 + 2)                      # re-bench lands
    _join(sched)
    assert cur.king_bench == SIX
    sched.tick(client, b0 + 3)
    assert ops.released == [cur.index]              # nothing left to do: released


def test_without_the_gate_the_pod_is_released_as_before(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    ops.bench_king = lambda entry, era: None
    sched.tick(FakeClient(), armed.round.rolling_from_block + 5)
    _join(sched)
    sched.tick(FakeClient(), armed.round.rolling_from_block + 6)
    assert ops.released == [sched.state.current.index]


class _Runner(RR._RollingMixin):
    def __init__(self, cfg, rent):
        self.cfg = cfg
        self.cascade_bench_plan = object()
        self._final_role_hosts = {}
        self._funded_king_lock = threading.Lock()
        self._funded_king_host = None
        self._rolling_king_host_era = None
        self._funded_wait_abort = threading.Event()
        self.rents: list[str] = []
        self._rent = rent
        self.benched = []

    def _rent_king_host(self, round_id):
        self.rents.append(round_id)
        return self._rent(self, round_id)

    def _fetch_checkpoint_dir(self, pointer):
        return f"/local/{pointer}"

    def _remote_bench_scores(self, host, entry, round_id, preset, *, role_dir, require_fev=False):
        self.benched.append((host, round_id, role_dir))
        return BenchScores(**SIX)


def _era():
    return EraState(index=12, start_block=12 * 4 * EB, seed_block=11 * 4 * EB, base_seed=4242)


@pytest.fixture
def pushes(monkeypatch):
    from cascade.trainer import bench_hook

    seen = []
    monkeypatch.setattr(bench_hook, "push_checkpoint",
                        lambda host, local, rid, preset, role, **kw: seen.append(
                            (host, local, rid, role)))
    monkeypatch.setattr(RR, "host_cannot_bench", lambda runner, host: "")
    return seen


def test_restart_king_without_a_host_is_benched_on_the_rerented_era_pod(cfg, pushes):
    host = SimpleNamespace(name="funded-king")

    def rent(r, rid):
        r._funded_king_host = host
        return host

    r = _Runner(_gated(cfg), rent)
    entry = SimpleNamespace(miner_hotkey="KING", trained_pointer="ptr", size="", role="king")
    out = r._rolling_bench_king(entry, _era())
    assert out == SIX
    assert r.rents == ["4242"]                                  # the era's own king pod name
    assert pushes == [(host, "/local/ptr", "4242", "king")]     # the checkpoint is pushed first
    assert r._rolling_king_hosts[12] is host                   # the rollover retires it
    assert r._rolling_king_host_era == 12
    # the second try reuses the mapped host: no second rent
    assert r._rolling_bench_king(entry, _era()) == SIX and r.rents == ["4242"]


def test_a_failed_bench_rent_never_releases_challenger_waits(cfg, pushes):
    def rent(r, rid):
        r._funded_wait_abort.set()           # what the king rent's give-up does
        raise RuntimeError("no capacity")

    r = _Runner(_gated(cfg), rent)
    entry = SimpleNamespace(miner_hotkey="KING", trained_pointer="ptr", size="", role="king")
    assert r._rolling_bench_king(entry, _era()) is None
    assert not r._funded_wait_abort.is_set()
    assert pushes == []


def test_a_later_eras_king_slot_is_never_taken(cfg, pushes):
    r = _Runner(_gated(cfg), lambda r, rid: pytest.fail("rented over the next era's slot"))
    r._funded_king_host, r._rolling_king_host_era = SimpleNamespace(), 13
    entry = SimpleNamespace(miner_hotkey="KING", trained_pointer="ptr", size="", role="king")
    assert r._rolling_bench_king(entry, _era()) is None
    assert r._rolling_king_host_era == 13


def test_no_rent_without_the_gate(cfg, pushes):
    r = _Runner(_armed(cfg), lambda r, rid: pytest.fail("rented with the gate off"))
    entry = SimpleNamespace(miner_hotkey="KING", trained_pointer="ptr", size="", role="king")
    assert r._rolling_bench_king(entry, _era()) is None
