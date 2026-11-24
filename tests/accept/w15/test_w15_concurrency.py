"""W15 prefetch cache on a shared root: concurrent fills never fail or serve a
partial entry (testnet E2E: two validators on one eval host filling the
shared lane D cache at once raised ``FileNotFoundError``; one cache opening
swept the other's live staging directory and a stale in-memory view replaced
an entry another process had just published).

Contract (README.md, "Shared root"): every operation holds an exclusive
``flock`` on ``root/.lane_d_cache.lock`` and re-reads the shared index first;
a ``put`` holds it through its ``fill``; entries and the index are published
by ``os.replace``. CPU only, small files, real processes (``fork``) and
threads."""

from __future__ import annotations

import multiprocessing
import os
import random
import threading
import time
import traceback
from pathlib import Path

import pytest

from .helpers import Entry, entry_dirs, read_tree

N_ENTRIES = 6


def _cache(root, budget, **kw):
    """A cache on ``root``; the shared-root contract's lock file is part of
    the surface (``LOCK_FILE``) and exists once a cache is open."""
    from cascade.eval.lane_d_cache import LOCK_FILE, PrefetchCache

    c = PrefetchCache(root, budget_bytes=budget, **kw)
    assert (Path(root) / LOCK_FILE).is_file()
    return c


class SlowEntry(Entry):
    """An entry whose fill writes one file at a time with a pause between
    them (a wide race window) and appends one line per call to ``log``."""

    def __init__(self, seed: int, log: Path, size: int = 1000):
        super().__init__(seed, size)
        self.log = log

    def fill(self, directory) -> None:
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(f"{self.digest}\n")
        for name, data in self.files.items():
            (Path(directory) / name).write_bytes(data)
            time.sleep(0.005)


def _worker(root: str, budget: int, log: str, out: str, seed: int, rounds: int,
            check_bytes: bool) -> None:
    try:
        entries = [SlowEntry(i, Path(log)) for i in range(N_ENTRIES)]
        rng = random.Random(seed)
        for _ in range(rounds):
            # A fresh cache per step, like a forecast request opening its own.
            c = _cache(root, budget)
            x = rng.choice(entries)
            path = c.ensure(x.digest, x.fill)
            assert path == Path(root) / x.digest
            if check_bytes:
                assert read_tree(path) == x.files, "a served entry is not its bytes"
        Path(out).write_text("ok", encoding="utf-8")
    except BaseException:
        Path(out).write_text(traceback.format_exc(), encoding="utf-8")


def _race(tmp_path, *, budget: int, workers: int, rounds: int, check_bytes: bool):
    root = tmp_path / "shared"
    log = tmp_path / "fills.log"
    log.touch()
    ctx = multiprocessing.get_context("fork")
    outs = [tmp_path / f"worker-{i}.out" for i in range(workers)]
    procs = [ctx.Process(target=_worker,
                         args=(str(root), budget, str(log), str(o), 1000 + i, rounds,
                               check_bytes))
             for i, o in enumerate(outs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
        assert p.exitcode == 0, f"worker exit {p.exitcode}"
    for o in outs:
        assert o.read_text(encoding="utf-8") == "ok", o.read_text(encoding="utf-8")
    return root, log.read_text(encoding="utf-8").split()


def test_two_processes_filling_one_root_never_fail_and_fill_each_digest_once(tmp_path):
    """Two validator processes race ensure() over the same digests on one
    root (a fresh cache object per step): no error, every served entry is its
    exact bytes, and each digest is fetched exactly once (the second filler
    waits on the lock, then hits)."""
    root, fills = _race(tmp_path, budget=1_000_000, workers=2, rounds=25, check_bytes=True)
    want = sorted(SlowEntry(i, tmp_path / "x").digest for i in range(N_ENTRIES))
    assert sorted(fills) == sorted(set(fills)), "a digest was filled twice"
    c = _cache(root, 1_000_000)
    assert sorted(c.entries()) == entry_dirs(root) == sorted(set(fills))
    assert set(fills) <= set(want)
    assert not [p.name for p in root.iterdir() if ".staging-" in p.name or ".tmp-" in p.name]


def test_racing_processes_under_eviction_pressure_stay_consistent(tmp_path):
    """Four processes, a budget of three entries: fills and evictions
    interleave across processes, nothing raises, and the shared index, the
    budget and the entry directories agree afterwards."""
    root, fills = _race(tmp_path, budget=3000, workers=4, rounds=20, check_bytes=False)
    assert fills
    c = _cache(root, 3000)
    assert c.total_bytes() <= 3000
    assert sorted(c.entries()) == entry_dirs(root)
    for d in c.entries():
        assert c.get(d) == root / d
    assert not [p.name for p in root.iterdir() if ".staging-" in p.name or ".tmp-" in p.name]


def test_opening_a_cache_never_sweeps_a_live_fill(tmp_path):
    """The testnet failure: a cache opened on the root while another is
    mid-fill swept the live staging directory as a crash leftover. Now the
    open waits for the fill to publish, and then sees the entry."""
    root = tmp_path / "shared"
    a = _cache(root, 10_000)
    x = Entry(1)
    started, release = threading.Event(), threading.Event()
    result: dict = {}

    def slow_fill(directory):
        x.fill(directory)
        started.set()
        assert release.wait(30)

    def filler():
        try:
            result["path"] = a.put(x.digest, slow_fill)
        except BaseException as e:      # surfaced by the asserts below
            result["error"] = e

    def opener():
        try:
            result["b"] = _cache(root, 10_000)
        except BaseException as e:
            result["b_error"] = e

    t1 = threading.Thread(target=filler)
    t1.start()
    assert started.wait(30)
    t2 = threading.Thread(target=opener)
    t2.start()
    time.sleep(0.3)
    assert "b" not in result, "a second cache opened (and swept staging) during a live fill"
    release.set()
    t1.join(30)
    t2.join(30)
    assert "error" not in result and "b_error" not in result, result
    assert result["path"] == root / x.digest and read_tree(root / x.digest) == x.files
    assert result["b"].entries() == [x.digest]
    assert result["b"].get(x.digest) == root / x.digest


def test_a_second_instance_sees_the_first_instances_fill(tmp_path):
    """Two cache objects opened on one root before either fills: an entry one
    admits is a hit for the other (no refill, no replace of a published
    entry), and an eviction by one is seen by the other."""
    root = tmp_path / "shared"
    one, two = _cache(root, 2500), _cache(root, 2500)
    a, b, c = Entry(1), Entry(2), Entry(3)
    one.put(a.digest, a.fill)
    assert two.get(a.digest) == root / a.digest
    assert two.put(a.digest, a.fill) == root / a.digest and a.calls == 1
    assert two.entries() == [a.digest]
    two.put(b.digest, b.fill)
    one.put(c.digest, c.fill)            # evicts a (LRU across both instances)
    assert one.entries() == two.entries() == [b.digest, c.digest]
    assert two.get(a.digest) is None
    assert entry_dirs(root) == sorted([b.digest, c.digest])


def test_threads_sharing_one_instance_do_not_interleave(tmp_path):
    root = tmp_path / "shared"
    cache = _cache(root, 1_000_000)
    entries = [Entry(i) for i in range(N_ENTRIES)]
    errors: list[str] = []

    def run(seed):
        rng = random.Random(seed)
        try:
            for _ in range(30):
                x = rng.choice(entries)
                assert read_tree(cache.ensure(x.digest, x.fill)) == x.files
        except BaseException:
            errors.append(traceback.format_exc())

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, errors[0]
    assert all(x.calls <= 1 for x in entries)
    assert sorted(cache.entries()) == entry_dirs(root)


def test_index_writes_are_atomic_for_a_reader(tmp_path):
    """Two cache objects in one process (same pid) admit entries from two
    threads while a reader polls the index file: no write collides with
    another's temp file and the reader always parses a complete index
    (unique temp file in root + os.replace, under the root lock)."""
    import json

    from cascade.eval.lane_d_cache import INDEX_FILE

    root = tmp_path / "shared"
    caches = [_cache(root, 1_000_000), _cache(root, 1_000_000)]
    stop = threading.Event()
    bad: list[str] = []

    def reader():
        while not stop.is_set():
            try:
                raw = (root / INDEX_FILE).read_text(encoding="utf-8")
                json.loads(raw)
            except FileNotFoundError:
                bad.append("index missing")
            except ValueError as e:
                bad.append(f"partial index: {e}")

    def writer(k):
        try:
            for i in range(20):
                x = Entry(100 * k + i)
                caches[k].put(x.digest, x.fill)
                caches[k].get(x.digest)
        except BaseException:
            bad.append(traceback.format_exc())

    t = threading.Thread(target=reader)
    t.start()
    writers = [threading.Thread(target=writer, args=(k,)) for k in (0, 1)]
    try:
        for w in writers:
            w.start()
        for w in writers:
            w.join(60)
    finally:
        stop.set()
        t.join(30)
    assert not bad, bad[:3]
    assert len(caches[0].entries()) == len(caches[1].entries()) == 40


@pytest.mark.parametrize("name", [".lane_d_cache.lock"])
def test_the_lock_file_is_never_an_entry(tmp_path, name):
    from cascade.eval.lane_d_cache import LOCK_FILE

    assert name == LOCK_FILE
    root = tmp_path / "c"
    c = _cache(root, 10_000)
    x = Entry(1)
    c.put(x.digest, x.fill)
    assert (root / LOCK_FILE).is_file()
    assert _cache(root, 10_000).entries() == [x.digest]
    assert os.path.basename(LOCK_FILE).startswith(".")


# ── shared pins and scoring leases ──────────────────────────────────────────
# A path the cache returns is read outside the lock (the forecast runs for
# minutes). Pins (KING_D, house) and every entry being fetched or scored are
# shared across processes: eviction by ANY process skips them.


def _lease_files(root: Path) -> list[str]:
    from cascade.eval.lane_d_cache import LEASE_PREFIX

    return sorted(p.name for p in root.iterdir()
                  if p.name.startswith(LEASE_PREFIX) and ".tmp-" not in p.name)


def _scorer_proc(root: str, flags: str, out: str) -> None:
    """Lease entry 1, admit it, signal ``ready``, then read its bytes until
    ``done`` appears (the evictor runs meanwhile) and record the verdict."""
    try:
        x = Entry(1)
        c = _cache(root, 2000)
        with c.lease(x.digest):
            path = c.ensure(x.digest, x.fill)
            (Path(flags) / "ready").touch()
            deadline = time.time() + 60
            while not (Path(flags) / "done").exists():
                assert read_tree(path) == x.files, "a leased entry changed mid-score"
                assert time.time() < deadline, "evictor never finished"
                time.sleep(0.01)
            assert read_tree(path) == x.files
        Path(out).write_text("ok", encoding="utf-8")
    except BaseException:
        Path(out).write_text(traceback.format_exc(), encoding="utf-8")


def _evictor_proc(root: str, flags: str, out: str) -> None:
    """Under a two-entry budget, admit entries 2..6: each admission must evict
    the LRU unprotected entry. Entry 1 (leased by the scorer) is the LRU."""
    try:
        deadline = time.time() + 60
        while not (Path(flags) / "ready").exists():
            assert time.time() < deadline, "scorer never got ready"
            time.sleep(0.01)
        c = _cache(root, 2000)
        for i in range(2, 7):
            y = Entry(i)
            c.put(y.digest, y.fill)
            assert Entry(1).digest in c.entries(), f"entry {i} evicted the leased entry"
        (Path(flags) / "done").touch()
        Path(out).write_text("ok", encoding="utf-8")
    except BaseException:
        (Path(flags) / "done").touch()
        Path(out).write_text(traceback.format_exc(), encoding="utf-8")


def test_a_scorer_lease_blocks_eviction_by_another_process(tmp_path):
    """An evictor process under budget pressure plus a scorer process holding a
    lease on the LRU entry: the leased entry is never evicted while the scorer
    reads it; once the lease is released it is evictable again."""
    root, flags = tmp_path / "shared", tmp_path / "flags"
    flags.mkdir()
    _cache(root, 2000)
    ctx = multiprocessing.get_context("fork")
    outs = [tmp_path / "scorer.out", tmp_path / "evictor.out"]
    procs = [ctx.Process(target=_scorer_proc, args=(str(root), str(flags), str(outs[0]))),
             ctx.Process(target=_evictor_proc, args=(str(root), str(flags), str(outs[1])))]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
        assert p.exitcode == 0
    for o in outs:
        assert o.read_text(encoding="utf-8") == "ok", o.read_text(encoding="utf-8")
    assert _lease_files(root) == [], "a released lease left its file"
    c = _cache(root, 2000)
    x = Entry(1)
    assert x.digest in c.entries() and c.total_bytes() <= 2000
    # Released: the next admissions evict it as the LRU entry it is.
    for i in (7, 8):
        y = Entry(i)
        c.put(y.digest, y.fill)
    assert x.digest not in c.entries()


def test_a_lease_taken_before_the_fill_protects_the_entry_once_admitted(tmp_path):
    root = tmp_path / "shared"
    scorer, other = _cache(root, 2000), _cache(root, 2000)
    x, y, z = Entry(1), Entry(2), Entry(3)
    with scorer.lease(x.digest):
        assert scorer.leased() == frozenset({x.digest})
        scorer.put(x.digest, x.fill)
        other.put(y.digest, y.fill)
        other.put(z.digest, z.fill)          # must evict y, not the leased (older) x
        assert other.entries() == [x.digest, z.digest]
    assert other.leased() == frozenset()


def test_stale_leases_are_reclaimed(tmp_path):
    """A lease whose owner pid is gone, or whose expiry passed, or that does
    not parse, is stale: eviction ignores it and the scan deletes it."""
    import json

    from cascade.eval.lane_d_cache import LEASE_PREFIX

    root = tmp_path / "shared"
    c = _cache(root, 2000)
    x, y, z = Entry(1), Entry(2), Entry(3)
    c.put(x.digest, x.fill)
    ctx = multiprocessing.get_context("fork")
    gone = ctx.Process(target=os._exit, args=(0,))
    gone.start()
    gone.join(30)
    dead_pid = gone.pid
    stale = {
        f"{LEASE_PREFIX}{x.digest}-{dead_pid}-a.json":
            {"digest": x.digest, "pid": dead_pid, "expires": time.time() + 3600},
        f"{LEASE_PREFIX}{x.digest}-{os.getpid()}-b.json":
            {"digest": x.digest, "pid": os.getpid(), "expires": time.time() - 1},
    }
    for name, body in stale.items():
        (root / name).write_text(json.dumps(body), encoding="utf-8")
    (root / f"{LEASE_PREFIX}{x.digest}-1-c.json").write_text("{not json", encoding="utf-8")
    assert c.leased() == frozenset()
    assert _lease_files(root) == []
    c.put(y.digest, y.fill)
    c.put(z.digest, z.fill)                  # x (LRU, no live lease) is evicted
    assert c.entries() == [y.digest, z.digest]


def test_lease_refuses_a_bad_digest_or_ttl(tmp_path):
    from cascade.eval.lane_d_cache import CacheError

    c = _cache(tmp_path / "c", 2000)
    with pytest.raises(CacheError), c.lease("../etc"):
        pass
    for bad in (0, -1, True, "1"):
        with pytest.raises(ValueError), c.lease(Entry(1).digest, ttl_s=bad):
            pass
    assert _lease_files(tmp_path / "c") == []


def test_pins_are_shared_across_instances_and_processes(tmp_path):
    """KING_D pinned by one cache is honoured by a pinless cache on the same
    root (another process's eviction must not drop the king); clearing the
    pins releases it."""
    root = tmp_path / "shared"
    king, y, z = Entry(1), Entry(2), Entry(3)
    pinned = _cache(root, 2000, king_d=king.digest)
    pinned.put(king.digest, king.fill)

    def evict_in_child(out):
        try:
            c = _cache(root, 2000)           # no pins of its own
            c.put(y.digest, y.fill)
            c.put(z.digest, z.fill)          # must evict y, never the shared-pinned king
            assert king.digest in c.entries()
            assert king.digest in c.protected()
            Path(out).write_text("ok", encoding="utf-8")
        except BaseException:
            Path(out).write_text(traceback.format_exc(), encoding="utf-8")

    out = tmp_path / "child.out"
    p = multiprocessing.get_context("fork").Process(target=evict_in_child, args=(str(out),))
    p.start()
    p.join(60)
    assert p.exitcode == 0 and out.read_text(encoding="utf-8") == "ok", out.read_text()
    assert pinned.entries() == [king.digest, z.digest]
    pinned.set_pinned(king_d=None, house=None)
    other = _cache(root, 2000)
    other.put(y.digest, y.fill)               # king is now the unprotected LRU
    assert king.digest not in other.entries()


def test_forecast_entry_holds_a_lease_while_scoring(tmp_path):
    """lane_d_forecasts.forecast_entry leases the entry from before the fetch
    until the forecast returns; the lease is gone afterwards, also when the
    scorer fails."""
    import numpy as np

    from cascade.eval.lane_d_forecasts import LaneDForecastError, forecast_entry

    root = tmp_path / "shared"
    cache = _cache(root, 10_000)
    x = Entry(1)
    cache.put(x.digest, x.fill)
    seen: list[frozenset] = []

    def scorer(ckpt, histories, *, context, horizon, contexts, device):
        seen.append(_cache(root, 10_000).leased())
        return np.zeros((len(histories), horizon, 9))

    wins = [{"window_id": "w0", "history": np.arange(64.0)}]
    q = forecast_entry(cache, x.digest, wins, context=4096, horizon=8, contexts=(4096,),
                       forecast=scorer)
    assert q.shape == (1, 8, 9)
    assert seen == [frozenset({x.digest})]
    assert cache.leased() == frozenset() and _lease_files(root) == []

    def crash(*a, **k):
        raise RuntimeError("CUDA OOM")

    with pytest.raises(LaneDForecastError):
        forecast_entry(cache, x.digest, wins, context=4096, horizon=8, contexts=(4096,),
                       forecast=crash)
    assert _lease_files(root) == []
