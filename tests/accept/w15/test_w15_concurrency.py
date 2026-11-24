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
