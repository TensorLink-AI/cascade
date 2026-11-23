"""W15 prefetch cache: byte-budget LRU, KING_D and the house pinned, content
addressing re-checked on every read, determinism, resumability and
invariant 1 (plan rev6 §3.0 W15 row, §1.2 step 5, §10 "Data paths" 2; the
contract is in README.md).

Every test imports ``cascade.eval.lane_d_cache`` inside its body, so on main
each one fails with ``No module named 'cascade.eval.lane_d_cache'`` (a PATHS
module)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from .helpers import Entry, digest_of, entry_dirs, failing_fill, lane_d_cfg, read_tree, wrong_fill

ERA = 3600
ARM = 10 * ERA


def _cache(root, budget, **kw):
    from cascade.eval.lane_d_cache import PrefetchCache

    return PrefetchCache(root, budget_bytes=budget, **kw)


def _subdirs(root: Path) -> list[str]:
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir())


# ── construction and the error type ─────────────────────────────────────────


def test_cache_codes_and_error_type():
    from cascade.eval.lane_d_cache import CACHE_CODES, CacheError

    assert set(CACHE_CODES) == {"bad_digest", "digest_mismatch", "over_budget"}
    e = CacheError("over_budget", "detail")
    assert isinstance(e, RuntimeError) and e.code == "over_budget"


def test_new_cache_is_empty_and_creates_its_root(tmp_path):
    root = tmp_path / "a" / "cache"
    c = _cache(root, 10_000)
    assert root.is_dir()
    assert c.root == root and c.budget_bytes == 10_000
    assert c.entries() == [] and c.total_bytes() == 0 and c.pinned == frozenset()


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "100"])
def test_budget_must_be_a_positive_int(tmp_path, budget):
    from cascade.eval.lane_d_cache import PrefetchCache

    with pytest.raises((ValueError, TypeError)):
        PrefetchCache(tmp_path / "c", budget_bytes=budget)


@pytest.mark.parametrize("bad", ["", "../" + "a" * 61, "A" * 64, "a" * 63, "a" * 65,
                                 "g" * 64, "a" * 32 + "/" + "a" * 31])
def test_bad_digest_refused_everywhere(tmp_path, bad):
    from cascade.eval.lane_d_cache import CacheError

    c = _cache(tmp_path / "c", 10_000)
    e = Entry(1)
    for call in (lambda: c.get(bad), lambda: c.put(bad, e.fill), lambda: c.ensure(bad, e.fill)):
        with pytest.raises(CacheError) as err:
            call()
        assert err.value.code == "bad_digest"
    if bad:
        with pytest.raises(CacheError) as err:
            c.set_pinned(king_d=bad, house=None)
        assert err.value.code == "bad_digest"
    assert e.calls == 0 and c.entries() == []
    assert _subdirs(tmp_path / "c") == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c"], \
        "a traversal digest never reaches outside the root"


# ── content addressing on write ─────────────────────────────────────────────


def test_put_admits_exact_bytes_under_root_slash_digest(tmp_path):
    c = _cache(tmp_path / "c", 10_000)
    e = Entry(1, 1000)
    out = c.put(e.digest, e.fill)
    assert out == tmp_path / "c" / e.digest
    assert read_tree(out) == e.files
    assert c.entries() == [e.digest] and c.total_bytes() == 1000
    assert e.calls == 1


def test_put_of_a_cached_digest_does_not_refill(tmp_path):
    c = _cache(tmp_path / "c", 10_000)
    e = Entry(1)
    c.put(e.digest, e.fill)
    assert c.put(e.digest, e.fill) == tmp_path / "c" / e.digest
    assert e.calls == 1 and c.total_bytes() == e.size and c.entries() == [e.digest]


def test_fill_whose_bytes_do_not_hash_to_the_digest_is_refused(tmp_path, caplog):
    from cascade.eval.lane_d_cache import CacheError

    c = _cache(tmp_path / "c", 10_000)
    good, other = Entry(1), Entry(2)
    tampered = dict(good.files)
    tampered["weights.safetensors"] = b"X" + tampered["weights.safetensors"][1:]
    extra = {**good.files, "evil.py": b"import os\n"}
    missing = {k: v for k, v in good.files.items() if k != "config.json"}
    with caplog.at_level(logging.WARNING, logger="cascade.eval.lane_d_cache"):
        for files in (tampered, extra, missing, other.files):
            fill = wrong_fill(files)
            with pytest.raises(CacheError) as err:
                c.put(good.digest, fill)
            assert err.value.code == "digest_mismatch"
            assert fill.calls == 1
    assert c.entries() == [] and c.total_bytes() == 0
    assert _subdirs(tmp_path / "c") == [], "a refused fill leaves nothing behind"
    assert c.get(good.digest) is None
    refused = [r for r in caplog.records if r.getMessage().startswith("lane_d_cache_refused")]
    assert len(refused) == 4 and all("digest_mismatch" in r.getMessage() for r in refused)
    assert all(r.levelno >= logging.WARNING for r in refused)


def test_fill_that_raises_propagates_and_leaves_nothing(tmp_path):
    c = _cache(tmp_path / "c", 10_000)
    keep = Entry(1)
    c.put(keep.digest, keep.fill)
    e = Entry(2)
    fill = failing_fill(OSError("network down"))
    with pytest.raises(OSError, match="network down"):
        c.put(e.digest, fill)
    assert fill.calls == 1
    assert c.entries() == [keep.digest] and c.total_bytes() == keep.size
    assert _subdirs(tmp_path / "c") == [keep.digest]


# ── byte budget and LRU order ───────────────────────────────────────────────


def test_lru_evicts_least_recently_used_first(tmp_path, caplog):
    c = _cache(tmp_path / "c", 3000)
    a, b, d, e = Entry(1), Entry(2), Entry(3), Entry(4)
    for x in (a, b, d):
        c.put(x.digest, x.fill)
    assert c.entries() == [a.digest, b.digest, d.digest] and c.total_bytes() == 3000
    with caplog.at_level(logging.INFO, logger="cascade.eval.lane_d_cache"):
        c.put(e.digest, e.fill)
    assert c.entries() == [b.digest, d.digest, e.digest]
    assert c.total_bytes() == 3000
    assert not (tmp_path / "c" / a.digest).exists(), "an evicted entry is deleted from disk"
    assert entry_dirs(tmp_path / "c") == sorted([b.digest, d.digest, e.digest])
    evicted = [r.getMessage() for r in caplog.records
               if r.getMessage().startswith("lane_d_cache_evict")]
    assert len(evicted) == 1 and a.digest in evicted[0]


def test_get_refreshes_recency(tmp_path):
    c = _cache(tmp_path / "c", 3000)
    a, b, d, e = Entry(1), Entry(2), Entry(3), Entry(4)
    for x in (a, b, d):
        c.put(x.digest, x.fill)
    assert c.get(a.digest) == tmp_path / "c" / a.digest
    assert c.entries() == [b.digest, d.digest, a.digest]
    c.put(e.digest, e.fill)
    assert c.entries() == [d.digest, a.digest, e.digest]


def test_put_hit_refreshes_recency(tmp_path):
    c = _cache(tmp_path / "c", 3000)
    a, b, d, e = Entry(1), Entry(2), Entry(3), Entry(4)
    for x in (a, b, d):
        c.put(x.digest, x.fill)
    c.put(a.digest, a.fill)
    assert a.calls == 1
    c.put(e.digest, e.fill)
    assert c.entries() == [d.digest, a.digest, e.digest]


def test_eviction_is_by_bytes_not_by_count(tmp_path):
    c = _cache(tmp_path / "c", 5000)
    small = [Entry(i, 1000) for i in range(1, 5)]
    for x in small:
        c.put(x.digest, x.fill)
    big = Entry(9, 3500)
    c.put(big.digest, big.fill)
    # 4000 + 3500 > 5000: the three oldest small entries go (1000 + 3500 fits).
    assert c.entries() == [small[3].digest, big.digest]
    assert c.total_bytes() == 4500 <= c.budget_bytes


def test_total_bytes_never_exceeds_budget(tmp_path):
    c = _cache(tmp_path / "c", 4000)
    sizes = [700, 1300, 900, 2100, 400, 1800, 1000, 3000, 650, 1200]
    for i, size in enumerate(sizes):
        x = Entry(100 + i, size)
        c.put(x.digest, x.fill)
        assert c.total_bytes() <= c.budget_bytes
        assert c.total_bytes() == sum(len(b) for d in c.entries()
                                      for b in read_tree(tmp_path / "c" / d).values())
        assert c.entries()[-1] == x.digest


def test_entry_larger_than_the_budget_is_refused_and_evicts_nothing(tmp_path):
    from cascade.eval.lane_d_cache import CacheError

    c = _cache(tmp_path / "c", 3000)
    a, b = Entry(1), Entry(2)
    c.put(a.digest, a.fill)
    c.put(b.digest, b.fill)
    huge = Entry(3, 3001)
    with pytest.raises(CacheError) as err:
        c.put(huge.digest, huge.fill)
    assert err.value.code == "over_budget"
    assert c.entries() == [a.digest, b.digest] and c.total_bytes() == 2000
    assert _subdirs(tmp_path / "c") == sorted([a.digest, b.digest])


# ── KING_D and the house are pinned ─────────────────────────────────────────


def test_king_d_and_house_are_never_evicted(tmp_path):
    king, house = Entry(1), Entry(2)
    c = _cache(tmp_path / "c", 4000, king_d=king.digest, house=house.digest)
    assert c.pinned == frozenset({king.digest, house.digest})
    c.put(king.digest, king.fill)
    c.put(house.digest, house.fill)
    for i in range(10, 30):
        x = Entry(i)
        c.put(x.digest, x.fill)
        assert king.digest in c.entries() and house.digest in c.entries()
        assert c.total_bytes() <= 4000
    assert (tmp_path / "c" / king.digest).is_dir() and (tmp_path / "c" / house.digest).is_dir()
    assert read_tree(tmp_path / "c" / king.digest) == king.files
    # The two pinned entries keep their LRU position (the oldest) but are skipped.
    assert c.entries()[:2] == [king.digest, house.digest]
    assert len(c.entries()) == 4


def test_pins_set_before_the_entry_arrives_apply_on_admission(tmp_path):
    king = Entry(1)
    c = _cache(tmp_path / "c", 2000)
    c.set_pinned(king_d=king.digest, house=None)
    assert c.pinned == frozenset({king.digest})
    c.put(king.digest, king.fill)
    for i in range(10, 15):
        x = Entry(i)
        c.put(x.digest, x.fill)
    assert king.digest in c.entries()


def test_new_entry_that_only_fits_by_evicting_a_pin_is_refused(tmp_path):
    from cascade.eval.lane_d_cache import CacheError

    king, house, other = Entry(1, 1500), Entry(2, 1500), Entry(3, 1000)
    c = _cache(tmp_path / "c", 4000, king_d=king.digest, house=house.digest)
    for x in (king, house, other):
        c.put(x.digest, x.fill)
    big = Entry(4, 1200)       # fits only by evicting a pinned entry
    with pytest.raises(CacheError) as err:
        c.put(big.digest, big.fill)
    assert err.value.code == "over_budget"
    # Nothing was evicted for an admission that could not happen.
    assert c.entries() == [king.digest, house.digest, other.digest]
    fits = Entry(5, 1000)      # fits by evicting the unpinned entry
    c.put(fits.digest, fits.fill)
    assert c.entries() == [king.digest, house.digest, fits.digest]


def test_pinned_set_follows_king_and_house_changes(tmp_path):
    k1, k2, h1, h2 = Entry(1), Entry(2), Entry(3), Entry(4)
    c = _cache(tmp_path / "c", 4000, king_d=k1.digest, house=h1.digest)
    for x in (k1, h1):
        c.put(x.digest, x.fill)
    # A new king and a new house: the old ones become ordinary LRU entries.
    c.set_pinned(king_d=k2.digest, house=h2.digest)
    assert c.pinned == frozenset({k2.digest, h2.digest})
    for x in (k2, h2):
        c.put(x.digest, x.fill)
    assert c.entries() == [k1.digest, h1.digest, k2.digest, h2.digest]
    n1 = Entry(5)
    c.put(n1.digest, n1.fill)
    assert k1.digest not in c.entries(), "the dethroned king is evictable"
    n2 = Entry(6)
    c.put(n2.digest, n2.fill)
    assert h1.digest not in c.entries(), "the replaced house is evictable"
    assert c.entries() == [k2.digest, h2.digest, n1.digest, n2.digest]
    for i in range(10, 20):
        x = Entry(i)
        c.put(x.digest, x.fill)
    assert {k2.digest, h2.digest} <= set(c.entries())


def test_clearing_a_pin_makes_it_evictable(tmp_path):
    king = Entry(1)
    c = _cache(tmp_path / "c", 2000, king_d=king.digest)
    c.put(king.digest, king.fill)
    c.set_pinned(king_d=None, house=None)
    assert c.pinned == frozenset()
    for i in range(10, 13):
        x = Entry(i)
        c.put(x.digest, x.fill)
    assert king.digest not in c.entries()


def test_king_equal_to_house_is_one_pinned_entry(tmp_path):
    same = Entry(1)
    c = _cache(tmp_path / "c", 2000, king_d=same.digest, house=same.digest)
    assert c.pinned == frozenset({same.digest})
    c.put(same.digest, same.fill)
    for i in range(10, 14):
        x = Entry(i)
        c.put(x.digest, x.fill)
    assert c.entries()[0] == same.digest and len(c.entries()) == 2


# ── re-hash on read: a corrupt entry is refused, never served ───────────────


def _flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))


CORRUPTIONS = {
    "flipped_byte": lambda d: _flip_last_byte(d / "weights.safetensors"),
    "truncated": lambda d: (d / "weights.safetensors").write_bytes(
        (d / "weights.safetensors").read_bytes()[:-10]),
    "appended": lambda d: (d / "model.py").write_bytes((d / "model.py").read_bytes() + b"\n"),
    "file_removed": lambda d: (d / "config.json").unlink(),
    "file_added": lambda d: (d / "evil.py").write_bytes(b"import os\n"),
    "nested_file_added": lambda d: ((d / "pkg").mkdir(), (d / "pkg" / "x.py").write_bytes(b"1")),
    "same_size_swap": lambda d: (d / "weights.safetensors").write_bytes(
        bytes(reversed((d / "weights.safetensors").read_bytes()))),
}


@pytest.mark.parametrize("corruption", sorted(CORRUPTIONS))
def test_corrupt_entry_is_refused_on_read_and_dropped(tmp_path, caplog, corruption):
    c = _cache(tmp_path / "c", 10_000)
    e, other = Entry(1), Entry(2)
    c.put(e.digest, e.fill)
    c.put(other.digest, other.fill)
    CORRUPTIONS[corruption](tmp_path / "c" / e.digest)
    with caplog.at_level(logging.WARNING, logger="cascade.eval.lane_d_cache"):
        assert c.get(e.digest) is None
    assert not (tmp_path / "c" / e.digest).exists(), "the corrupt entry is removed"
    assert c.entries() == [other.digest] and c.total_bytes() == other.size
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(m.startswith("lane_d_cache_corrupt") and e.digest in m for m in msgs), msgs
    # The intact neighbour is still served.
    assert read_tree(c.get(other.digest)) == other.files


@pytest.mark.parametrize("corruption", sorted(CORRUPTIONS))
def test_ensure_refetches_a_corrupt_entry_and_serves_only_verified_bytes(tmp_path, corruption):
    c = _cache(tmp_path / "c", 10_000)
    e = Entry(1)
    assert c.ensure(e.digest, e.fill) == tmp_path / "c" / e.digest
    assert e.calls == 1
    assert c.ensure(e.digest, e.fill) == tmp_path / "c" / e.digest
    assert e.calls == 1, "a verified hit does not refetch"
    CORRUPTIONS[corruption](tmp_path / "c" / e.digest)
    out = c.ensure(e.digest, e.fill)
    assert e.calls == 2, "a corrupt entry is refetched"
    assert read_tree(out) == e.files
    assert c.entries() == [e.digest] and c.total_bytes() == e.size


def test_refetch_that_is_also_wrong_is_refused_not_served(tmp_path):
    from cascade.eval.lane_d_cache import CacheError

    c = _cache(tmp_path / "c", 10_000)
    e = Entry(1)
    c.put(e.digest, e.fill)
    _flip_last_byte(tmp_path / "c" / e.digest / "weights.safetensors")
    bad = dict(e.files)
    bad["weights.safetensors"] = bad["weights.safetensors"][:-1] + b"\x00"
    with pytest.raises(CacheError) as err:
        c.ensure(e.digest, wrong_fill(bad))
    assert err.value.code == "digest_mismatch"
    assert c.get(e.digest) is None and c.entries() == []
    assert _subdirs(tmp_path / "c") == []


def test_corrupt_pinned_king_is_refused_and_refetched(tmp_path):
    king = Entry(1)
    c = _cache(tmp_path / "c", 10_000, king_d=king.digest)
    c.put(king.digest, king.fill)
    _flip_last_byte(tmp_path / "c" / king.digest / "weights.safetensors")
    assert c.get(king.digest) is None, "a pin is never a reason to serve corrupt bytes"
    assert c.pinned == frozenset({king.digest}), "the pin itself stays"
    out = c.ensure(king.digest, king.fill)
    assert read_tree(out) == king.files and king.calls == 2
    for i in range(10, 25):
        x = Entry(i)
        c.put(x.digest, x.fill)
    assert king.digest in c.entries()


def test_every_read_rehashes_not_only_the_first(tmp_path):
    c = _cache(tmp_path / "c", 10_000)
    e = Entry(1)
    c.put(e.digest, e.fill)
    for _ in range(3):
        assert c.get(e.digest) == tmp_path / "c" / e.digest
    _flip_last_byte(tmp_path / "c" / e.digest / "weights.safetensors")
    assert c.get(e.digest) is None


def test_rehash_does_not_trust_mtimes_or_sizes(tmp_path):
    c = _cache(tmp_path / "c", 10_000)
    e = Entry(1)
    c.put(e.digest, e.fill)
    w = tmp_path / "c" / e.digest / "weights.safetensors"
    st = w.stat()
    _flip_last_byte(w)
    os.utime(w, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert w.stat().st_size == st.st_size and w.stat().st_mtime_ns == st.st_mtime_ns
    assert c.get(e.digest) is None


def test_digest_is_the_w4_manifest_digest(tmp_path):
    from cascade.lane_d.intake import checkpoint_manifest, manifest_digest

    c = _cache(tmp_path / "c", 10_000)
    e = Entry(1)
    out = c.put(e.digest, e.fill)
    assert manifest_digest(checkpoint_manifest(out)) == e.digest == digest_of(read_tree(out))


# ── resumable: reopen keeps verified entries and their order ────────────────


def test_reopen_keeps_entries_bytes_and_lru_order(tmp_path):
    root = tmp_path / "c"
    c = _cache(root, 3000)
    a, b, d = Entry(1), Entry(2), Entry(3)
    for x in (a, b, d):
        c.put(x.digest, x.fill)
    c.get(a.digest)
    order = c.entries()
    assert order == [b.digest, d.digest, a.digest]
    del c
    c2 = _cache(root, 3000)
    assert c2.entries() == order and c2.total_bytes() == 3000
    assert read_tree(c2.get(d.digest)) == d.files
    e = Entry(4)
    c2.put(e.digest, e.fill)
    assert c2.entries() == [a.digest, d.digest, e.digest]


def test_reopen_order_does_not_come_from_mtimes(tmp_path):
    root = tmp_path / "c"
    c = _cache(root, 10_000)
    xs = [Entry(i) for i in range(1, 5)]
    for x in xs:
        c.put(x.digest, x.fill)
    order = c.entries()
    # Reverse every file and directory mtime (newest becomes oldest).
    for i, x in enumerate(xs):
        t = 1_000_000_000 + (len(xs) - i) * 1000
        d = root / x.digest
        for p in [*d.rglob("*"), d]:
            os.utime(p, (t, t))
    assert _cache(root, 10_000).entries() == order


def test_reopen_ignores_staging_leftovers_from_a_crash(tmp_path):
    root = tmp_path / "c"
    c = _cache(root, 10_000)
    a, b = Entry(1), Entry(2)
    c.put(a.digest, a.fill)
    leftover = root / f".{b.digest}.staging-4242"
    leftover.mkdir()
    b.fill(leftover)
    c2 = _cache(root, 10_000)
    assert c2.entries() == [a.digest] and c2.total_bytes() == a.size
    assert c2.get(b.digest) is None
    assert c2.ensure(b.digest, b.fill) == root / b.digest and b.calls == 2


def test_reopen_with_a_smaller_budget_evicts_lru_but_keeps_pins(tmp_path):
    root = tmp_path / "c"
    king = Entry(1)
    c = _cache(root, 4000, king_d=king.digest)
    xs = [king] + [Entry(i) for i in range(2, 5)]
    for x in xs:
        c.put(x.digest, x.fill)
    c2 = _cache(root, 2000, king_d=king.digest)
    assert c2.total_bytes() <= 2000
    assert c2.entries() == [king.digest, xs[3].digest]
    assert entry_dirs(root) == sorted([king.digest, xs[3].digest])


# ── determinism ─────────────────────────────────────────────────────────────


def _script(c, entries):
    king, house = entries[0], entries[1]
    c.set_pinned(king_d=king.digest, house=house.digest)
    log = []
    for step, i in enumerate([0, 1, 2, 3, 4, 2, 5, 6, 3, 7, 8, 2, 9, 1, 10, 11]):
        x = entries[i]
        if step == 8:
            c.set_pinned(king_d=entries[5].digest, house=house.digest)
        if i in (2, 3) and step > 4:
            c.get(x.digest)
        else:
            c.put(x.digest, x.fill)
        log.append((tuple(c.entries()), c.total_bytes()))
    return log


def test_same_operations_give_the_same_cache(tmp_path):
    sizes = [900, 600, 700, 800, 500, 1000, 650, 750, 550, 850, 950, 400]
    one = _script(_cache(tmp_path / "one", 3500), [Entry(i, s) for i, s in enumerate(sizes)])
    two = _script(_cache(tmp_path / "two", 3500), [Entry(i, s) for i, s in enumerate(sizes)])
    assert one == two
    assert entry_dirs(tmp_path / "one") == entry_dirs(tmp_path / "two")
    final = list(one[-1][0])
    assert entry_dirs(tmp_path / "one") == sorted(final)
    # Reopening either root reproduces the same order.
    assert _cache(tmp_path / "one", 3500).entries() == final


# ── invariant 1: unarmed, nothing changes ───────────────────────────────────


def test_inv1_prefetch_cache_unchanged_when_unarmed(tmp_path):
    from cascade.eval.lane_d_cache import open_prefetch_cache

    root = tmp_path / "eval" / "lane_d_cache"
    unarmed = lane_d_cfg(from_block=0)
    assert unarmed.from_block == 0
    for boundary in (None, 0, ARM, ARM + ERA, 10 ** 9):
        assert open_prefetch_cache(unarmed, boundary, root, budget_bytes=10_000) is None
    # Armed for a later era: not yet active, same as unarmed.
    later = lane_d_cfg(from_block=ARM + 10 * ERA)
    assert open_prefetch_cache(later, ARM + ERA, root, budget_bytes=10_000) is None
    # Closed (past until_block): no cache either.
    closed = lane_d_cfg(from_block=ARM, until_block=ARM + 2 * ERA)
    assert open_prefetch_cache(closed, ARM + 3 * ERA, root, budget_bytes=10_000) is None
    assert not (tmp_path / "eval").exists(), "unarmed: no cache directory is created"


def test_inv1_prefetch_cache_armed_nonempty_lane_d_effect(tmp_path):
    from cascade.eval.lane_d_cache import PrefetchCache, open_prefetch_cache

    root = tmp_path / "eval" / "lane_d_cache"
    armed = lane_d_cfg(from_block=ARM)
    king, house = Entry(1), Entry(2)
    c = open_prefetch_cache(armed, ARM + ERA, root, budget_bytes=3000,
                            king_d=king.digest, house=house.digest)
    assert isinstance(c, PrefetchCache)
    assert root.is_dir() and c.budget_bytes == 3000
    assert c.pinned == frozenset({king.digest, house.digest})
    for x in (king, house, Entry(3), Entry(4)):
        c.put(x.digest, x.fill)
    assert c.entries()[:2] == [king.digest, house.digest] and len(c.entries()) == 3
