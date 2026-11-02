"""ONE space manager for every large local write on the box (2026-11-02).

The checkpoint volume filled to 100 %: the validator's eval checkpoint cache
held 29 x 14 GB that nothing ever evicted, and from then on every finished
training leg failed its harvest with ENOSPC. Every big local cache now goes
through this module:

* the validator eval cache (``[validator] eval_ckpt_cache_dir``, digest dirs),
* the trainer's ``_bench_ckpts`` / ``_funded_ckpt_guard`` fetch caches,
* harvested checkpoints under ``_funded_harvest`` whose push is settled
  (no ``.pending.json`` record) and no harvest is writing,
* hand-made ``_funded_harvest.box-*`` leftovers.

Policy: a byte budget per source and a free-space floor per filesystem, met
by evicting the least-recently-used RECLAIMABLE item first. Never
reclaimable: a checkpoint referenced by validator state (the king), the
cascade state or the warm-start init; a checkpoint of a manifest not yet
settled (the validator's pin file); a harvest whose upload is not confirmed
(its pending-upload record) or that is being written now; anything touched
within the grace. When the state that decides protection cannot be read the
manager evicts NOTHING (fail closed) and says so.

Eviction only ever deletes LOCAL COPIES of digest-pinned remote objects (a
re-fetch is self-verifying) or harvest copies whose push already landed, so
it never changes what any validator computes. Pure policy + a thin
filesystem layer; tests drive it on tmp dirs with a fake ``disk_usage``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("cascade.disk")

GIB = 1 << 30
# Digest-named cache dirs: HubRef digests with ":" -> "-" (sha256 OCI, or the
# HF-fallback "hf-<sha1>" form). Nothing else in a cache root is ever touched.
DIGEST_DIR_RE = re.compile(r"^(?:sha256|hf)-[0-9a-f]{40,64}$")
# A digest token anywhere in a state file ("repo@sha256:<hex>", "hf:<sha1>").
_REF_TOKEN_RE = re.compile(r"(?<![0-9a-z])(sha256|hf)[:-]([0-9a-f]{40,64})(?![0-9a-f])")
PINS_FILE = ".pinned.json"
PENDING_SUFFIX = ".pending.json"

# Resolved through this attribute so tests swap in a fake filesystem.
DISK_USAGE: Callable | None = None


def _usage(path: Path):
    return (DISK_USAGE or shutil.disk_usage)(path)


def free_bytes(path: Path | str) -> int | None:
    """Free bytes on ``path``'s filesystem (its nearest existing ancestor), or
    ``None`` when unmeasurable."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return int(_usage(p).free)
    except OSError:
        return None


def tree_bytes(path: Path) -> int:
    """Bytes of every regular file under ``path`` (symlinks not followed)."""
    p = Path(path)
    if p.is_file() and not p.is_symlink():
        return p.stat().st_size
    total = 0
    for dirpath, _dirs, files in os.walk(p):
        for name in files:
            f = Path(dirpath) / name
            with contextlib.suppress(OSError):
                if not f.is_symlink():
                    total += f.stat().st_size
    return total


_MTIME_WALK_LIMIT = 4096


def _last_used(path: Path) -> float:
    """The newest mtime anywhere in ``path``'s tree (bounded walk; LRU key and
    grace) — a file being written deep inside counts (review F11)."""
    t = path.stat().st_mtime
    if not path.is_dir():
        return t
    seen = 0
    for dirpath, dirs, files in os.walk(path):
        for name in dirs + files:
            seen += 1
            with contextlib.suppress(OSError):
                t = max(t, (Path(dirpath) / name).lstat().st_mtime)
            if seen >= _MTIME_WALK_LIMIT:
                return t
    return t


def touch(path: Path | str) -> None:
    """Mark a cache entry used now (the LRU clock). Best-effort."""
    with contextlib.suppress(OSError):
        os.utime(path, None)


def digest_dir_name(token: str) -> str | None:
    """``"repo@sha256:ab.."`` / ``"sha256:ab.."`` / ``"hf:..."`` → the cache dir
    name it lands in (``"sha256-ab.."``), or ``None``."""
    m = _REF_TOKEN_RE.search(str(token or ""))
    return f"{m.group(1)}-{m.group(2)}" if m else None


def referenced_digest_dirs(texts: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for text in texts:
        for m in _REF_TOKEN_RE.finditer(text or ""):
            out.add(f"{m.group(1)}-{m.group(2)}")
    return out


class ProtectionUnknown(RuntimeError):
    """A file that decides what must be kept exists but could not be read."""


def read_state_refs(paths: Iterable[Path | str]) -> set[str]:
    """Digest dirs referenced by the given state files. A missing file is
    fine; an unreadable one raises :class:`ProtectionUnknown` (fail closed)."""
    texts = []
    for raw in paths:
        p = Path(raw)
        try:
            texts.append(p.read_text(encoding="utf-8", errors="replace"))
        except FileNotFoundError:
            continue
        except OSError as e:
            raise ProtectionUnknown(f"{p}: {e}") from e
    return referenced_digest_dirs(texts)


def validator_state_paths(cfg) -> list[Path]:
    """Every validator-side file whose refs must survive eviction: the
    champion state (+ backup + settlement sidecar), the cascade reign state
    and the warm-start init (+ backups)."""
    v = getattr(cfg, "validator", None)
    if v is None:
        return []
    out: list[Path] = []
    for raw in (getattr(v, "state_db_path", ""), getattr(v, "cascade_state_db_path", ""),
                getattr(v, "warm_start_init_path", "")):
        if not raw:
            continue
        p = Path(raw)
        out += [p, p.with_name(p.name + ".bak"), p.with_name(p.name + ".settlement.json")]
    return out


def read_pins(root: Path | str) -> set[str]:
    """Digest dirs pinned by the validator for the manifest it is settling.
    Unreadable/corrupt pins raise :class:`ProtectionUnknown`."""
    p = Path(root) / PINS_FILE
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    except (OSError, ValueError) as e:
        raise ProtectionUnknown(f"{p}: {e}") from e
    return {str(x) for x in (doc.get("pinned") or []) if isinstance(x, str)}


def write_pins(root: Path | str, names: Iterable[str]) -> None:
    """Persist the unsettled-manifest pins (atomic). An empty set removes the file."""
    from .atomic import write_text_atomic

    p = Path(root) / PINS_FILE
    names = sorted({n for n in names if DIGEST_DIR_RE.match(n)})
    if not names:
        with contextlib.suppress(FileNotFoundError):
            p.unlink()
        return
    Path(root).mkdir(parents=True, exist_ok=True)
    write_text_atomic(p, json.dumps({"pinned": names, "at": time.time()}))


@dataclass(frozen=True)
class Candidate:
    path: Path
    nbytes: int
    last_used: float
    source: str


@dataclass
class Source:
    """One family of reclaimable items. ``list_items`` yields the item paths;
    ``reclaimable(path)`` is the per-item protection verdict."""

    name: str
    root: Path
    list_items: Callable[[Path], Iterable[Path]]
    reclaimable: Callable[[Path], bool]
    max_bytes: int = 0                 # 0 = no byte budget for this source

    def candidates(self, *, now: float, grace: float) -> tuple[list[Candidate], int]:
        """(reclaimable candidates, total bytes of ALL items in the source)."""
        root = Path(self.root)
        if not root.exists():
            return [], 0
        cands: list[Candidate] = []
        total = 0
        for item in self.list_items(root):
            try:
                if item.is_symlink():
                    continue
                size = tree_bytes(item)
                used = _last_used(item)
            except OSError:
                continue
            total += size
            if now - used < grace or not self.reclaimable(item):
                continue
            cands.append(Candidate(item, size, used, self.name))
        return cands, total


@dataclass
class Report:
    evicted: list[Candidate] = field(default_factory=list)
    free_after: int | None = None
    short_bytes: int = 0               # still missing to reach floor + need
    blocked: str = ""                  # why nothing could be evicted

    @property
    def freed(self) -> int:
        return sum(c.nbytes for c in self.evicted)


def plan(candidates: list[Candidate], *, totals: dict[str, int], budgets: dict[str, int],
         free: int | None, floor: int, need: int) -> tuple[list[Candidate], int]:
    """Pure eviction plan: LRU-first, first every over-budget source down to
    its budget, then globally until ``free >= floor + need``. Returns the
    plan and the bytes still short (0 when satisfied)."""
    order = sorted(candidates, key=lambda c: (c.last_used, str(c.path)))
    chosen: list[Candidate] = []
    totals = dict(totals)
    for c in order:
        cap = budgets.get(c.source, 0)
        if cap and totals.get(c.source, 0) > cap:
            chosen.append(c)
            totals[c.source] -= c.nbytes
    gained = sum(c.nbytes for c in chosen)
    if free is None:
        return chosen, 0
    want = floor + need
    for c in order:
        if free + gained >= want:
            break
        if c in chosen:
            continue
        chosen.append(c)
        gained += c.nbytes
    return chosen, max(0, want - (free + gained))


def _same_fs(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


_LOCK = threading.Lock()


def ensure_space(target: Path | str, sources: list[Source], *, floor_bytes: int,
                 need_bytes: int = 0, grace_seconds: float = 3600.0,
                 now: float | None = None, why: str = "") -> Report:
    """Evict reclaimable items (LRU first) so ``target``'s filesystem has at
    least ``floor_bytes + need_bytes`` free and every source sits within its
    byte budget. Only items on ``target``'s filesystem count toward the floor.
    Returns what happened; never raises for a filesystem problem."""
    t = time.time() if now is None else float(now)
    target = Path(target)
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    with _LOCK:
        cands: list[Candidate] = []
        totals: dict[str, int] = {}
        budgets: dict[str, int] = {}
        for s in sources:
            got, total = s.candidates(now=t, grace=grace_seconds)
            totals[s.name] = total
            budgets[s.name] = int(s.max_bytes or 0)
            cands += got
        free = free_bytes(probe)
        # Only items on the target's filesystem help its floor; an item
        # elsewhere is still evicted down to its own source's byte budget.
        on_target = [c for c in cands if _same_fs(c.path, probe)]
        off_target = [c for c in cands if c not in on_target]
        chosen, short = plan(on_target, totals=totals, budgets=budgets, free=free,
                             floor=int(floor_bytes), need=int(need_bytes))
        over, _ = plan(off_target, totals=totals, budgets=budgets, free=None, floor=0, need=0)
        report = Report()
        for c in chosen + over:
            try:
                _remove(c.path)
            except OSError as e:
                log.warning("disk: could not evict %s (%s)", c.path, e)
                continue
            report.evicted.append(c)
            log.warning("disk: evicted %s %s (%.1f GB, last used %s)%s", c.source, c.path.name,
                        c.nbytes / 1e9, time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime(c.last_used)),
                        f" — {why}" if why else "")
        report.free_after = free_bytes(probe)
        if report.free_after is not None:
            report.short_bytes = max(0, int(floor_bytes) + int(need_bytes) - report.free_after)
        else:
            report.short_bytes = short
        return report


# ── the concrete sources ─────────────────────────────────────────────────────


def _children(root: Path) -> Iterable[Path]:
    with contextlib.suppress(OSError):
        yield from sorted(root.iterdir())


def digest_cache_source(name: str, root: Path | str, protected: set[str], *,
                        max_bytes: int = 0) -> Source:
    """A digest-named fetch cache (validator eval cache, bench/guard caches)."""
    def items(r: Path):
        return [p for p in _children(r) if p.is_dir() and DIGEST_DIR_RE.match(p.name)]

    return Source(name=name, root=Path(root), list_items=items,
                  reclaimable=lambda p: p.name not in protected, max_bytes=max_bytes)


def harvest_source(root: Path | str, active: Callable[[], set[str]]) -> Source:
    """``_funded_harvest/<seed>/<item>``: reclaimable once no upload is
    pending for it (``<item>.pending.json`` absent) and no harvest is
    writing it (``active()`` — resolved paths)."""
    def items(r: Path):
        out = []
        for seed_dir in _children(r):
            if seed_dir.is_dir() and not seed_dir.is_symlink():
                out += [p for p in _children(seed_dir)
                        if p.is_dir() and not p.name.endswith(PENDING_SUFFIX)]
        return out

    def ok(p: Path) -> bool:
        # ``<item>.partial-<pid>-<tid>`` is a harvest being extracted (or a
        # crash leftover): judged as its item — active ⇒ never reclaimed.
        base = p.with_name(p.name.split(".partial-", 1)[0])
        if base.with_name(base.name + PENDING_SUFFIX).exists():
            return False                    # push not confirmed: never reclaimable
        return str(base.resolve()) not in active()

    return Source(name="harvest", root=Path(root), list_items=items, reclaimable=ok)


def leftover_source(work_root: Path | str, active: Callable[[], set[str]] = set) -> list[Source]:
    """Hand-made ``_funded_harvest.box-*`` copies beside the harvest dir —
    judged exactly like the harvest dir (a pending upload record keeps an
    item; review F11)."""
    out = []
    for p in sorted(Path(work_root).glob("_funded_harvest.box-*")):
        src = harvest_source(p.resolve(), active)
        src.name = f"leftover:{p.name}"
        out.append(src)
    return out


def eval_cache_root(cfg) -> Path:
    return Path(getattr(getattr(cfg, "validator", None), "eval_ckpt_cache_dir", "")
                or "_eval_ckpts")


def eval_cache_protected(cfg, cache_root: Path | str, extra: Iterable[str] = ()) -> set[str]:
    """Never-evict set for the eval cache: refs in validator / cascade / warm-
    start state, the validator's unsettled-manifest pins, plus ``extra``.
    Raises :class:`ProtectionUnknown` when any of it cannot be read — and
    when the champion state file is MISSING while the cache holds
    checkpoints (a wrong working directory or a moved state file must never
    read as "no king": review F11)."""
    state = Path(getattr(getattr(cfg, "validator", None), "state_db_path", "") or "")
    if str(state) and not state.is_file():
        root = Path(cache_root)
        if root.is_dir() and any(DIGEST_DIR_RE.match(p.name) for p in root.iterdir()):
            raise ProtectionUnknown(f"validator state {state} not found (cwd {Path.cwd()}) "
                                    "— protecting everything")
    return (read_state_refs(validator_state_paths(cfg)) | read_pins(cache_root)
            | {n for n in extra if n})


@contextlib.contextmanager
def space_lock(root: Path | str):
    """Cross-process lock shared by the trainer's reclaim and the
    validator's pin-then-trim (``<eval cache>/.space.lock``). A missing root
    takes no lock (nothing there to evict)."""
    import fcntl

    root = Path(root)
    if not root.is_dir():
        yield
        return
    with open(root / ".space.lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def box_sources(cfg, work_root: Path | str | None = None, *,
                active_harvests: Callable[[], set[str]] = set,
                extra_protected: Iterable[str] = ()) -> list[Source]:
    """Every reclaimable source on this box, under ONE protection set. Raises
    :class:`ProtectionUnknown` when the protection state is unreadable."""
    v = getattr(cfg, "validator", None)
    cache = eval_cache_root(cfg)
    protected = eval_cache_protected(cfg, cache, extra_protected)
    srcs = [digest_cache_source(
        "eval_cache", cache, protected,
        max_bytes=int(float(getattr(v, "eval_ckpt_cache_max_gb", 0.0) or 0.0) * GIB))]
    if work_root is not None:
        wr = Path(work_root)
        srcs += [digest_cache_source("bench_ckpts", wr / "_bench_ckpts", protected),
                 digest_cache_source("ckpt_guard", wr / "_funded_ckpt_guard", protected),
                 harvest_source(wr / "_funded_harvest", active_harvests)]
        srcs += leftover_source(wr, active_harvests)
    return srcs


def reclaim(cfg, target: Path | str, *, need_bytes: int = 0, floor_bytes: int | None = None,
            work_root: Path | str | None = None, active_harvests: Callable[[], set[str]] = set,
            extra_protected: Iterable[str] = (), why: str = "") -> Report:
    """The single entry point: make room on ``target``'s filesystem.
    ``floor_bytes`` None = ``[validator] eval_ckpt_cache_min_free_gb``."""
    v = getattr(cfg, "validator", None)
    if floor_bytes is None:
        floor_bytes = int(float(getattr(v, "eval_ckpt_cache_min_free_gb", 0.0) or 0.0) * GIB)
    grace = float(getattr(v, "eval_ckpt_cache_grace_seconds", 3600) or 0)
    with space_lock(eval_cache_root(cfg)):
        try:
            sources = box_sources(cfg, work_root, active_harvests=active_harvests,
                                  extra_protected=extra_protected)
        except ProtectionUnknown as e:
            log.error("disk: protection state unreadable (%s) — evicting NOTHING", e)
            free = free_bytes(target)
            short = 0 if free is None else max(0, int(floor_bytes) + int(need_bytes) - free)
            return Report(free_after=free, short_bytes=short, blocked=str(e))
        return ensure_space(target, sources, floor_bytes=int(floor_bytes),
                            need_bytes=int(need_bytes), grace_seconds=grace, why=why)
