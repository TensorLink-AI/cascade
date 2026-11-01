"""Release a leg's pod the moment its harvested checkpoint is verified on the box.

Owner 2026-11-01: a funded / operator-rented leg's pod idled (~$23/h) through
the whole box → Hippius upload of its ~14.5 GB checkpoint (5FbUnQh5: harvest
done 03:24:48, ``lium rm`` only at 03:48:29). Nothing needs the pod after a
complete harvest unless the TRAINER benches the checkpoint there, so a leg
path that would tear the pod down at leg end anyway arms a release callback
(:func:`release_after_verified_harvest`); the harvest
(``TrainerRunner._harvest_funded_checkpoint``) then

1. pulls the checkpoint dir to the box (unchanged),
2. checks the box copy against the POD's own listing of that dir — every
   regular file present with the same size and sha256
   (:func:`cascade.trainer.remote.remote_file_manifest`); one re-harvest on a
   mismatch, and on a second mismatch the pod is simply kept (today's path),
3. runs the existing ingest guards (tensor digests, checkpoint guard,
   finiteness) on that copy,
4. releases the pod (:func:`release_if_verified`) — and only then uploads
   from the box copy. The box copy is kept until the upload returns a pinned
   ref; a failed upload leaves the pending-upload record the requeued leg's
   ``_reupload_pending`` re-uploads from, no pod needed.

The manifest is a COMPLETENESS check (no truncated or missing file), not a
trust anchor: the pod is miner-controlled, so the ingest guards stay the
integrity gate either way. Any doubt — no callback armed, the knob off, an
empty copy, an unreachable pod, a mismatch — keeps the pod, i.e. exactly the
pre-change behaviour (torn down when the leg returns).
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("cascade.trainer")

RELEASE_LOG = ("pod released after verified harvest (%d files, %.2f GB) — "
               "upload continues from the box copy")


@dataclass(frozen=True)
class VerifiedHarvest:
    """A box copy that matched the pod's listing file for file."""

    files: int
    nbytes: int


def _local(runner) -> threading.local:
    return runner.__dict__.setdefault("_leg_local_obj", threading.local())


@contextmanager
def release_after_verified_harvest(runner, release: Callable[[], None]):
    """Arm ``release`` (tears the leg's pod down) for every harvest this
    thread runs inside the block. Only a caller that would tear the pod down
    at leg end anyway may arm it — never for a pod the trainer still benches
    on, a shared operator lane, or the era king pod."""
    prev = arm_release(runner, release)
    try:
        yield
    finally:
        disarm_release(runner, prev)


def arm_release(runner, release: Callable[[], None]):
    """Arm ``release`` on this thread; returns the token :func:`disarm_release`
    restores (the non-``with`` form of :func:`release_after_verified_harvest`)."""
    local = _local(runner)
    prev = getattr(local, "harvest_release", None)
    local.harvest_release = release
    return prev


def disarm_release(runner, prev) -> None:
    _local(runner).harvest_release = prev


def armed_release(runner) -> Callable[[], None] | None:
    """The armed release callback, or ``None`` (knob off / nothing armed)."""
    loc = runner.__dict__.get("_leg_local_obj")
    fn = getattr(loc, "harvest_release", None) if loc is not None else None
    if fn is None:
        return None
    rnd = getattr(getattr(runner, "cfg", None), "round", None)
    if not bool(getattr(rnd, "funded_release_after_harvest", True)):
        return None
    return fn


def local_file_manifest(dest: Path | str) -> dict[str, tuple[int, str]]:
    """``{relative path: (size, sha256)}`` of every regular (non-symlink)
    file under ``dest`` — the box-side twin of the pod's listing."""
    from .ckpt_digest import file_sha256

    root = Path(dest)
    out: dict[str, tuple[int, str]] = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            p = Path(dirpath) / name
            if p.is_symlink() or not p.is_file():
                continue
            out[p.relative_to(root).as_posix()] = (p.stat().st_size, file_sha256(p))
    return out


def manifest_mismatches(pod: dict, box: dict) -> list[str]:
    """Why the box copy is not the pod's checkpoint, one reason per file."""
    bad: list[str] = []
    for path in sorted(pod):
        size, digest = pod[path]
        got = box.get(path)
        if got is None:
            bad.append(f"{path}: missing from the box copy")
        elif got[0] != size:
            bad.append(f"{path}: {got[0]} bytes on the box, {size} on the pod")
        elif got[1] != digest:
            bad.append(f"{path}: sha256 differs from the pod's")
    bad += [f"{path}: on the box but not on the pod" for path in sorted(set(box) - set(pod))]
    return bad


def verify_harvest_complete(runner, host, remote_dir: str, dest: Path,
                            reharvest: Callable[[], None]) -> VerifiedHarvest | None:
    """Check the box copy at ``dest`` against the pod's listing of
    ``remote_dir``. Returns the verified summary, or ``None`` (keep the pod)
    when no release is armed, the copy is empty, the pod cannot list its
    dir, or the copy still differs after one ``reharvest()``. Never raises
    for a listing problem; ``reharvest`` failures propagate (the harvest
    itself failed, as today)."""
    if armed_release(runner) is None:
        return None
    from concurrent.futures import ThreadPoolExecutor

    from . import remote

    for attempt in (1, 2):
        if not any(p.is_file() for p in Path(dest).rglob("*")):
            log.warning("harvest of %s: the box copy is empty — pod kept", remote_dir)
            return None
        # Hash both sides at once: the pod's sha256 over ssh while the box
        # hashes its copy (each ~disk speed over ~14.5 GB).
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="harvest-hash") as pool:
            box_f = pool.submit(local_file_manifest, dest)
            try:
                pod = remote.remote_file_manifest(host, remote_dir)
            except Exception as e:  # noqa: BLE001 — unverifiable ⇒ keep the pod
                box_f.result()
                log.warning("harvest of %s: pod listing unavailable (%s) — pod kept "
                            "until the upload ends", remote_dir, str(e)[-200:])
                return None
            box = box_f.result()
        if not pod:
            log.warning("harvest of %s: the pod lists no files — pod kept", remote_dir)
            return None
        bad = manifest_mismatches(pod, box)
        if not bad:
            return VerifiedHarvest(files=len(pod), nbytes=sum(s for s, _ in pod.values()))
        log.warning("harvest of %s (attempt %d): box copy differs from the pod's "
                    "checkpoint (%d file(s): %s)%s", remote_dir, attempt, len(bad),
                    "; ".join(bad[:3]),
                    " — re-harvesting" if attempt == 1 else " — pod kept, not released")
        if attempt == 1:
            reharvest()
    return None


def release_if_verified(runner, verified: VerifiedHarvest | None) -> bool:
    """Run the armed release for a verified harvest; True once released. A
    release that raises is logged and reported unreleased (the leg's own
    teardown then runs as before)."""
    fn = armed_release(runner)
    if verified is None or fn is None:
        return False
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — the leg-end teardown still runs
        log.error("pod release after verified harvest failed (%s) — torn down at "
                  "leg end instead", e)
        return False
    log.info(RELEASE_LOG, verified.files, verified.nbytes / 1e9)
    return True
