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

With the r2 transport (``[round] funded_harvest_transport = "r2"``,
:mod:`cascade.trainer.r2_transfer`) nothing lands on the box: the verified
copy is the R2 objects, so the release runs once every pushed file on R2
matches the pod's listing by size and sha256
(:func:`verify_r2_harvest_complete`), after the tensor digests and the
contract guard passed, and before the Hippius push.
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
RELEASE_LOG_R2 = ("pod released after verified r2 harvest (%d files, %.2f GB) — "
                  "Hippius push continues from R2")


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
                            reharvest: Callable[[], None],
                            release: Callable[[], None] | None = None
                            ) -> VerifiedHarvest | None:
    """Check the box copy at ``dest`` against the pod's listing of
    ``remote_dir``. Returns the verified summary, or ``None`` (keep the pod)
    when no release is armed, the copy is empty, the pod cannot list its
    dir, or the copy still differs after one ``reharvest()``. Never raises
    for a listing problem; ``reharvest`` failures propagate (the harvest
    itself failed, as today)."""
    # Set only when the pod LISTED its checkpoint and the copy still differs
    # after the re-harvest: the harvest is then not a success (harvest_guard).
    _local(runner).harvest_mismatch = False
    if release_callback(runner, release) is None:
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
        else:
            _local(runner).harvest_mismatch = True
    return None


def r2_manifest_mismatches(pod: dict, r2_files: dict) -> list[str]:
    """Why the R2 copy (``R2Checkpoint.files``: {name: {key, size, sha256}})
    is not the pod's checkpoint. Only the files the r2 transport carries are
    in scope — top-level regular files on the push allow-list
    (:func:`cascade.trainer.r2_transfer.checkpoint_file_allowed`); the
    transport deliberately leaves anything else on the pod, and so does
    the Hippius push. Every in-scope pod file must be on R2 with the same
    size and sha256, and R2 must hold nothing the pod does not list."""
    from .r2_transfer import checkpoint_file_allowed

    scoped = {p: v for p, v in pod.items()
              if "/" not in p and checkpoint_file_allowed(p)}
    on_r2 = {name: (int(m["size"]), str(m["sha256"])) for name, m in r2_files.items()}
    bad = manifest_mismatches(scoped, on_r2)
    return [b.replace("the box copy", "R2").replace("on the box", "on R2") for b in bad]


def verify_r2_harvest_complete(runner, host, remote_dir: str, r2_files: dict,
                               release: Callable[[], None] | None = None
                               ) -> VerifiedHarvest | None:
    """r2 transport (#398) twin of :func:`verify_harvest_complete`: the
    "verified copy" is the set of R2 objects (sizes + sha256 already streamed
    back by the box), checked against the POD's own listing of
    ``remote_dir``. ``None`` (keep the pod until leg end) when no release is
    armed (``release`` given or thread-armed, knob on), R2 holds nothing,
    the pod cannot list its dir, or anything differs. No re-harvest here: a
    second multipart upload costs as much as the first, and keeping the pod
    is the safe pre-change path. Never raises for a listing problem."""
    if release_callback(runner, release) is None:
        return None
    from . import remote

    if not r2_files:
        log.warning("r2 harvest of %s: nothing on R2 — pod kept", remote_dir)
        return None
    try:
        # Same ssh seam as the r2 part uploads (tests script the pod there).
        pod = remote.remote_file_manifest(
            host, remote_dir, runner=getattr(runner, "_r2_ssh_runner", None))
    except Exception as e:  # noqa: BLE001 — unverifiable ⇒ keep the pod
        log.warning("r2 harvest of %s: pod listing unavailable (%s) — pod kept "
                    "until the push ends", remote_dir, str(e)[-200:])
        return None
    bad = r2_manifest_mismatches(pod, r2_files)
    if bad:
        log.warning("r2 harvest of %s: R2 copy differs from the pod's checkpoint "
                    "(%d file(s): %s) — pod kept, not released", remote_dir, len(bad),
                    "; ".join(bad[:3]))
        return None
    return VerifiedHarvest(files=len(r2_files),
                           nbytes=sum(int(m["size"]) for m in r2_files.values()))


def release_callback(runner, release: Callable[[], None] | None = None
                     ) -> Callable[[], None] | None:
    """The release to run: an explicitly passed ``release`` (the r2 harvest's
    keyword), else the thread-armed one — either way ``None`` while
    ``[round] funded_release_after_harvest`` is off."""
    if release is None:
        return armed_release(runner)
    rnd = getattr(getattr(runner, "cfg", None), "round", None)
    if not bool(getattr(rnd, "funded_release_after_harvest", True)):
        return None
    return release


def release_if_verified(runner, verified: VerifiedHarvest | None,
                        release: Callable[[], None] | None = None, *,
                        where: str = "box") -> bool:
    """Run the armed (or given) release for a verified harvest; True once
    released. A release that raises is logged and reported unreleased (the
    leg's own teardown then runs as before)."""
    fn = release_callback(runner, release)
    if verified is None or fn is None:
        return False
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — the leg-end teardown still runs
        log.error("pod release after verified harvest failed (%s) — torn down at "
                  "leg end instead", e)
        return False
    # From here the box copy (or R2) is the only copy: harvest_guard never
    # deletes it on a later local-disk error.
    _local(runner).harvest_released = True
    log.info(RELEASE_LOG if where == "box" else RELEASE_LOG_R2,
             verified.files, verified.nbytes / 1e9)
    return True
