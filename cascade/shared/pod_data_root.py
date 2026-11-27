"""Pod-side cache dirs on the pod's large data filesystem (2026-11-27).

The lane B settlement rebase failed on ``valeval-blackwell4``: "pod disk
short … 6.8 GB free on the lane B cache filesystem < 26.0 GB needed" — its
``/`` overlay (where ``<workdir>/_lane_b_offload`` lived) is 43 GB while
``/workspace`` (159 GB) sat empty. Fixed by hand with a symlink; this module
makes that placement structural for every eval-pod cache dir (lane B's
``_lane_b_offload``, the validator's ``_eval_offload`` and
``_private_eval_offload``):

* a host's explicit ``cache_root`` (hosts toml, e.g. ``cache_root =
  "/workspace"``) roots the dirs — the callers' paths change, nothing is
  linked;
* otherwise the dir stays at ``<workdir>/<name>`` (every path the box uses
  is unchanged) and :func:`place_on_data_mount_command` makes it a symlink
  into the largest writable data mount (:data:`DATA_MOUNT_CANDIDATES`, e.g.
  ``/workspace``) when the dir does not exist yet (or is empty) and that
  mount is another filesystem with more free space. A non-empty real dir is
  never moved or removed (another process may be using it): its disk-short
  error names the alternative mount instead (:func:`alternative_mounts_command`,
  :func:`alternatives_text`).

Pure stdlib, shell strings only; nothing here runs anything.
"""

from __future__ import annotations

import os
import shlex

# Data mounts a rented GPU pod may carry beside its root overlay (Lium /
# RunPod-style volumes), in preference order.
DATA_MOUNT_CANDIDATES: tuple[str, ...] = ("/workspace",)
ALT_TAG = "CASCADE_ALT="


def base_dir(host) -> str:
    """``cache_root`` when the host sets one, else its ``workdir``."""
    root = str(getattr(host, "cache_root", "") or "").strip()
    return (root.rstrip("/") or "/") if root else str(host.workdir).rstrip("/") or "/"


def _avail(path_expr: str) -> str:
    return f"$(df -PB1 {path_expr} 2>/dev/null | awk 'NR==2{{print $4}}')"


def place_on_data_mount_command(link: str, *,
                                candidates: tuple[str, ...] = DATA_MOUNT_CANDIDATES) -> str:
    """Shell (run on the pod, never fails) that puts ``link`` on the largest
    writable data mount: when ``link`` does not exist (an empty real dir is
    removed first with ``rmdir``, which refuses a non-empty one) and a
    candidate is a writable dir on ANOTHER filesystem with more free bytes
    than ``link``'s parent, ``<candidate>/<basename>`` is created and
    ``link`` becomes a symlink to it. Anything else leaves ``link`` alone."""
    q = shlex.quote
    name = os.path.basename(link.rstrip("/"))
    parent = os.path.dirname(link.rstrip("/")) or "/"
    cands = " ".join(q(c) for c in candidates)
    body = (
        f"L={q(link)}; P={q(parent)}; mkdir -p \"$P\"; "
        "if [ -d \"$L\" ] && [ ! -L \"$L\" ]; then rmdir \"$L\" 2>/dev/null; fi; "
        "if [ ! -e \"$L\" ] && [ ! -L \"$L\" ]; then "
        f"best=; bestf={_avail(chr(34) + '$P' + chr(34))}; "
        f"for c in {cands}; do "
        "[ -d \"$c\" ] && [ -w \"$c\" ] || continue; "
        "[ \"$(stat -c %d \"$c\")\" = \"$(stat -c %d \"$P\")\" ] && continue; "
        f"f={_avail(chr(34) + '$c' + chr(34))}; "
        "[ \"${f:-0}\" -gt \"${bestf:-0}\" ] && { best=$c; bestf=$f; }; done; "
        f"[ -n \"$best\" ] && mkdir -p \"$best\"/{q(name)} && "
        f"ln -sfn \"$best\"/{q(name)} \"$L\"; "
        "fi"
    )
    return "{ " + body + "; } 2>/dev/null || true"


def alternative_mounts_command(path: str, *,
                               candidates: tuple[str, ...] = DATA_MOUNT_CANDIDATES) -> str:
    """Shell (never fails) printing ``CASCADE_ALT=<mount>:<free bytes>`` for
    each candidate data mount on another filesystem than ``path``."""
    q = shlex.quote
    cands = " ".join(q(c) for c in candidates)
    return ("{ " f"for c in {cands}; do [ -d \"$c\" ] || continue; "
            f"[ \"$(stat -L -c %d \"$c\")\" = \"$(stat -L -c %d {q(path)})\" ] && continue; "
            f"echo {ALT_TAG}$c:{_avail(chr(34) + '$c' + chr(34))}; done; "
            "} 2>/dev/null || true")


def parse_alternatives(stdout: str | None) -> list[tuple[str, int]]:
    """The ``(mount, free bytes)`` pairs :func:`alternative_mounts_command` printed."""
    out: list[tuple[str, int]] = []
    for line in str(stdout or "").splitlines():
        if not line.startswith(ALT_TAG):
            continue
        mount, _, free = line[len(ALT_TAG):].rpartition(":")
        try:
            out.append((mount, int(free)))
        except ValueError:
            continue
    return out


def alternatives_text(alts: list[tuple[str, int]], need: int = 0) -> str:
    """" — /workspace has 159.0 GB free on another filesystem: set
    cache_root = "/workspace" for this host in the hosts toml" ("" = none)."""
    fit = [(m, f) for m, f in alts if f > 0 and (not need or f >= need)] or \
        [(m, f) for m, f in alts if f > 0]
    if not fit:
        return ""
    mount, free = max(fit, key=lambda mf: mf[1])
    return (f" — {mount} has {free / 1e9:.1f} GB free on another filesystem: set "
            f"cache_root = \"{mount}\" for this host in the hosts toml (or move the "
            "cache dir there and symlink it)")


def local_alternatives(path: str, *,
                       candidates: tuple[str, ...] = DATA_MOUNT_CANDIDATES,
                       disk_usage=None) -> list[tuple[str, int]]:
    """Pod-side Python twin of :func:`alternative_mounts_command`."""
    import shutil

    du = disk_usage or shutil.disk_usage
    try:
        dev = os.stat(path).st_dev
    except OSError:
        return []
    out = []
    for c in candidates:
        try:
            if not os.path.isdir(c) or os.stat(c).st_dev == dev:
                continue
            out.append((c, int(du(c).free)))
        except OSError:
            continue
    return out
