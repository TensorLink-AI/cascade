"""Pull a pod's worker logs onto the box BEFORE the pod is removed (owner
2026-11-26).

Every abort / reject path that tears a pod down (THROUGHPUT ABORT and the
first-step deadline, short pod, lemon / not-ready, NCCL preflight, stale
image, worker crash) used to lose the pod's own evidence with the pod. Before
the teardown, :func:`capture` copies, best effort:

* the worker's ``stdout`` / ``stderr`` and ``progress.json`` (plus ``pid`` /
  ``exit_code``) of the newest detached run dirs on the pod
  (``_train_work/_dispatch/*``, the run dir of the leg first when known),
  each tailed to a per-file cap;
* ``nvidia-smi`` output;
* ``dmesg | tail`` (``journalctl -k`` when dmesg is not readable);
* uptime / memory / disk facts,

into ``<root>/<pod-name>-<utc>/`` on the box (default root
``/root/cascade/_pod_postmortem``: the trainer work root's parent; override
with ``CASCADE_POD_POSTMORTEM_DIR``). Small text only — never a checkpoint:
the pod side only ``tail -c``s the named files.

The capture runs in its own daemon thread under a hard wall
(:data:`TIMEOUT_SECONDS`): the caller waits at most that long and then tears
down regardless; the ssh child is killed at the same deadline. The stream is
capped at :data:`MAX_BYTES` on the box side too (a hostile pod cannot flood
it). Any failure is logged and swallowed — a capture never blocks, delays
past its timeout, or fails a teardown. Postmortems older than
:data:`KEEP_DAYS` are pruned on every capture.

The pod is untrusted (a payer controls it): file names it could influence
are sanitised, nothing it sends is executed or extracted as an archive.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import secrets
import select
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("cascade.trainer.postmortem")

TIMEOUT_SECONDS = 60.0
MAX_BYTES = 50 * 1024 * 1024
KEEP_DAYS = 14
RUN_DIRS = 3
DIRNAME = "_pod_postmortem"
ENV_DIR = "CASCADE_POD_POSTMORTEM_DIR"
_RUN_ROOT = "_train_work/_dispatch"
_NAME_OK = re.compile(r"[^A-Za-z0-9_.-]")
_DIR_RE = re.compile(r"^[A-Za-z0-9_.-]+-\d{8}T\d{6}Z(-\d+)?$")


def _utc(t: float) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(float(t)))


def _safe(name: str, default: str = "pod") -> str:
    s = _NAME_OK.sub("_", str(name or ""))[:96].lstrip(".")
    return s or default


def root_for(work_root) -> Path:
    """``$CASCADE_POD_POSTMORTEM_DIR`` or ``<work root's parent>/_pod_postmortem``
    (the live trainer's ``./_train_work`` in /root/cascade ⇒
    ``/root/cascade/_pod_postmortem``)."""
    env = os.environ.get(ENV_DIR, "").strip()
    if env:
        return Path(env)
    return Path(work_root).resolve().parent / DIRNAME


def remote_command(workdir: str, *, run_dir: str | None, nonce: str,
                   per_file_bytes: int) -> str:
    """The read-only pod-side script: each section is announced by a
    ``@@PM-<nonce>@@ <name>`` line (the random nonce keeps log content from
    forging a section)."""
    mark = f"@@PM-{nonce}@@"
    cap = int(max(1024, per_file_bytes))
    dirs = (f"{shlex.quote(run_dir)} " if run_dir else "")
    return (
        f"cd {shlex.quote(workdir or '.')} 2>/dev/null; "
        f"em(){{ printf '\\n{mark} %s\\n' \"$1\"; }}; "
        f"for d in {dirs}$(ls -1dt {_RUN_ROOT}/*/ 2>/dev/null | head -n {RUN_DIRS}); do "
        "d=${d%/}; [ -d \"$d\" ] || continue; b=$(basename \"$d\"); "
        "for f in stdout stderr progress.json exit_code pid; do "
        f"[ -f \"$d/$f\" ] && {{ em \"$b.$f\"; tail -c {cap} \"$d/$f\" 2>/dev/null; }}; "
        "done; done; "
        "em nvidia-smi.txt; (timeout 20 nvidia-smi 2>&1 || echo 'nvidia-smi unavailable') "
        "| head -c 1048576; "
        "em dmesg_tail.txt; ((timeout 10 dmesg -T 2>/dev/null || "
        "timeout 10 journalctl -k --no-pager 2>/dev/null) | tail -n 300) "
        "|| echo 'dmesg unavailable'; "
        "em pod_facts.txt; date -u; uptime 2>&1; free -m 2>&1; df -h . /dev/shm 2>&1; "
        f"ls -la --time-style=+%FT%T {_RUN_ROOT} 2>&1 | tail -n 20; true")


def stream_capped(argv: list[str], timeout: float, max_bytes: int) -> bytes:
    """Run ``argv``; its stdout, at most ``max_bytes`` (the child is killed
    past the cap or the deadline). Never raises for the child's own failure."""
    deadline = time.monotonic() + float(timeout)
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    out = bytearray()
    try:
        fd = proc.stdout.fileno()
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            ready, _, _ = select.select([fd], [], [], min(left, 1.0))
            if not ready:
                if proc.poll() is not None:
                    break
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            out += chunk[: max(0, max_bytes - len(out))]
            if len(out) >= max_bytes:
                break
    finally:
        with contextlib.suppress(Exception):
            if proc.poll() is None:
                proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
        with contextlib.suppress(Exception):
            proc.stdout.close()
    return bytes(out)


def split_sections(data: bytes, nonce: str) -> list[tuple[str, bytes]]:
    """``(file name, bytes)`` per announced section, names sanitised and
    de-duplicated; text before the first marker is ``preamble.txt``."""
    mark = f"@@PM-{nonce}@@ ".encode()
    out: list[tuple[str, bytes]] = []
    name, buf = "preamble.txt", bytearray()
    for line in data.splitlines(keepends=True):
        if line.startswith(mark):
            out.append((name, bytes(buf)))
            name = _safe(line[len(mark):].decode("utf-8", "replace").strip(), "section")
            buf = bytearray()
        else:
            buf += line
    out.append((name, bytes(buf)))
    seen: dict[str, int] = {}
    named = []
    for n, b in out:
        if n == "preamble.txt" and not b.strip():
            continue
        k = seen.get(n, 0)
        seen[n] = k + 1
        named.append((n if k == 0 else f"{n}.{k}", b))
    return named


def prune(root, *, keep_days: int = KEEP_DAYS, now: float | None = None) -> int:
    """Remove postmortem dirs under ``root`` older than ``keep_days`` (by
    mtime; only names this module writes). Returns how many were removed."""
    root = Path(root)
    if not root.is_dir():
        return 0
    cut = (time.time() if now is None else float(now)) - keep_days * 86400.0
    n = 0
    for d in root.iterdir():
        try:
            if d.is_dir() and not d.is_symlink() and _DIR_RE.match(d.name) \
                    and d.stat().st_mtime < cut:
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        except OSError:
            continue
    return n


def _write_capture(dest: Path, data: bytes, nonce: str, max_bytes: int) -> int:
    total = 0
    for name, body in split_sections(data, nonce):
        body = body[: max(0, max_bytes - total)]
        (dest / name).write_bytes(body)
        total += len(body)
    return total


def capture(host, *, root, pod_name: str, reason: str = "", run_dir: str | None = None,
            timeout: float | None = None, max_bytes: int | None = None, ssh=None,
            clock=time.time) -> str | None:
    """Best-effort postmortem of ``host`` (a :class:`~cascade.trainer.remote.
    RemoteHost`) into ``root/<pod_name>-<utc>/``. Returns that path (also when
    the pod part timed out or failed: ``meta.json`` says so), or None when
    nothing could be written. Waits at most ``timeout`` seconds.

    ``ssh(argv, timeout, max_bytes) -> bytes`` is the transport
    (:func:`stream_capped`; a test seam)."""
    timeout = TIMEOUT_SECONDS if timeout is None else float(timeout)
    max_bytes = MAX_BYTES if max_bytes is None else int(max_bytes)
    t0 = time.monotonic()
    try:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(Exception):
            prune(root, now=float(clock()))
        base = f"{_safe(pod_name)}-{_utc(clock())}"
        dest = root / base
        k = 1
        while dest.exists():
            dest = root / f"{base}-{k}"
            k += 1
        dest.mkdir(parents=True)
    except Exception as e:  # noqa: BLE001 — evidence never blocks a teardown
        log.warning("pod postmortem for %s not written: %s", pod_name, e)
        return None
    meta = {"pod": str(pod_name), "reason": str(reason)[:2000], "host": "",
            "run_dir": run_dir or "", "started_utc": _utc(clock()),
            "timeout_s": timeout, "max_bytes": max_bytes, "status": "running"}

    def _meta(**kw) -> None:
        meta.update(kw)
        with contextlib.suppress(Exception):
            (dest / "meta.json").write_text(json.dumps(meta, sort_keys=True, indent=1))

    _meta()
    nonce = secrets.token_hex(8)
    done = threading.Event()

    def _work() -> None:
        try:
            from .remote import build_ssh_argv

            # ``host`` may be a resolver (a provider lookup): it runs here,
            # inside the bounded thread
            h = host() if callable(host) else host
            if h is None or not getattr(h, "host", ""):
                _meta(status="no_address")
                return
            _meta(host=f"{getattr(h, 'host', '')}:{getattr(h, 'port', '')}")
            cmd = remote_command(str(getattr(h, "workdir", ".") or "."), run_dir=run_dir,
                                 nonce=nonce, per_file_bytes=max_bytes // 8)
            argv = build_ssh_argv(h, cmd)
            argv[1:1] = ["-o", "ConnectTimeout=15"]
            left = max(1.0, timeout - (time.monotonic() - t0) - 1.0)
            data = (ssh or stream_capped)(argv, left, max_bytes)
            if done.is_set():
                return                      # the caller gave up: do not write late
            n = _write_capture(dest, data or b"", nonce, max_bytes)
            _meta(status="ok", bytes=n)
        except Exception as e:  # noqa: BLE001
            _meta(status=f"failed: {str(e)[:300]}")
        finally:
            done.set()

    th = threading.Thread(target=_work, name=f"postmortem-{_safe(pod_name)[:24]}", daemon=True)
    th.start()
    finished = done.wait(max(0.0, timeout - (time.monotonic() - t0)))
    if not finished:
        done.set()
        _meta(status="timeout")
        log.warning("pod postmortem %s: timed out after %.0fs — teardown proceeds (partial "
                    "evidence at %s)", pod_name, timeout, dest)
    else:
        log.warning("pod postmortem %s (%s): saved to %s [%s]", pod_name,
                    str(reason)[:120], dest, meta.get("status"))
    return str(dest)


def provider_resolver(provider, pod_id: str, profile, *, host_key: str = ""):
    """A lazy ``host`` for :func:`capture`: the pod's ssh address from the
    provider (looked up inside the bounded capture thread), as a read-only
    :class:`~cascade.trainer.remote.RemoteHost` on ``profile``'s user / key /
    workdir. Without a pinned ``host_key`` (a pod that failed before its pin)
    a throwaway known-hosts file is used."""

    def resolve():
        from .remote import RemoteHost

        addr = provider.get_ip(pod_id)
        if addr is None or not getattr(addr, "ip", ""):
            return None
        extra = tuple(getattr(profile, "ssh_options", ()) or ())
        if not host_key:
            extra = ("UserKnownHostsFile=/dev/null", *extra)
        return RemoteHost(name=f"postmortem-{pod_id}"[:64], host=addr.ip,
                          port=int(getattr(addr, "ssh_port", 22) or 22),
                          user=getattr(profile, "user", "root"),
                          key_path=getattr(profile, "key_path", None),
                          workdir=getattr(profile, "workdir", "."),
                          pinned_host_key=str(host_key or ""), ssh_options=extra,
                          isolated=True)

    return resolve


def before_teardown(runner, host, *, pod_name: str, reason: str,
                    run_dir: str | None = None) -> str | None:
    """:func:`capture` under the runner's work root; never raises."""
    try:
        flag = os.environ.get("CASCADE_POD_POSTMORTEM", "")
        if flag == "0":
            return None
        if os.environ.get("PYTEST_CURRENT_TEST") and flag != "1":
            # test doubles' hosts are not real pods: never ssh out of a test
            # run (set CASCADE_POD_POSTMORTEM=1 to exercise it there)
            return None
        root = root_for(getattr(runner, "work_root", None) or ".")
        return capture(host, root=root, pod_name=pod_name, reason=reason, run_dir=run_dir)
    except Exception as e:  # noqa: BLE001 — evidence never blocks a teardown
        log.warning("pod postmortem for %s skipped: %s", pod_name, e)
        return None
