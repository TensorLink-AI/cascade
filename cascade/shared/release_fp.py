"""One release fingerprint, checked and re-synced at the start of every run
on every remote host (retro P4, failure class C3).

Every remote host used to be configured once (bootstrap) and then reset by
hand on each release: the pool builder ran old code for a day (#976, 11-28
02:12), the lane D eval host was reset by hand on every validator release
(4× on 11-28), and a running provisioner kept its old pins. #994 fixed the
builder alone (``.release_sha`` + re-sync before every build); the lane D
handshake refuses a stale eval host but nothing brings it back.

This module is the one implementation:

* :class:`Fingerprint` — the release's git ``HEAD`` (``+dirty.<diff sha>``
  when the checkout has uncommitted edits) and :func:`package_hash` (the same
  bytes-only hash the lane D handshake uses). Two hosts run the same code iff
  their package hashes match; the head names the release.
* :func:`probe_command` — a stdlib-only probe a host runs (whatever code it
  has, even none) that prints its own ``package_hash``, git head / dirtiness
  and the ``.release_fp`` recorded by the last sync.
* :func:`ensure_release` — at the start of a run: probe; in sync ⇒ proceed;
  skewed ⇒ ``release_sync = "sync"`` re-syncs (a git checkout is moved to the
  release's commit, refused when it carries hand edits; any other workdir
  gets the release's ``cascade/`` package by a tar stream swapped in
  atomically) and probes again, so the run never starts on old code; a failed
  sync raises :class:`ReleaseSyncFailed` (named) and the run is refused.
  ``"check"`` only logs ``release_skew`` at ERROR (never blocks a run);
  ``"off"`` (the default without ``CASCADE_RELEASE_SYNC_DEFAULT``) skips it. Venv / dependency drift is out of scope here (the image /
  ``uv sync --frozen`` path owns it: #999 lockdrift).

Pure stdlib; importing it imports nothing heavy (box guard).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shlex
import subprocess
import tarfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger("cascade.release_fp")

RELEASE_FP_FILE = ".release_fp"
SYNC_MODES = ("off", "check", "sync")
# A host entry without ``release_sync`` gets this process's default: ``off``
# unless ``CASCADE_RELEASE_SYNC_DEFAULT`` says otherwise (the operator box's
# units set it; external validators running the same code are untouched).
DEFAULT_SYNC_MODE = "off"
DEFAULT_ENV = "CASCADE_RELEASE_SYNC_DEFAULT"
PACKAGE_DIR = "cascade"
STATUS_DIR_ENV = "CASCADE_RELEASE_STATUS_DIR"
_GIT_TIMEOUT_S = 30


class ReleaseSyncFailed(RuntimeError):
    """A host could not be brought onto the release; the run is refused."""


class ReleaseUnreachable(OSError):
    """The host's probe failed (transport): transient, like any I/O error."""


# ── the fingerprint ─────────────────────────────────────────────────────────


def package_hash(root: str | os.PathLike) -> str:
    """sha256 over the source files under ``<root>/cascade/`` (relative paths
    and bytes; no ``__pycache__``, bytecode or mtimes). Identical to
    :func:`cascade.eval.lane_d_handshake.package_hash` (pinned by a test)."""
    pkg = Path(root) / PACKAGE_DIR
    if not pkg.is_dir():
        raise FileNotFoundError(f"no {PACKAGE_DIR}/ package under {root}")
    files = []
    for dirpath, dirnames, filenames in os.walk(pkg):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in filenames:
            if name.endswith((".pyc", ".pyo")):
                continue
            p = Path(dirpath) / name
            if p.is_file():
                files.append((p.relative_to(pkg).as_posix(), p))
    h = hashlib.sha256()
    for rel, p in sorted(files):
        rb = rel.encode("utf-8")
        h.update(len(rb).to_bytes(8, "big"))
        h.update(rb)
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _git(root: Path, *args: str, text: bool = True):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=text,
                          env=env, timeout=_GIT_TIMEOUT_S, check=False)
    return proc.stdout if proc.returncode == 0 else ("" if text else b"")


def git_head(root: str | os.PathLike) -> tuple[str, str]:
    """``(HEAD, dirty)`` of ``root`` when it is a git checkout or worktree:
    ``dirty`` is the first 16 hex of sha256(``git diff HEAD``) when tracked
    files are modified (``""`` when clean); ``("", "")`` when not a checkout."""
    r = Path(root)
    try:
        head = _git(r, "rev-parse", "--verify", "HEAD^{commit}").strip()
        if len(head) != 40:
            return "", ""
        if _git(r, "status", "--porcelain", "--untracked-files=no").strip():
            return head, hashlib.sha256(_git(r, "diff", "HEAD", text=False)).hexdigest()[:16]
        return head, ""
    except (OSError, subprocess.SubprocessError):
        return "", ""


@dataclass(frozen=True)
class Fingerprint:
    """What a release (or a host) runs."""

    git_head: str
    dirty: str
    package_hash: str

    @property
    def token(self) -> str:
        head = (self.git_head[:12] or "nogit") + (f"+dirty.{self.dirty}" if self.dirty else "")
        return f"{head}/{self.package_hash[:16]}"

    def to_dict(self) -> dict:
        return {**asdict(self), "token": self.token}


def local_fingerprint(root: str | os.PathLike | None = None) -> Fingerprint:
    """This process's release (``root`` defaults to the directory holding
    the imported ``cascade`` package)."""
    if root is None:
        import cascade

        root = Path(os.path.realpath(cascade.__file__)).parent.parent
    head, dirty = git_head(root)
    return Fingerprint(git_head=head, dirty=dirty, package_hash=package_hash(root))


def builder_token(release_dir: str | os.PathLike, chain_toml: str | os.PathLike) -> str:
    """The pool builder's ``.release_sha`` value (#994 R13 format, unchanged:
    ``<HEAD>[+dirty.<16 hex>] chain.<16 hex of chain.toml>``)."""
    head, dirty = git_head(release_dir)
    if not head:
        raise ReleaseSyncFailed(f"release_dir {release_dir} is not a git checkout")
    chain = hashlib.sha256(Path(chain_toml).read_bytes()).hexdigest()[:16]
    return f"{head}{'+dirty.' + dirty if dirty else ''} chain.{chain}"


def log_service_fingerprint(service: str, root: str | os.PathLike | None = None,
                            environ=None) -> Fingerprint | None:
    """Log a box service's fingerprint at start; with
    ``CASCADE_RELEASE_STATUS_DIR`` set, also write ``<dir>/<service>.json``
    (cascade-ops reads it for its release-skew check). Never raises."""
    env = os.environ if environ is None else environ
    try:
        fp = local_fingerprint(root)
    except Exception as e:  # noqa: BLE001 — a label, never a startup failure
        log.warning("%s: release fingerprint unreadable (%s)", service, e)
        return None
    log.info("%s: release fingerprint %s", service, fp.token)
    out = (env.get(STATUS_DIR_ENV) or "").strip()
    if out:
        try:
            d = Path(out)
            d.mkdir(parents=True, exist_ok=True)
            doc = {**fp.to_dict(), "service": service, "pid": os.getpid(),
                   "started": int(time.time())}
            tmp = d / f".{service}.json.tmp"
            tmp.write_text(json.dumps(doc, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, d / f"{service}.json")
        except OSError as e:
            log.warning("%s: release status file not written (%s)", service, e)
    return fp


# ── the host probe (runs on the host, any code or none) ─────────────────────

# Stdlib only and self-contained: a host with old (or no) cascade code must
# still answer. Same hash as package_hash() (pinned by a test).
PROBE_PY = r"""
import hashlib, json, os, subprocess, sys
root = sys.argv[1]
pkg = os.path.join(root, "cascade")
out = {"package_hash": "", "git_head": "", "dirty": "", "is_git": False, "recorded": ""}
if os.path.isdir(pkg):
    files = []
    for dp, dn, fn in os.walk(pkg):
        dn[:] = sorted(d for d in dn if d != "__pycache__")
        for n in fn:
            if n.endswith((".pyc", ".pyo")):
                continue
            p = os.path.join(dp, n)
            if os.path.isfile(p):
                files.append((os.path.relpath(p, pkg).replace(os.sep, "/"), p))
    h = hashlib.sha256()
    for rel, p in sorted(files):
        rb = rel.encode("utf-8")
        h.update(len(rb).to_bytes(8, "big")); h.update(rb)
        with open(p, "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    out["package_hash"] = h.hexdigest()
if os.path.exists(os.path.join(root, ".git")):
    def git(*a):
        r = subprocess.run(["git", "-C", root, *a], capture_output=True, timeout=30)
        return r.stdout if r.returncode == 0 else b""
    head = git("rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    if len(head) == 40:
        out["is_git"], out["git_head"] = True, head
        if git("status", "--porcelain", "--untracked-files=no").strip():
            out["dirty"] = hashlib.sha256(git("diff", "HEAD")).hexdigest()[:16]
try:
    with open(os.path.join(root, ".release_fp")) as f:
        out["recorded"] = f.read().strip()
except OSError:
    pass
print("RELEASE_PROBE " + json.dumps(out, sort_keys=True))
"""


def probe_command(workdir: str, python: str = "python3") -> str:
    """The remote shell command printing the host's ``RELEASE_PROBE`` line."""
    return (f"{shlex.quote(python)} -I -c {shlex.quote(PROBE_PY)} {shlex.quote(workdir)} "
            f"|| python3 -I -c {shlex.quote(PROBE_PY)} {shlex.quote(workdir)}")


def parse_probe(stdout: str) -> dict:
    for line in reversed((stdout or "").splitlines()):
        if line.startswith("RELEASE_PROBE "):
            return json.loads(line[len("RELEASE_PROBE "):])
    raise ValueError("no RELEASE_PROBE line in the host's reply")


def host_matches(probe: dict, want: Fingerprint) -> tuple[bool, str]:
    """``(in sync?, why not)`` for a host's probe against the release."""
    have = str(probe.get("package_hash") or "")
    if not have:
        return False, "no cascade/ package in the host's workdir"
    if have != want.package_hash:
        return False, f"package {have[:16]} != {want.package_hash[:16]}"
    if probe.get("is_git"):
        if want.git_head and probe.get("git_head") != want.git_head:
            return False, f"git head {str(probe.get('git_head'))[:12]} != {want.git_head[:12]}"
        if probe.get("dirty"):
            return False, f"host checkout has uncommitted edits ({probe['dirty']})"
    return True, ""


# ── the sync ────────────────────────────────────────────────────────────────


def package_tarball(root: str | os.PathLike) -> bytes:
    """Deterministic ``tar.gz`` of ``<root>/cascade/`` (no bytecode)."""
    import gzip

    base = Path(root)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tar:
        for p in sorted((base / PACKAGE_DIR).rglob("*")):
            rel = p.relative_to(base)
            if not p.is_file() or "__pycache__" in rel.parts or p.suffix in (".pyc", ".pyo"):
                continue
            data = p.read_bytes()
            ti = tarfile.TarInfo(rel.as_posix())
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid = len(data), 0, 0o644, 0, 0
            tar.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def git_sync_command(workdir: str, want: Fingerprint) -> str:
    """Move a git checkout host onto ``want``'s commit; refuses hand edits."""
    wd = shlex.quote(workdir)
    head = shlex.quote(want.git_head)
    return ("set -e; "
            f"if [ -n \"$(git -C {wd} status --porcelain --untracked-files=no)\" ]; then "
            "echo 'release_sync refused: the host checkout has uncommitted edits' >&2; exit 3; fi; "
            f"git -C {wd} cat-file -e {head}^{{commit}} 2>/dev/null || git -C {wd} fetch -q origin; "
            f"git -C {wd} checkout -q --detach {head}")


def tar_sync_command(workdir: str, token: str) -> str:
    """Swap ``<workdir>/cascade`` for the tarball on stdin (atomically: the
    new tree is complete before the rename) and record ``.release_fp``; the
    previous tree is kept as ``cascade.prev``."""
    wd = shlex.quote(workdir)
    return ("set -e; "
            f"d={wd}/.release_sync.$$; rm -rf \"$d\"; mkdir -p \"$d\"; "
            "tar -xzf - -C \"$d\"; test -d \"$d/cascade\"; "
            f"rm -rf {wd}/cascade.prev; "
            f"if [ -e {wd}/cascade ]; then mv {wd}/cascade {wd}/cascade.prev; fi; "
            f"mv \"$d/cascade\" {wd}/cascade; rm -rf \"$d\"; "
            f"printf '%s' {shlex.quote(token)} > {wd}/{RELEASE_FP_FILE}")


Runner = Callable[[object, str, bytes | None, float], subprocess.CompletedProcess]


def ssh_runner(host, command: str, stdin: bytes | None, timeout: float):
    """Run ``command`` on ``host`` over the trainer's ssh argv builder."""
    from ..trainer.remote import build_ssh_argv

    return subprocess.run(build_ssh_argv(host, command), input=stdin, capture_output=True,
                          timeout=timeout, check=False)


def _out(proc) -> str:
    o = getattr(proc, "stdout", b"") or b""
    return o.decode("utf-8", "replace") if isinstance(o, bytes) else str(o)


def _err(proc) -> str:
    e = getattr(proc, "stderr", b"") or b""
    return (e.decode("utf-8", "replace") if isinstance(e, bytes) else str(e))[-400:]


def sync_mode_of(host, environ=None) -> str:
    """The host's ``release_sync`` (hosts TOML), else ``$CASCADE_RELEASE_SYNC_DEFAULT``,
    else ``off``."""
    env = os.environ if environ is None else environ
    m = str(getattr(host, "release_sync", "") or env.get(DEFAULT_ENV, "")
            or DEFAULT_SYNC_MODE).strip().lower()
    if m not in SYNC_MODES:
        log.error("release_sync %r is not one of %s; using check", m, SYNC_MODES)
        return "check"
    return m


def ensure_release(host, *, role: str, release_root: str | os.PathLike | None = None,
                   want: Fingerprint | None = None, run: Runner | None = None,
                   mode: str | None = None, timeout: float = 300.0) -> dict:
    """Bring ``host`` onto this process's release before a run (see module
    doc). Returns ``{"host", "role", "mode", "status", "have", "want"}`` with
    status ``in_sync`` | ``synced`` | ``skewed`` / ``unknown`` (check mode) |
    ``off``.
    Raises :class:`ReleaseSyncFailed` when a sync was due and did not land,
    :class:`ReleaseUnreachable` when the host cannot be probed."""
    mode = (mode or sync_mode_of(host)).lower()
    name = str(getattr(host, "name", host))
    rec = {"host": name, "role": role, "mode": mode, "status": "off", "have": "", "want": ""}
    if mode == "off":
        return rec
    root = Path(release_root) if release_root is not None else None
    want = want or local_fingerprint(root)
    rec["want"] = want.token
    run = run or ssh_runner
    workdir = str(getattr(host, "workdir", ".") or ".")
    python = str(getattr(host, "remote_python", "") or "python3")

    def probe() -> dict:
        try:
            proc = run(host, probe_command(workdir, python), None, timeout)
        except (OSError, subprocess.SubprocessError) as e:
            raise ReleaseUnreachable(f"{role} host {name}: release probe failed ({e})") from e
        if getattr(proc, "returncode", 1) != 0:
            raise ReleaseUnreachable(f"{role} host {name}: release probe exited "
                                     f"{proc.returncode}: {_err(proc)}")
        try:
            return parse_probe(_out(proc))
        except ValueError as e:
            raise ReleaseUnreachable(f"{role} host {name}: {e}") from e

    try:
        got = probe()
    except ReleaseUnreachable as e:
        if mode == "check":
            rec["status"] = "unknown"
            log.warning("release_skew unknown: %s", e)
            return rec
        raise
    ok, why = host_matches(got, want)
    rec["have"] = Fingerprint(str(got.get("git_head") or ""), str(got.get("dirty") or ""),
                              str(got.get("package_hash") or "")).token
    if ok:
        rec["status"] = "in_sync"
        log.info("release %s host %s in sync on %s", role, name, want.token)
        return rec
    if mode == "check":
        rec["status"] = "skewed"
        log.error("release_skew: %s host %s runs %s, this release is %s (%s); set "
                  "release_sync = \"sync\" on the host entry to re-sync it at every run",
                  role, name, rec["have"], want.token, why)
        return rec
    log.warning("release_skew: %s host %s runs %s, release is %s (%s): re-syncing",
                role, name, rec["have"], want.token, why)
    if got.get("is_git"):
        if want.dirty or not want.git_head:
            raise ReleaseSyncFailed(
                f"release_sync_failed: {role} host {name} is a git checkout but this release "
                f"({want.token}) has uncommitted edits: commit and push it first")
        cmd, stdin = git_sync_command(workdir, want), None
    else:
        if root is None:
            import cascade

            root = Path(os.path.realpath(cascade.__file__)).parent.parent
        cmd, stdin = tar_sync_command(workdir, want.token), package_tarball(root)
    try:
        proc = run(host, cmd, stdin, timeout)
    except (OSError, subprocess.SubprocessError) as e:
        raise ReleaseSyncFailed(f"release_sync_failed: {role} host {name}: {e}") from e
    if getattr(proc, "returncode", 1) != 0:
        raise ReleaseSyncFailed(f"release_sync_failed: {role} host {name} (exit "
                                f"{proc.returncode}): {_err(proc)}")
    got = probe()
    ok, why = host_matches(got, want)
    if not ok:
        raise ReleaseSyncFailed(f"release_sync_failed: {role} host {name} still skewed after "
                                f"the sync ({why}); the run is refused, never started on "
                                f"old code")
    rec["status"] = "synced"
    log.warning("release %s host %s re-synced to %s", role, name, want.token)
    return rec


__all__ = ["DEFAULT_SYNC_MODE", "RELEASE_FP_FILE", "SYNC_MODES", "Fingerprint",
           "ReleaseSyncFailed", "ReleaseUnreachable", "builder_token", "ensure_release",
           "sync_mode_of",
           "git_head", "host_matches", "local_fingerprint", "log_service_fingerprint",
           "package_hash", "package_tarball", "parse_probe", "probe_command"]
