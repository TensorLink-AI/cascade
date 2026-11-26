"""Lane D worker handshake (LANE_D_PLAN W4, §10.4/§10.5, T19/T28).

Before an eval worker's forecasts are used, the validator compares the
worker's code identity with its own: the git ``HEAD`` of the release
directory and a hash of the ``cascade`` package's source bytes. Any
difference — or anything unclear — fails closed with a CRITICAL
``lane_d_handshake_mismatch`` line and nothing is scored, so a stale or
hot-patched eval host can never feed a settlement (T19). The documented
upgrade order (§10.5 S5: eval host first, then the validator) therefore
passes through mismatched steps that score nothing and ends matched (T28).

The git head alone is not enough (an edited checkout keeps its head) and the
package hash alone does not name a release, so both must match and neither
may be empty.

**Scoring environment (rev6.18 B5/B7).** The eval host's ``hello`` (the
``lane_d_forecasts hello`` CLI) also carries the host it runs on: the named
scoring environment ``env``, ``torch_build``, ``gpu_model``, ``driver`` and a
``fingerprint`` of the physical GPU (:func:`probe_environment`).
:func:`check_worker` checks the code identity AND that the environment is
qualified at the settlement boundary in the committed list
(``deploy/laned/scoring_envs.toml``,
:func:`cascade.shared.config.lane_d.load_scoring_envs`); it returns the
identity and the host record the signed report carries (``report.host``),
whose ``identity`` keys the score cache (B6). Anything unqualified or
unclear is a CRITICAL ``lane_d_handshake_env_refused`` and nothing is scored.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

log = logging.getLogger("cascade.eval.lane_d_handshake")

HANDSHAKE_SCHEMA = "lane-d-handshake/1"
PACKAGE_DIR = "cascade"
# Code identity (W4) and, rev6.18 B7, the scoring environment fields.
CODE_FIELDS = ("git_head", "package_hash", "schema")
ENV_FIELDS = ("env", "torch_build", "gpu_model", "driver", "fingerprint")
HANDSHAKE_FIELDS = CODE_FIELDS + ENV_FIELDS
# The eval host names its environment with ``hello --env`` or this variable.
ENV_VAR = "CASCADE_LANE_D_SCORING_ENV"
_SMI_TIMEOUT_S = 30
_GIT_TIMEOUT_S = 30
# Environment that would point git somewhere other than ``root``.
_GIT_ENV_DROP = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                 "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CEILING_DIRECTORIES",
                 "GIT_COMMON_DIR", "GIT_NAMESPACE")

T = TypeVar("T")


@dataclass(frozen=True)
class CodeIdentity:
    """What a host runs: the release checkout's ``HEAD`` (``""`` when the
    root is not the top of a git checkout) and :func:`package_hash`."""

    git_head: str
    package_hash: str


class HandshakeMismatch(RuntimeError):
    """The worker's code identity differs from the validator's (or cannot be
    established); ``fields`` is the sorted tuple of mismatching fields."""

    def __init__(self, fields, detail: str = ""):
        self.fields = tuple(sorted(set(fields)))
        msg = f"lane D handshake mismatch on {list(self.fields)}"
        super().__init__(f"{msg}: {detail}" if detail else msg)


def _source_files(pkg: Path) -> list[tuple[str, Path]]:
    out = []
    for dirpath, dirnames, filenames in os.walk(pkg):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in filenames:
            if name.endswith((".pyc", ".pyo")):
                continue
            p = Path(dirpath) / name
            if p.is_file():
                out.append((p.relative_to(pkg).as_posix(), p))
    return sorted(out)


def package_hash(root: str | os.PathLike) -> str:
    """sha256 hex over the source files under ``<root>/cascade/``: their
    relative paths and bytes only (no mtimes, no location, no bytecode)."""
    pkg = Path(root) / PACKAGE_DIR
    if not pkg.is_dir():
        raise FileNotFoundError(f"no {PACKAGE_DIR}/ package under {root}")
    h = hashlib.sha256()
    for rel, p in _source_files(pkg):
        rel_b = rel.encode("utf-8")
        h.update(len(rel_b).to_bytes(8, "big"))
        h.update(rel_b)
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _git(root: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in _GIT_ENV_DROP}
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          env=env, timeout=_GIT_TIMEOUT_S, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _git_head(root: Path) -> str:
    """The full ``HEAD`` when ``root`` is itself the top of a git checkout."""
    try:
        top = _git(root, "rev-parse", "--show-toplevel")
        if not top or os.path.realpath(top) != os.path.realpath(root):
            return ""
        head = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    except (OSError, subprocess.SubprocessError):
        return ""
    if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        return ""
    return head


def _default_root() -> Path:
    import cascade

    return Path(os.path.realpath(cascade.__file__)).parent.parent


def local_identity(root: str | os.PathLike | None = None) -> CodeIdentity:
    """This host's :class:`CodeIdentity`; ``root`` defaults to the directory
    containing the imported ``cascade`` package."""
    r = _default_root() if root is None else Path(os.path.realpath(root))
    return CodeIdentity(git_head=_git_head(r), package_hash=package_hash(r))


def worker_hello(root: str | os.PathLike | None = None, *,
                 environment: dict | None = None) -> dict:
    """The eval worker's hello: ``{"schema", "git_head", "package_hash"}``,
    plus (rev6.18 B7) the :data:`ENV_FIELDS` of ``environment`` when given
    (:func:`probe_environment`)."""
    ident = local_identity(root)
    hello = {"schema": HANDSHAKE_SCHEMA, "git_head": ident.git_head,
             "package_hash": ident.package_hash}
    if environment is not None:
        hello.update({k: str(environment.get(k) or "") for k in ENV_FIELDS})
    return hello


def _nvidia_smi(run: Callable = subprocess.run) -> list[tuple[str, str, str]]:
    """``(name, driver_version, uuid)`` per visible GPU (``[]`` on failure)."""
    argv = ["nvidia-smi", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if cvd:
        argv.insert(1, f"--id={cvd}")
    try:
        proc = run(argv, capture_output=True, text=True, timeout=_SMI_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    out = []
    for line in (proc.stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and all(parts):
            out.append((parts[0], parts[1], parts[2]))
    return out


def _torch_build() -> str:
    try:
        import torch  # noqa: PLC0415 — eval host only; the box never probes
    except ImportError:
        return ""
    cuda = getattr(getattr(torch, "version", None), "cuda", None) or ""
    return f"{torch.__version__}/cuda-{cuda}" if cuda else ""


def probe_environment(env: str | None = None, *, envs=None,
                      smi: Callable[[], list] | None = None,
                      torch_build: Callable[[], str] | None = None) -> dict:
    """This eval host's scoring environment, the :data:`ENV_FIELDS`:
    ``gpu_model`` / ``driver`` from ``nvidia-smi`` (the first visible GPU),
    ``torch_build`` = ``"<torch.__version__>/cuda-<torch.version.cuda>"``,
    ``fingerprint`` = sha256 of the GPU's UUID (the physical card, never
    published in clear) and ``env``: ``env``, else ``$CASCADE_LANE_D_SCORING_ENV``,
    else the one entry of ``envs`` (default: the committed list) whose torch
    build and GPU fit. Unknown values are ``""``: the validator refuses them."""
    gpus = (smi or _nvidia_smi)()
    name, driver, uuid = gpus[0] if gpus else ("", "", "")
    tb = (torch_build or _torch_build)()
    chosen = env or os.environ.get(ENV_VAR, "")
    if not chosen:
        try:
            if envs is None:
                from ..shared.config.lane_d import load_scoring_envs

                envs = load_scoring_envs()
            fits = [e.name for e in envs if e.torch_build == tb and name in e.gpu_models]
        except ValueError:
            fits = []
        chosen = fits[0] if len(fits) == 1 else ""
    fp = hashlib.sha256(uuid.encode("utf-8")).hexdigest() if uuid else ""
    return {"env": chosen, "torch_build": tb, "gpu_model": name, "driver": driver,
            "fingerprint": fp}


def _mismatches(hello, expected: CodeIdentity) -> list[str]:
    if not isinstance(hello, dict):
        return list(CODE_FIELDS)
    bad = []
    if hello.get("schema") != HANDSHAKE_SCHEMA:
        bad.append("schema")
    for field in ("git_head", "package_hash"):
        got, want = hello.get(field), getattr(expected, field)
        if not isinstance(got, str) or not got or not want or got != want:
            bad.append(field)
    return bad


def _short(value) -> str:
    return value[:12] if isinstance(value, str) and value else repr(value)[:24]


def check_handshake(hello, *, expected: CodeIdentity | None = None) -> CodeIdentity:
    """Compare a worker's hello with ``expected`` (default: this host);
    return the identity on a match, else log CRITICAL and raise
    :class:`HandshakeMismatch`. Fails closed on anything unclear."""
    want = local_identity() if expected is None else expected
    bad = _mismatches(hello, want)
    if bad:
        fields = sorted(bad)
        got = hello if isinstance(hello, dict) else {}
        log.critical("lane_d_handshake_mismatch fields=%s worker_git_head=%s "
                     "worker_package_hash=%s expected_git_head=%s expected_package_hash=%s",
                     ",".join(fields), _short(got.get("git_head")),
                     _short(got.get("package_hash")), _short(want.git_head),
                     _short(want.package_hash))
        raise HandshakeMismatch(fields)
    log.info("lane_d_handshake_ok git_head=%s package_hash=%s", want.git_head,
             want.package_hash[:12])
    return want


def check_worker(hello, *, envs, boundary: int,
                 expected: CodeIdentity | None = None) -> tuple[CodeIdentity, dict]:
    """rev6.18 B7: :func:`check_handshake` (code identity), then the scoring
    environment: the hello must carry every :data:`ENV_FIELDS` value and fit
    an entry of ``envs`` qualified at ``boundary``
    (:func:`cascade.shared.config.lane_d.host_problem`). Returns the identity
    and the report's ``host`` record (``identity`` = the B6 host identity).
    Otherwise logs CRITICAL ``lane_d_handshake_env_refused`` and raises
    :class:`HandshakeMismatch` naming the environment fields."""
    from ..shared.config.lane_d import host_identity, host_problem

    ident = check_handshake(hello, expected=expected)
    missing = [k for k in ENV_FIELDS if not isinstance(hello.get(k), str) or not hello[k]]
    if missing:
        log.critical("lane_d_handshake_env_refused fields=%s: the eval host reports no "
                     "scoring environment", ",".join(sorted(missing)))
        raise HandshakeMismatch(missing, "the eval host reports no scoring environment")
    host = {k: hello[k] for k in ("env", "gpu_model", "driver", "torch_build", "fingerprint")}
    host["identity"] = host_identity(host["env"], host["gpu_model"], host["driver"],
                                     host["torch_build"], host["fingerprint"])
    why = host_problem(host, envs, int(boundary))
    if why:
        log.critical("lane_d_handshake_env_refused env=%s gpu=%s driver=%s torch=%s: %s",
                     host["env"], host["gpu_model"], host["driver"], host["torch_build"], why)
        raise HandshakeMismatch(["env"], why)
    log.info("lane_d_handshake_env_ok env=%s gpu=%s driver=%s torch=%s identity=%s",
             host["env"], host["gpu_model"], host["driver"], host["torch_build"],
             host["identity"][:12])
    return ident, host


def with_handshake(hello, work: Callable[[], T], *,
                   expected: CodeIdentity | None = None) -> T:
    """Run ``work()`` only after :func:`check_handshake` passes; on a
    mismatch nothing runs (T19)."""
    check_handshake(hello, expected=expected)
    return work()


__all__ = [
    "CODE_FIELDS", "ENV_FIELDS", "ENV_VAR", "check_worker", "probe_environment",
    "CodeIdentity", "HANDSHAKE_FIELDS", "HANDSHAKE_SCHEMA", "HandshakeMismatch",
    "check_handshake", "local_identity", "package_hash", "with_handshake", "worker_hello",
]
