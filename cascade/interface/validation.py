"""Schema + runtime checks for a submitted generator repo.

Used by the miner CLI (``cascade verify``) and by the trainer before it
imports and runs a generator in the sandbox. A submission MAY carry model
weights (the generator can be a trained model behind ``generate()``), but only
as **safetensors** — pickle-based checkpoints are rejected because loading them
executes arbitrary code. The whole submission is size-capped (``max_repo_mb``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# A generator HF repo must contain at least these files; it MAY additionally
# ship safetensors weights (the generator can be a trained model).
REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "generator.py",
    "requirements.txt",
)

# Pickle-based weight globs that must NOT appear: loading them (torch.load et al.)
# unpickles — i.e. runs arbitrary code from untrusted miner data. Ship weights as
# ``*.safetensors`` instead, a safe code-free tensor container.
FORBIDDEN_PICKLE_GLOBS: tuple[str, ...] = ("*.bin", "*.pt", "*.pth", "*.ckpt")

# requirements.txt line: ``pkg==1.2.3 --hash=sha256:abc...`` (one or more hash flags).
_REQ_LINE = re.compile(
    r"""
    ^
    (?P<name>[A-Za-z0-9_.\-]+)        # package name
    \s*==\s*                          # ==
    (?P<version>[A-Za-z0-9_.\-+]+)    # version
    (?P<hashes>(\s+--hash=sha256:[A-Fa-f0-9]{64})+)
    \s*(\#.*)?$
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str | None = None
    details: dict | None = None

    @classmethod
    def fail(cls, reason: str, **details) -> ValidationResult:
        return cls(ok=False, reason=reason, details=details or None)

    @classmethod
    def pass_(cls) -> ValidationResult:
        return cls(ok=True)


def check_repo_layout(repo_dir: Path | str) -> ValidationResult:
    """Required files present and no pickle-based weight files.

    safetensors weights are allowed (a generator may be a trained model); only
    pickle checkpoints are rejected. The size cap is :func:`check_repo_size`.
    """
    d = Path(repo_dir)
    if not d.is_dir():
        return ValidationResult.fail("not_a_directory", path=str(d))
    missing = [name for name in REQUIRED_FILES if not (d / name).is_file()]
    if missing:
        return ValidationResult.fail("missing_files", missing=missing)
    pickled = sorted({p.name for g in FORBIDDEN_PICKLE_GLOBS for p in d.rglob(g)})
    if pickled:
        return ValidationResult.fail("pickle_weights_forbidden", files=pickled)
    return ValidationResult.pass_()


def check_repo_size(repo_dir: Path | str, max_repo_mb: int) -> ValidationResult:
    """Total size of the fetched submission must be ``<= max_repo_mb``.

    Counts every file in the tree (code + config + any safetensors weights), so a
    generator that ships a model is bounded — it keeps download/storage/audit
    cost sane and caps how large a model a miner can submit as a "generator".
    """
    d = Path(repo_dir)
    if not d.is_dir():
        return ValidationResult.fail("not_a_directory", path=str(d))
    total = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    cap = int(max_repo_mb) * 1024 * 1024
    if total > cap:
        return ValidationResult.fail(
            "repo_too_large", total_bytes=total, max_bytes=cap, max_repo_mb=int(max_repo_mb)
        )
    return ValidationResult.pass_()


def check_config(repo_dir: Path | str) -> ValidationResult:
    """``config.json`` is present and parses as a JSON object."""
    d = Path(repo_dir)
    config_p = d / "config.json"
    if not config_p.is_file():
        return ValidationResult.fail("missing_config_json")
    try:
        obj = json.loads(config_p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return ValidationResult.fail("config_json_invalid", error=str(e))
    if not isinstance(obj, dict):
        return ValidationResult.fail("config_json_not_object")
    return ValidationResult.pass_()


def check_requirements_hash_locked(
    requirements_path: Path | str,
    allowed: tuple[str, ...] | None,
    max_packages: int,
) -> ValidationResult:
    """Reject if any line isn't hash-pinned, count exceeds limit, or a package
    name is outside the allowlist (when an allowlist is supplied)."""
    p = Path(requirements_path)
    if not p.is_file():
        return ValidationResult.fail("missing_requirements")
    lines = [
        ln.strip()
        for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    # Join continuation lines (``\``-terminated) — pip allows them.
    joined: list[str] = []
    buf = ""
    for ln in lines:
        if ln.endswith("\\"):
            buf += ln[:-1] + " "
        else:
            joined.append(buf + ln)
            buf = ""
    if buf:
        joined.append(buf)

    if len(joined) > max_packages:
        return ValidationResult.fail(
            "too_many_packages", count=len(joined), max=max_packages
        )

    names = []
    for ln in joined:
        m = _REQ_LINE.match(ln)
        if not m:
            return ValidationResult.fail("requirement_not_hash_locked", line=ln)
        names.append(m.group("name").lower())

    if allowed is not None:
        allow_set = {n.lower() for n in allowed}
        bad = [n for n in names if n not in allow_set]
        if bad:
            return ValidationResult.fail("requirement_not_allowlisted", packages=bad)

    return ValidationResult.pass_()


# ----- on-chain commit format --------------------------------------------------

# Single pointer string: ``metro-v1:gen:hippius:<repo>@<digest>``. The generator
# repo (code + config + any safetensors weights) is pushed to the Hippius Hub OCI
# registry; the immutable ``repo@digest`` reference both *locates* and *pins* the
# submission — the OCI manifest digest is the content hash, so no separate
# revision is needed. The ``gen`` tag distinguishes a miner's submission from the
# trainer's ``trained`` pointers in the manifest.
COMMIT_RE = re.compile(r"^metro-v1:gen:hippius:(?P<ref>.+)$")


@dataclass(frozen=True)
class ParsedCommit:
    """A parsed generator pointer. ``ref`` is the Hippius Hub ``repo@digest``."""

    ref: str


def parse_commit(payload: str) -> ParsedCommit | None:
    """Return None for malformed payloads. The trainer treats None as a
    permanent rejection of the submission. The reference is validated against the
    Hub ``repo@digest`` grammar so a garbage payload never reaches a fetch.
    """
    from ..shared.hippius import is_hub_ref

    m = COMMIT_RE.match(payload.strip())
    if not m:
        return None
    ref = m.group("ref").strip()
    if not is_hub_ref(ref):
        return None
    return ParsedCommit(ref=ref)


def format_commit(ref: str) -> str:
    """Build the on-chain payload from a Hub ``repo@digest`` reference. Raises if
    it would not round-trip through :func:`parse_commit`."""
    payload = f"metro-v1:gen:hippius:{ref.strip()}"
    if parse_commit(payload) is None:
        raise ValueError(f"refusing to emit malformed commit: {payload!r}")
    return payload
