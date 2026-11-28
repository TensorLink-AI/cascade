"""Installed venv vs ``uv.lock``: the reproducibility contract, checked.

The contract is ``uv sync --frozen`` (Python 3.11, torch 2.4.1+cu124): every
pod's venv must equal the lock. The worker images up to rc5 built with
``uv pip install -e``, which re-resolves everything not pinned by hand. The
lock stayed the same but the image did not: rc5 shipped hippius_hub 0.7.2
against a pinned 0.5.1, and 0.6+ pushes chunked-v2 checkpoints that no
validator on 0.5.1 can read (2026-11-28, #998).

This module compares an installed distribution set with a lock file:

* ``version``      a package the lock pins, installed at another version;
* ``not_in_lock``  a package installed that the lock does not know (strict mode
                   only; ``uv sync --inexact`` hosts keep profile extras);
* ``git_rev``      a git-sourced package (timestral) installed from another
                   commit (only when the installed ``direct_url.json`` is
                   readable).

It does NOT check for missing packages (that would need the lock's marker and
extra graph); ``uv sync --frozen`` installs everything, and an import failure
catches the rest.

Stdlib only (``tomllib``, ``importlib.metadata``) so it runs in a bare image
layer, in the provisioner's health gate, and in tests without torch.

CLI (exit 0 = venv equals the lock, 1 = drift, 2 = usage/lock error)::

    .venv/bin/python -P -m cascade.shared.lockdrift --lock uv.lock \
        [--allow-extraneous] [--write-marker .venv/.cascade-lock-verified]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = [
    "LOCK_MARKER",
    "NUMERICS_PACKAGES",
    "Drift",
    "compare",
    "installed_from_dist_info_names",
    "installed_in_process",
    "lock_sha256",
    "normalize",
    "parse_lock",
]

# Written into the venv by the image build / pod bootstrap once the check has
# passed. The provisioner's health gate enforces the lock only on pods that
# carry it; images built before this guard (up to rc5 / mkt-rc3) are reported,
# not refused, until the repin retires them.
LOCK_MARKER = ".cascade-lock-verified"

# Packages whose version can move training/eval numerics or the on-wire
# checkpoint format. A drift here is flagged first in every report.
NUMERICS_PACKAGES = frozenset({
    "torch", "triton", "numpy", "scipy", "safetensors", "einops", "timestral",
    "transformers", "accelerate", "tokenizers", "pandas", "pyarrow", "numba",
    "llvmlite", "scikit-learn", "statsmodels", "gpytorch", "linear-operator",
    "networkx", "sympy", "hippius-hub", "huggingface-hub",
})

_NORMALIZE_RE = re.compile(r"[-_.]+")


def normalize(name: str) -> str:
    """PEP 503 name normalisation (``Hippius_Hub`` → ``hippius-hub``)."""
    return _NORMALIZE_RE.sub("-", name).lower()


def is_numerics(name: str) -> bool:
    n = normalize(name)
    return n in NUMERICS_PACKAGES or n.startswith("nvidia-")


@dataclass(frozen=True)
class Drift:
    name: str
    kind: str            # "version" | "not_in_lock" | "git_rev"
    installed: str
    locked: str          # "" for not_in_lock
    numerics: bool

    def line(self) -> str:
        tag = " [numerics]" if self.numerics else ""
        if self.kind == "not_in_lock":
            return f"{self.name} {self.installed} not in uv.lock{tag}"
        return f"{self.name} {self.installed} != uv.lock {self.locked}{tag}"


@dataclass(frozen=True)
class LockedPackage:
    versions: frozenset[str]
    git_rev: str = ""    # the locked commit for a git source, else ""


def parse_lock(text: str) -> dict[str, LockedPackage]:
    """``uv.lock`` TOML → normalised name → accepted versions (+ git commit).

    A forked resolution can lock one name at several versions (one per marker
    set, e.g. torch ``2.4.1`` on aarch64 and ``2.4.1+cu124`` elsewhere); any of
    them is accepted.
    """
    data = tomllib.loads(text)
    out: dict[str, LockedPackage] = {}
    for pkg in data.get("package", []):
        name = normalize(str(pkg.get("name", "")))
        version = str(pkg.get("version", ""))
        if not name or not version:
            continue
        rev = ""
        git = (pkg.get("source") or {}).get("git", "")
        if git and "#" in git:
            rev = git.rsplit("#", 1)[1].strip()
        prev = out.get(name)
        versions = {version} | (set(prev.versions) if prev else set())
        out[name] = LockedPackage(frozenset(versions), rev or (prev.git_rev if prev else ""))
    if not out:
        raise ValueError("uv.lock has no [[package]] entries")
    return out


def lock_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


_DIST_INFO_RE = re.compile(r"^(?P<name>[^/]+?)-(?P<version>[^-/]+)\.dist-info/?$")


def installed_from_dist_info_names(names: Iterable[str]) -> dict[str, str]:
    """``hippius_hub-0.7.2.dist-info`` directory names (or paths) → versions.

    This is what a pod reports without running Python: ``find`` over
    site-packages. Wheel installers write ``{name}-{version}.dist-info``.
    """
    out: dict[str, str] = {}
    for raw in names:
        base = raw.strip().rstrip("/").rsplit("/", 1)[-1]
        m = _DIST_INFO_RE.match(base)
        if m:
            out[normalize(m.group("name"))] = m.group("version")
    return out


def installed_in_process() -> tuple[dict[str, str], dict[str, str]]:
    """``(name → version, name → git commit)`` for the running interpreter."""
    from importlib import metadata

    versions: dict[str, str] = {}
    revs: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name") if dist.metadata else None
        if not name:
            continue
        key = normalize(name)
        versions[key] = dist.version
        try:
            raw = dist.read_text("direct_url.json")
        except Exception:  # noqa: BLE001 — absent/unreadable ⇒ no commit claim
            raw = None
        if raw:
            try:
                commit = (json.loads(raw).get("vcs_info") or {}).get("commit_id", "")
            except ValueError:
                commit = ""
            if commit:
                revs[key] = commit
    return versions, revs


def compare(lock: Mapping[str, LockedPackage], installed: Mapping[str, str], *,
            installed_revs: Mapping[str, str] | None = None,
            allow_extraneous: bool = False) -> list[Drift]:
    """Every drift of ``installed`` from ``lock``, numerics-relevant first."""
    drifts: list[Drift] = []
    revs = {normalize(k): v for k, v in (installed_revs or {}).items()}
    for raw_name, version in installed.items():
        name = normalize(raw_name)
        locked = lock.get(name)
        if locked is None:
            if not allow_extraneous:
                drifts.append(Drift(name, "not_in_lock", version, "", is_numerics(name)))
            continue
        if version not in locked.versions:
            drifts.append(Drift(name, "version", version,
                                " | ".join(sorted(locked.versions)), is_numerics(name)))
            continue
        got_rev = revs.get(name, "")
        if locked.git_rev and got_rev and got_rev != locked.git_rev:
            drifts.append(Drift(name, "git_rev", f"{version}@{got_rev[:12]}",
                                f"{version}@{locked.git_rev[:12]}", is_numerics(name)))
    drifts.sort(key=lambda d: (not d.numerics, d.name))
    return drifts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -P -m cascade.shared.lockdrift",
        description="Fail unless the running venv equals uv.lock.")
    ap.add_argument("--lock", default="uv.lock", help="path to uv.lock")
    ap.add_argument("--allow-extraneous", action="store_true",
                    help="tolerate packages the lock does not know (uv sync --inexact hosts)")
    ap.add_argument("--write-marker", default="",
                    help="on success, write the lock's sha256 to this path")
    args = ap.parse_args(argv)
    try:
        text = Path(args.lock).read_text()
        lock = parse_lock(text)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as e:
        print(json.dumps({"lock_drift": "error", "detail": str(e)}), file=sys.stderr)
        return 2
    versions, revs = installed_in_process()
    drifts = compare(lock, versions, installed_revs=revs,
                     allow_extraneous=args.allow_extraneous)
    report = {"lock_drift": "ok" if not drifts else "FAIL",
              "lock_sha256": lock_sha256(text), "installed": len(versions),
              "drifts": [asdict(d) for d in drifts]}
    print(json.dumps(report, sort_keys=True))
    if drifts:
        for d in drifts:
            print(f"lock drift: {d.line()}", file=sys.stderr)
        return 1
    if args.write_marker:
        Path(args.write_marker).write_text(report["lock_sha256"] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
