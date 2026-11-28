#!/usr/bin/env python3
"""Report stale release checkouts and restart scripts on the box (retro P4).

REPORT-ONLY: nothing is deleted, moved or edited. On 2026-11-28 the box held 36
``/root/cascade-val-*`` checkouts and 63 ``restart_*.sh`` scripts. The live
validator script was ``restart_validator_3718f82a.sh``, but it pointed at
e15993f7, and the watchdog had to be "updated if the validator release
changes". This tool shows which of them are in use:

* **release dirs**. Each ``<root>/cascade-val-*`` / ``cascade-rc*`` /
  ``cascade-release*`` checkout is reported with its HEAD and what references
  it:
  - a RUNNING process (its ``PYTHONPATH``; nothing else is read from a
    process environment);
  - a systemd unit or drop-in;
  - a restart script;
  - a ``/root/cascade-releases/*.current`` symlink.

  A dir nothing references is ``STALE``.
* **restart scripts**. Each script is reported with the service it restarts
  and the PYTHONPATH it launches.
  - ``LIVE``: it launches the release the running service uses (several
    scripts can).
  - ``STALE``: it launches a release nothing runs, or a dir that no longer
    exists.

  Scripts that other scripts or units call (``validator_watchdog.sh`` ->
  ``restart_validator_3718f82a.sh``) are ``REFERENCED``. Never retire them
  without updating the caller.

Usage::

    python3 -I scripts/release_sprawl.py [--root /root] [--scripts /root/cascade]
        [--units /etc/systemd/system] [--json]

The report ends with the stale lists. Retiring an entry is the owner's call:
rename a script to ``*.retired``, and remove a worktree with
``git worktree remove``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

RELEASE_GLOBS = ("cascade-val-*", "cascade-rc*", "cascade-release*")
SERVICE_OF = (("validator", re.compile(r"cascade-validator|cascade\.validator")),
              ("trainer", re.compile(r"cascade-trainer|cascade\.trainer")),
              ("intake", re.compile(r"cascade\.funding|cascade-intake")),
              ("provisioner", re.compile(r"cascade\.provision|cascade-provisioner")),
              ("laned", re.compile(r"cascade\.lane_d\b|cascade\.laned")))
_PYTHONPATH = re.compile(r"PYTHONPATH=([^\s\"';]+)")


def release_dirs(root: Path) -> list[Path]:
    out: set[Path] = set()
    for g in RELEASE_GLOBS:
        out.update(p for p in root.glob(g) if p.is_dir() and (p / "cascade").is_dir())
    return sorted(out)


def git_head(d: Path) -> str:
    try:
        r = subprocess.run(["git", "-C", str(d), "rev-parse", "--short=8", "HEAD"],
                           capture_output=True, text=True, timeout=10, check=False)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _service(text: str) -> str:
    for name, pat in SERVICE_OF:
        if pat.search(text):
            return name
    return "?"


def _paths(text: str) -> list[str]:
    out = []
    for m in _PYTHONPATH.finditer(text):
        out.extend(p for p in m.group(1).split(":") if p)
    return out


def running_releases(proc_root: Path = Path("/proc")) -> list[dict]:
    """``{"pid", "service", "pythonpath"}`` per running cascade process.
    Only ``PYTHONPATH`` is taken from a process environment (secrets live
    there); a process without one reports its cwd-less command line only."""
    out = []
    for d in proc_root.iterdir() if proc_root.is_dir() else ():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "cascade" not in cmd or "release_sprawl" in cmd:
            continue
        pp = []
        try:
            for kv in (d / "environ").read_bytes().split(b"\0"):
                if kv.startswith(b"PYTHONPATH="):
                    pp = [p for p in kv[len(b"PYTHONPATH="):].decode().split(":") if p]
        except OSError:
            pass
        out.append({"pid": int(d.name), "service": _service(cmd), "pythonpath": pp})
    return out


def scan_scripts(scripts_dir: Path) -> list[dict]:
    out = []
    for p in sorted(scripts_dir.glob("*.sh")):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        launch = [ln for ln in text.splitlines()
                  if not ln.lstrip().startswith("#") and _PYTHONPATH.search(ln)]
        calls = sorted(set(re.findall(r"(restart_[\w.-]+\.sh)", text)) - {p.name})
        out.append({"script": p.name, "path": str(p),
                    "service": _service(" ".join(launch)) if launch else "?",
                    "pythonpath": _paths(" ".join(launch)), "calls": calls,
                    "restart": p.name.startswith("restart_")})
    return out


def scan_units(units_dir: Path) -> list[dict]:
    out = []
    files = list(units_dir.glob("*.service")) + list(units_dir.glob("*.service.d/*.conf")) \
        + list(units_dir.glob("*.timer"))
    for p in sorted(files):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "cascade" not in text:
            continue
        refs = sorted(set(re.findall(r"/root/[\w./-]+", text)))
        out.append({"unit": str(p.relative_to(units_dir)), "pythonpath": _paths(text),
                    "refs": refs})
    return out


def symlinks(releases_dir: Path) -> dict[str, str]:
    out = {}
    for p in sorted(releases_dir.glob("*.current")) if releases_dir.is_dir() else ():
        with contextlib.suppress(OSError):
            out[p.name] = os.path.realpath(p)
    return out


def report(*, root: Path, scripts_dir: Path, units_dir: Path,
           proc_root: Path = Path("/proc"), releases_dir: Path | None = None) -> dict:
    procs = running_releases(proc_root)
    scripts = scan_scripts(scripts_dir)
    units = scan_units(units_dir)
    links = symlinks(releases_dir or root / "cascade-releases")
    live_paths = {os.path.realpath(p) for pr in procs for p in pr["pythonpath"]}
    live_by_service: dict[str, set[str]] = {}
    for pr in procs:
        live_by_service.setdefault(pr["service"], set()).update(
            os.path.realpath(p) for p in pr["pythonpath"])
    # A script is REFERENCED when a unit calls it, or a script a unit calls
    # does (the watchdog); calls between old restart scripts do not count.
    by_name = {s["script"]: s for s in scripts}
    called = {Path(r).name for u in units for r in u["refs"] if r.endswith(".sh")}
    todo = list(called)
    while todo:
        for c in by_name.get(todo.pop(), {}).get("calls", ()):
            if c not in called:
                called.add(c)
                todo.append(c)
    for s in scripts:
        real = [os.path.realpath(p) for p in s["pythonpath"]]
        if not s["restart"] and not s["pythonpath"]:
            s["verdict"] = "OTHER"
        elif s["script"] in called:
            s["verdict"] = "REFERENCED"
        elif real and any(r in live_by_service.get(s["service"], set()) for r in real):
            s["verdict"] = "LIVE"
        elif real and not all(Path(r).exists() for r in real):
            s["verdict"] = "STALE"
            s["why"] = "launches a release dir that no longer exists"
        else:
            s["verdict"] = "STALE"
            s["why"] = "launches a release no running service uses"
    dirs = []
    for d in release_dirs(root):
        rd = os.path.realpath(d)
        why = []
        if rd in live_paths:
            why.append("running: " + ",".join(sorted({pr["service"] for pr in procs
                                                      if rd in {os.path.realpath(p) for p in pr["pythonpath"]}})))
        why += [f"unit {u['unit']}" for u in units
                if rd in {os.path.realpath(p) for p in u["pythonpath"] + u["refs"]}]
        why += [f"symlink {n}" for n, t in links.items() if t == rd]
        live_scripts = [s["script"] for s in scripts if s.get("verdict") in ("LIVE", "REFERENCED")
                        and rd in {os.path.realpath(p) for p in s["pythonpath"]}]
        why += [f"script {n}" for n in live_scripts]
        stale_scripts = [s["script"] for s in scripts if s.get("verdict") == "STALE"
                         and rd in {os.path.realpath(p) for p in s["pythonpath"]}]
        dirs.append({"dir": str(d), "head": git_head(d), "used_by": why,
                     "stale_scripts": stale_scripts,
                     "worktree": (d / ".git").is_file(),
                     "verdict": "IN USE" if why else "STALE"})
    return {"processes": procs, "release_dirs": dirs, "scripts": scripts, "units": units,
            "symlinks": links,
            "stale_dirs": [d["dir"] for d in dirs if d["verdict"] == "STALE"],
            "stale_scripts": [s["script"] for s in scripts if s.get("verdict") == "STALE"]}


def render(rep: dict) -> str:
    lines = ["release dirs:"]
    for d in rep["release_dirs"]:
        lines.append(f"  {d['verdict']:<7} {d['dir']} @ {d['head'] or '?'}"
                     f"{' (worktree)' if d['worktree'] else ''}"
                     f"{': ' + '; '.join(d['used_by']) if d['used_by'] else ''}")
    lines.append("restart scripts:")
    for s in rep["scripts"]:
        if s["verdict"] == "OTHER":
            continue
        lines.append(f"  {s['verdict']:<10} {s['script']} [{s['service']}] -> "
                     f"{':'.join(s['pythonpath']) or '-'}"
                     f"{' (' + s['why'] + ')' if s.get('why') else ''}")
    lines.append(f"STALE: {len(rep['stale_dirs'])} release dir(s), "
                 f"{len(rep['stale_scripts'])} script(s) — report only, nothing changed")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path("/root"))
    ap.add_argument("--scripts", type=Path, default=Path("/root/cascade"))
    ap.add_argument("--units", type=Path, default=Path("/etc/systemd/system"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = report(root=a.root, scripts_dir=a.scripts, units_dir=a.units)
    print(json.dumps(rep, indent=2, sort_keys=True) if a.json else render(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
