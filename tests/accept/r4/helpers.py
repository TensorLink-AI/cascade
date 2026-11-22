"""Shared helpers for the R4 acceptance tests (stake-weighted upgrade tracker).

The tracker is driven as a subprocess,
``python -P -m cascade.laned.upgrade_tracker ...``, so the tests pin the CLI
contract (README.md), not internals. The cross-checks call the REAL tally in
``cascade/shared/activation.py`` (``tally``, ``parse_signals``) and the REAL
``cfg-`` naming in ``cascade/shared/netconfig.py``; nothing is mocked.
Fixtures are written to ``tmp_path`` inside test bodies.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODULE = "cascade.laned.upgrade_tracker"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TARGET = "cfg-1a2b3c4d"
OLD_CFG = "cfg-99887766"
RELEASE = "rolling-era-king"

# The S6 shape (plan §5 row 11): six permit-holding validators, the largest
# (5E2LP6…) at 49% of the eligible stake. Integer stakes, total 1,000,000.
LARGEST = "5E2LP6largestholder"
S6_STAKES = {
    LARGEST: 490_000,
    "5Bvalidator2": 180_000,
    "5Cvalidator3": 120_000,
    "5Dvalidator4": 100_000,
    "5Evalidator5": 70_000,
    "5Fvalidator6": 40_000,
}
# Not counted: no permit (however large), and a permit with zero stake.
NO_PERMIT = "5Gnopermitbig"
ZERO_STAKE = "5Hzerostake"


def activation():
    import cascade.shared.activation as act

    assert Path(act.__file__).resolve().is_relative_to(REPO), act.__file__
    return act


def netconfig():
    import cascade.shared.netconfig as nc

    assert Path(nc.__file__).resolve().is_relative_to(REPO), nc.__file__
    return nc


def note_of(*segments: str) -> str:
    return "cascade-ready:1:" + ":".join(segments)


def on_note(cfg: str = TARGET, release: str = RELEASE) -> str:
    """A capability-release node's note: primary, one extra, the cfg- tail."""
    return note_of(f"{release}:9374400:9378000", "prelaunch-95-r2:9486000:9486900",
                   f"{cfg}:0:0")


def off_note(release: str = RELEASE) -> str:
    """A node that never adopted any overlay: no cfg- segment."""
    return note_of(f"{release}:9374400:9378000", "prelaunch-95-r2:9486000:9486900")


def s6_metagraph(extra: bool = True) -> dict:
    mg = {hk: {"stake": st, "validator_permit": True} for hk, st in S6_STAKES.items()}
    if extra:
        mg[NO_PERMIT] = {"stake": 300_000, "validator_permit": False}
        mg[ZERO_STAKE] = {"stake": 0, "validator_permit": True}
    return mg


def s6_notes(off: tuple[str, ...] = ()) -> dict:
    notes = {hk: (off_note() if hk in off else on_note()) for hk in S6_STAKES}
    notes[NO_PERMIT] = on_note()
    notes[ZERO_STAKE] = on_note()
    return notes


def write_inputs(tmp_path: Path, metagraph, notes) -> tuple[Path, Path]:
    mg = tmp_path / "metagraph.json"
    nt = tmp_path / "notes.json"
    mg.write_text(metagraph if isinstance(metagraph, str) else json.dumps(metagraph))
    nt.write_text(notes if isinstance(notes, str) else json.dumps(notes))
    return mg, nt


def run_tracker(*args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    return subprocess.run([sys.executable, "-P", "-m", MODULE, *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120, check=False)


def track(tmp_path: Path, metagraph, notes, cfg: str = TARGET, *extra: str,
          expect_rc: int) -> dict:
    """Run the tracker on fixture files; assert its exit code and ONE JSON object."""
    mg, nt = write_inputs(tmp_path, metagraph, notes)
    args = ["--metagraph", str(mg), "--notes", str(nt), "--cfg", cfg, *extra]
    proc = run_tracker(*args)
    assert proc.returncode == expect_rc, (
        f"`python -P -m {MODULE} {' '.join(args)}` exited {proc.returncode}, want "
        f"{expect_rc}\nstdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}")
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(f"stdout is not one JSON object ({e}): {proc.stdout[-2000:]!r}\n"
                             f"stderr: {proc.stderr[-2000:]}") from None
    assert isinstance(out, dict), out
    return out


def track_bad(tmp_path: Path, metagraph, notes, cfg: str = TARGET, *extra: str,
              mg_path: str | None = None) -> subprocess.CompletedProcess:
    """Run on bad input: assert exit 2 with a message on stderr."""
    mg, nt = write_inputs(tmp_path, metagraph, notes)
    args = ["--metagraph", mg_path or str(mg), "--notes", str(nt), "--cfg", cfg, *extra]
    proc = run_tracker(*args)
    assert proc.returncode == 2, (
        f"`python -P -m {MODULE} {' '.join(args)}` exited {proc.returncode}, want 2\n"
        f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}")
    assert proc.stderr.strip(), "exit 2 must carry a message on stderr"
    return proc


def by_hotkey(out: dict) -> dict:
    return {row["hotkey"]: row for row in out["hotkeys"]}


def laggard_names(out: dict) -> list[str]:
    return [row["hotkey"] for row in out["laggards"]]
