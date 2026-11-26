"""Shared helpers for the R2 acceptance tests (note-budget decoder, S0b).

The decoder is driven as a subprocess, ``python -m cascade.laned.notes ...``,
so the tests pin the CLI contract (README.md), not internals. The
cross-checks build notes with the REAL builder in
``cascade/shared/activation.py`` (``own_signal_payload`` /
``note_trimmed_segments``) from the REAL shipped ``chain.toml``; nothing is
mocked.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
MODULE = "cascade.laned.notes"

# The in-process cross-checks must import THIS tree's cascade, never an
# installed release.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def activation():
    import cascade.shared.activation as act

    assert Path(act.__file__).resolve().is_relative_to(REPO), act.__file__
    return act


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def run_notes(*args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    return subprocess.run([sys.executable, "-m", MODULE, *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120, check=False)


def run_json(*args: str, expect_rc: int) -> dict:
    """Run the decoder; assert its exit code and that stdout is ONE JSON object."""
    proc = run_notes(*args)
    assert proc.returncode == expect_rc, (
        f"`python -m {MODULE} {' '.join(args)}` exited {proc.returncode}, want {expect_rc}\n"
        f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}")
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(f"stdout is not one JSON object ({e}): {proc.stdout[-2000:]!r}\n"
                             f"stderr: {proc.stderr[-2000:]}") from None
    assert isinstance(out, dict), out
    return out


def simulate(note: str, head_block: int, cfg_feature: str, lock: int | None = None,
             act: int | None = None, *, expect_rc: int, as_hex: bool = True,
             worst_case: bool = False) -> dict:
    src = ["--hex", note.encode("utf-8").hex()] if as_hex else ["--text", note]
    args = ["simulate", *src, "--head-block", str(head_block), "--cfg", cfg_feature]
    if lock is not None:
        args += ["--cfg-lock", str(lock)]
    if act is not None:
        args += ["--cfg-activation", str(act)]
    if worst_case:
        args.append("--worst-case")
    return run_json(*args, expect_rc=expect_rc)


def segment(feature: str, lock: int = 0, act: int = 0) -> str:
    return f"{feature}:{lock}:{act}"


def note_of(*segments: str) -> str:
    return "cascade-ready:1:" + ":".join(segments)


# ── real configs ─────────────────────────────────────────────────────────────


def shipped_config():
    from cascade.shared.config import load_chain_config

    return load_chain_config(REPO / "chain.toml")


def _predecision_module():
    spec = importlib.util.spec_from_file_location(
        "r2_predecision_main", REPO / "tests" / "unit" / "predecision_main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def crashloop_config(tmp_path: Path):
    """The release chain.toml as the 2026-11-13 external validator ran it: the
    rollover vote untyped (the pre-decision release text) and the v2b vote
    still open (``image_decoupled_from_block = 0``, the line PR #457 typed);
    forfeit / margin-v2 typed as main had them since #410."""
    act = activation()
    cfg = _predecision_module().predecision_main(tmp_path)
    cfg = replace(cfg, v2=replace(cfg.v2, image_decoupled_from_block=0))
    assert act.image_decoupled_feature(cfg).decided_on_chain
    return cfg


def all_votes_open_config(tmp_path: Path):
    """The pre-decision release with the forfeit, margin-v2 and v2b blocks also
    untyped: every secondary vote rides the note (the widest shape a node that
    never pulled the typed blocks carries)."""
    from cascade.shared.config import load_chain_config

    text = _predecision_module().predecision_text()
    for key in ("forfeit_from_block", "margin_v2_from_block", "image_decoupled_from_block"):
        text, n = re.subn(rf"(?m)^{key}\s*=.*$", f"{key} = 0", text)
        assert n == 1, key
    p = tmp_path / "all_votes_open.toml"
    p.write_text(text)
    return load_chain_config(p)


def builder_note_and_trim(cfg, records: dict, cfg_seg):
    """What the REAL builder does: (untrimmed note without cfg-, final note with
    ``cfg_seg``, trimmed feature names in trim order). ``records`` maps the
    own_signal_payload keyword (``record``, ``forfeit_record``, ...) to an
    ActivationRecord."""
    act = activation()
    kw = {k: records.get(k) for k in ("forfeit_record", "margin_v2_record",
                                       "prelaunch_record", "image_decoupled_record",
                                       "shared_ladder_record")}
    record = records.get("record") or act.ActivationRecord()
    parts = act._own_note_parts(cfg, record, netconfig_segment=None, in_force=None, **kw)
    assert parts is not None, "the builder signals nothing for this config"
    head, extras, _tail = parts
    untrimmed = act.format_signals(head, *extras)
    final = act.own_signal_payload(cfg, record, netconfig_segment=cfg_seg, in_force=None, **kw)
    trimmed = act.note_trimmed_segments(cfg, record, netconfig_segment=cfg_seg,
                                        in_force=None, **kw)
    return untrimmed, final, trimmed
