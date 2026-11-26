"""Shared helpers for the R11 acceptance tests (shipped-config startup gates, T26).

The harness is driven as a subprocess,
``python -P -m cascade.laned.startup_gates --config F --head-block N --role R``,
under a guard that refuses every network connection and every ``torch``
import (the harness reads no chain, no storage, and loads no model). Planted
configs are built from the REAL shipped ``chain.toml`` text; planted
activation state is written with the REAL ``ActivationStore``. Nothing in the
component is mocked.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODULE = "cascade.laned.startup_gates"
R2_CRASH_FIXTURE = REPO / "tests" / "accept" / "r2" / "fixtures" / "crashloop_20261113.json"
SHIPPED = ("chain.toml", "chain.testnet.toml")
ROLES = ("validator", "trainer")

# A real mainnet head on 2026-11-21: between the 9547200 era start (KING CHECK
# settlement, OPSLOG 2026-11-20 10:26 UTC) and the 9561600 king check, at 12 s
# per block. HEAD_WORST adds NOTE_BLOCK_HORIZON (216000), the reach of the
# worst-case note rule.
HEAD = 9_557_440
HEAD_TS = int(datetime(2026, 11, 21, 20, 33, tzinfo=UTC).timestamp())
NOTE_BLOCK_HORIZON = 216_000
HEAD_WORST = HEAD + NOTE_BLOCK_HORIZON
HEAD_WORST_TS = HEAD_TS + 12 * NOTE_BLOCK_HORIZON

LINE_RE = re.compile(r"^(PASS|FAIL|WARN|SKIP) ([a-z0-9_]+)(?:: (.*))?$")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_GUARD = '''\
import runpy
import socket
import sys


class _NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("R11 guard: the startup-gate harness must not import torch")
        return None


def _no_network(*args, **kwargs):
    raise OSError("R11 guard: the startup-gate harness must not open a network connection")


sys.meta_path.insert(0, _NoTorch())
socket.socket.connect = _no_network
socket.socket.connect_ex = _no_network
socket.create_connection = _no_network
sys.argv = [MODULE] + sys.argv[1:]
runpy.run_module(MODULE, run_name="__main__", alter_sys=True)
'''


def _guard_script(tmp_path: Path) -> Path:
    p = tmp_path / "r11_guard.py"
    if not p.exists():
        p.write_text(f"MODULE = {MODULE!r}\n" + _GUARD)
    return p


def run_gates(tmp_path: Path, config: Path | str, role: str, head: int | None,
              *extra: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run the harness under the no-network / no-torch guard, cwd = repo root.
    ``PYTHONPATH`` = this repo only (so the code-path gate declares this tree)."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("CASCADE_RELEASE_PATH", "CASCADE_ALLOW_CODE_PATH_MISMATCH")}
    env["PYTHONPATH"] = str(REPO)
    if env_extra:
        env.update(env_extra)
    args = ["--config", str(config), "--role", role]
    if head is not None:
        args += ["--head-block", str(head)]
    args += list(extra)
    return subprocess.run([sys.executable, "-P", str(_guard_script(tmp_path)), *args],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
                          check=False)


def parse_lines(proc: subprocess.CompletedProcess) -> list[tuple[str, str, str]]:
    """stdout -> [(status, gate, reason)]; every non-empty stdout line must be a gate line."""
    out = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        m = LINE_RE.match(line)
        assert m, (f"stdout line is not `<PASS|FAIL|WARN|SKIP> <gate>[: <reason>]`: {line!r}\n"
                   f"stdout: {proc.stdout[-3000:]}\nstderr: {proc.stderr[-3000:]}")
        out.append((m.group(1), m.group(2), m.group(3) or ""))
    return out


def describe(proc: subprocess.CompletedProcess) -> str:
    return (f"exit {proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
            f"stderr:\n{proc.stderr[-3000:]}")


def harness():
    """The harness module, imported from THIS tree."""
    import importlib

    mod = importlib.import_module(MODULE)
    assert Path(mod.__file__).resolve().is_relative_to(REPO), mod.__file__
    return mod


def activation():
    import cascade.shared.activation as act

    assert Path(act.__file__).resolve().is_relative_to(REPO), act.__file__
    return act


def load_crash_fixture() -> dict:
    return json.loads(R2_CRASH_FIXTURE.read_text())


# ── planted configs (from the real shipped text) ─────────────────────────────


def set_key(text: str, section: str, key: str, value: str) -> str:
    """Rewrite ``key`` inside ``[section]`` only (exactly one line)."""
    head = re.search(rf"(?m)^\[{re.escape(section)}\]\s*$", text)
    assert head, section
    nxt = re.search(r"(?m)^\[", text[head.end():])
    end = head.end() + (nxt.start() if nxt else len(text) - head.end())
    body, n = re.subn(rf"(?m)^{re.escape(key)}\s*=[^\n]*$", f"{key} = {value}",
                      text[head.end():end], count=1)
    assert n == 1, (section, key)
    return text[:head.end()] + body + text[end:]


def read_source(rel: str) -> str:
    return (REPO / rel).read_text()


def shipped_text(name: str = "chain.toml") -> str:
    return (REPO / name).read_text()


def write_config(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def _predecision_module():
    spec = importlib.util.spec_from_file_location(
        "r11_predecision_main", REPO / "tests" / "unit" / "predecision_main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def predecision_text() -> str:
    """The release chain.toml with the 2026-11-13 decided blocks untyped (the
    rollover vote open, ``[v2] from_block = 0`` so the dated launch is pending)."""
    return _predecision_module().predecision_text()


def crashloop_text() -> str:
    """The chain.toml the 2026-11-13 external validator ran: the pre-decision
    release with the v2b vote still open (``image_decoupled_from_block = 0``,
    the line PR #457 typed)."""
    return set_key(predecision_text(), "v2", "image_decoupled_from_block", "0")


def plant_crashloop_state(state_dir: Path, fx: dict) -> None:
    """The external validator's persisted activation records, written with the
    real store under the real file names the validator reads."""
    act = activation()
    state_dir.mkdir(parents=True, exist_ok=True)
    roll = fx["records"]["rolling-era-king"]
    pre = fx["records"]["prelaunch-95-r2"]
    act.ActivationStore(state_dir / "activation_state.json").save(act.ActivationRecord(
        feature="rolling-era-king", lock_block=roll[0], activation_block=roll[1],
        source="tally"))
    act.ActivationStore(state_dir / "activation_prelaunch_burn_state.json").save(
        act.ActivationRecord(feature="prelaunch-95-r2", lock_block=pre[0],
                             activation_block=pre[1], source="tally"))


# ── the startup path, read from the source (AST) ─────────────────────────────

# Where each role's real startup sequence lives: (file, function qualname).
# For a ``run_forever`` only the statements before its first top-level
# ``while`` (the poll loop) are startup.
STARTUP_SCOPES = {
    "validator": (
        ("cascade/validator/main.py", "main"),
        ("cascade/validator/loop.py", "ValidatorRunner.run_forever"),
        ("cascade/validator/activation_mixin.py", "ActivationMixin._activation_startup"),
        ("cascade/validator/activation_mixin.py", "ActivationMixin._v2b_typed_notice_startup"),
    ),
    "trainer": (
        ("cascade/trainer/main.py", "main"),
        ("cascade/trainer/loop.py", "TrainerRunner.run_forever"),
        ("cascade/trainer/loop.py", "TrainerRunner._image_decoupled_startup"),
    ),
}

# A startup gate call: a callee whose last name segment matches this.
GATE_NAME_RE = re.compile(r"^(verify_|require_|assert_|check_)|_problem$|^load_chain_config$")

# Calls on the startup path that match the pattern but are not startup gates
# of the live service, with the reason.
NOT_GATES = {
    "validator": {
        # `--direct-receipt FILE`: an offline replay mode that returns before
        # any service starts.
        "verify_direct_receipt",
    },
    "trainer": set(),
}


def _find(tree: ast.AST, qualname: str) -> ast.AST:
    node = tree
    for part in qualname.split("."):
        found = [n for n in ast.iter_child_nodes(node)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                 and n.name == part]
        assert found, f"startup scope {qualname!r}: {part!r} not found"
        node = found[0]
    return node


def _callee(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def startup_calls_in(source: str, qualname: str) -> list[str]:
    """Every callee name in the startup part of ``qualname`` in ``source``."""
    fn = _find(ast.parse(source), qualname)
    names: list[str] = []
    for stmt in fn.body:
        if isinstance(stmt, ast.While):
            break
        for n in ast.walk(stmt):
            if isinstance(n, ast.Call):
                name = _callee(n)
                if name:
                    names.append(name)
    return names


def startup_calls(role: str, sources: dict[str, str] | None = None) -> list[str]:
    """All callee names on ``role``'s startup path; ``sources`` overrides a
    file's text (relpath -> source) for the scanner self-test."""
    sources = sources or {}
    out: list[str] = []
    for rel, qual in STARTUP_SCOPES[role]:
        text = sources.get(rel)
        if text is None:
            text = (REPO / rel).read_text()
        out.extend(startup_calls_in(text, qual))
    return out


def startup_gate_calls(role: str, sources: dict[str, str] | None = None) -> set[str]:
    return {n for n in startup_calls(role, sources)
            if GATE_NAME_RE.search(n) and n not in NOT_GATES[role]}
