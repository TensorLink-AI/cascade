"""Shared fixtures for the R3 overlay-lint acceptance tests.

Bodies are built with the real ``cascade.shared.netconfig.next_overlay``
against the shipped ``chain.toml`` (W2 is on main). The module under test is
imported only through :func:`lint_module`, called inside test bodies.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import subprocess
import sys
import tomllib
from functools import cache
from pathlib import Path

from cascade.shared import netconfig as N
from cascade.shared.config import parse_chain_config

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "chain.toml"
MODULE = "cascade.laned.overlay_lint"

ERA = 3600                              # mainnet era (900 x 4): every block below is one
EFF1 = 2690 * ERA                       # seq 1 (lane A tuning)
EFF2 = 2698 * ERA                       # seq 2 (S6 activation)
EFF3 = 2720 * ERA                       # seq 3 (kill switch)
ARM_BLOCK = 2700 * ERA
UNTIL_BLOCK = 2724 * ERA
ARM = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a"}
KILL_UNTIL = {**ARM, "until_block": UNTIL_BLOCK}
KILL_SHARE0 = {**ARM, "share": 0.0}
LANE_A = {"scoring": {"win_margin_start": 0.015, "min_windows": 300}}


def lint_module():
    return importlib.import_module(MODULE)


@cache
def base_cfg():
    return parse_chain_config(tomllib.loads(BASE.read_text()))


def network() -> dict:
    return N.network_identity(base_cfg())


def body(overrides: dict, effective_block: int, previous: dict | None) -> dict:
    return N.next_overlay(copy.deepcopy(overrides), effective_block, previous, network=network())


def raw_body(overrides: dict, effective_block: int, previous: dict | None) -> dict:
    """A body built WITHOUT ``check_body`` (shapes today's release refuses)."""
    return {
        "protocol": N.PROTOCOL,
        "network": network(),
        "seq": 1 if previous is None else int(previous["seq"]) + 1,
        "prev": None if previous is None else N.body_digest(previous),
        "effective_block": int(effective_block),
        "overrides": copy.deepcopy(overrides),
    }


def seq1() -> dict:
    """Lane A tuning only."""
    return body(LANE_A, EFF1, None)


def seq2() -> dict:
    """The S6 activation: lane A restated byte-identically + [lane_d] armed."""
    return body({**LANE_A, "lane_d": ARM}, EFF2, seq1())


def lane_d_over(lane_a: dict, lane_d: dict) -> dict:
    return {**copy.deepcopy(lane_a), "lane_d": dict(lane_d)}


def policy(lane_d: bool = True) -> dict:
    p = copy.deepcopy(N.overlay_policy())
    if not lane_d:
        p["overlay_keys"].pop("lane_d", None)
    return p


def fleet(**nodes: bool) -> dict:
    """name -> has lane_d in its policy."""
    return {"nodes": [{"name": name.replace("_", "-"), "policy": policy(has)}
                      for name, has in nodes.items()]}


def _reorder(obj):
    if isinstance(obj, dict):
        return {k: _reorder(obj[k]) for k in reversed(list(obj))}
    if isinstance(obj, list):
        return [_reorder(v) for v in obj]
    return obj


def write_json(path: Path, obj, *, reverse: bool = False) -> Path:
    path.write_text(json.dumps(_reorder(obj) if reverse else obj, indent=1))
    return path


def argv_for(tmp: Path, overlay: dict, previous: dict | None = None,
             policies: dict | None = None, *, reverse: bool = False) -> list[str]:
    tmp.mkdir(parents=True, exist_ok=True)
    argv = ["--overlay", str(write_json(tmp / "overlay.json", overlay, reverse=reverse))]
    if previous is not None:
        argv += ["--previous", str(write_json(tmp / "previous.json", previous, reverse=reverse))]
    argv += ["--base", str(BASE)]
    if policies is not None:
        argv += ["--policies", str(write_json(tmp / "policies.json", policies, reverse=reverse))]
    return argv


def run_main(capsys, argv: list[str]) -> tuple[int, list[str]]:
    mod = lint_module()
    capsys.readouterr()
    code = mod.main(list(argv))
    out = capsys.readouterr().out
    return code, out.splitlines()


def lint(capsys, tmp: Path, overlay: dict, previous: dict | None = None,
         policies: dict | None = None) -> tuple[int, list[str]]:
    return run_main(capsys, argv_for(tmp, overlay, previous, policies))


def findings(lines: list[str]) -> list[str]:
    return [ln for ln in lines if ln.startswith("REFUSED ")]


def with_rule(lines: list[str], rule: str) -> list[str]:
    return [ln for ln in findings(lines) if ln.startswith(f"REFUSED {rule} ")
            or ln.startswith(f"REFUSED {rule}:")]


def assert_ok(code: int, lines: list[str]) -> None:
    assert code == 0, f"lint exit {code}, expected 0:\n" + "\n".join(lines)
    assert lines and lines[-1] == "RESULT OK", "last line is not 'RESULT OK':\n" + "\n".join(lines)
    assert not findings(lines), "findings on an accepted overlay:\n" + "\n".join(lines)


def assert_refused(code: int, lines: list[str], rule: str, prefix: str = "") -> list[str]:
    assert code == 1, f"lint exit {code}, expected 1:\n" + "\n".join(lines)
    assert lines and lines[-1] == f"RESULT REFUSED {len(findings(lines))}", (
        "last line is not 'RESULT REFUSED <n>':\n" + "\n".join(lines))
    hits = with_rule(lines, rule)
    if prefix:
        hits = [ln for ln in hits if ln.startswith(f"REFUSED {rule} {prefix}")]
    assert hits, f"no 'REFUSED {rule} {prefix}' line:\n" + "\n".join(lines)
    return hits


def run_cli(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, "-P", "-m", MODULE, *argv], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)
