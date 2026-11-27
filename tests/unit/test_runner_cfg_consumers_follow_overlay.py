"""Every config-holding component the trainer hangs off its runner follows the overlay.

2026-11-26 (#951): ``TrainerRunner._point_cfg_consumers`` re-pointed only the
rolling scheduler and promotion. ``LaneBDriver`` / ``LaneCDriver`` were built
in ``cascade/trainer/main.py`` with the STARTUP ``ChainConfig`` (lanes B/C off
in the base file; the owner's overlay turned them on), so ``lane_b.active()``
stayed False and the first funded lane B entry sat queued for hours.

The class: a component built with ``cfg`` and stored on the runner keeps the
config it was built with. This test reads ``main.py`` and fails for any
``runner.<attr> = Cls(cfg, ...)`` whose ``<attr>`` the re-point hook never
names. A new lane driver therefore cannot ship overlay-blind. Torch-free (AST
only); tests/unit/test_lane_drivers_follow_overlay.py proves the behaviour.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _passes_cfg(call: ast.Call) -> bool:
    vals = [*call.args, *(k.value for k in call.keywords)]
    return any(isinstance(v, ast.Name) and v.id == "cfg" for v in vals)


def runner_cfg_attrs(src: str) -> dict[str, int]:
    """attr -> line for every `runner.<attr> = Something(cfg, ...)` in src."""
    out = {}
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "runner" and _passes_cfg(node.value)):
                out[tgt.attr] = node.lineno
    return out


def repointed_names(src: str) -> set[str]:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_point_cfg_consumers":
            names = set()
            for n in ast.walk(node):
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    names.add(n.value)
                elif isinstance(n, ast.Attribute):
                    names.add(n.attr)
            return names
    raise AssertionError("TrainerRunner._point_cfg_consumers is gone: who re-points the drivers?")


def test_every_cfg_component_on_the_runner_is_repointed():
    built = runner_cfg_attrs((ROOT / "cascade/trainer/main.py").read_text())
    assert {"lane_b_driver", "lane_c_driver"} <= set(built), built
    named = repointed_names((ROOT / "cascade/trainer/loop.py").read_text())
    blind = {a: line for a, line in built.items() if a not in named}
    assert not blind, (
        "built with the startup cfg in cascade/trainer/main.py but never re-pointed by "
        f"TrainerRunner._point_cfg_consumers (overlay-blind, the #951 class): {blind}")


def test_the_guard_catches_the_pre_951_hook():
    pre_951 = textwrap.dedent('''
        class TrainerRunner:
            def _point_cfg_consumers(self, cfg):
                sched = self.__dict__.get("_rolling_sched")
                if sched is not None:
                    sched.update_config(cfg)
                promo = getattr(self, "promotion", None)
                if promo is not None:
                    promo.round_cfg = cfg.round
        ''')
    built = runner_cfg_attrs(textwrap.dedent('''
        def main(cfg, runner):
            runner.lane_b_driver = LaneBDriver(cfg, ops, path)
        '''))
    assert set(built) - repointed_names(pre_951) == {"lane_b_driver"}


def test_the_hook_reaches_the_driver_cfg():
    from cascade.trainer.loop import TrainerRunner

    src = inspect.getsource(TrainerRunner._point_cfg_consumers)
    assert "drv.cfg = cfg" in src or ".cfg = cfg" in src
