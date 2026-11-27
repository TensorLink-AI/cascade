"""The trainer's lane B leg contract == the contract the worker trains, for every
funding variant (2026-11-27 RCA guard, docs/launch-lessons-2026-11.md).

2026-11-27 13:47: the rescued 5GMzsP leg was rejected because the trainer
applied default levers to a no-spec leg (expected ebfed637…) while the worker
folds levers only ``if lane_b_spec:`` (receipt a3ea79b2…). #980 fixed the
no-spec case and pinned it against a test-side re-implementation of the
worker. This test runs the worker's OWN expert-contract block — extracted from
``cascade/trainer/worker.py`` by AST, so the worker module (and torch) is never
imported — on the exact ``--recipe-json`` / ``--block`` / ``--contract-block``
the trainer's dispatch hands it, for: no spec, a WB23 length only
(``train_hours``), levers only, and length + levers. Any drift between the two
derivations (the class behind #980) fails here, not on a paid pod.

Torch-free: torch imports are refused for the whole module.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.accept.wb27 import helpers as h

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "cascade" / "trainer" / "worker.py"


def _worker_expert_block():
    """``def _expert(cfg, contract, args, expert_hours, log)`` built from the
    body of worker.main's ``elif expert_hours is not None:`` branch."""
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    branch = None
    for node in ast.walk(main):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "expert_hours"
                and isinstance(node.test.ops[0], ast.IsNot)):
            branch = node
            break
    assert branch is not None, ("worker.main has no `elif expert_hours is not None:` branch: "
                                "update this parity test to the worker's lane B contract code")
    fn = ast.FunctionDef(
        name="_expert",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=a) for a in
                                                 ("cfg", "contract", "args", "expert_hours", "log")],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[*branch.body, ast.Return(value=ast.Name(id="contract", ctx=ast.Load()))],
        decorator_list=[], returns=None, type_params=[])
    mod = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    ns = {"__name__": "cascade.trainer.worker", "__package__": "cascade.trainer", "json": json}
    exec(compile(mod, str(WORKER), "exec"), ns)   # noqa: S102 — our own source, no import
    return ns["_expert"]


def _worker_contract(cfg, *, contract_block: int, block: int, recipe_json: str | None):
    """What a pod running ``worker --role expert`` trains (worker.py's steps,
    from its ``--contract-block`` base to the expert branch)."""
    contract = cfg.training.at_block(contract_block).primary_size
    args = SimpleNamespace(recipe_json=recipe_json, block=block)
    out = _worker_expert_block()(cfg, contract, args, float(cfg.lane_b.leg_hours),
                                 logging.getLogger("parity"))
    assert not isinstance(out, int), f"the worker refused the leg spec (exit {out})"
    return out

GEN = "alice/metro-gen@sha256:" + "a" * 64
PTR = "metro-v1:trained:hippius:org/ckpt@sha256:" + "b" * 64


def _runner(tmp_path, monkeypatch, run):
    """A TrainerRunner whose funded-leg call is ``run`` (the pod stand-in)."""
    from cascade.trainer.loop import TrainerRunner
    from tests.accept.wb23.helpers import SKU, STEP_RATES_ENV, table

    cfg = h.chain_cfg()
    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    era = SimpleNamespace(base_seed=12345, start_block=int(cfg.training.v2_from_block), index=3)
    r.cfg = dataclasses.replace(cfg, round=dataclasses.replace(cfg.round, funded_mode="required"))
    r.__dict__["_rolling_sched"] = SimpleNamespace(state=SimpleNamespace(current=era))
    monkeypatch.setattr(TrainerRunner, "_dispatcher", lambda self: object())
    monkeypatch.setattr(TrainerRunner, "_rolling_lane_pool",
                        lambda self: pytest.fail("operator pool used"))
    monkeypatch.setattr(TrainerRunner, "_era_contract_block",
                        lambda self, era: int(era.start_block))
    monkeypatch.setattr(TrainerRunner, "_run_funded_leg", run)
    p = tmp_path / "rates.json"
    p.write_text(json.dumps(table([SKU])), encoding="utf-8")
    monkeypatch.setenv(STEP_RATES_ENV, str(p))
    return r


def _leg(extra: dict):
    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg
    from tests.accept.wb23.helpers import SKU

    funding = {"entry_ref": "x", "gpu": SKU, "gpu_allow_fallback": False,
               "gpu_max_price_per_hour": 30.0, **extra}
    return OpenLeg(sub=LaneBSubmission("alice", 7, GEN, "energy", 5),
                   selection={1: [9]}, base_king_ref="kingref", base_era=3,
                   exclude={}, funding=funding)


FUNDING = {
    "no-spec": {},
    "train-hours": {"train_hours": 2.0, "leg_steps": 100, "leg_guard_seconds": 9600},
    "levers": {"funded_levers": {"lr_scale": 1.5, "ema": 0.05}},
    "train-hours+levers": {"train_hours": 2.0, "leg_steps": 100, "leg_guard_seconds": 9600,
                           "funded_levers": {"lr_scale": 1.5, "ema": 0.05}},
}


@pytest.mark.parametrize("variant", sorted(FUNDING))
def test_dispatch_contract_is_what_the_worker_trains(variant, tmp_path, monkeypatch):
    with h.torch_blocked():
        from cascade.shared.manifest import contract_digest

        seen = {}

        def run(self, disp, gen, seeds, block, contract, suffix, **kw):
            seen.update(contract=contract, block=int(block), kw=kw)
            return SimpleNamespace(trained_pointer=PTR, corpus_digest="cd",
                                   train_block=int(block), gpu_name="NVIDIA H200")

        r = _runner(tmp_path, monkeypatch, run)
        era_start = int(r.cfg.training.v2_from_block)
        leg = _leg(FUNDING[variant])
        block = h.effective_block(r.cfg)
        from cascade.trainer.lane_b_driver import RunnerLaneBOps

        ops = RunnerLaneBOps(r, client=None, window_source=None)
        ev, out = threading.Event(), []

        def done(leg, ptr, *a, failure=None, **kw):
            out.append((ptr, failure))
            ev.set()

        ops.dispatch_leg(leg, 3, block, done)
        assert ev.wait(10), "the leg never returned"
        ptr, failure = out[0]
        assert failure is None and ptr == PTR, failure
        assert seen["kw"]["contract_block"] == era_start
        recipe = seen["kw"].get("finetune_recipe")
        if variant == "no-spec":
            assert recipe is None, "a no-spec leg must not pass --recipe-json"
        else:
            assert recipe, f"{variant}: the dispatch passed no --recipe-json"
        trainer = seen["contract"]
        worker = _worker_contract(r.cfg, contract_block=seen["kw"]["contract_block"],
                                  block=seen["block"], recipe_json=recipe)
        diff = {f.name: (getattr(trainer, f.name), getattr(worker, f.name))
                for f in dataclasses.fields(trainer)
                if getattr(trainer, f.name) != getattr(worker, f.name)}
        assert not diff, f"{variant}: trainer vs worker contract fields differ: {diff}"
        assert contract_digest(trainer) == contract_digest(worker)


def test_the_extracted_worker_block_folds_levers_only_with_a_spec():
    # the #980 rule, read from the worker's own code: no spec → no levers
    with h.torch_blocked():
        cfg, b = h.armed_cfg()
        base_block = int(cfg.training.v2_from_block)
        plain = _worker_contract(cfg, contract_block=base_block, block=b, recipe_json=None)
        levered = _worker_contract(cfg, contract_block=base_block, block=b,
                                   recipe_json=json.dumps({"levers": {"lr_scale": 1.5}}))
        assert getattr(levered, "lane_b_lr_scale", 1.0) == 1.5
        assert getattr(plain, "lane_b_lr_scale", 1.0) == 1.0
