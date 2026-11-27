"""The box-side entry points import without torch / safetensors.

Owner rule (2026-10-31, restated 2026-11-26): nothing loads a model, a
checkpoint or torch on the 7 GB trainer/validator box. The lane B launch broke
it five times (#952-#955 trainer routing probe, judge, merge scores, rebase,
bench staging; #965 validator judge / merged-init / acceptance). Those fixes
moved model work to eval pods and import torch only inside pod-side functions.

This guard imports every long-running box-side module in a subprocess whose
import system refuses torch, safetensors, transformers and triton, and records
each attempt. A module-level ``import torch`` (even one wrapped in
``try/except ImportError``) on any of these paths fails here. The test never
imports torch itself, so it is safe on the box (torch-free, no model load).

The function-level paths (the judge / probe / merge refusals without an eval
host) are pinned by tests/accept/wb27/test_wb27_validator_remote.py and
tests/unit/test_lane_b_offload.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Long-running box-side processes: validator, satellite, intake, lane D
# service, trainer (non-pod modules), provisioner, audit — and the lane B/C/D
# driver / judge / offload modules they load.
BOX_MODULES = (
    "cascade.validator.main",
    "cascade.validator.loop",
    "cascade.validator.satellite",
    "cascade.validator.lane_b",
    "cascade.validator.expert_lane",
    "cascade.validator.merge_acceptance",
    "cascade.validator.era_gate",
    "cascade.validator.lane_d_inputs",
    "cascade.validator.lane_d_wiring",
    "cascade.validator.mkt_offload",
    "cascade.validator.eval_offload",
    "cascade.funding.main",
    "cascade.funding.intake",
    "cascade.lane_d.main",
    "cascade.lane_d.service",
    "cascade.trainer.main",
    "cascade.trainer.lane_b_driver",
    "cascade.trainer.lane_c_driver",
    "cascade.trainer.lane_b_offload",
    "cascade.trainer.lane_b_model",
    "cascade.trainer.lane_b_bench",
    "cascade.trainer.merge_acceptance",
    "cascade.trainer.runner_funded_legs",
    "cascade.trainer.runner_funded_pods",
    "cascade.trainer.runner_operator_pods",
    "cascade.trainer.money_guard",
    "cascade.trainer.cost_guards",
    "cascade.provision.main",
    "cascade.audit.main",
)

PROBE = r"""
import importlib, json, sys, traceback
HEAVY = ("torch", "safetensors", "transformers", "triton")
seen = []
class Refuse:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in HEAVY:
            where = [f"{f.filename.rsplit('/cascade/', 1)[-1]}:{f.lineno}"
                     for f in traceback.extract_stack() if "/cascade/" in f.filename]
            seen.append({"module": name, "from": where[-3:]})
            raise ImportError(f"box-side import of {name} refused")
        return None
sys.meta_path.insert(0, Refuse())
out = {}
for mod in sys.argv[1:]:
    seen.clear()
    try:
        importlib.import_module(mod)
        err = ""
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:300]
    out[mod] = {"heavy": list(seen), "error": err}
print(json.dumps(out))
"""


def test_box_side_modules_never_import_torch():
    proc = subprocess.run([sys.executable, "-P", "-c", PROBE, *BOX_MODULES],
                          capture_output=True, text=True, cwd=str(ROOT), timeout=600,
                          env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    heavy = {m: r["heavy"] for m, r in got.items() if r["heavy"]}
    assert not heavy, ("box-side modules import torch/safetensors at import time (owner rule: "
                       f"nothing heavy on the box; move it into the pod-side function): {heavy}")
    broken = {m: r["error"] for m, r in got.items() if r["error"]}
    assert not broken, f"box-side modules failed to import: {broken}"
