"""WB19 test helpers: the tool loader, the independent contract derivation,
GPU fixtures and pod-result fixtures. Contract: README.md.

Nothing here imports a PATHS module at import time. ``load_tool`` loads
``scripts/laned_wb19_measure.py`` by path when a test calls it; the reader
``cascade.funding.step_rates`` is imported inside test bodies only.
Stdlib + cascade config only: no torch, no GPU, no network.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "scripts" / "laned_wb19_measure.py"
CHAIN_TOML = REPO / "chain.toml"

RUN_SCHEMA = "laned_wb19_run/1"
EVIDENCE_SCHEMA = "laned_wb19/1"
TABLE_SCHEMA = "laned_wb19_table/1"
LEG_SCHEMA = "laned_wb19_leg/1"
IMAGE_ENV = "CASCADE_TRAIN_IMAGE_DIGEST"

RUN_REFUSALS = frozenset({
    "prod_host", "wrong_contract", "gpu_not_candidate", "wrong_image", "no_gpu",
    "gpu_mismatch", "bad_selection", "bad_steps", "fixture_not_allowed",
    "leg_failed", "non_finite",
})
COMBINE_REFUSALS = frozenset({
    "bad_run", "not_pod", "wrong_image", "wrong_contract", "mixed_block", "duplicate_sku",
})
PROD_MARKERS = (
    "etc/cascade-ops.env",
    "etc/systemd/system/cascade-ops-follow.service",
    "etc/systemd/system/cascade-intake.service",
    "root/cascade/validator_state.db",
)
SECRET_ENV = ("HIPPIUS_ACCESS_KEY", "HIPPIUS_SECRET_KEY", "HF_TOKEN", "WANDB_API_KEY",
              "LIUM_API_KEY", "BT_WALLET_PASSWORD", "R2_SECRET_ACCESS_KEY")
SECRET = "wb19-SECRET-sentinel-9f3c2a71"

SELECTION = {"3": [9, 2, 5], "0": [4, 1]}
SELECTION_CANON = {"0": [1, 4], "3": [2, 5, 9]}
STEPS = 40
TORCH_PIN = "2.4.1+cu124"

_SHA = re.compile(r"sha256:[0-9a-f]{64}")
HEX64 = re.compile(r"[0-9a-f]{64}")

GIB = 1024 ** 3
GPU_NAMES = {
    "H100": ("NVIDIA H100 80GB HBM3", 80 * GIB),
    "H200": ("NVIDIA H200", 141 * GIB),
    "L40S": ("NVIDIA L40S", 48 * GIB),
    "L40": ("NVIDIA L40", 48 * GIB),
    "RTX4090": ("NVIDIA GeForce RTX 4090", 24 * GIB),
    "A6000": ("NVIDIA RTX A6000", 48 * GIB),
}


# ── the tool ────────────────────────────────────────────────────────────────

def load_tool():
    """``scripts/laned_wb19_measure.py`` loaded by path (a fresh module)."""
    assert TOOL.is_file(), "scripts/laned_wb19_measure.py is missing (WB19 builds it)"
    name = "laned_wb19_measure_under_test"
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def host_is_prod() -> bool:
    """The same markers the tool's guard checks, on this host's real ``/``."""
    if str(os.environ.get("CASCADE_HOST_ROLE", "")).strip().lower() == "prod":
        return True
    return any(marker_present(Path("/") / m) for m in PROD_MARKERS)


def marker_present(path: Path) -> bool:
    """Stat only. An OSError (e.g. a CI runner may not stat /root) = absent."""
    try:
        return path.exists()
    except OSError:
        return False


def clean_env(tmp_path: Path, **extra: str) -> dict:
    """A minimal non-prod env: the pinned image digest, an empty PATH dir
    (no nvidia-smi), no secrets."""
    bindir = tmp_path / "emptybin"
    bindir.mkdir(exist_ok=True)
    env = {"PATH": str(bindir), IMAGE_ENV: pinned_image(), "HOME": str(tmp_path)}
    env.update(extra)
    return env


def fs_root(tmp_path: Path) -> Path:
    root = tmp_path / "fsroot"
    root.mkdir(exist_ok=True)
    return root


def run_tool(tool, argv, env, root, capsys):
    """``tool.main(argv, env=env, fs_root=root)`` → (rc, stdout, stderr)."""
    try:
        rc = tool.main([str(a) for a in argv], env=env, fs_root=root)
    except SystemExit as e:   # argparse usage errors
        rc = e.code if isinstance(e.code, int) else 2
    if capsys is None:
        return rc, "", ""
    out = capsys.readouterr()
    return rc, out.out, out.err


def refusal_code(stderr: str) -> str | None:
    """The code on the last non-empty stderr line ``REFUSED <code>: ...``."""
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    if not lines:
        return None
    m = re.match(r"REFUSED ([a-z_]+):", lines[-1])
    return m.group(1) if m else None


# ── independent derivations from the shipped chain.toml ─────────────────────

def chain_cfg(path: Path = CHAIN_TOML):
    from cascade.shared.config.loader import load_chain_config

    return load_chain_config(path)


def contract_block(cfg=None) -> int:
    cfg = cfg or chain_cfg()
    return int(cfg.training.v2_from_block)


def leg_contract(cfg, block: int, mode: str):
    """The lane B driver's derivation (``lane_b_driver.py``) for ``expert``;
    the unreduced throne contract for ``full``."""
    c = cfg.throne_contracts_at(int(block))[0]
    if mode == "full":
        return c
    return c.for_hours(cfg.lane_b.leg_hours, guard_factor=cfg.round.heat_guard_factor,
                       guard_floor_seconds=cfg.round.heat_guard_floor_seconds)


def leg_contract_digest(cfg, block: int, mode: str) -> str:
    from cascade.shared.manifest import contract_digest

    return contract_digest(leg_contract(cfg, block, mode))


def pinned_image(cfg=None) -> str:
    cfg = cfg or chain_cfg()
    m = _SHA.search(str(leg_contract(cfg, contract_block(cfg), "expert").train_image_digest))
    assert m, "the shipped [v2.training] train_image_digest carries no sha256 pin"
    return m.group(0)


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def body_sha256(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "body_sha256"}
    return hashlib.sha256(canonical(body)).hexdigest()


def rehash(record: dict) -> dict:
    record = dict(record)
    record["body_sha256"] = body_sha256(record)
    return record


def expected_leg_config_digest(*, mode, sku, contract_digest, image_digest, steps, selection):
    leg = {"schema": LEG_SCHEMA, "mode": mode, "sku": sku,
           "contract_digest": contract_digest, "image_digest": image_digest,
           "steps": int(steps), "selection": selection}
    return hashlib.sha256(canonical(leg)).hexdigest()


def split_layout(sku: str) -> tuple[str, int]:
    m = re.match(r"^\s*(\d+)\s*[xX×]\s*(\S+)\s*$", sku)
    if m and int(m.group(1)) >= 1:
        return m.group(2), int(m.group(1))
    return sku.strip(), 1


# ── fixtures ────────────────────────────────────────────────────────────────

def gpu_fixture(sku: str, *, count: int | None = None, name: str | None = None,
                driver: str = "570.86.15", cuda: str = "12.4") -> dict:
    base, n = split_layout(sku)
    nm, mem = GPU_NAMES[base]
    return {"gpus": [{"name": name or nm, "memory_total_bytes": mem}
                     for _ in range(n if count is None else count)],
            "driver": driver, "cuda": cuda}


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True))
    return path


def dry_run_argv(tmp_path: Path, sku: str, *, mode: str = "expert", steps: int | None = STEPS,
                 selection=SELECTION, fixture: dict | None = None, out: Path | None = None,
                 block: int | None = None, extra=()) -> list:
    fx = write_json(tmp_path / f"gpu-{sku}-{mode}.json", fixture or gpu_fixture(sku))
    argv = ["run", "--chain-toml", CHAIN_TOML, "--contract-block",
            block if block is not None else contract_block(), "--sku", sku, "--mode", mode,
            "--out", out or tmp_path / f"run-{sku}-{mode}.json", "--dry-run",
            "--gpu-fixture", fx]
    if steps is not None:
        argv += ["--steps", steps]
    if selection is not None and mode == "expert":
        argv += ["--selection-json", json.dumps(selection)]
    return argv + list(extra)


def good_result(steps: int, *, peak: int = 40 * GIB) -> dict:
    return {"ok": True, "steps": int(steps), "error": None, "deadline_hit": False,
            "final_loss": 0.4375, "torch": TORCH_PIN, "peak_memory_bytes": int(peak),
            "wall": {"total_seconds": 1000.0, "train_seconds": 900.0,
                     "overhead_seconds": {"boot": 40.0, "corpus": 30.0, "harvest": 20.0}}}


def make_plan(tool, tmp_path, capsys, sku, *, mode="expert", **kw) -> dict:
    """A dry-run plan record from the tool (in-process, clean fs root)."""
    out = tmp_path / f"plan-{sku}-{mode}.json"
    steps = kw.pop("steps", STEPS)
    argv = dry_run_argv(tmp_path, sku, mode=mode, out=out,
                        steps=None if mode == "full" else steps, **kw)
    rc, _, err = run_tool(tool, argv, clean_env(tmp_path), fs_root(tmp_path), capsys)
    assert rc == 0, f"dry-run for {sku}/{mode} failed (rc {rc}): {err}"
    assert out.is_file(), f"dry-run for {sku}/{mode} wrote no record"
    return json.loads(out.read_text())


def make_pod_record(tool, tmp_path, capsys, sku, *, mode="expert", peak=40 * GIB) -> dict:
    """A finished record as the pod path builds it: a plan whose source is
    ``pod`` (the real run's GPU probe) finished with a clean leg result."""
    plan = make_plan(tool, tmp_path, capsys, sku, mode=mode)
    plan["source"] = "pod"
    return tool.finish_record(plan, good_result(plan["steps"], peak=peak))


def is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
