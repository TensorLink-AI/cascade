"""Shared spec, fixture builders and the T7 evidence verifier for W5d.

Copied from docs/plans/LANE_D_PLAN.md (rev6): the W5d row, §1.1 (entry,
context variants, no padding, 128-patch training rows, horizons ≤ 512), §1.6
(launch eval host, ≤ 50% of a settlement), §4 T7, §17.1 L4. README.md states
the module contract. Nothing here imports a W5d module at import time, and
nothing here imports torch at import time: the torch-only builders import it
inside their bodies, and only tests marked ``torch`` call them.

:func:`evidence_problems` is the T7 verifier. It never trusts the pod run: it
re-derives ``fits``, ``armed``, ``short_window.identical`` and ``passed`` from
the recorded numbers, so a hand-edited ``evidence/t7.json`` fails. Since RV3
it also refuses a row that did not exercise its variant (``input_len`` must
equal the variant, ``history_min`` must reach it) and a row whose memory is
not the variant's own CUDA peak (``peak_mem_source``).
"""

from __future__ import annotations

import importlib.util
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]

# §1.1: the 9-level scoring grid of lane A under [v2] quantile_head_after.
SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# The pyramid head's 21 levels (pyramid_model.QUANTILE_LEVELS), written out so
# the non-torch tests never import the model module.
HEAD_LEVELS = (0.01, *(i / 20 for i in range(1, 20)), 0.99)
PATCH = 32                     # pyramid_model patch_size (§1.1)
TRAINED_CONTEXT = 4096         # 128 patches: the training row length (§1.1, rev5)
HORIZON = 64                   # lane D scores h = 64 at launch (§1.1)
MAX_HORIZON = 2048             # 64 patches x 32: the band cap (owner 2026-11-27; was 512)
LAUNCH_CONTEXTS = (4096, 8192, 16384)
ALL_CONTEXTS = (4096, 8192, 16384, 32768, 65536, 131072)
DECLARED_128K = 131072
SEASONAL_PERIOD = 24

# T7 evidence (schema laned_t7/1).
T7_SCHEMA = "laned_t7/1"
WP = "W5d"
S8_HOST = "valeval-laned"              # 96 GB, rented only at S8 for 32k-128k
PINNED_TORCH = "2.4.1+cu124"           # the reproducibility contract
SETTLEMENT_BLOCKS = 900
BLOCK_SECONDS = 12.0
BUDGET_FRACTION = 0.5                  # §1.6: arm only variants that fit ≤ 50% of a settlement
MAX_BUDGET_WALL_S = SETTLEMENT_BLOCKS * BLOCK_SECONDS * BUDGET_FRACTION   # 5400 s
MIN_MODELS = 10                        # up to 8 entries + KING_D + house per settlement (§1.6)
TOP_KEYS = ("schema", "wp", "source", "torch", "tool", "measured_at", "rows",
            "short_window", "armed", "passed")
ROW_KEYS = ("host", "gpu", "variant", "input_len", "history_min", "history_max", "wall_s",
            "peak_mem_gb", "peak_mem_source", "budget", "fits", "n_models", "n_windows")
# RV3: a row's memory is either the device peak of THIS variant's run
# (torch.cuda.max_memory_allocated after reset_peak_memory_stats) or, on CPU,
# the process-lifetime ru_maxrss, which every later row inherits. Only the
# first is a budget input, so only it may appear in pod evidence.
POD_PEAK_MEM_SOURCES = ("cuda_max_allocated",)
CPU_PEAK_MEM_SOURCE = "process_peak_rss"
PEAK_MEM_SOURCES = (*POD_PEAK_MEM_SOURCES, CPU_PEAK_MEM_SOURCE)
SHORT_KEYS = ("base_context", "declared_context", "n_windows", "max_abs_diff", "identical")
SHA_RE = re.compile(r"[0-9a-f]{40}")


def lane_d_cfg(**over):
    """The real W2 parse of an armed ``[lane_d]`` table (launch contexts)."""
    from cascade.shared.config.lane_d import parse_lane_d

    raw = {"from_block": 2700 * 3600, "share": 0.25, "takes_from": "lane_a",
           "remainder": "burn", "contexts": list(LAUNCH_CONTEXTS), "horizon": HORIZON,
           "horizon_bands": []}  # owner 2026-11-27: the fixed-horizon rule
    return parse_lane_d({**raw, **over}, era_blocks_at=lambda _block: 3600)


def series(n_points: int, seed: int, level: float = 20.0) -> np.ndarray:
    """A positive hourly-looking float64 series of ``n_points`` points."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=np.float64)
    return level + 3.0 * np.sin(2 * np.pi * t / SEASONAL_PERIOD) + rng.normal(0, 0.5, n_points)


def sorted_quantiles(rng: np.random.Generator, horizon: int, centre: float) -> np.ndarray:
    """A monotone ``(horizon, 9)`` quantile forecast around ``centre``."""
    base = centre + rng.normal(0, 1.0, (horizon, 1))
    steps = np.cumsum(np.abs(rng.normal(0.3, 0.1, (horizon, len(SCORING_LEVELS)))), axis=1)
    return base + steps - steps[:, 4:5]


# ── torch-only builders (call only from tests marked ``torch``) ──────────────


def pyramid_contract():
    """The smallest pyramid size (``cascade-moe-v2-2m``), via the repo's own
    ``scripts/pyramid_config.py`` (loaded by path)."""
    from cascade.shared.config import parse_chain_config

    spec = importlib.util.spec_from_file_location(
        "w5d_pyramid_config", REPO / "scripts" / "pyramid_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return parse_chain_config(mod.local_config("2m")).training


def tiny_checkpoint(directory: Path, contract, *, seed: int, median_shift: float,
                    bf16: bool) -> Path:
    """A random-init 2m pyramid checkpoint written by the real trainer code.
    ``median_shift`` raises the head's median-channel bias (the house gets
    +3, as in W1.5); ``bf16`` casts floating tensors to BF16 (D14)."""
    import torch

    from cascade.trainer.pyramid_model import HEAD_CHANNELS, CascadeMoEConfig, CascadeMoEModel
    from cascade.trainer.pyramid_trainer import save_checkpoint

    config = CascadeMoEConfig.from_contract(contract)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = CascadeMoEModel(config)
    if median_shift:
        with torch.no_grad():
            model.head.bias.view(config.patch_size, HEAD_CHANNELS)[:, 0] += median_shift
    state = {}
    for name, tensor in model.state_dict().items():
        t = tensor.detach()
        if bf16 and t.is_floating_point():
            t = t.to(torch.bfloat16)
        state[name] = t.contiguous().clone()
    save_checkpoint(directory, state, config, contract)
    return directory


class ModelInputs:
    """Records the patch tensor every ``CascadeMoEModel`` forward receives
    (a global forward pre-hook: observation only, the forward is untouched)."""

    def __init__(self):
        self.shapes: list[tuple[int, ...]] = []
        self._handle = None

    def __enter__(self):
        import torch

        def hook(module, args):
            if type(module).__name__ == "CascadeMoEModel" and args:
                self.shapes.append(tuple(int(d) for d in args[0].shape))

        self._handle = torch.nn.modules.module.register_module_forward_pre_hook(hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        return False


# ── T7 evidence verifier ─────────────────────────────────────────────────────


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                          check=False)


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _tool_problems(tool: object, check_git: bool) -> list[str]:
    if not isinstance(tool, dict):
        return [f"tool: want an object, got {tool!r}"]
    out = []
    sha = tool.get("git_sha")
    if not isinstance(sha, str) or not SHA_RE.fullmatch(sha):
        out.append(f"tool.git_sha: want 40 lowercase hex, got {sha!r}")
    elif check_git:
        if _git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
            out.append(f"tool.git_sha {sha} is not a commit in this repository")
        elif _git("cat-file", "-e", f"{sha}:cascade/eval/lane_d_t7.py").returncode != 0:
            out.append(f"tool.git_sha {sha} has no cascade/eval/lane_d_t7.py")
    if tool.get("git_dirty") is not False:
        out.append(f"tool.git_dirty: the run must be from a clean checkout, "
                   f"got {tool.get('git_dirty')!r}")
    return out


def fits(row: dict) -> bool:
    """§1.6 / T7: a variant fits when its settlement wall time and its peak
    memory are both inside the budget."""
    b = row["budget"]
    return float(row["wall_s"]) <= float(b["wall_s"]) and float(row["peak_mem_gb"]) <= float(b["mem_gb"])


def _row_problems(i: int, row: object) -> list[str]:
    if not isinstance(row, dict):
        return [f"rows[{i}]: want an object"]
    missing = [k for k in ROW_KEYS if k not in row]
    if missing:
        return [f"rows[{i}]: missing {missing}"]
    out = []
    v = row["variant"]
    if type(v) is not int or v not in ALL_CONTEXTS:
        out.append(f"rows[{i}].variant: want one of {list(ALL_CONTEXTS)}, got {v!r}")
    for k in ("host", "gpu"):
        if not isinstance(row[k], str) or not row[k].strip():
            out.append(f"rows[{i}].{k}: want a non-empty string, got {row[k]!r}")
    for k in ("wall_s", "peak_mem_gb"):
        if not _num(row[k]) or float(row[k]) <= 0:
            out.append(f"rows[{i}].{k}: want a positive finite number, got {row[k]!r}")
    b = row["budget"]
    if (not isinstance(b, dict) or not _num(b.get("wall_s")) or not _num(b.get("mem_gb"))
            or not 0 < float(b["wall_s"]) <= MAX_BUDGET_WALL_S or float(b["mem_gb"]) <= 0):
        out.append(f"rows[{i}].budget: want wall_s in (0, {MAX_BUDGET_WALL_S}] and mem_gb > 0, "
                   f"got {b!r}")
    if type(row["n_models"]) is not int or row["n_models"] < MIN_MODELS:
        out.append(f"rows[{i}].n_models: want an int ≥ {MIN_MODELS}, got {row['n_models']!r}")
    if type(row["n_windows"]) is not int or row["n_windows"] < 1:
        out.append(f"rows[{i}].n_windows: want an int ≥ 1, got {row['n_windows']!r}")
    # RV3: the row must have exercised its variant. The no-pad scorer feeds a
    # window min(n, context) points, so a variant row is only a measurement of
    # that variant when every window had ≥ variant points of history and the
    # model input that ran was exactly ``variant`` points.
    if type(row["input_len"]) is not int or row["input_len"] != v:
        out.append(f"rows[{i}].input_len: the model input that ran must be the variant "
                   f"({v!r} points), got {row['input_len']!r}")
    hmin, hmax = row["history_min"], row["history_max"]
    if type(hmin) is not int or type(v) is not int or hmin < v:
        out.append(f"rows[{i}].history_min: every window needs ≥ {v!r} points of history, "
                   f"got {hmin!r}")
    elif type(hmax) is not int or hmax < hmin:
        out.append(f"rows[{i}].history_max: want an int ≥ history_min {hmin}, got {hmax!r}")
    if row["peak_mem_source"] not in POD_PEAK_MEM_SOURCES:
        out.append(f"rows[{i}].peak_mem_source: want one of {list(POD_PEAK_MEM_SOURCES)} "
                   f"(the variant's own device peak), got {row['peak_mem_source']!r}")
    if out:
        return out
    if row["fits"] is not fits(row):
        out.append(f"rows[{i}].fits: recorded {row['fits']!r}, the numbers give {fits(row)}")
    if v <= 16384 and row["host"] == S8_HOST:
        out.append(f"rows[{i}].host: variant {v} must be measured on the launch eval host, "
                   f"not {S8_HOST}")
    if v > 16384 and row["host"] != S8_HOST:
        out.append(f"rows[{i}].host: variant {v} is an S8 variant, measured on {S8_HOST}, "
                   f"got {row['host']!r}")
    return out


def _short_problems(sw: object) -> list[str]:
    if not isinstance(sw, dict) or any(k not in sw for k in SHORT_KEYS):
        return [f"short_window: want an object with {list(SHORT_KEYS)}, got {sw!r}"]
    out = []
    if sw["base_context"] != TRAINED_CONTEXT or sw["declared_context"] != DECLARED_128K:
        out.append(f"short_window: want base_context {TRAINED_CONTEXT} vs declared_context "
                   f"{DECLARED_128K}, got {sw['base_context']!r} vs {sw['declared_context']!r}")
    if type(sw["n_windows"]) is not int or sw["n_windows"] < 1:
        out.append(f"short_window.n_windows: want an int ≥ 1, got {sw['n_windows']!r}")
    if not _num(sw["max_abs_diff"]) or float(sw["max_abs_diff"]) < 0:
        out.append(f"short_window.max_abs_diff: want a finite number ≥ 0, got {sw['max_abs_diff']!r}")
    elif sw["identical"] is not (float(sw["max_abs_diff"]) == 0.0):
        out.append(f"short_window.identical: recorded {sw['identical']!r}, "
                   f"max_abs_diff {sw['max_abs_diff']!r} gives {float(sw['max_abs_diff']) == 0.0}")
    return out


def expected_passed(data: dict) -> bool:
    """T7's verdict from the numbers: identical short-window scores, every
    launch variant measured, and the trained 4096 context fits."""
    variants = {r["variant"] for r in data["rows"]}
    return (data["short_window"]["identical"] is True
            and set(LAUNCH_CONTEXTS) <= variants
            and TRAINED_CONTEXT in data["armed"])


def evidence_problems(data: object, *, check_git: bool = True) -> list[str]:
    """Every problem with a T7 evidence object; ``[]`` means it verifies."""
    if not isinstance(data, dict):
        return ["evidence: want a JSON object"]
    missing = [k for k in TOP_KEYS if k not in data]
    if missing:
        return [f"evidence: missing {missing}"]
    out = []
    if data["schema"] != T7_SCHEMA:
        out.append(f"schema: want {T7_SCHEMA!r}, got {data['schema']!r}")
    if data["wp"] != WP:
        out.append(f"wp: want {WP!r}, got {data['wp']!r}")
    if data["source"] != "pod":
        out.append(f"source: want 'pod' (a measured run), got {data['source']!r}")
    if data["torch"] != PINNED_TORCH:
        out.append(f"torch: want the pinned {PINNED_TORCH!r}, got {data['torch']!r}")
    out += _tool_problems(data["tool"], check_git)
    try:
        datetime.fromisoformat(str(data["measured_at"]).replace("Z", "+00:00"))
    except ValueError:
        out.append(f"measured_at: want ISO 8601, got {data['measured_at']!r}")
    rows = data["rows"]
    if not isinstance(rows, list) or not rows:
        return out + ["rows: want a non-empty list"]
    row_probs = [p for i, r in enumerate(rows) for p in _row_problems(i, r)]
    out += row_probs
    short = _short_problems(data["short_window"])
    out += short
    if row_probs or short:
        return out
    variants = [r["variant"] for r in rows]
    if len(set(variants)) != len(variants):
        out.append(f"rows: duplicate variants {variants}")
    absent = [v for v in LAUNCH_CONTEXTS if v not in variants]
    if absent:
        out.append(f"rows: launch variants not measured: {absent}")
    armed = sorted(r["variant"] for r in rows if fits(r))
    if data["armed"] != armed:
        out.append(f"armed: recorded {data['armed']!r}, the fitting variants are {armed}")
        return out
    want = expected_passed(data)
    if data["passed"] is not want:
        out.append(f"passed: recorded {data['passed']!r}, the numbers give {want}")
    return out
