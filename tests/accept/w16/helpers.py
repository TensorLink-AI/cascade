"""Shared spec and fixture builders for the W16 acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6): the W16 row ("Reference
fine-tune script (miner kit)", ci = CPU tiny run), §10.2 "Miner kit" (the
latest public KING_A as a start point, a reference fine-tune script that runs
on the miner's GPU), §4 T3 (an honest fine-tune of the house on public data)
and the W4 checkpoint format (exactly four files, BF16 weights). README.md
states the module contract.

Nothing here imports the W16 module (a PATHS module) at import time: on main
every test fails at its own first import or at its subprocess's exit code.
"""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODULE = "cascade.miner.lane_d_finetune"

# W4 / §1.1: a lane D checkpoint is exactly these files, nothing else.
ALLOWED = ("config.json", "forecast_wrapper.py", "model.py", "weights.safetensors")

# README "Refusals": every typed refusal code, in check order.
REFUSAL_CODES = (
    "base_files", "base_code", "base_config", "base_header",
    "data_missing", "data_bad", "data_empty",
    "out_not_empty", "torch_unavailable",
)
REFUSED_PREFIX = "lane_d_finetune_refused"
REFUSAL_EXIT = 3

# A small torch-free fixture base: the repo's real pyramid code bytes, a
# config.json shaped like save_checkpoint's, and a hand-written safetensors.
FIXTURE_MOE = {"context_length": 4096, "d_model": 16, "n_heads": 2, "n_layers": 1,
               "patch_size": 32}
FIXTURE_TENSORS = {
    "blocks.0.attn.weight": (16, 16),
    "embed.weight": (16, 4),
    "head.bias": (8,),
    "head.weight": (8, 16),
}

# Blocks torch in a subprocess: the package raises on import.
_TORCH_BLOCK = "raise ImportError('W16 test: torch is blocked on this run')\n"


def repo_code() -> dict[str, bytes]:
    trainer = REPO / "cascade" / "trainer"
    return {"model.py": (trainer / "pyramid_model.py").read_bytes(),
            "forecast_wrapper.py": (trainer / "pyramid_wrapper.py").read_bytes()}


def _numel(shape) -> int:
    n = 1
    for x in shape:
        n *= int(x)
    return n


def tensor_bytes(dtype: str, shape) -> bytes:
    n = _numel(shape)
    values = [((i % 13) - 6) / 8.0 for i in range(n)]
    if dtype == "BF16":
        return b"".join(struct.pack("<f", v)[2:4] for v in values)
    if dtype == "F32":
        return struct.pack(f"<{n}f", *values)
    raise ValueError(dtype)


def safetensors_bytes(tensors: dict[str, tuple[str, tuple, bytes]], *,
                      offsets: dict[str, list[int]] | None = None) -> bytes:
    header, data = {}, b""
    for name in sorted(tensors):
        dtype, shape, raw = tensors[name]
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    for name, offs in (offsets or {}).items():
        header[name]["data_offsets"] = list(offs)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    return struct.pack("<Q", len(blob)) + blob + data


def fixture_weights(dtype: str = "F32", **kw) -> bytes:
    return safetensors_bytes(
        {name: (dtype, shape, tensor_bytes(dtype, shape))
         for name, shape in FIXTURE_TENSORS.items()}, **kw)


def config_json(*, base_arch: str = "cascade_pyramid", moe=None, drop_moe: bool = False) -> bytes:
    body = {"arch": "cascade-moe-v2-2m", "base_arch": base_arch,
            "input_transform": "arcsinh_causal",
            "moe": dict(FIXTURE_MOE if moe is None else moe),
            "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    if drop_moe:
        body.pop("moe")
    return json.dumps(body, sort_keys=True).encode("utf-8")


def write_base(directory: Path, *, dtype: str = "F32") -> Path:
    """A well-formed torch-free base checkpoint (KING_A is F32, a lane D
    king or house is BF16)."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in repo_code().items():
        (directory / name).write_bytes(data)
    (directory / "config.json").write_bytes(config_json())
    (directory / "weights.safetensors").write_bytes(fixture_weights(dtype))
    return directory


def write_data(path: Path, series=None) -> Path:
    """Public-data JSONL: one ``{"target": [float, ...]}`` object per line."""
    if series is None:
        series = [[((i * (k + 3)) % 17) / 4.0 for i in range(640)] for k in range(4)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"target": s}) + "\n" for s in series))
    return path


def snapshot(directory: Path) -> dict[str, bytes]:
    return {str(p.relative_to(directory)): p.read_bytes()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def _sandbox(tmp_path: Path) -> Path:
    d = tmp_path / "w16_sandbox"
    (d / "torch").mkdir(parents=True, exist_ok=True)
    (d / "torch" / "__init__.py").write_text(_TORCH_BLOCK)
    return d


def run_script(tmp_path: Path, *args: str, block_torch: bool = True,
               extra_env: dict[str, str] | None = None,
               timeout: int = 120) -> subprocess.CompletedProcess:
    """Runs ``python -P -m cascade.miner.lane_d_finetune ARGS`` with cwd =
    ``tmp_path``. With ``block_torch``, ``import torch`` raises."""
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("HIPPIUS_") or k.startswith("AWS_") or k == "HF_TOKEN"
                   or k == "WANDB_API_KEY")}
    paths = [str(_sandbox(tmp_path))] if block_torch else []
    env["PYTHONPATH"] = os.pathsep.join([*paths, str(REPO)])
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-P", "-m", MODULE, *args], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=timeout,
                          check=False)


def refusal_codes(stderr: str) -> list[str]:
    """The codes of every ``lane_d_finetune_refused <code>`` line on stderr."""
    out = []
    for line in stderr.splitlines():
        if line.startswith(REFUSED_PREFIX + " "):
            out.append(line[len(REFUSED_PREFIX) + 1:].split(":", 1)[0].split()[0])
    return out


def last_json_line(stdout: str) -> dict:
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    assert lines, "nothing on stdout"
    obj = json.loads(lines[-1])
    assert isinstance(obj, dict), lines[-1]
    return obj


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


# ── torch helpers (the ``torch``-marked tests only; CI, never the box) ───────


def pyramid_contract():
    """The smallest pyramid size (``cascade-moe-v2-2m``) via the repo's own
    ``scripts/pyramid_config.py`` (loaded by path), as W1.5 and W4 do."""
    from cascade.shared.config import parse_chain_config

    spec = importlib.util.spec_from_file_location(
        "w16_pyramid_config", REPO / "scripts" / "pyramid_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return parse_chain_config(mod.local_config("2m")).training


def tiny_base(directory: Path, contract, *, seed: int, bf16: bool) -> Path:
    """A random-init 2m pyramid checkpoint written by the real trainer code
    (F32 = a lane A KING_A, BF16 = a lane D king or house)."""
    import torch

    from cascade.trainer.pyramid_model import CascadeMoEConfig, CascadeMoEModel
    from cascade.trainer.pyramid_trainer import save_checkpoint

    config = CascadeMoEConfig.from_contract(contract)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = CascadeMoEModel(config)
    state = {}
    for name, tensor in model.state_dict().items():
        t = tensor.detach().clone()
        if bf16 and t.is_floating_point():
            t = t.to(torch.bfloat16)
        state[name] = t.contiguous()
    save_checkpoint(directory, state, config, contract)
    return directory
