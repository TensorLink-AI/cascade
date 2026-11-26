"""Reference fine-tune script of the lane D miner kit (LANE_D_PLAN §3.0 W16,
§10.2 "Miner kit", §4 T3).

Start from a published checkpoint (the latest public KING_A, or a lane D
king / house), fine-tune it on your own **public** data on your GPU, and
write a lane D entry in the W4 format (exactly ``config.json``,
``forecast_wrapper.py``, ``model.py``, ``weights.safetensors``; BF16
weights), ready for ``cascade lane-d verify | submit``::

    python -P -m cascade.miner.lane_d_finetune --base KING_DIR \\
        --data public.jsonl --out ENTRY_DIR --steps 200 --lr 3e-4

``--data`` is JSONL, one ``{"target": [float, ...]}`` object per line.

Every input is checked before torch is imported, so ``--help``,
``--dry-run`` and every typed refusal work on a box without torch. Nothing
in ``--base`` is ever imported or executed: its code files are compared as
bytes with the repo's ``cascade_pyramid`` sources, its safetensors header is
parsed by hand, and the model is built from this repo's own
``cascade.trainer.pyramid_model``. A refusal prints one stderr line
``lane_d_finetune_refused <code>: <detail>`` and exits 3, leaving ``--out``
as it was. A run writes the entry into a temporary sibling directory and
renames it into place, and prints a JSON summary ``{"out", "steps",
"final_loss", "digest"}`` as the last stdout line (``digest`` is the entry
digest W10 submits).

The objective is the trainer's marginal quantile (pinball) loss under the
same causal arcsinh standardisation, plus the MoE router's own auxiliary
loss, with AdamW; it is a reference starting point, not a tuned recipe.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

REFUSAL_CODES = (
    "base_files", "base_code", "base_config", "base_header",
    "data_missing", "data_bad", "data_empty",
    "out_not_empty", "torch_unavailable",
)
REFUSED_PREFIX = "lane_d_finetune_refused"
REFUSAL_EXIT = 3

BASE_FILES = ("config.json", "forecast_wrapper.py", "model.py", "weights.safetensors")
WEIGHTS_FILE = "weights.safetensors"
CONFIG_FILE = "config.json"
LANE_D_ARCH = "cascade_pyramid"
OUT_DTYPE = "BF16"
# The base files' code must be the repo's lane D architecture, byte for byte.
_TRAINER_DIR = Path(__file__).resolve().parents[1] / "trainer"
CODE_SOURCES = {"model.py": _TRAINER_DIR / "pyramid_model.py",
                "forecast_wrapper.py": _TRAINER_DIR / "pyramid_wrapper.py"}
# Upper bound on the safetensors JSON header (the W4 ingest's bound).
MAX_HEADER_BYTES = 4 * 1024 * 1024
# safetensors dtype -> bytes per element (the format's own table).
_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2,
                "I8": 1, "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1,
                "F8_E4M3": 1, "F8_E5M2": 1}
# Base tensors in these dtypes are written as BF16; any other dtype is kept.
_TO_BF16 = ("F32", "BF16")
# A training window needs a context patch and a target patch.
MIN_PATCHES = 2
DEVICES = ("auto", "cpu", "cuda")


class FinetuneRefused(Exception):
    """A typed refusal; ``code`` is one of :data:`REFUSAL_CODES`."""

    def __init__(self, code: str, detail: str = ""):
        if code not in REFUSAL_CODES:
            raise ValueError(f"FinetuneRefused: unknown code {code!r}")
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ── arguments ────────────────────────────────────────────────────────────────


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from e
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} must be >= 1")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from e
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be finite and > 0")
    return value


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -P -m cascade.miner.lane_d_finetune",
        description="Fine-tune a published cascade_pyramid checkpoint on your own public "
                    "data and write a lane D entry (four files, BF16 weights).")
    p.add_argument("--base", type=Path, required=True,
                   help="Base checkpoint directory (KING_A, or a lane D king / house).")
    p.add_argument("--data", type=Path, required=True,
                   help='Public-data JSONL: one {"target": [float, ...]} object per line.')
    p.add_argument("--out", type=Path, required=True,
                   help="Output entry directory (must not exist, or be empty).")
    p.add_argument("--steps", type=_positive_int, default=200, help="Optimizer steps.")
    p.add_argument("--lr", type=_positive_float, default=3e-4, help="AdamW learning rate.")
    p.add_argument("--seed", type=int, default=0, help="Seed for sampling and init of RNGs.")
    p.add_argument("--batch-size", type=_positive_int, default=8,
                   help="Series windows per step.")
    p.add_argument("--device", choices=DEVICES, default="auto",
                   help="auto = cuda when available, else cpu.")
    p.add_argument("--context-length", type=_positive_int, default=None,
                   help="Window length in points (default: the base config's).")
    p.add_argument("--dry-run", action="store_true",
                   help="Check every input, print the plan as JSON, write nothing.")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    return _parser().parse_args(argv)


# ── input checks (no torch) ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Inputs:
    config: dict
    header: dict[str, tuple[str, tuple[int, ...]]]
    series: list[list[float]]


def _check_base_files(base: Path) -> None:
    if not base.is_dir():
        raise FinetuneRefused("base_files", f"{base} is not a directory")
    names = sorted(p.name for p in base.iterdir())
    if names != sorted(BASE_FILES):
        extra = sorted(set(names) - set(BASE_FILES))
        missing = sorted(set(BASE_FILES) - set(names))
        raise FinetuneRefused("base_files", f"{base}: extra {extra}, missing {missing}; "
                                            f"a base is exactly {list(BASE_FILES)}")
    for name in BASE_FILES:
        p = base / name
        if p.is_symlink() or not p.is_file():
            raise FinetuneRefused("base_files", f"{p} is not a regular file")


def _check_base_code(base: Path) -> None:
    for name, source in CODE_SOURCES.items():
        if (base / name).read_bytes() != source.read_bytes():
            raise FinetuneRefused("base_code", f"{base / name} differs from the repo's "
                                               f"{source.name} ({LANE_D_ARCH} only)")


def _positive_int_value(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 1


def _check_base_config(base: Path) -> dict:
    path = base / CONFIG_FILE
    try:
        config = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise FinetuneRefused("base_config", f"{path}: not JSON ({e})") from e
    if not isinstance(config, dict):
        raise FinetuneRefused("base_config", f"{path}: not a JSON object")
    moe = config.get("moe")
    if not isinstance(moe, dict):
        raise FinetuneRefused("base_config", f"{path}: no \"moe\" object")
    if config.get("base_arch") != LANE_D_ARCH:
        raise FinetuneRefused("base_config", f"{path}: base_arch {config.get('base_arch')!r} "
                                             f"is not {LANE_D_ARCH!r}")
    for key in ("patch_size", "context_length"):
        if not _positive_int_value(moe.get(key)):
            raise FinetuneRefused("base_config", f"{path}: moe.{key} {moe.get(key)!r} is not "
                                                 "a positive integer")
    return config


def read_safetensors_header(path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """``{name: (dtype, shape)}`` of a safetensors file, parsed by hand
    (u64 LE length, JSON header): every tensor's dtype known, its offsets in
    bounds, sized by dtype x shape, contiguous and covering the data."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise FinetuneRefused("base_header", f"{path}: truncated length prefix")
        (n,) = struct.unpack("<Q", raw)
        if n == 0 or n > MAX_HEADER_BYTES or 8 + n > size:
            raise FinetuneRefused("base_header", f"{path}: header length {n} out of bounds")
        blob = f.read(n)
    try:
        header = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise FinetuneRefused("base_header", f"{path}: unparseable header") from e
    if not isinstance(header, dict):
        raise FinetuneRefused("base_header", f"{path}: header is not an object")
    header.pop("__metadata__", None)
    if not header:
        raise FinetuneRefused("base_header", f"{path}: no tensors")
    data_len = size - 8 - n
    spans, out = [], {}
    for name, entry in header.items():
        label = str(name)[:64]
        if not isinstance(entry, dict):
            raise FinetuneRefused("base_header", f"tensor {label!r}: malformed entry")
        dtype, shape, offs = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
            raise FinetuneRefused("base_header", f"tensor {label!r}: unknown dtype {dtype!r}")
        if not isinstance(shape, list) or not all(
                isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in shape):
            raise FinetuneRefused("base_header", f"tensor {label!r}: malformed shape")
        if (not isinstance(offs, list) or len(offs) != 2
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in offs)
                or not 0 <= offs[0] <= offs[1] <= data_len):
            raise FinetuneRefused("base_header", f"tensor {label!r}: malformed data_offsets")
        numel = math.prod(shape)
        if offs[1] - offs[0] != numel * _DTYPE_BYTES[dtype]:
            raise FinetuneRefused("base_header", f"tensor {label!r}: data_offsets disagree "
                                                 f"with {dtype} x {shape}")
        spans.append((offs[0], offs[1]))
        out[str(name)] = (dtype, tuple(shape))
    pos = 0
    for begin, end in sorted(spans):
        if begin != pos:
            raise FinetuneRefused("base_header", f"{path}: tensor data overlaps or leaves gaps")
        pos = end
    if pos != data_len:
        raise FinetuneRefused("base_header", f"{path}: tensor data does not cover the file")
    return out


def _refuse_constant(name: str):
    raise ValueError(f"non-finite constant {name}")


def _finite_number(x) -> bool:
    return (isinstance(x, int | float) and not isinstance(x, bool)
            and math.isfinite(float(x)))


def read_series(path: Path, *, min_points: int) -> list[list[float]]:
    """The ``target`` lists of a public-data JSONL file (blank lines skipped)."""
    if not path.is_file():
        raise FinetuneRefused("data_missing", f"{path} does not exist")
    series = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line, parse_constant=_refuse_constant)
            except ValueError as e:
                raise FinetuneRefused("data_bad", f"{path} line {lineno}: not JSON ({e})") from e
            if not isinstance(obj, dict) or not isinstance(obj.get("target"), list):
                raise FinetuneRefused("data_bad", f"{path} line {lineno}: not an object with "
                                                  "a \"target\" list")
            target = obj["target"]
            if not all(_finite_number(x) for x in target):
                raise FinetuneRefused("data_bad", f"{path} line {lineno}: \"target\" holds a "
                                                  "non-numeric or non-finite value")
            if len(target) < min_points:
                raise FinetuneRefused("data_bad", f"{path} line {lineno}: {len(target)} points; "
                                                  f"a series needs at least {min_points} "
                                                  f"({MIN_PATCHES} patches)")
            series.append([float(x) for x in target])
    if not series:
        raise FinetuneRefused("data_empty", f"{path} holds no series")
    return series


def _check_out(out: Path, base: Path) -> None:
    if out.exists() or out.is_symlink():
        if out.resolve() == base.resolve():
            raise FinetuneRefused("out_not_empty", f"{out} is the base directory")
        if not out.is_dir() or out.is_symlink() or any(out.iterdir()):
            raise FinetuneRefused("out_not_empty", f"{out} exists and is not an empty directory")


def check_inputs(cfg) -> Inputs:
    """Every input check, in :data:`REFUSAL_CODES` order (torch excluded)."""
    base = Path(cfg.base)
    _check_base_files(base)
    _check_base_code(base)
    config = _check_base_config(base)
    header = read_safetensors_header(base / WEIGHTS_FILE)
    series = read_series(Path(cfg.data), min_points=MIN_PATCHES * int(config["moe"]["patch_size"]))
    _check_out(Path(cfg.out), base)
    return Inputs(config=config, header=header, series=series)


def plan(cfg, inputs: Inputs) -> dict:
    return {"base": str(Path(cfg.base).resolve()), "data": str(Path(cfg.data).resolve()),
            "out": str(Path(cfg.out).resolve()), "base_arch": inputs.config["base_arch"],
            "n_tensors": len(inputs.header), "n_series": len(inputs.series),
            "steps": cfg.steps, "lr": cfg.lr, "seed": cfg.seed, "batch_size": cfg.batch_size,
            "device": cfg.device, "context_length": cfg.context_length,
            "out_dtype": OUT_DTYPE, "files": sorted(BASE_FILES)}


# ── training (torch from here on) ────────────────────────────────────────────


def _import_torch():
    try:
        import torch
    except Exception as e:  # noqa: BLE001 - any import failure means "no usable torch"
        raise FinetuneRefused("torch_unavailable", f"cannot import torch ({e})") from e
    return torch


def window_points(config: dict, context_length: int | None) -> int:
    """The training window: whole patches, at most the base's context."""
    moe = config["moe"]
    ps = int(moe["patch_size"])
    ctx = min(int(moe["context_length"]), int(context_length or moe["context_length"]))
    return ps * max(MIN_PATCHES, ctx // ps)


def sample_batch(series, rng, batch_size: int, length: int, patch_size: int):
    """``(values, mask)`` numpy arrays ``(B, 1, length)`` / ``(B, 1, P)``: a
    random window per item; a series shorter than the window keeps its last
    whole patches and is left-padded with masked (unobserved) patches."""
    import numpy as np

    p = length // patch_size
    values = np.zeros((batch_size, 1, length), dtype=np.float64)
    mask = np.zeros((batch_size, 1, p), dtype=bool)
    for b in range(batch_size):
        s = series[int(rng.integers(len(series)))]
        if len(s) >= length:
            start = int(rng.integers(len(s) - length + 1))
            values[b, 0] = s[start:start + length]
        else:
            keep = (len(s) // patch_size) * patch_size
            values[b, 0, length - keep:] = s[len(s) - keep:]
            mask[b, 0, :p - keep // patch_size] = True
    return values, mask


def _masked_pinball(torch, model, values, mask):
    """The trainer's marginal pinball loss over observed next-patch targets."""
    from ..trainer.pyramid_model import QUANTILE_LEVELS, quantiles
    from ..trainer.pyramid_trainer import prediction

    b, c, length = values.shape
    ps = model.cfg.patch_size
    p = length // ps
    params, loc, scale = prediction(model, values, mask)
    truth = ((values.reshape(b, c, p, ps)[:, :, 1:].double() - loc.double())
             / scale.double()).float()
    q = quantiles(params)
    residual = torch.asinh(truth).clamp(-64, 64)[..., None] - torch.asinh(q)
    levels = q.new_tensor(QUANTILE_LEVELS)
    pin = torch.maximum(levels * residual, (levels - 1) * residual).mean(-1)
    keep = (~mask[:, :, 1:])[..., None].expand(b, c, p - 1, ps).to(pin.dtype)
    return 90 * (pin * keep).sum() / keep.sum().clamp_min(1.0)


def _write_entry(torch, base: Path, staging: Path, state: dict, header: dict) -> None:
    from safetensors.torch import save_file

    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64,
              "F16": torch.float16, "I64": torch.int64, "I32": torch.int32,
              "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
    out = {}
    for name, (dtype, _) in header.items():
        want = OUT_DTYPE if dtype in _TO_BF16 else dtype
        out[name] = state[name].detach().to("cpu", dtypes[want]).contiguous()
    staging.mkdir(parents=True)
    for name in BASE_FILES:
        if name != WEIGHTS_FILE:
            shutil.copyfile(base / name, staging / name)
    save_file(out, str(staging / WEIGHTS_FILE))


def run(cfg, inputs: Inputs) -> dict:
    torch = _import_torch()
    import numpy as np
    from safetensors.torch import load_file

    from ..lane_d.intake import checkpoint_manifest, manifest_digest
    from ..trainer.pyramid_model import CascadeMoEConfig, CascadeMoEModel

    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed) % (2**63), 16]))

    base, out = Path(cfg.base), Path(cfg.out)
    model_cfg = CascadeMoEConfig.from_dict(inputs.config["moe"])
    model = CascadeMoEModel(model_cfg)
    model.load_state_dict(load_file(str(base / WEIGHTS_FILE)), strict=True)
    model = model.to(device=device, dtype=torch.float32)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    length = window_points(inputs.config, cfg.context_length)
    final_loss = float("nan")
    for step in range(cfg.steps):
        values, mask = sample_batch(inputs.series, rng, cfg.batch_size, length,
                                    model_cfg.patch_size)
        values_t = torch.from_numpy(values).to(device)
        mask_t = torch.from_numpy(mask).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = _masked_pinball(torch, model, values_t, mask_t)
        final_loss = float(loss.detach())
        if model.aux_loss is not None:
            loss = loss + model.aux_loss
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_routing_bias()

    staging = out.parent / f".{out.name}.lane_d_finetune-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        _write_entry(torch, base, staging, model.state_dict(), inputs.header)
        if out.exists():
            out.rmdir()   # checked empty above
        os.replace(staging, out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    digest = manifest_digest(checkpoint_manifest(out))
    return {"out": str(out.resolve()), "steps": cfg.steps, "final_loss": final_loss,
            "digest": digest}


def main(argv=None) -> int:
    cfg = parse_args(argv)
    try:
        inputs = check_inputs(cfg)
        if cfg.dry_run:
            print(json.dumps(plan(cfg, inputs), sort_keys=True))
            return 0
        summary = run(cfg, inputs)
    except FinetuneRefused as e:
        print(f"{REFUSED_PREFIX} {e.code}: {e.detail}", file=sys.stderr)
        return REFUSAL_EXIT
    print(json.dumps(summary, sort_keys=True))
    return 0


__all__ = ["REFUSAL_CODES", "FinetuneRefused", "check_inputs", "main", "parse_args",
           "read_safetensors_header", "read_series"]


if __name__ == "__main__":
    sys.exit(main())
