"""Checkpoint ingest guard: a trained checkpoint is DATA, never code.

Every scorer of a checkpoint — the validator's verdict, the post-publish bench,
the audit replay — historically imported and executed two Python files
shipped INSIDE the checkpoint directory (``forecast_wrapper.py`` and
``model.py``) and built the model from the checkpoint's own ``config.json``.
That was sound while every checkpoint came off an operator pod. Under
miner-funded compute (DEC-CA-0036) the checkpoint is produced on a pod the
MINER controls, so an unguarded load is arbitrary code execution on every
validator and on the operator's king pod. This module closes that:

* the two Python files must be **byte-identical to this repo's own copies**
  (``toto2_model.py`` — whose bytes are already folded into the contract's
  ``base_arch_digest`` — and the wrapper template ``_FORECAST_WRAPPER_PY``);
  any other ``.py`` in the checkpoint is refused outright. Executing the
  checkpoint's copies after this check is exactly executing repo code;
* ``config.json``'s model config must equal the config the contract derives
  for the entry's size (``Toto2Config.from_contract``) — no shape games;
* ``weights.safetensors`` is checked by its HEADER (tensor names, dtypes,
  shapes must equal the pinned model's state dict) and by size (a byte
  budget from the pinned parameter count, PRISM's ``n_params × k`` shape)
  BEFORE anything is loaded — a header declaring absurd tensors is refused
  without allocating them.

Fail-closed: any deviation raises :class:`CheckpointTampered`. Callers map it
to their own failure class (the validator scores the entry as unloadable; the
trainer's funded leg settles it as ``tamper``).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CheckpointReport",
    "CheckpointTampered",
    "WEIGHTS_FILE",
    "expected_checkpoint_code",
    "safetensors_header",
    "verify_checkpoint",
    "verify_checkpoint_code",
]

WEIGHTS_FILE = "weights.safetensors"
CODE_FILES = ("model.py", "forecast_wrapper.py")
# The safetensors header is JSON metadata; 4M-param models need a few KB.
MAX_HEADER_BYTES = 4 * 1024 * 1024
# Byte budget multiplier over fp32 param bytes: weights are fp32 (×4/param);
# the ×3 covers dtype/alignment/metadata slack with room to spare, and stays
# far below anything that could be a payload.
BYTES_PER_PARAM = 4
BUDGET_FACTOR = 3
_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
                "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


class CheckpointTampered(RuntimeError):
    """The checkpoint deviates from what the contract could have produced."""


@dataclass(frozen=True)
class CheckpointReport:
    n_params: int
    weights_bytes: int
    n_tensors: int


def expected_checkpoint_code() -> dict[str, bytes]:
    """The exact bytes this repo writes into every checkpoint's code files."""
    from ..trainer import toto2_model, toto2_trainer

    return {
        "model.py": Path(toto2_model.__file__).read_bytes(),
        "forecast_wrapper.py": toto2_trainer._FORECAST_WRAPPER_PY.encode("utf-8"),
    }


def verify_checkpoint_code(checkpoint_dir: Path | str) -> None:
    """Refuse a checkpoint whose Python is not byte-identical to the repo's.

    Also refuses ANY other ``.py`` file (there is no legitimate one) and a
    missing wrapper — the scorer would otherwise fail later with a less
    honest error.
    """
    d = Path(checkpoint_dir)
    expected = expected_checkpoint_code()
    extra = sorted(p.name for p in d.rglob("*.py") if p.name not in expected)
    if extra:
        raise CheckpointTampered(f"unexpected python in checkpoint: {extra}")
    for name, want in expected.items():
        p = d / name
        if not p.is_file():
            raise CheckpointTampered(f"checkpoint is missing {name}")
        if p.read_bytes() != want:
            raise CheckpointTampered(
                f"{name} differs from this release's copy — a checkpoint's "
                "code is never executed unless it is byte-identical")


def safetensors_header(path: Path | str) -> dict:
    """The tensor header of a safetensors file, WITHOUT reading any tensor.

    Format: 8-byte little-endian header length, then that many bytes of
    JSON mapping tensor name → {dtype, shape, data_offsets} (plus an
    optional ``__metadata__`` entry).
    """
    p = Path(path)
    with open(p, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise CheckpointTampered(f"{p.name}: truncated safetensors header")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > MAX_HEADER_BYTES:
            raise CheckpointTampered(f"{p.name}: safetensors header size {n} out of bounds")
        try:
            header = json.loads(f.read(n).decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as e:
            raise CheckpointTampered(f"{p.name}: unparseable safetensors header") from e
    if not isinstance(header, dict):
        raise CheckpointTampered(f"{p.name}: safetensors header is not an object")
    header.pop("__metadata__", None)
    return header


def _expected_state_spec(contract) -> tuple[dict[str, tuple[str, tuple[int, ...]]], int]:
    """``{name: (dtype, shape)}`` of the pinned model for ``contract`` and its
    parameter count — built on CPU from the repo's own model code."""
    from ..trainer.toto2_model import Toto2Config, Toto2Model

    model = Toto2Model(Toto2Config.from_contract(contract))
    spec = {}
    n_params = 0
    for name, t in model.state_dict().items():
        dtype = str(t.dtype).replace("torch.", "")
        spec[name] = ({"float32": "F32", "float16": "F16", "bfloat16": "BF16",
                       "float64": "F64", "int64": "I64", "int32": "I32",
                       "bool": "BOOL"}.get(dtype, dtype.upper()), tuple(t.shape))
    n_params = sum(p.numel() for p in model.parameters())
    return spec, n_params


def verify_checkpoint(checkpoint_dir: Path | str, contract=None, *,
                      max_bytes: int | None = None) -> CheckpointReport:
    """Full ingest check; returns what was verified, raises on any deviation.

    ``contract`` is the entry's ``TrainingContractConfig`` (the validator
    knows it from the entry's size). With ``None`` only the code-identity
    check runs — enough to make executing the checkpoint safe, but not to
    bound its shape; every live scorer should pass the contract.
    """
    d = Path(checkpoint_dir)
    verify_checkpoint_code(d)
    weights = d / WEIGHTS_FILE
    if not weights.is_file():
        raise CheckpointTampered(f"checkpoint is missing {WEIGHTS_FILE}")
    header = safetensors_header(weights)
    size = weights.stat().st_size
    if contract is None:
        return CheckpointReport(n_params=0, weights_bytes=size, n_tensors=len(header))

    # config.json must be exactly what the contract derives — the validator
    # rebuilds the model from ITS config, never from the checkpoint's.
    from ..trainer.toto2_model import Toto2Config

    try:
        cfg_obj = json.loads((d / "config.json").read_text(encoding="utf-8"))
        shipped = dict(cfg_obj["toto2"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise CheckpointTampered(f"config.json unreadable or malformed: {e}") from e
    want = Toto2Config.from_contract(contract).to_dict()
    if shipped != want:
        diff = sorted(k for k in set(shipped) | set(want) if shipped.get(k) != want.get(k))
        raise CheckpointTampered(f"config.json model config differs from the "
                                 f"contract on {diff}")

    spec, n_params = _expected_state_spec(contract)
    budget = max_bytes if max_bytes is not None else n_params * BYTES_PER_PARAM * BUDGET_FACTOR
    if size > budget:
        raise CheckpointTampered(f"{WEIGHTS_FILE} is {size} bytes; budget for "
                                 f"{n_params} params is {budget}")
    got = {}
    for name, meta in header.items():
        if not isinstance(meta, dict):
            raise CheckpointTampered(f"tensor {name!r}: malformed header entry")
        got[name] = (str(meta.get("dtype")), tuple(int(x) for x in meta.get("shape", ())))
        offs = meta.get("data_offsets")
        n_elem = 1
        for x in got[name][1]:
            n_elem *= x
        expect_bytes = n_elem * _DTYPE_BYTES.get(got[name][0], 0)
        if (not isinstance(offs, list) or len(offs) != 2
                or int(offs[1]) - int(offs[0]) != expect_bytes):
            raise CheckpointTampered(f"tensor {name!r}: data_offsets do not match "
                                     "its declared shape/dtype")
    if got != spec:
        missing = sorted(set(spec) - set(got))
        extra = sorted(set(got) - set(spec))
        wrong = sorted(k for k in set(spec) & set(got) if spec[k] != got[k])
        raise CheckpointTampered(
            f"{WEIGHTS_FILE} tensors differ from the pinned model: "
            f"missing={missing[:5]} extra={extra[:5]} shape/dtype={wrong[:5]}")
    return CheckpointReport(n_params=n_params, weights_bytes=size, n_tensors=len(got))
