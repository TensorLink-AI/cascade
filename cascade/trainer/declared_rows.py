"""Miner-declared expert rows for a lane B leg (plan §12b.2 LB2, WB14).

A lane B miner may declare WHICH expert rows its leg trains:
``declared_rows = {ordinal: [expert index, ...]}``. The keys are MoE ORDINALS
``0 .. len(experts_per_layer) - 1`` (one per ``moe_layers()``, the worker's
``--selection-json`` keys), not block indices: ordinal ``i`` has
``experts_per_layer[i]`` experts.

Per layer with at least one row, the count must sit inside the clamp
``ceil(min_frac * E) .. floor(max_frac * E)`` (``[lane_b] min_experts_frac`` /
``max_experts_frac``). A layer may carry no rows (ESFT likewise may take none
on a layer); at least one row in total is required. A miner who declares
nothing does not send the field at all, and selection stays with ESFT.

Pure Python: the funding intake runs on a box without torch, so the model's
geometry is read from ``pyramid_model.py``'s source, never imported.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path

__all__ = [
    "DECLARED_ROWS_BAD_LAYER",
    "DECLARED_ROWS_CODES",
    "DECLARED_ROWS_COUNT",
    "DECLARED_ROWS_MALFORMED",
    "DECLARED_ROWS_OUT_OF_RANGE",
    "DeclaredRowsError",
    "canonical_declared_rows",
    "declared_rows_digest",
    "layer_clamp",
    "model_experts_per_layer",
    "parse_declared_rows",
]

DECLARED_ROWS_OUT_OF_RANGE = "declared_rows_out_of_range"
DECLARED_ROWS_BAD_LAYER = "declared_rows_bad_layer"
DECLARED_ROWS_COUNT = "declared_rows_count"
DECLARED_ROWS_MALFORMED = "declared_rows_malformed"
DECLARED_ROWS_CODES = (DECLARED_ROWS_OUT_OF_RANGE, DECLARED_ROWS_BAD_LAYER,
                       DECLARED_ROWS_COUNT, DECLARED_ROWS_MALFORMED)

_PYRAMID_MODEL = Path(__file__).with_name("pyramid_model.py")
_MODEL_CONFIG_CLASS = "CascadeMoEConfig"
_ORDINAL_RE = re.compile(r"0|[1-9][0-9]*")


class DeclaredRowsError(ValueError):
    """A refused declaration; ``code`` is one of :data:`DECLARED_ROWS_CODES`."""

    def __init__(self, code: str, message: str) -> None:
        if code not in DECLARED_ROWS_CODES:
            raise ValueError(f"unknown declared-rows code {code!r}")
        super().__init__(message)
        self.code = code


@cache
def model_experts_per_layer() -> tuple[int, ...]:
    """``CascadeMoEConfig.experts_per_layer`` of the live MoE arch, read from
    ``pyramid_model.py``'s source by AST (that module imports torch)."""
    tree = ast.parse(_PYRAMID_MODEL.read_text(encoding="utf-8"), str(_PYRAMID_MODEL))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == _MODEL_CONFIG_CLASS:
            for st in node.body:
                if (isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
                        and st.target.id == "experts_per_layer" and st.value is not None):
                    value = tuple(int(e) for e in ast.literal_eval(st.value))
                    if not value or any(e <= 0 for e in value):
                        raise RuntimeError(f"{_PYRAMID_MODEL}: bad experts_per_layer {value!r}")
                    return value
    raise RuntimeError(f"{_PYRAMID_MODEL}: {_MODEL_CONFIG_CLASS}.experts_per_layer not found")


def layer_clamp(n_experts: int, min_frac: float, max_frac: float) -> tuple[int, int]:
    """LB2 per-layer row count bounds: ``ceil(min_frac E) .. floor(max_frac E)``."""
    return math.ceil(float(min_frac) * n_experts), math.floor(float(max_frac) * n_experts)


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _ordinal(key, n_layers: int) -> int:
    if _is_int(key):
        idx = key
    elif isinstance(key, str) and _ORDINAL_RE.fullmatch(key):
        idx = int(key)
    else:
        raise DeclaredRowsError(DECLARED_ROWS_BAD_LAYER,
                                f"layer key {key!r} is not an MoE ordinal 0..{n_layers - 1}")
    if not 0 <= idx < n_layers:
        raise DeclaredRowsError(DECLARED_ROWS_BAD_LAYER,
                                f"layer key {key!r} is outside the MoE ordinals "
                                f"0..{n_layers - 1} (keys are ordinals, not block indices)")
    return idx


def _unique_pairs(pairs):
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise DeclaredRowsError(DECLARED_ROWS_BAD_LAYER, f"layer key {k!r} appears twice")
        out[k] = v
    return out


def parse_declared_rows(raw, *, experts_per_layer, min_frac: float,
                        max_frac: float) -> dict[int, list[int]]:
    """Validate a declaration and return ``{ordinal: sorted rows}`` for every
    layer with rows (layers with none are left out).

    ``raw`` is JSON text or a mapping; keys are ordinals as ints or decimal
    strings, values are lists of expert indices. Raises
    :class:`DeclaredRowsError`. ``raw`` is never mutated."""
    epl = tuple(int(e) for e in experts_per_layer)
    if not epl or any(e <= 0 for e in epl):
        raise ValueError(f"bad experts_per_layer {experts_per_layer!r}")
    lo_frac, hi_frac = float(min_frac), float(max_frac)
    if not (math.isfinite(lo_frac) and math.isfinite(hi_frac) and 0.0 <= lo_frac <= hi_frac):
        raise ValueError(f"bad clamp fractions {min_frac!r} / {max_frac!r}")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw, object_pairs_hook=_unique_pairs)
        except DeclaredRowsError:
            raise
        except ValueError as e:
            raise DeclaredRowsError(DECLARED_ROWS_MALFORMED,
                                    f"declared rows are not valid JSON: {e}") from None
    if not isinstance(raw, Mapping):
        raise DeclaredRowsError(DECLARED_ROWS_MALFORMED,
                                "declared rows must be a JSON object keyed by MoE ordinal, "
                                f"got {type(raw).__name__}")
    out: dict[int, list[int]] = {}
    seen: set[int] = set()
    for key, value in raw.items():
        ordinal = _ordinal(key, len(epl))
        if ordinal in seen:
            raise DeclaredRowsError(DECLARED_ROWS_BAD_LAYER,
                                    f"MoE ordinal {ordinal} is declared twice")
        seen.add(ordinal)
        if not isinstance(value, (list, tuple)):
            raise DeclaredRowsError(DECLARED_ROWS_MALFORMED,
                                    f"layer {ordinal}: rows must be a list of expert indices, "
                                    f"got {type(value).__name__}")
        n_experts = epl[ordinal]
        for x in value:
            if not _is_int(x) or not 0 <= x < n_experts:
                raise DeclaredRowsError(DECLARED_ROWS_OUT_OF_RANGE,
                                        f"layer {ordinal}: row {x!r} is not an expert index "
                                        f"0..{n_experts - 1}")
        if not value:
            continue
        rows = sorted(set(value))
        if len(rows) != len(value):
            raise DeclaredRowsError(DECLARED_ROWS_COUNT,
                                    f"layer {ordinal}: duplicate rows ({len(value)} given, "
                                    f"{len(rows)} distinct)")
        lo, hi = layer_clamp(n_experts, lo_frac, hi_frac)
        if not lo <= len(rows) <= hi:
            raise DeclaredRowsError(DECLARED_ROWS_COUNT,
                                    f"layer {ordinal}: {len(rows)} rows outside the clamp "
                                    f"{lo}..{hi} for {n_experts} experts")
        out[ordinal] = rows
    if not out:
        raise DeclaredRowsError(DECLARED_ROWS_COUNT,
                                "at least one declared row in total is required (omit the "
                                "field to let the trainer pick)")
    return dict(sorted(out.items()))


def canonical_declared_rows(rows: Mapping[int, list[int]]) -> str:
    """The canonical JSON of parsed rows: string ordinal keys in order,
    sorted rows, empty layers dropped, no whitespace. This is also the
    ``--selection-json`` form the worker reads."""
    clean = {int(k): sorted(int(x) for x in v) for k, v in rows.items() if v}
    return json.dumps({str(k): clean[k] for k in sorted(clean)}, separators=(",", ":"))


def declared_rows_digest(rows: Mapping[int, list[int]]) -> str:
    """sha256 hex of :func:`canonical_declared_rows`: the value a lane B fund
    request signs, so the hotkey signature binds the rows."""
    return hashlib.sha256(canonical_declared_rows(rows).encode()).hexdigest()
