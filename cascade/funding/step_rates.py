"""The trainer-side per-GPU step-rate table (WB19 → WB22 / WB23).

WB19's measurement tool (``scripts/laned_wb19_measure.py combine``) writes a
``laned_wb19_table/1`` table, either as its own file (``--table-out``) or as
the ``table`` of the ``laned_wb19/1`` evidence. This module reads it for the
trainer: WB22 keys lane B legs on the measured expert-only step rate of the
GPU type a leg rents, and WB23 fixes a leg's step count at admission as
``floor(train_hours × 3600 × measured_steps_per_s[gpu])`` (D42b (a)).

Stdlib only: importing it never imports torch.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

TABLE_SCHEMA = "laned_wb19_table/1"
EVIDENCE_SCHEMA = "laned_wb19/1"

_IMAGE = re.compile(r"sha256:[0-9a-f]{64}")


class StepRateTableError(ValueError):
    """A table the trainer must not use. ``code``: ``bad_table`` (malformed),
    ``not_passed`` (evidence with a failing check other than coverage, rev6.16) or
    ``wrong_image`` (measured on another train image than the pinned one)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class StepRate:
    """One measured GPU type (a layout SKU such as ``8xH100``)."""

    sku: str
    gpu_name: str
    gpu_count: int
    steps_per_s: float
    tokens_per_s: float
    overhead_seconds: float
    peak_memory_bytes: int
    memory_total_bytes: int


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _positive_rate(row: dict, key: str, sku: str) -> float:
    v = row.get(key)
    if not _number(v) or not math.isfinite(v) or v <= 0:
        raise StepRateTableError("bad_table", f"{sku}: {key} must be a finite number > 0; "
                                              f"got {v!r}")
    return float(v)


def _row(sku: str, row) -> StepRate:
    if not isinstance(row, dict):
        raise StepRateTableError("bad_table", f"{sku}: a row must be an object")
    overhead = row.get("overhead_seconds")
    if not _number(overhead) or not math.isfinite(overhead) or overhead < 0:
        raise StepRateTableError("bad_table", f"{sku}: overhead_seconds must be a finite "
                                              f"number >= 0; got {overhead!r}")
    ints = {}
    for key in ("gpu_count", "peak_memory_bytes", "memory_total_bytes"):
        v = row.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise StepRateTableError("bad_table", f"{sku}: {key} must be a non-negative int; "
                                                  f"got {v!r}")
        ints[key] = v
    name = row.get("gpu_name")
    if not isinstance(name, str):
        raise StepRateTableError("bad_table", f"{sku}: gpu_name must be a string")
    return StepRate(sku=sku, gpu_name=name, gpu_count=ints["gpu_count"],
                    steps_per_s=_positive_rate(row, "steps_per_s", sku),
                    tokens_per_s=_positive_rate(row, "tokens_per_s", sku),
                    overhead_seconds=float(overhead),
                    peak_memory_bytes=ints["peak_memory_bytes"],
                    memory_total_bytes=ints["memory_total_bytes"])


def _body_sha256(doc: dict) -> str:
    """WB19's seal: sha256 of the canonical JSON without ``body_sha256``."""
    body = {k: v for k, v in doc.items() if k != "body_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _partial_ok(doc: dict) -> bool:
    """Partial WB19 evidence (owner 2026-11-23, plan rev6.16): usable when it is
    sealed and ``coverage`` is its only failing check. Its rows are the measured
    types; every other type stays unmeasured."""
    if not isinstance(doc.get("body_sha256"), str):
        return False
    checks = doc.get("checks")
    if not isinstance(checks, list) or not checks:
        return False
    failing = []
    for c in checks:
        if not (isinstance(c, dict) and isinstance(c.get("name"), str)
                and isinstance(c.get("passed"), bool)):
            return False
        if not c["passed"]:
            failing.append(c["name"])
    return bool(failing) and all(n == "coverage" for n in failing)


def load_step_rate_table(path: Path | str, *,
                         image_digest: str | None = None) -> dict[str, StepRate]:
    """The table at ``path`` (a bare ``laned_wb19_table/1`` or a
    ``laned_wb19/1`` evidence file) as ``{sku: StepRate}``.

    ``image_digest`` (the pinned ``train_image_digest``) refuses a table
    measured on another image with ``wrong_image``: a repinned image makes
    the table stale."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise StepRateTableError("bad_table", f"{path}: unreadable JSON ({e})") from e
    if not isinstance(doc, dict):
        raise StepRateTableError("bad_table", f"{path}: not a JSON object")
    if doc.get("schema") == EVIDENCE_SCHEMA:
        sha = doc.get("body_sha256")
        if sha is not None and sha != _body_sha256(doc):
            raise StepRateTableError("bad_table", f"{path}: the evidence body_sha256 does not "
                                                  "match its body (hand-edited)")
        if doc.get("passed") is not True and not _partial_ok(doc):
            raise StepRateTableError("not_passed", f"{path}: the WB19 evidence did not pass "
                                                   "every check (only a failing coverage "
                                                   "check is allowed, rev6.16)")
        doc = doc.get("table")
        if not isinstance(doc, dict):
            raise StepRateTableError("bad_table", f"{path}: the evidence carries no table")
    if doc.get("schema") != TABLE_SCHEMA:
        raise StepRateTableError("bad_table", f"{path}: schema {doc.get('schema')!r} is not "
                                              f"{TABLE_SCHEMA!r}")
    image = doc.get("image_digest")
    if not isinstance(image, str) or not _IMAGE.fullmatch(image):
        raise StepRateTableError("bad_table", f"{path}: image_digest {image!r} is not "
                                              "sha256:<64 hex>")
    rows = doc.get("rows")
    if not isinstance(rows, dict) or not rows:
        raise StepRateTableError("bad_table", f"{path}: no rows")
    table = {str(sku): _row(str(sku), row) for sku, row in rows.items()}
    if image_digest is not None and image != image_digest:
        raise StepRateTableError("wrong_image", f"{path}: measured on {image}, the pinned "
                                                f"train image is {image_digest}")
    return table


def rate_for(table: dict[str, StepRate], sku: str) -> StepRate | None:
    """The row for ``sku``, case-insensitive and whitespace-tolerant."""
    want = str(sku or "").strip().lower()
    for key, rate in table.items():
        if key.strip().lower() == want:
            return rate
    return None


def steps_for_hours(rate: StepRate, hours: float) -> int:
    """D42b (a): ``floor(hours × 3600 × rate.steps_per_s)``."""
    h = float(hours)
    if not math.isfinite(h) or h <= 0:
        raise ValueError(f"hours must be a finite number > 0; got {hours!r}")
    return int(math.floor(h * 3600.0 * rate.steps_per_s))
