"""Load vendored data files (baselines, manifests) shipped under ``data/``."""

from __future__ import annotations

import json
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data"


def load_json(filename: str):
    return json.loads((_DATA / filename).read_text(encoding="utf-8"))
