"""The shipped chain.toml WITHOUT its ``[[activation.pinned]]`` lock-ins.

DEC-CA-0050 pins the fleet's three decided activations (rollover, forfeiture,
margin v2) into chain.toml, so loading it arms the mainnet grid switch, the
forfeiture and the v2 bar at mainnet block heights. Fixtures that model the
generic pre-lock-in fleet (pytest's tiny block numbers, a feature still to be
decided on chain) load this copy instead; tests of the pins load the real file.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

from cascade.shared.config import ChainConfig, load_chain_config

REPO_ROOT = Path(__file__).resolve().parents[1]
_PIN_RE = re.compile(r"^\[\[activation\.pinned\]\][^\n]*\n(?:[ \t]*[a-z_]+[ \t]*=[^\n]*\n)*",
                     re.M)


def strip_pins(text: str) -> str:
    return _PIN_RE.sub("", text)


def load_unpinned(path: Path | str | None = None) -> ChainConfig:
    src = Path(path) if path is not None else REPO_ROOT / "chain.toml"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "chain.toml"
        p.write_text(strip_pins(src.read_text(encoding="utf-8")), encoding="utf-8")
        return load_chain_config(p)
