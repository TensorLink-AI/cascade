"""[static_guard] packed_sources = "reject": every module must be a .py file.

"scan" (legacy) unpacks Python packed into string constants (plain, base64,
zlib, hex; nested) and scans its imports. "reject" makes such a constant
itself a rejection — at ADMISSION (rolling intake) and in `cascade verify` —
so the code that runs is the code the duplicate screen fingerprints. Pinned:

* every encoding the decoder knows is rejected in reject mode, at any depth;
* scan mode is byte-for-byte the legacy behaviour;
* long non-Python strings (data) still pass in reject mode;
* the loader parses the knob (default "scan") and fails loud on junk;
* the per-leg sandbox preflight is untouched (it never passes the knob), so a
  seated king's own legs cannot be affected — rules apply at the door only.
"""
from __future__ import annotations

import base64
import inspect
import zlib
from dataclasses import replace
from pathlib import Path

import pytest

from cascade.interface.static_guard import (
    scan_source,
    scan_tree,
    validate_packed_sources_mode,
)
from cascade.shared.config import StaticGuardConfig, load_chain_config

BLOCKED = ("socket", "ctypes")
CLEAN = (
    "import numpy as np\n"
    "from cascade.interface import DataGenerator\n"
    "class Generator(DataGenerator):\n"
    "    def generate(self):\n"
    "        yield np.zeros(64)\n"
)
PAD = "\n".join(f"x{i} = {i}" for i in range(80))
INNER = ("import math\n" + PAD)      # clean inner module — rejected for BEING packed, not for imports


def _packed_variants(inner: str = INNER):
    b = inner.encode()
    yield "plain", f"_S = {inner!r}\nexec(compile(_S, '<p>', 'exec'))\n"
    yield "b64", f"_B = {base64.b64encode(b).decode()!r}\n"
    yield "b64zlib", f"_Z = {base64.b64encode(zlib.compress(b)).decode()!r}\n"
    yield "hex", f"_H = {b.hex()!r}\n"
    yield "bytes", f"_Y = {b!r}\n"


@pytest.mark.parametrize("name,src", list(_packed_variants()))
def test_reject_mode_rejects_every_known_packing(name, src):
    res = scan_source(src, BLOCKED, packed_sources="reject")
    assert not res.ok and res.reason == "packed_source[1]" and res.blocked_module is None, name


@pytest.mark.parametrize("name,src", list(_packed_variants()))
def test_scan_mode_is_the_legacy_behaviour(name, src):
    assert scan_source(src, BLOCKED).ok, name                         # default = scan
    assert scan_source(src, BLOCKED, packed_sources="scan").ok, name


@pytest.mark.parametrize("name,src", list(_packed_variants(INNER.replace("import math", "import socket"))))
def test_scan_mode_still_catches_blocked_imports_inside_packing(name, src):
    res = scan_source(src, BLOCKED)
    assert not res.ok and res.blocked_module == "socket", name


def test_nested_packing_is_rejected_at_the_outer_layer():
    # the middle layer must itself look like Python to the decoder (an import)
    inner_packed = f"import os\n_I = {base64.b64encode(INNER.encode()).decode()!r}\n" + PAD
    outer = f"_O = {base64.b64encode(inner_packed.encode()).decode()!r}\n"
    res = scan_source(outer, BLOCKED, packed_sources="reject")
    assert not res.ok and res.reason == "packed_source[1]"


def test_long_data_strings_still_pass_in_reject_mode():
    data = "0.1,0.2,0.3," * 200
    src = CLEAN + f"_CSV = {data!r}\n_B = {base64.b64encode(data.encode()).decode()!r}\n"
    assert scan_source(src, BLOCKED, packed_sources="reject").ok


def test_scan_tree_names_the_file(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    (d / "generator.py").write_text(CLEAN)
    (d / "config.json").write_text("{}")
    (d / "requirements.txt").write_text("")
    (d / "helpers.py").write_text(f"_S = {INNER!r}\n")
    assert scan_tree(d, BLOCKED).ok
    res = scan_tree(d, BLOCKED, packed_sources="reject")
    assert not res.ok and res.file == "helpers.py" and res.reason == "packed_source[1]"


def test_preflight_never_passes_the_knob():
    """The per-leg sandbox preflight keeps the legacy scan whatever the config
    says: rejection is an admission-time rule, never a leg-time one."""
    from cascade.trainer import sandbox

    src = inspect.getsource(sandbox._preflight)
    assert "scan_tree(repo, tuple(blocked))" in src
    assert "packed_sources" not in src


# ── config ────────────────────────────────────────────────────────────────────

def test_validate_mode():
    assert validate_packed_sources_mode("scan") == "scan"
    assert validate_packed_sources_mode("reject") == "reject"
    for bad in ("", "REJECT", "shadow", "enforce"):
        with pytest.raises(ValueError):
            validate_packed_sources_mode(bad)


def test_default_is_scan_and_loader_round_trips(tmp_path):
    assert StaticGuardConfig(blocked=()).packed_sources == "scan"
    root = Path(__file__).resolve().parents[2]
    text = (root / "chain.toml").read_text()
    assert "packed_sources" not in text.split("[static_guard]")[1].split("[storage]")[0]
    assert load_chain_config(root / "chain.toml").static_guard.packed_sources == "scan"
    armed = text.replace("[static_guard]\n", '[static_guard]\npacked_sources = "reject"\n', 1)
    p = tmp_path / "chain.toml"
    p.write_text(armed)
    assert load_chain_config(p).static_guard.packed_sources == "reject"
    bad = tmp_path / "bad.toml"
    bad.write_text(armed.replace('"reject"', '"shadow"'))
    with pytest.raises(ValueError, match="packed_sources"):
        load_chain_config(bad)
    assert load_chain_config(root / "chain.testnet.toml").static_guard.packed_sources == "reject"


def test_replace_keeps_frozen_dataclass_semantics(cfg):
    sg = replace(cfg.static_guard, packed_sources="reject")
    assert sg.blocked == cfg.static_guard.blocked and sg.packed_sources == "reject"
