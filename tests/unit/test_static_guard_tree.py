"""Static guard over the whole tree + packed sources; binary members rejected."""

from __future__ import annotations

import base64
import zlib

from cascade.interface.static_guard import scan_source, scan_tree
from cascade.interface.validation import check_repo_layout

BLOCKED = ("socket", "subprocess", "ctypes", "cascade.shared.chain")
CLEAN = (
    "import numpy as np\n"
    "from cascade.interface import DataGenerator\n"
    "class Generator(DataGenerator):\n"
    "    def generate(self):\n"
    "        yield np.zeros(8)\n"
)
PAD = "\n".join(f"x{i} = {i}" for i in range(80))   # pushes packed sources past MIN_PACKED_CHARS


def _repo(tmp_path, generator=CLEAN, **extra):
    d = tmp_path / "repo"
    d.mkdir(parents=True)
    (d / "generator.py").write_text(generator)
    (d / "config.json").write_text("{}")
    (d / "requirements.txt").write_text("")
    for name, body in extra.items():
        f = d / name          # keys may carry "/" (passed via **{...})
        f.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, bytes):
            f.write_bytes(body)
        else:
            f.write_text(body)
    return d


def test_sibling_module_is_scanned(tmp_path):
    d = _repo(tmp_path, generator=CLEAN + "import helper\n",
              **{"helper.py": "import socket\n"})
    res = scan_tree(d, BLOCKED)
    assert not res.ok and res.blocked_module == "socket" and res.file == "helper.py"


def test_nested_package_module_is_scanned(tmp_path):
    d = _repo(tmp_path, **{"pkg/__init__.py": "", "pkg/net.py": "from subprocess import run\n"})
    res = scan_tree(d, BLOCKED)
    assert not res.ok and res.blocked_module == "subprocess" and res.file == "pkg/net.py"


def test_clean_tree_passes_and_generator_first(tmp_path):
    d = _repo(tmp_path, **{"util.py": "import math\n" + PAD})
    assert scan_tree(d, BLOCKED).ok
    (d / "generator.py").unlink()
    assert scan_tree(d, BLOCKED).reason == "missing_file"


def test_plain_string_packed_module_is_unpacked():
    packed = "import ctypes\n" + PAD
    src = f"_SRC = {packed!r}\nexec(compile(_SRC, '<p>', 'exec'))\n"
    res = scan_source(src, BLOCKED)
    assert not res.ok and res.blocked_module == "ctypes" and res.reason.startswith("packed[1]")


def test_base64_zlib_packed_module_is_unpacked():
    inner = ("import socket\n" + PAD).encode()
    b64z = base64.b64encode(zlib.compress(inner)).decode()
    src = f"import base64, zlib\n_B = {b64z!r}\nexec(zlib.decompress(base64.b64decode(_B)))\n"
    assert scan_source(src, BLOCKED).blocked_module == "socket"
    b64 = base64.b64encode(inner).decode()
    assert scan_source(f"_B = {b64!r}\n", BLOCKED).blocked_module == "socket"
    hx = inner.hex()
    assert scan_source(f"_H = {hx!r}\n", BLOCKED).blocked_module == "socket"


def test_packed_inside_packed_is_unpacked():
    innermost = "import socket\n" + PAD
    middle = f"_I = {innermost!r}\nexec(_I)\n" + PAD
    outer = f"_M = {middle!r}\nexec(_M)\n"
    res = scan_source(outer, BLOCKED)
    assert not res.ok and res.reason.startswith("packed[2]")


def test_bytes_constant_is_unpacked():
    inner = ("import socket\n" + PAD).encode()
    src = f"_B = {inner!r}\nexec(_B)\n"
    assert scan_source(src, BLOCKED).blocked_module == "socket"


def test_long_benign_strings_pass():
    lorem = "lorem ipsum " * 100
    noise = base64.b64encode(bytes(range(256)) * 4).decode()
    docs = "def is a keyword; import is too; " * 20    # looks like python, is prose
    src = f"A = {lorem!r}\nB = {noise!r}\nC = {docs!r}\nimport numpy\n"
    assert scan_source(src, BLOCKED).ok


def test_packed_clean_module_passes():
    inner = "import numpy\n" + PAD
    assert scan_source(f"_S = {inner!r}\nexec(_S)\n", BLOCKED).ok


def test_unpack_can_be_disabled():
    src = "_S = " + repr("import socket\n" + PAD) + "\n"
    assert scan_source(src, BLOCKED, unpack=False).ok
    assert not scan_source(src, BLOCKED).ok


def test_unparseable_sibling_is_reported(tmp_path):
    d = _repo(tmp_path, **{"broken.py": "def (:\n"})
    res = scan_tree(d, BLOCKED)
    assert not res.ok and res.reason.startswith("syntax_error") and res.file == "broken.py"


def test_layout_rejects_compiled_modules(tmp_path):
    d = _repo(tmp_path, **{"fast.cpython-311-x86_64-linux-gnu.so": b"\x7fELF"})
    res = check_repo_layout(d)
    assert not res.ok and res.reason == "binary_modules_forbidden"
    assert res.details["files"] == ["fast.cpython-311-x86_64-linux-gnu.so"]
    # a sourceless bytecode module next to the code is importable → rejected;
    # the interpreter's own __pycache__ (written when the scanned source is
    # imported) is not.
    d2 = _repo(tmp_path / "b", **{"helper.pyc": b"\x00"})
    assert check_repo_layout(d2).reason == "binary_modules_forbidden"
    d3 = _repo(tmp_path / "c", **{"__pycache__/generator.cpython-311.pyc": b"\x00"})
    assert check_repo_layout(d3).ok
