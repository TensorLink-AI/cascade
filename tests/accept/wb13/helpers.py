"""Shared spec and fixtures for the WB13 acceptance tests (intake contamination check, LB1).

Everything here comes from docs/plans/LANE_D_PLAN.md: the WB13 row (§3.0
lane B track), §12b.1 LB1, D38, the T-rows TB10 / TB11 / TB12 (§4) and the
WB11 hash-set contract (tests/accept/wb11/README.md). Contract: README.md.

Nothing here imports a PATHS module at module level: every helper that needs
one imports it in its body, so on main each failure is an import error for a
PATHS module (or an AssertionError) inside a test.

Miner code is untrusted. The fixture repos are written as plain files in
``tmp_path`` and are never imported or run by these tests: the scanner must
read them (AST, JSON, bytes) without executing them, which the
``never_executed`` fixture checks with a side-effect marker.

The suite fixture is tiny, seeded and in memory. The WB11 module
(``cascade.eval.contamination``, on main) builds the hash set from it; no
benchmark data is downloaded and no torch is loaded.
"""

from __future__ import annotations

import importlib
import importlib.abc
import json
import random
import struct
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
FUNDING_MAIN = REPO / "cascade" / "funding" / "main.py"
CHAIN_TOML = REPO / "chain.toml"
PLAN = REPO / "docs" / "plans" / "LANE_D_PLAN.md"

SCAN_MODULE = "cascade.interface.contamination_scan"
WB11_MODULE = "cascade.eval.contamination"

EMBEDDED = "benchmark_series_embedded"
TOO_LARGE = "repo_too_large"
RECEIPT_KEYS = {"hashset_version", "result", "suite", "window_id"}
MIB = 1024 * 1024

# ── the suite fixture (four tiny seeded suites) ──────────────────────────────

VERSIONS = {
    "gift": "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0",
    "boom": "69325b544c45ff0d6c43c7a99c49a6601a01725b",
    "time": "83e3d0b3be28d11c7182bffcc1892d19b36c4da1",
    "fev": "6de245af1c0ffee0wb13fixturemanifestdigest",
}

GIFT_ITEM = "gift/electricity/7"
FLAT_ITEM = "time/flat/0"
SEG_START, SEG_LEN = 20, 40           # the embedded test window: 40 points, 9 sub-windows
FLAT_VALUE = 3


def _walk(rng: random.Random, n: int, level: float, scale: float) -> list[float]:
    out, x = [], level
    for _ in range(n):
        x += rng.gauss(0.0, scale)
        out.append(x + rng.gauss(0.0, scale * 0.5))
    return out


def fixture_suites() -> dict:
    """``{suite: {"version", "series": {item: [float]}}}`` for gift / boom /
    time / fev. ``GIFT_ITEM`` is the series TB10 and TB11 embed. Its level
    (5000) and step (40) are electricity-like. See the README's TB11 note
    for why the scale matters to the WB11 normalised hash."""
    rng = random.Random("cascade wb13 fixture suites")
    return {
        "gift": {"version": VERSIONS["gift"],
                 "series": {GIFT_ITEM: _walk(rng, 72, 5000.0, 40.0),
                            "gift/m4_hourly/0": _walk(rng, 48, 100.0, 3.0)}},
        "boom": {"version": VERSIONS["boom"],
                 "series": {"boom/svc-a/cpu": _walk(rng, 48, 0.4, 0.02)}},
        "time": {"version": VERSIONS["time"],
                 "series": {FLAT_ITEM: [float(FLAT_VALUE)] * 40,
                            "time/traffic/3": _walk(rng, 56, 300.0, 9.0),
                            COUNTS_ITEM: [float(v) for v in counts_series()]}},
        "fev": {"version": VERSIONS["fev"],
                "series": {"fev/solar/2": _walk(rng, 44, 7.0, 0.3)}},
    }


# RV6: an integer-valued, informative suite series (event counts). The int
# byte decodings are pinned on it: the flat series is low-information under
# WB11's RV6 B5 rule, so it is no longer contamination evidence.
COUNTS_ITEM = "time/counts/5"
COUNTS_WINDOW_ID = f"{COUNTS_ITEM}@0"


def counts_series() -> list[int]:
    rng = random.Random("cascade wb13 counts")
    return [rng.randrange(0, 60) for _ in range(40)]


def segment() -> list[float]:
    """TB10's exact test window: ``GIFT_ITEM[20:60]``."""
    s = fixture_suites()["gift"]["series"][GIFT_ITEM]
    return list(s[SEG_START:SEG_START + SEG_LEN])


F32_ITEM = "gift/m4_hourly/0"


def f32_source() -> list[float]:
    """The whole ``F32_ITEM`` series (level 100), the float32 bytes' source.
    float32 keeps about 7 digits, which is enough for the WB11 normalised
    hash at this level but not at level 5000."""
    return list(fixture_suites()["gift"]["series"][F32_ITEM])


SEG_WINDOW_ID = f"{GIFT_ITEM}@{SEG_START}"
FLAT_WINDOW_ID = f"{FLAT_ITEM}@0"


def tb11(values) -> list[float]:
    """TB11: "Same window scaled ×3.7, offset +100 and rounded to 3 dp"."""
    return [round(3.7 * x + 100.0, 3) for x in values]


def shifted(values) -> list[float]:
    """A shifted copy with negative values (``x − 1e6``)."""
    return [x - 1.0e6 for x in values]


def float32_copy(values) -> list[float]:
    return list(struct.unpack(f"<{len(values)}f", struct.pack(f"<{len(values)}f", *values)))


def clean_series(n: int = 120) -> list[float]:
    return _walk(random.Random("cascade wb13 clean"), n, 42.0, 2.0)


# ── the WB11 hash set (on main) ──────────────────────────────────────────────


def wb11():
    return importlib.import_module(WB11_MODULE)


def artefact(version: int = 1) -> dict:
    return wb11().build_hashset(fixture_suites(), version=version)


def pin_of(art: dict) -> str:
    return wb11().artefact_sha256(art)


def expected_first(art: dict, values) -> dict:
    """The WB11 lookup's first hit of ``values`` (the reference for a
    receipt's ``suite`` / ``window_id``)."""
    first = wb11().match_series(art, values)["first"]
    assert first is not None, "fixture error: the sequence has no hit in the hash set"
    return first


# ── [lane_b] config through the real parser (WB10) ───────────────────────────


def lane_b_cfg(*, pin: str = "", suite_targets: bool = False, cap_mb: int | None = None,
               min_hits: dict | None = None):
    """``parse_lane_b`` of a two-domain lane B with the contamination keys.
    ``pin = ""`` is the shipped default (unarmed). Suite targets need the pin
    (the parser refuses them without one)."""
    from cascade.shared.config import parse_lane_b

    raw: dict = {"from_block": 1, "domains": ["energy", "web"], "open_tokens": 100,
                 "routing_jaccard": 0.7, "legs_per_domain": 4, "max_base_age_eras": 20}
    if pin:
        raw["contamination_hashset"] = pin
    if suite_targets:
        raw["suite_targets"] = True
    if cap_mb is not None:
        raw["suite_target_max_repo_mb"] = int(cap_mb)
    if min_hits is not None:
        raw["contamination_min_hits"] = dict(min_hits)
    return parse_lane_b(raw)


# ── fixture repos: plain files, never executed ───────────────────────────────

CLEAN_GENERATOR = '''\
"""A clean code-only generator: random walks with a seasonal term."""
import math
import random

LENGTHS = (64, 128, 256)
WEIGHTS = [0.25, 0.5, 0.25]


def generate(seed, n_series):
    rng = random.Random(seed)
    out = []
    for i in range(n_series):
        n = LENGTHS[i % len(LENGTHS)]
        x, period = 0.0, rng.choice((7, 24, 168))
        row = []
        for t in range(n):
            x += rng.gauss(0.0, 1.0)
            row.append(x + math.sin(2 * math.pi * t / period))
        out.append(row)
    return out
'''

CLEAN_CONFIG = {"name": "walks", "params": {"lengths": [64, 128, 256], "noise": 1.0,
                                            "periods": [7, 24, 168]}}
REQUIREMENTS = ("numpy==1.26.4 --hash=sha256:"
                "2ffbc1ea1ae59cf9c2f6c4f2e1b1dd8f0f8e4b3e1a3d5f8d8a7c6b5e4d3c2b1a\n")
README = "Random walks with seasonality.\nNo data files.\n"


def clean_files() -> dict[str, bytes]:
    return {
        "generator.py": CLEAN_GENERATOR.encode(),
        "config.json": json.dumps(CLEAN_CONFIG, indent=2).encode(),
        "requirements.txt": REQUIREMENTS.encode(),
        "README.md": README.encode(),
    }


def py_list(name: str, values) -> str:
    return f"{name} = [\n" + "".join(f"    {v!r},\n" for v in values) + "]\n"


def marker_preamble(marker: Path) -> str:
    """Top-level code that leaves a marker file if the module is ever run."""
    return f"import pathlib\npathlib.Path({str(marker)!r}).write_text('executed')\n"


def write_repo(root: Path, files: dict[str, bytes | str], order=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel in (order or sorted(files)):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        data = files[rel]
        p.write_bytes(data.encode() if isinstance(data, str) else data)
    return root


def repo(tmp: Path, name: str = "repo", **extra: bytes | str) -> Path:
    """The clean repo plus ``extra`` files (``relpath=content``; ``/`` in a
    path is written as ``__`` in the keyword)."""
    files = clean_files()
    for k, v in extra.items():
        files[k.replace("__", "/")] = v
    return write_repo(tmp / name, files)


def repo_with(tmp: Path, files: dict[str, bytes | str], name: str = "repo") -> Path:
    base = clean_files()
    base.update(files)
    return write_repo(tmp / name, base)


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def pad_to(root: Path, total: int, rel: str = "notes.py") -> None:
    """Add a comment-only ``.py`` file so the tree totals ``total`` bytes."""
    need = total - tree_bytes(root)
    assert need > 2, "fixture error: tree already too large"
    line = "#" + "a" * 78 + "\n"
    body = line * ((need - 1) // len(line))
    body += "#" * (need - len(body) - 1) + "\n"
    (root / rel).write_text(body, encoding="ascii")
    assert tree_bytes(root) == total


# ── the scanner (PATHS module, lazy) ─────────────────────────────────────────


class _TorchBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(
                f"WB13: '{name}' must not be imported (the contamination check is "
                "a static, torch-free scan at intake)", name=name)
        return None


@contextmanager
def torch_blocked():
    blocker = _TorchBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


def scan_mod():
    return importlib.import_module(SCAN_MODULE)


def scan(repo_dir: Path, art: dict, min_hits=None) -> dict:
    out = scan_mod().scan_repo(repo_dir, art, min_hits=min_hits)
    assert isinstance(out, dict), f"scan_repo returned {type(out).__name__}, not a dict"
    assert set(out) == RECEIPT_KEYS, (
        f"scan_repo keys {sorted(out)} != the receipt field {sorted(RECEIPT_KEYS)}")
    return out


def receipt(version: int, result: str, suite=None, window_id=None) -> dict:
    return {"hashset_version": version, "result": result, "suite": suite,
            "window_id": window_id}


# ── funding intake ───────────────────────────────────────────────────────────

HK = "5FakeHotkeyWB13AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
REF = "vault/direct@sha256:" + "b" * 64
API_KEY = "sk-live-wb13"


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class Fetcher:
    """The injected repo fetch: ``ref -> local tree``. It records every
    call. ``tree=None`` answers ``None`` (unfetchable); ``raises`` raises."""

    def __init__(self, tree: Path | None, raises: Exception | None = None) -> None:
        self.tree = tree
        self.raises = raises
        self.calls: list[str] = []

    def __call__(self, ref: str):
        self.calls.append(ref)
        if self.raises is not None:
            raise self.raises
        return self.tree


def intake(tmp_path: Path, *, cfg, hashset, fetch, sig: bool = False, verify=None,
           reveal: int | None = 700):
    """A real :class:`FundingIntake` with a real lane B queue on disk. Only
    the chain (the reveal oracle), the signature verifier and the repo fetch
    are stand-ins."""
    import inspect

    from cascade.funding.intake import FundingIntake
    from cascade.funding.queue import FundedQueue, LaneBQueue, lane_b_queue_path
    from cascade.funding.vault import PayerKeyVault

    params = inspect.signature(FundingIntake.__init__).parameters
    for kw in ("lane_b_hashset", "fetch_lane_b_repo"):
        assert kw in params, (
            f"FundingIntake takes {kw} (WB13: the loaded hash set and the submitted-repo "
            "fetch the contamination check scans)")
    clock = Clock()
    lane_a = FundedQueue(tmp_path / "queue.json", clock=clock)
    qpath = lane_b_queue_path(tmp_path / "queue.json")
    lane_b = LaneBQueue(qpath, clock=clock)
    vault = PayerKeyVault(dir=None, clock=clock)
    it = FundingIntake(
        lane_a, vault, resolve_reveal=lambda hk, ref: None, require_signature=sig,
        verify=verify or (lambda hk, msg, s: False), clock=clock,
        lane_b_queue=lane_b, resolve_lane_b_reveal=lambda hk, t, r: reveal,
        lane_b_targets=tuple(cfg.targets), lane_b_max_targets=1, lane_b_cfg=cfg,
        lane_b_hashset=hashset, fetch_lane_b_repo=fetch)
    return SimpleNamespace(intake=it, lane_b=lane_b, vault=vault, clock=clock, qpath=qpath)


def headers(target: str = "energy", **kw) -> dict:
    h = {"X-Miner-Hotkey": HK, "X-Commit-Ref": REF, "X-Lane-B-Targets": target,
         "X-Lium-Api-Key": API_KEY}
    h.update(kw)
    return {k: v for k, v in h.items() if v is not None}


def queue_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def entry_contamination(lane_b) -> dict | None:
    e = lane_b.get(HK)
    assert e is not None, "no lane B entry queued"
    return getattr(e, "contamination", None)


# ── RV6 B7: packaging forms of an embedded window ────────────────────────────


def npy_header(descr: str, n: int) -> bytes:
    """A version-1.0 ``.npy`` header of a 1-D array of ``n`` items (by hand: no numpy)."""
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': ({n},), }}"
    header += " " * (63 - (10 + len(header)) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1")


def npy_bytes(values, descr: str = "<f8") -> bytes:
    """A ``.npy`` of a 1-D float array."""
    code = {"<f8": "<d", ">f8": ">d", "<f4": "<f", ">f4": ">f"}[descr]
    return npy_header(descr, len(values)) + struct.pack(f"{code[0]}{len(values)}{code[1]}",
                                                         *values)


def pickle_trap_npy(marker: Path) -> bytes:
    """An object-dtype ``.npy`` whose payload, IF UNPICKLED, writes
    ``marker``. The scan must never unpickle (parse, never execute)."""
    import pickle

    class _Trap:
        def __reduce__(self):
            return (exec, (f"import pathlib; pathlib.Path({str(marker)!r}).write_text('x')",))

    return npy_header("|O", 1) + pickle.dumps([_Trap()], protocol=2)


def npz_bytes(members: dict[str, bytes], *, compressed: bool) -> bytes:
    """A ``.npz``: a zip of ``.npy`` members (``np.savez`` /
    ``np.savez_compressed`` layout), built with :mod:`zipfile`."""
    import io
    import zipfile

    buf = io.BytesIO()
    method = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(buf, "w", compression=method) as z:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = method
            z.writestr(info, data)
    return buf.getvalue()


def zip_bomb_npz(inflated_mib: int) -> bytes:
    """A deflated ``.npz`` whose one member inflates to ``inflated_mib`` MiB of
    zeros (header included), built in 1 MiB chunks: small on disk, never
    held inflated here."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo("big.npy", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        n = inflated_mib * MIB // 8 - 16
        with z.open(info, "w", force_zip64=True) as f:
            f.write(npy_header("<f8", n))
            chunk = bytes(MIB)
            left = n * 8
            while left > 0:
                f.write(chunk[:min(MIB, left)])
                left -= MIB
    return buf.getvalue()


# ── RV6 B6: an adversarial edge-window file ──────────────────────────────────

# 32 values with mean 0 and population std exactly 1 whose z-scores are all
# bin edges of WB11's cascade-zgrid/2 (GRID 0.5): 6 x 1.5, 10 x 0.5, 10 x -0.5,
# 6 x -1.5. Every value is within TOLERANCE of an edge, so one window costs the
# full 2 x 2**12 probes (measured on main: ~21 ms per window vs ~0.19 ms for a
# random window). Repeated with period 32, every sliding window is a cyclic
# shift of the same multiset, so every window is crafted. It is not
# low-information (4 distinct values, mode 10 of 32, not affine).
EDGE_BLOCK = (1.5, -0.5, 0.5, -1.5, 0.5, 1.5, -0.5, -0.5, 0.5, -1.5, 1.5, 0.5, -0.5, 0.5, -1.5,
              -0.5, 1.5, 0.5, -0.5, -1.5, 0.5, 1.5, -0.5, 0.5, -1.5, -0.5, 1.5, 0.5, -0.5,
              -1.5, 0.5, -0.5)
EDGE_MS_PER_WINDOW_MAIN = 21.0
RANDOM_MS_PER_WINDOW_MAIN = 0.19


def edge_values(n_windows: int) -> list[float]:
    vals = list(EDGE_BLOCK) * (n_windows // 32 + 2)
    return vals[:n_windows + 31]


def edge_repo(tmp: Path, n_windows: int, name: str = "edge") -> Path:
    vals = edge_values(n_windows)
    return repo_with(tmp, {"data/edge.f64": struct.pack(f"<{len(vals)}d", *vals)}, name=name)


SCAN_CHILD = """
import json, sys, time
from pathlib import Path
from tests.accept.wb13 import helpers as h
kw = json.loads(sys.argv[2])
t0, c0 = time.monotonic(), time.process_time()
with h.torch_blocked():
    out = h.scan_mod().scan_repo(Path(sys.argv[1]), h.artefact(1), **kw)
import resource
print(json.dumps({"receipt": out, "wall": time.monotonic() - t0, "cpu": time.process_time() - c0,
                  "maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}))
"""

INTAKE_CHILD = """
import json, sys, time
from pathlib import Path
from tests.accept.wb13 import helpers as h
tmp, tree, deadline = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
art = h.artefact(1)
cfg = h.lane_b_cfg(pin=h.pin_of(art))
fetch = h.Fetcher(tree)
with h.torch_blocked():
    m = h.scan_mod()
    m.SCAN_DEADLINE_S = deadline
    t = h.intake(tmp, cfg=cfg, hashset=art, fetch=fetch)
    t0 = time.monotonic()
    status, body = t.intake.lane_b_fund(h.headers())
    wall = time.monotonic() - t0
print(json.dumps({"status": status, "body": body, "wall": wall,
                  "queued": t.lane_b.get(h.HK) is not None, "vaulted": t.vault.has(h.HK)}))
"""


def run_child(code: str, *args: str, timeout: float) -> dict:
    """Run ``code`` in a fresh, niced interpreter; an overrun is a test
    failure (the child is killed), never a hung test."""
    import os
    import subprocess

    env = dict(os.environ, PYTHONPATH=str(REPO))
    try:
        p = subprocess.run(["nice", "-n", "19", sys.executable, "-P", "-c", code, *args],
                           capture_output=True, text=True, env=env, cwd=str(REPO),
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"the scan did not return within {timeout:.0f} s (killed): "
                             "no scan budget bounds it") from None
    assert p.returncode == 0, f"child rc={p.returncode}\n{p.stderr[-3000:]}"
    return json.loads(p.stdout.strip().splitlines()[-1])


# ── hash-seed entry point (determinism, fresh interpreters) ──────────────────


def _main(argv: list[str]) -> int:
    """``python -P -m tests.accept.wb13.helpers REPO_DIR`` prints the
    scan of REPO_DIR against the fixture hash set as one JSON line."""
    with torch_blocked():
        out = scan_mod().scan_repo(Path(argv[0]), artefact(1), min_hits=None)
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
