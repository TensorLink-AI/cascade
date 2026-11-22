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
                            "time/traffic/3": _walk(rng, 56, 300.0, 9.0)}},
        "fev": {"version": VERSIONS["fev"],
                "series": {"fev/solar/2": _walk(rng, 44, 7.0, 0.3)}},
    }


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
