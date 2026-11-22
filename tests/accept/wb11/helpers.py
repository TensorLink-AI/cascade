"""Shared spec and fixtures for the WB11 acceptance tests (contamination hash set).

Everything here comes from docs/plans/LANE_D_PLAN.md: the WB11 row (§3.0),
§12b.1 LB1 (exact + normalised sub-window hashes, 32-point sub-windows,
z-score + 4 significant figures, a versioned digest-pinned artefact refreshed
only when a suite identity changes) and the W3b4 suite identity formula
(tests/accept/w3b4/README.md). Contract: README.md.

Nothing here imports the PATHS module ``cascade.eval.contamination`` at module
level: every helper that needs it imports it in its body, so on main each
failure is a ``ModuleNotFoundError`` for a PATHS module inside a test. The
fixture suites are tiny, deterministic (seeded ``random.Random``) and built in
memory: no benchmark data is downloaded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import struct
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODULE = "cascade.eval.contamination"

SCHEMA = "cascade_contamination_hashset/1"
WINDOW = 32          # §12b.1: sliding sub-windows of 32 points
STRIDE = 1           # §12b.1: sliding
SIG_FIGS = 4         # §12b.1: z-score, rounded to 4 significant figures
HASH = "sha256"
SUITE_KEYS = ("boom", "fev", "gift", "time")   # cascade.eval.benchmarks.ITEM_SUITE_KEY values
MIN_HITS = {"exact": 1, "normalised": 3}       # §12b.1 recommended / WB10 default

# §12b.1: "the suite identities recorded on the bench pod (gift 30841734 /
# boom 69325b54 / time 83e3d0b3 / fev 6de245af, OPSLOG)".
PLAN_IDENTITIES = {"gift": "30841734", "boom": "69325b54", "time": "83e3d0b3",
                   "fev": "6de245af"}

# Fixture suite versions: the pinned revisions in
# benchmarks/cascade_benchmark/datasets.py, and an FEV manifest digest with the
# plan's prefix.
VERSIONS = {
    "gift": "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0",
    "boom": "69325b544c45ff0d6c43c7a99c49a6601a01725b",
    "time": "83e3d0b3be28d11c7182bffcc1892d19b36c4da1",
    "fev": "6de245af" + hashlib.sha256(b"cascade wb11 fev manifest").hexdigest()[8:],
}

FLAT_VALUE = 3.0
NAN_AT = 10


def _walk(rng: random.Random, n: int, level: float, scale: float) -> list[float]:
    out, x = [], level
    for _ in range(n):
        x += rng.gauss(0.0, scale)
        out.append(x + rng.gauss(0.0, scale * 0.5))
    return out


def fixture_suites() -> dict:
    """``{suite: {"version": str, "series": {item_id: [float | None]}}}``.

    - gift: three random walks (80, 64, 48 points);
    - boom: two series (64, 40 points);
    - time: a random walk, a flat series (constant ``FLAT_VALUE``, 40 points)
      and a series with a NaN at index ``NAN_AT`` (48 points);
    - fev: a 31-point series (shorter than a window: no hashes) and a copy of
      boom's first series' first 40 points (cross-suite duplicates).
    """
    rng = random.Random("cascade wb11 fixture suites")
    gift = {"gift/m4_hourly/0": _walk(rng, 80, 100.0, 3.0),
            "gift/m4_hourly/1": _walk(rng, 64, -20.0, 0.7),
            "gift/electricity/7": _walk(rng, 48, 5000.0, 40.0)}
    boom = {"boom/svc-a/cpu": _walk(rng, 64, 0.4, 0.02),
            "boom/svc-b/mem": _walk(rng, 40, 1.0e6, 5.0e3)}
    nan_series: list[float | None] = _walk(rng, 48, 12.0, 1.0)
    nan_series[NAN_AT] = None
    time = {"time/traffic/3": _walk(rng, 56, 300.0, 9.0),
            "time/flat/0": [FLAT_VALUE] * 40,
            "time/gappy/1": nan_series}
    fev = {"fev/short/0": _walk(rng, 31, 7.0, 0.3),
           "fev/copy/0": list(boom["boom/svc-a/cpu"][:40])}
    return {
        "gift": {"version": VERSIONS["gift"], "series": gift},
        "boom": {"version": VERSIONS["boom"], "series": boom},
        "time": {"version": VERSIONS["time"], "series": time},
        "fev": {"version": VERSIONS["fev"], "series": fev},
    }


def shuffled(suites: dict, seed: str) -> dict:
    """The same suites with suite and item insertion order shuffled."""
    rng = random.Random(seed)
    keys = list(suites)
    rng.shuffle(keys)
    out = {}
    for k in keys:
        items = list(suites[k]["series"].items())
        rng.shuffle(items)
        out[k] = {"series": dict(items), "version": suites[k]["version"]}
    return out


def clean_series(n: int = 120, seed: str = "cascade wb11 clean") -> list[float]:
    """A series unrelated to any fixture suite."""
    return _walk(random.Random(seed), n, 42.0, 2.0)


def as_float(v) -> float:
    return float("nan") if v is None else float(v)


# ── independent reference values (not the module) ─────────────────────────────


def exact_hash(window) -> str:
    """§12b.1 exact form: sha256 of the float64 values (little-endian IEEE 754)."""
    vals = [as_float(v) for v in window]
    return hashlib.sha256(struct.pack(f"<{len(vals)}d", *vals)).hexdigest()


def sub_windows(values, window: int = WINDOW, stride: int = STRIDE):
    """``(offset, window)`` for every full sub-window without a NaN/None."""
    vals = [as_float(v) for v in values]
    for off in range(0, len(vals) - window + 1, stride):
        w = vals[off:off + window]
        if all(math.isfinite(x) for x in w):
            yield off, w


def expected_exact(suites: dict, window: int = WINDOW, stride: int = STRIDE) -> dict:
    """``{hash: [suite, "<item>@<offset>"]}``: the smallest (suite, item, offset)."""
    best: dict[str, tuple] = {}
    for suite, spec in suites.items():
        for item, values in spec["series"].items():
            for off, w in sub_windows(values, window, stride):
                h = exact_hash(w)
                key = (suite, item, off)
                if h not in best or key < best[h]:
                    best[h] = key
    return {h: [s, f"{i}@{o}"] for h, (s, i, o) in best.items()}


def window_count(values, window: int = WINDOW, stride: int = STRIDE) -> int:
    return sum(1 for _ in sub_windows(values, window, stride))


def nonconstant_count(values, window: int = WINDOW) -> int:
    return sum(1 for _, w in sub_windows(values, window) if max(w) != min(w))


def expected_items_sha256(ids) -> str:
    """W3b4 ``item_set_digest``, computed independently."""
    canon = json.dumps(sorted(set(ids)), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode("ascii")).hexdigest()


def expected_identity(suite: str, version: str, ids) -> dict:
    """W3b4 ``suite_identity`` of a suite with these item ids and version."""
    sha = expected_items_sha256(ids)
    canon = json.dumps({"items_sha256": sha, "suite": suite, "version": version},
                       sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {"suite": suite, "version": version, "items_sha256": sha,
            "n_items": len(set(ids)),
            "digest": hashlib.sha256(canon.encode("ascii")).hexdigest()}


def expected_identities(suites: dict) -> dict:
    return {k: expected_identity(k, v["version"], list(v["series"])) for k, v in suites.items()}


def canonical_bytes(obj) -> bytes:
    """The pinned canonical artefact encoding, computed independently."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def sha256_pin(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ── the module under test (lazy imports) ──────────────────────────────────────


def mod():
    import importlib

    return importlib.import_module(MODULE)


def build(suites: dict | None = None, *, version: int = 1, **kw) -> dict:
    return mod().build_hashset(fixture_suites() if suites is None else suites,
                               version=version, **kw)


def match(artefact: dict, values) -> dict:
    return mod().match_series(artefact, values)


# ── CLI ───────────────────────────────────────────────────────────────────────


def suites_json(suites: dict) -> str:
    """The CLI ``--input`` file: ``{"suites": {...}}``; a missing value is null."""
    return json.dumps({"suites": suites}, allow_nan=False)


def run_cli(*args: str, hashseed: str = "0", cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONHASHSEED"] = hashseed
    return subprocess.run([sys.executable, "-P", "-m", MODULE, *args],
                          capture_output=True, text=True, env=env,
                          cwd=str(cwd or REPO), timeout=300)


def cli_json(proc: subprocess.CompletedProcess) -> dict:
    assert proc.stdout.strip(), f"no stdout (rc={proc.returncode}):\n{proc.stderr[-3000:]}"
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(f"stdout is not one JSON object: {e}\n{proc.stdout[-2000:]}"
                             f"\n{proc.stderr[-2000:]}") from None
    assert isinstance(out, dict), out
    return out


# ── the owner's pin: evidence/wb11.json ───────────────────────────────────────

EVIDENCE_SCHEMA = "laned_wb11_pin/1"
ARTEFACT_NAME = re.compile(r"^benchmarks/contamination/hashset-v([1-9][0-9]*)\.json(\.gz)?$")
SHA_PIN = re.compile(r"^sha256:[0-9a-f]{64}$")


def lane_b_pin(root: Path, name: str) -> str | None:
    p = root / name
    if not p.is_file():
        return None
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    return (raw.get("lane_b") or {}).get("contamination_hashset")


def evidence_problems(ev, root: Path = REPO) -> list[str]:
    """Every reason the owner's WB11 pin evidence is not acceptable ([] = ok)."""
    probs: list[str] = []
    if not isinstance(ev, dict):
        return ["evidence is not a JSON object"]
    if ev.get("wp") != "WB11":
        probs.append(f"wp={ev.get('wp')!r}, expected 'WB11'")
    if ev.get("passed") is not True:
        probs.append("passed is not true")
    if ev.get("schema") != EVIDENCE_SCHEMA:
        probs.append(f"schema={ev.get('schema')!r}, expected {EVIDENCE_SCHEMA!r}")
    src = ev.get("source")
    if not isinstance(src, str) or not src or src == "fixture":
        probs.append(f"source={src!r}: must be the bench pod run, never 'fixture'")
    sha = ev.get("sha256")
    if not (isinstance(sha, str) and SHA_PIN.match(sha)):
        probs.append(f"sha256={sha!r} is not sha256:<64 hex>")
    if ev.get("rebuild_sha256") != sha:
        probs.append(f"rebuild_sha256={ev.get('rebuild_sha256')!r} != sha256 {sha!r} "
                     "(a second build from the same suite data must reproduce it)")
    version = ev.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        probs.append(f"version={version!r} is not an int >= 1")
    suites = ev.get("suites")
    if not isinstance(suites, dict) or set(suites) != set(PLAN_IDENTITIES):
        probs.append(f"suites must cover exactly {sorted(PLAN_IDENTITIES)}, got "
                     f"{sorted(suites) if isinstance(suites, dict) else suites!r}")
    else:
        for k, prefix in PLAN_IDENTITIES.items():
            ident = suites[k]
            v = ident.get("version") if isinstance(ident, dict) else None
            if not (isinstance(v, str) and re.search(rf"(?<![0-9a-f]){prefix}", v.lower())):
                probs.append(f"suite {k} version={v!r} is not the pinned identity {prefix}")
    for name, allow_empty in (("chain.toml", False), ("chain.testnet.toml", True)):
        pin = lane_b_pin(root, name)
        ok = pin == sha or (allow_empty and pin in ("", None))
        if not ok:
            probs.append(f"{name} [lane_b] contamination_hashset={pin!r} != evidence sha256 {sha!r}")
    art = ev.get("artefact")
    m = ARTEFACT_NAME.match(art) if isinstance(art, str) else None
    if m is None:
        probs.append(f"artefact={art!r} is not benchmarks/contamination/hashset-v<n>.json[.gz]")
        return probs
    if int(m.group(1)) != version:
        probs.append(f"artefact {art} names v{m.group(1)}, evidence version is {version!r}")
    path = root / art
    if not path.is_file():
        probs.append(f"artefact {art} is not in the repo")
        return probs
    m_ = mod()
    try:
        a = m_.load_artefact(path)
    except (ValueError, TypeError, KeyError, OSError) as e:
        probs.append(f"artefact {art} does not load: {type(e).__name__}: {e}")
        return probs
    got = m_.artefact_sha256(a)
    if got != sha:
        probs.append(f"artefact {art} sha256 {got} != pinned {sha}")
    if canonical_bytes(a) != m_.artefact_bytes(a) or sha256_pin(canonical_bytes(a)) != got:
        probs.append("artefact_sha256 is not sha256 over the canonical JSON")
    if a.get("schema") != SCHEMA:
        probs.append(f"artefact schema={a.get('schema')!r}")
    if a.get("version") != version:
        probs.append(f"artefact version={a.get('version')!r} != evidence {version!r}")
    if a.get("suites") != suites:
        probs.append("artefact suite identities differ from the evidence's")
    if a.get("window") != WINDOW or a.get("stride") != STRIDE or a.get("hash") != HASH:
        probs.append(f"artefact window/stride/hash = {a.get('window')}/{a.get('stride')}/"
                     f"{a.get('hash')}, expected {WINDOW}/{STRIDE}/{HASH}")
    norm = a.get("normalisation") or {}
    if norm.get("method") != "zscore" or norm.get("sig_figs") != SIG_FIGS:
        probs.append(f"artefact normalisation={norm!r}")
    if not a.get("exact") or not a.get("normalised"):
        probs.append("artefact has an empty exact or normalised set")
    return probs


def good_evidence(root: Path, *, version: int = 1, source: str = "bench_pod") -> dict:
    """Build an acceptable evidence set in ``root`` from the fixture suites:
    the artefact under benchmarks/contamination/, chain.toml pinned to it."""
    m_ = mod()
    a = m_.build_hashset(fixture_suites(), version=version)
    rel = f"benchmarks/contamination/hashset-v{version}.json"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    m_.write_artefact(a, root / rel)
    sha = m_.artefact_sha256(a)
    write_pins(root, sha, "")
    return {"wp": "WB11", "passed": True, "schema": EVIDENCE_SCHEMA, "source": source,
            "artefact": rel, "version": version, "sha256": sha, "rebuild_sha256": sha,
            "suites": copy.deepcopy(a["suites"])}


def write_pins(root: Path, main: str, testnet: str) -> None:
    (root / "chain.toml").write_text(
        f'[lane_b]\ncontamination_hashset = "{main}"\n', encoding="utf-8")
    (root / "chain.testnet.toml").write_text(
        f'[lane_b]\ncontamination_hashset = "{testnet}"\n', encoding="utf-8")
