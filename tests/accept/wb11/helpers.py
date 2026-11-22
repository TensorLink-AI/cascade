"""Shared spec and fixtures for the WB11 acceptance tests (contamination hash set).

Everything here comes from docs/plans/LANE_D_PLAN.md: the WB11 row (§3.0),
§12b.1 LB1 (exact + normalised sub-window hashes, 32-point sub-windows,
a z-score normalisation robust to scale, offset and rounding — schema 2, see
README — a versioned digest-pinned artefact refreshed only when a suite
identity changes) and the W3b4 suite identity formula
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

# Schema 2: the normalised encoding changed from v1's "z-score at 4 significant
# figures" (which misses TB11 at small scales: README "Normalised matching at
# every scale"). A v1 artefact is refused; none was ever pinned.
SCHEMA = "cascade_contamination_hashset/2"
SCHEMA_V1 = "cascade_contamination_hashset/1"
WINDOW = 32          # §12b.1: sliding sub-windows of 32 points
STRIDE = 1           # §12b.1: sliding
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

# ── low-information windows (RV6 B5) ─────────────────────────────────────────
#
# A window that carries almost no shape is not evidence of contamination: all
# zeros, a one-hot list, a counter ramp or a value plus float noise also turn
# up in benign code. Such a window is in NEITHER set at build time, and a
# submission window that is low-information is never counted as a hit (exact
# or normalised), whatever the artefact holds. README "Low-information
# windows" states the rule and why both sides apply it. A window is
# low-information when ANY of these holds:
MIN_DISTINCT = 4            # (a) fewer than 4 distinct values (constant, one-hot, binary, 3-level)
MODE_SHARE = (3, 4)         # (b) one value fills >= 3/4 of the window (24 of 32)
STD_REL_FLOOR = 2.0 ** -46  # (c) std <= 2^-46 * |mean|: under 64 float64 ulps of its level
AFFINE_RESIDUAL = 1e-6      # (d) a least-squares line in the index explains all but 1e-6 of the variance


def low_information(w) -> bool:
    """The pinned low-information rule, computed independently of the module."""
    n = len(w)
    if len(set(w)) < MIN_DISTINCT:
        return True
    counts: dict[float, int] = {}
    for x in w:
        counts[x] = counts.get(x, 0) + 1
    if max(counts.values()) * MODE_SHARE[1] >= MODE_SHARE[0] * n:
        return True
    mean = math.fsum(w) / n
    dev = [x - mean for x in w]
    ss_tot = math.fsum(d * d for d in dev)
    std = math.sqrt(ss_tot / n)
    if not std > STD_REL_FLOOR * abs(mean):
        return True
    t = [i - (n - 1) / 2 for i in range(n)]
    slope = math.fsum(ti * d for ti, d in zip(t, dev, strict=True)) / math.fsum(ti * ti for ti in t)
    ss_res = math.fsum((d - slope * ti) ** 2 for ti, d in zip(t, dev, strict=True))
    return ss_res <= AFFINE_RESIDUAL * ss_tot


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


# ── scale fixture: TB11 evasions at every series level ───────────────────────

# (suite, item, level, step scale): the four levels WB13's fixture embeds
# (boom 0.4, fev 7, gift 100, gift 5000), with the same walk parameters.
SCALE_ITEMS = (
    ("boom", "boom/scale/0.4", 0.4, 0.02),
    ("fev", "fev/scale/7", 7.0, 0.3),
    ("gift", "gift/scale/100", 100.0, 3.0),
    ("time", "time/scale/5000", 5000.0, 40.0),
)
SCALE_LEN = 48       # 17 sub-windows per item
ROUND_DP = 3         # TB11: "rounded to 3 dp"
ROUND_STEP = 10.0 ** -ROUND_DP


def scale_suites() -> dict:
    """One random-walk item per level, in its own suite (seeded, in memory)."""
    rng = random.Random("cascade wb11 scale suites")
    out = {}
    for suite, item, level, scale in SCALE_ITEMS:
        out[suite] = {"version": VERSIONS[suite],
                      "series": {item: _walk(rng, SCALE_LEN, level, scale)}}
    return out


def unrelated_series(level: float, scale: float, k: int) -> list[float]:
    """The k-th series unrelated to any fixture suite at this level: a random
    walk with the suite's own walk parameters (even k) or i.i.d. Gaussian noise
    around the level (odd k). Different seeds from every suite series."""
    rng = random.Random(f"cascade wb11 unrelated {level} {k}")
    if k % 2 == 0:
        return _walk(rng, SCALE_LEN, level, scale)
    return [level + rng.gauss(0.0, scale * 3.0) for _ in range(SCALE_LEN)]


def to_float32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def window_std(w) -> float:
    """Population std of one window."""
    n = len(w)
    mean = math.fsum(w) / n
    return math.sqrt(math.fsum((x - mean) ** 2 for x in w) / n)


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
    """``{hash: [suite, "<item>@<offset>"]}``: the smallest (suite, item, offset),
    low-information windows left out (RV6 B5)."""
    best: dict[str, tuple] = {}
    for suite, spec in suites.items():
        for item, values in spec["series"].items():
            for off, w in sub_windows(values, window, stride):
                if low_information(w):
                    continue
                h = exact_hash(w)
                key = (suite, item, off)
                if h not in best or key < best[h]:
                    best[h] = key
    return {h: [s, f"{i}@{o}"] for h, (s, i, o) in best.items()}


def window_count(values, window: int = WINDOW, stride: int = STRIDE) -> int:
    return sum(1 for _ in sub_windows(values, window, stride))


def nonconstant_count(values, window: int = WINDOW) -> int:
    return sum(1 for _, w in sub_windows(values, window) if max(w) != min(w))


def informative_count(values, window: int = WINDOW) -> int:
    """Sub-windows that are not low-information: the ones that can hit."""
    return sum(1 for _, w in sub_windows(values, window) if not low_information(w))


# ── benign-hazard fixture (RV6 B5) ───────────────────────────────────────────


def hazard_suites() -> dict:
    """Suites with the low-information shapes real GIFT/BOOM data has (a zero
    run: solar at night; a sparse 0/1 counter; an exact index counter; a value
    plus float noise) next to informative series a benign repo can resemble (a
    noisy spike train, a noisy trend). Seeded, in memory."""
    rng = random.Random("cascade wb11 hazard suites")
    solar = _walk(rng, 100, 50.0, 2.0)
    solar[30:70] = [0.0] * 40                                  # a 40-zero run
    spikes = [rng.gauss(0.0, 1e-3) for _ in range(96)]         # noisy spike train
    for i in (0, 40, 80):
        spikes[i] = 1.0
    counter = [1.0 if i % 40 == 0 else 0.0 for i in range(96)] # sparse exact counter
    ramp = [float(i) for i in range(64)]                       # an exact index counter
    trend = [0.5 * i + rng.gauss(0.0, 0.05) for i in range(64)]
    still = [1000.0 + rng.gauss(0.0, 1e-12) for _ in range(48)]  # 1000 + float noise
    return {
        "gift": {"version": VERSIONS["gift"], "series": {"gift/solar/0": solar}},
        "boom": {"version": VERSIONS["boom"],
                 "series": {"boom/spikes/0": spikes, "boom/counter/0": counter}},
        "time": {"version": VERSIONS["time"],
                 "series": {"time/ramp/0": ramp, "time/trend/0": trend}},
        "fev": {"version": VERSIONS["fev"], "series": {"fev/still/0": still}},
    }


def benign_inputs() -> dict:
    """Value streams a benign generator carries, as WB13 would extract them
    (README "Low-information windows")."""
    one_hot = []
    for hot in (0, 5, 17):
        one_hot += [1 if i == hot else 0 for i in range(WINDOW)]
    return {
        "zeros_1kb_f64": [0.0] * (1024 // 8),
        "x_eq_0_lines": [0] * WINDOW,
        "one_hot_lists": one_hot,
        "linspace_ramp": [i / (WINDOW - 1) for i in range(WINDOW)],
        "range_ints": list(range(48)),
        "near_constant": [1000.0 + 1e-12 * ((i * 7919) % 13 - 6) for i in range(64)],
    }


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


# ── memory budget of the intake's lookup (RV6 B8, README "Memory budget") ─────

# The intake runs on the 7 GB production box next to the validator. The
# lookup structure (load_lookup) may use at most this much Python heap at its
# peak while loading, per suite window, plus a fixed allowance.
LOOKUP_PEAK_BYTES_PER_WINDOW = 64
LOOKUP_FIXED_BYTES = 1 << 20
# Process RSS growth over the load (catches allocations tracemalloc misses and
# mapped file pages): twice the heap budget plus a fixed allowance.
RSS_BYTES_PER_WINDOW = 2 * LOOKUP_PEAK_BYTES_PER_WINDOW
RSS_FIXED_BYTES = 16 << 20
# The committed artefact as write_artefact writes hashset-v<n>.json.gz.
FILE_BYTES_PER_WINDOW = 96
FILE_FIXED_BYTES = 64 << 10
# Absolute cap for the box whatever N is: 1.5 GiB. With the per-window budget
# this admits at most (1.5 GiB - 1 MiB) / 64 B ~ 25.1M suite windows.
BOX_LOOKUP_CAP_BYTES = 3 << 29
SYNTH_WINDOWS = 50_000          # 4 items x 12 531 points: 4 x 12 500 windows
SYNTH_ITEM_LEN = SYNTH_WINDOWS // 4 + WINDOW - 1


def lookup_budget(n_windows: int) -> int:
    return LOOKUP_PEAK_BYTES_PER_WINDOW * n_windows + LOOKUP_FIXED_BYTES


def file_budget(n_windows: int) -> int:
    return FILE_BYTES_PER_WINDOW * n_windows + FILE_FIXED_BYTES


def synth_suites(item_len: int = SYNTH_ITEM_LEN) -> dict:
    """Four seeded random-walk items (one per suite): informative windows only,
    so ``item_len - 31`` windows per item reach both sets."""
    rng = random.Random("cascade wb11 synthetic scale")
    return {s: {"version": VERSIONS[s], "series": {f"{s}/synth/0": _walk(rng, item_len, 100.0, 3.0)}}
            for s in SUITE_KEYS}


# Child processes: build in one, load in a fresh one, so the measured process
# holds nothing but the lookup.
_BUILD_CHILD = """
import json, sys
from cascade.eval import contamination as c
from tests.accept.wb11.helpers import synth_suites
a = c.build_hashset(synth_suites(), version=1)
sha = c.write_artefact(a, sys.argv[1])
print(json.dumps({"sha256": sha, "n_exact": len(a["exact"]), "n_normalised": len(a["normalised"])}))
"""

_LOAD_CHILD = """
import gc, json, resource, sys, tracemalloc
from cascade.eval import contamination as c
from tests.accept.wb11.helpers import synth_suites, WINDOW
seg = synth_suites()["gift"]["series"]["gift/synth/0"][100:100 + WINDOW + 8]
load = getattr(c, "load_lookup", None)
if not callable(load):
    print(json.dumps({"missing": "load_lookup"}))
    sys.exit(0)
gc.collect()
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
tracemalloc.start()
lk = load(sys.argv[1], expect_sha256=sys.argv[2])
peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()
rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
r = c.match_series(lk, seg)
print(json.dumps({"peak": peak, "rss_delta": rss1 - rss0, "version": lk["version"],
                  "window": lk["window"], "match": r,
                  "contaminated": c.is_contaminated(lk, seg)}))
"""


def run_child(code: str, *args: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    proc = subprocess.run([sys.executable, "-P", "-c", code, *args], capture_output=True,
                          text=True, env=env, cwd=str(REPO), timeout=600)
    assert proc.returncode == 0, f"child rc={proc.returncode}\n{proc.stderr[-3000:]}"
    return cli_json(proc)


def lookup_peak_bytes(path, sha256: str) -> int:
    """Traced heap peak of ``load_lookup(path, expect_sha256=sha256)`` in this
    process (small artefacts only: the evidence fixture)."""
    import tracemalloc

    m_ = mod()
    load = getattr(m_, "load_lookup", None)
    assert callable(load), ("cascade.eval.contamination has no load_lookup: the intake's "
                            "memory-bounded lookup (RV6 B8, README 'Memory budget')")
    tracemalloc.start()
    try:
        load(path, expect_sha256=sha256)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


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
    if norm.get("method") != "zscore" or norm.get("scope") != "subwindow":
        probs.append(f"artefact normalisation={norm!r}")
    if not a.get("exact") or not a.get("normalised"):
        probs.append("artefact has an empty exact or normalised set")
    probs += _budget_problems(ev, a, path)
    return probs


def _budget_problems(ev: dict, a: dict, path: Path) -> list[str]:
    """RV6 B8: the pin records the artefact's scale and the lookup's measured
    load peak, and both stay inside the box's budget."""
    probs: list[str] = []
    counts = {}
    for key, set_name in (("n_exact", "exact"), ("n_normalised", "normalised")):
        v = ev.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            probs.append(f"{key}={v!r} is not an int >= 1")
            continue
        counts[key] = v
        have = len(a.get(set_name) or {})
        if v != have:
            probs.append(f"{key}={v} but the artefact's {set_name} set has {have}")
    n = max(counts.values()) if counts else 0
    size = path.stat().st_size
    fb = ev.get("file_bytes")
    if fb != size or isinstance(fb, bool):
        probs.append(f"file_bytes={fb!r} != the artefact file's {size} bytes")
    if n and size > file_budget(n):
        probs.append(f"artefact file {size} B > {FILE_BYTES_PER_WINDOW} B/window x {n} + "
                     f"{FILE_FIXED_BYTES} B")
    peak = ev.get("lookup_peak_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 1:
        probs.append(f"lookup_peak_bytes={peak!r} is not a measured int >= 1")
    else:
        if n and peak > lookup_budget(n):
            probs.append(f"lookup_peak_bytes={peak} > {LOOKUP_PEAK_BYTES_PER_WINDOW} B/window x "
                         f"{n} + {LOOKUP_FIXED_BYTES} B")
        if peak > BOX_LOOKUP_CAP_BYTES:
            probs.append(f"lookup_peak_bytes={peak} > the box cap {BOX_LOOKUP_CAP_BYTES} B")
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
            "suites": copy.deepcopy(a["suites"]),
            "n_exact": len(a["exact"]), "n_normalised": len(a["normalised"]),
            "file_bytes": (root / rel).stat().st_size,
            "lookup_peak_bytes": lookup_peak_bytes(root / rel, sha)}


def write_pins(root: Path, main: str, testnet: str) -> None:
    (root / "chain.toml").write_text(
        f'[lane_b]\ncontamination_hashset = "{main}"\n', encoding="utf-8")
    (root / "chain.testnet.toml").write_text(
        f'[lane_b]\ncontamination_hashset = "{testnet}"\n', encoding="utf-8")
