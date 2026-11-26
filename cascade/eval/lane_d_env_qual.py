"""B5 qualification kit for a lane D scoring environment (LANE_D_PLAN rev6.18
B5; deploy/laned/scoring_envs.toml header).

A ``candidate`` environment becomes ``qualified`` only with an evidence file
``evidence/lane_d_env_<name>.json`` (schema :data:`EVIDENCE_SCHEMA`) that
shows, on that GPU:

1. **W0b** context equivalence passes in the environment (a pytest junit XML
   of ``tests/accept/w0b``: every required torch test ran and passed);
2. **golden set**: :data:`HALF_MIN` windows per half (500 live + 500 public),
   :data:`MIN_MODELS` models, every :data:`CONTEXTS` value: each per-window
   ``crps`` and ``mase`` is within relative :data:`REL_TOL` of the
   ``cu124-torch241`` reference run, and the replayed cohort (model 0 =
   KING_D, the others members; clusters = source; the settlement's max-T
   bootstrap :func:`cascade.validator.lane_d.judge_cohort`) gives identical
   crown decisions at both ends of the crown bar;
3. **A9 timing**: a ``laned_t7/1`` pod document measured on that GPU at
   :data:`A9_WINDOWS` windows x :data:`A9_MODELS` models that passes (each
   row re-derived through :func:`cascade.eval.lane_d_t7.build_evidence`).

Subcommands (``python -P -m cascade.eval.lane_d_env_qual ...``):

``golden``    build the golden window set from ``pool_d/1`` directories, one
              per half (numpy only; any host).
``forecast``  forecast the golden set with N checkpoints at every context in
              the current venv on the current GPU (POD ONLY: needs torch).
``compare``   score both runs, replay the cohort, attach W0b + A9 and write
              the evidence file (numpy only).
``check``     re-derive an evidence file against its scoring_envs.toml entry.

:func:`evidence_problem` is what the W4 acceptance test runs on every
committed qualified entry, so a hand-edited or stale evidence file (another
lock, another GPU, a failed threshold) cannot qualify an environment. It
recomputes every verdict from the per-window scores the file carries; no
pass/fail flag is trusted.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

GOLDEN_SCHEMA = "laned_golden_set/1"
RUN_SCHEMA = "laned_golden_run/1"
EVIDENCE_SCHEMA = "laned_env_qual/1"
REFERENCE_ENV = "cu124-torch241"
REL_TOL = 1e-3
HALF_MIN = {"live": 500, "public": 500}
MIN_MODELS = 3
CONTEXTS = (4096, 8192, 16384)       # the launch variants (lane_d_t7.LAUNCH_VARIANTS)
A9_WINDOWS = 15000
A9_MODELS = 10
MIN_HISTORY = 64
# W0b tests that must run AND pass (torch-marked: a skip means torch was absent).
W0B_REQUIRED = tuple(
    f"{t}[{k}]" for t in ("test_scorer_at_4096_equals_pyramid_wrapper_quantile_head_per_window",
                          "test_scorer_at_4096_equals_pyramid_wrapper_quantile_head_batched",
                          "test_4096_checkpoint_loads_strict_at_every_context_variant")
    for k in ("f32", "bf16")) + ("test_tolerance_discriminates_the_wrapper_edge_pad",)
REPO = Path(__file__).resolve().parents[2]


class QualError(ValueError):
    """An input cannot produce honest evidence; nothing is written."""


# ── hashing ─────────────────────────────────────────────────────────────────


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _arrays_sha(arrays: Sequence[np.ndarray]) -> str:
    h = hashlib.sha256(b"laned-golden-arrays/1")
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(_canon(list(a.shape)))
        h.update(a.tobytes())
    return h.hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── golden set ──────────────────────────────────────────────────────────────


def _usable(w: dict, horizon: int) -> bool:
    h = np.asarray(w["history"], dtype=np.float64)
    y = np.asarray(w["target"], dtype=np.float64)
    if h.ndim not in (1, 2) or y.ndim != h.ndim or y.shape[-1] != horizon:
        return False
    if h.ndim == 2 and (h.shape[0] != y.shape[0] or h.shape[0] < 2):
        return False
    if h.shape[-1] < MIN_HISTORY or not (np.isfinite(h).all() and np.isfinite(y).all()):
        return False
    return bool((np.abs(y).sum(axis=-1) > 0).any())


def pick_golden(halves: Mapping[str, list[dict]], *, horizon: int, per_half: int,
                seed: str) -> list[dict]:
    """Deterministic pick: per half, the usable windows ranked by
    ``sha256(seed|half|window key)``, first ``per_half``. Refuses a short half."""
    out = []
    for half in sorted(halves):
        cands = [w for w in halves[half] if _usable(w, horizon)]

        def rank(w, half=half):
            key = w.get("window_id") or w["series_id"]
            return hashlib.sha256(f"{seed}|{half}|{key}".encode()).hexdigest()

        cands.sort(key=rank)
        if len(cands) < per_half:
            raise QualError(f"half {half!r}: {len(cands)} usable windows < {per_half}")
        for w in cands[:per_half]:
            out.append({"half": half, "series_id": str(w["series_id"]),
                        "window_id": str(w.get("window_id") or ""),
                        "source": str(w["source"]),
                        "seasonal_period": int(w.get("seasonal_period") or 1),
                        "history": np.asarray(w["history"], dtype=np.float64),
                        "target": np.asarray(w["target"], dtype=np.float64)})
    return out


def golden_digest(meta_windows: list[dict], hists, tgts, horizon: int) -> str:
    h = hashlib.sha256(b"laned-golden-set/1")
    h.update(_canon({"horizon": int(horizon), "windows": meta_windows}))
    h.update(_arrays_sha(hists).encode())
    h.update(_arrays_sha(tgts).encode())
    return h.hexdigest()


def write_golden(windows: list[dict], out_dir, *, horizon: int, seed: str,
                 pools: Mapping[str, dict]) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_w = [{k: w[k] for k in ("half", "series_id", "window_id", "source", "seasonal_period")}
              for w in windows]
    hists = [w["history"] for w in windows]
    tgts = [w["target"] for w in windows]
    np.savez(out / "golden.npz", **{f"h{i}": a for i, a in enumerate(hists)},
             **{f"t{i}": a for i, a in enumerate(tgts)})
    meta = {"schema": GOLDEN_SCHEMA, "horizon": int(horizon), "seed": str(seed),
            "pools": {k: dict(v) for k, v in sorted(pools.items())}, "windows": meta_w,
            "sha256": golden_digest(meta_w, hists, tgts, horizon)}
    (out / "golden.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return meta


def load_golden(golden_dir) -> tuple[dict, list[np.ndarray], list[np.ndarray]]:
    d = Path(golden_dir)
    meta = json.loads((d / "golden.json").read_text(encoding="utf-8"))
    if meta.get("schema") != GOLDEN_SCHEMA:
        raise QualError(f"{d}: golden schema {meta.get('schema')!r} != {GOLDEN_SCHEMA}")
    n = len(meta["windows"])
    with np.load(d / "golden.npz") as z:
        hists = [np.asarray(z[f"h{i}"], dtype=np.float64) for i in range(n)]
        tgts = [np.asarray(z[f"t{i}"], dtype=np.float64) for i in range(n)]
    if golden_digest(meta["windows"], hists, tgts, meta["horizon"]) != meta["sha256"]:
        raise QualError(f"{d}: golden set bytes do not hash to its sha256")
    return meta, hists, tgts


# ── forecast run (pod) ──────────────────────────────────────────────────────


def _host_record(env: str, probe: Callable | None) -> dict:
    from ..shared.config.lane_d import host_identity
    from .lane_d_handshake import probe_environment

    p = (probe or probe_environment)(env)
    host = {k: str(p.get(k) or "") for k in ("env", "gpu_model", "driver", "torch_build",
                                              "fingerprint")}
    host["identity"] = host_identity(host["env"], host["gpu_model"], host["driver"],
                                     host["torch_build"], host["fingerprint"])
    return host


def forecast_run(golden_dir, checkpoints: Mapping[str, Path], *, env: str, out_dir,
                 contexts: Sequence[int] = CONTEXTS, device: str = "cuda",
                 forecast: Callable | None = None, probe: Callable | None = None,
                 envs=None, git_head: str = "") -> dict:
    """Forecast the golden set with every checkpoint at every context; write
    ``run.npz`` (quantiles, key ``<model>@<context>``) and ``run.json``."""
    from ..shared.config.lane_d import host_record_problem, load_scoring_envs

    meta, hists, _ = load_golden(golden_dir)
    if len(checkpoints) < MIN_MODELS:
        raise QualError(f"{len(checkpoints)} checkpoints < {MIN_MODELS}")
    by_name = {e.name: e for e in (envs if envs is not None else load_scoring_envs())}
    if env not in by_name:
        raise QualError(f"env {env!r} is not in the scoring environment list")
    e = by_name[env]
    lock_sha = file_sha256(REPO / e.lock) if e.lock else ""
    host = _host_record(env, probe)
    why = host_record_problem(host)
    if why or host["torch_build"] != e.torch_build or host["gpu_model"] not in e.gpu_models:
        raise QualError(f"this host does not run {env}: {why or host}")
    if forecast is None:
        from .lane_d_scorer import forecast_quantiles as forecast
    arrays, models = {}, []
    for name, d in checkpoints.items():
        d = Path(d)
        models.append({"name": str(name),
                       "weights_sha256": file_sha256(d / "weights.safetensors"),
                       "config_sha256": file_sha256(d / "config.json")})
        for c in contexts:
            q = np.asarray(forecast(d, hists, context=int(c), horizon=int(meta["horizon"]),
                                    device=device), dtype=np.float64)
            if not np.isfinite(q).all():
                raise QualError(f"{name}@{c}: non-finite quantiles")
            arrays[f"{name}@{int(c)}"] = q
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "run.npz", **arrays)
    run = {"schema": RUN_SCHEMA, "env": env, "host": host, "lock": e.lock,
           "lock_sha256": lock_sha, "golden_sha256": meta["sha256"], "device": device,
           "contexts": [int(c) for c in contexts], "models": models, "git_head": git_head,
           "quantiles_sha256": _arrays_sha([arrays[k] for k in sorted(arrays)])}
    (out / "run.json").write_text(json.dumps(run, indent=1, sort_keys=True) + "\n")
    return run


def load_run(run_dir) -> tuple[dict, dict[str, np.ndarray]]:
    d = Path(run_dir)
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    if run.get("schema") != RUN_SCHEMA:
        raise QualError(f"{d}: run schema {run.get('schema')!r} != {RUN_SCHEMA}")
    with np.load(d / "run.npz") as z:
        arrays = {k: np.asarray(z[k], dtype=np.float64) for k in z.files}
    if _arrays_sha([arrays[k] for k in sorted(arrays)]) != run["quantiles_sha256"]:
        raise QualError(f"{d}: quantiles do not hash to quantiles_sha256")
    return run, arrays


# ── scores, cohort replay, W0b, A9 ──────────────────────────────────────────


def score_run(meta: dict, hists, tgts, arrays: Mapping[str, np.ndarray]) -> dict:
    """``{"<model>@<ctx>": {"crps": [...], "mase": [...]}}`` via the lane D
    ``score_window`` (forecast rows are window-major, channel-minor)."""
    from .lane_d_scorer import forecast_rows, score_window

    offs = np.concatenate([[0], np.cumsum(forecast_rows(hists))]).astype(int)
    out = {}
    for key in sorted(arrays):
        q = arrays[key]
        if q.shape[0] != offs[-1]:
            raise QualError(f"{key}: {q.shape[0]} forecast rows != {offs[-1]}")
        crps, mase = [], []
        for i, w in enumerate(meta["windows"]):
            qi = q[offs[i]:offs[i + 1]]
            qi = qi[0] if hists[i].ndim == 1 else qi
            c, m = score_window(qi, tgts[i], hists[i], int(w["seasonal_period"]))
            crps.append(float(c))
            mase.append(float(m))
        out[key] = {"crps": crps, "mase": mase}
    return out


def rel_diff(ref, cand) -> float:
    r = np.asarray(ref, dtype=np.float64)
    c = np.asarray(cand, dtype=np.float64)
    den = np.maximum(np.maximum(np.abs(r), np.abs(c)), 1e-12)
    return float(np.max(np.abs(c - r) / den)) if r.size else 0.0


def replay_crowns(scores: Mapping[str, dict], models: Sequence[str], contexts, clusters,
                  lane_d: Mapping, seed: str) -> dict:
    """Per context and bar: the cohort verdict (model 0 = KING_D)."""
    from ..shared.config.lane_d import LaneDConfig
    from ..validator.lane_d import judge_cohort

    # rev6.18 A3 (F4): the judge reads the per-window ratio clamp and the
    # public weight. The golden set is live windows only, so the replay judges
    # the live half alone (public_weight 0); the clamp is the deployed
    # config's when given, else the release default (LaneDConfig).
    cfg = argparse.Namespace(
        alpha=float(lane_d["alpha"]), bootstrap_B=int(lane_d["bootstrap_B"]),
        d_margin=float(lane_d["d_margin"]), public_weight=0.0,
        window_ratio_floor=float(lane_d.get("window_ratio_floor",
                                            LaneDConfig.window_ratio_floor)))
    out = {}
    for c in contexts:
        def sc(m, c=c):
            s = scores[f"{m}@{int(c)}"]
            return (np.asarray(s["crps"]), np.asarray(s["mase"]))

        row = {}
        for bar_name in ("d_margin", "d_margin_start"):
            res = judge_cohort(sc(models[0]), [sc(m) for m in models[1:]], clusters=list(clusters),
                               cfg=cfg, seed=f"{seed}|{int(c)}", bar=float(lane_d[bar_name]))
            row[bar_name] = res
        out[str(int(c))] = row
    return out


def parse_junit(path) -> dict:
    root = ET.parse(str(path)).getroot()
    cases = []
    for tc in root.iter("testcase"):
        status = "passed"
        for tag in ("failure", "error", "skipped"):
            if tc.find(tag) is not None:
                status = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
        cases.append({"name": tc.get("name", ""), "classname": tc.get("classname", ""),
                      "status": status})
    return {"file_sha256": file_sha256(path), "cases": cases}


def w0b_problem(w0b) -> str | None:
    if not isinstance(w0b, dict) or not isinstance(w0b.get("cases"), list) or not w0b["cases"]:
        return "W0b: no test cases recorded"
    bad = [c["name"] for c in w0b["cases"] if c.get("status") != "passed"]
    if bad:
        return f"W0b: not passed: {bad[:5]}"
    names = {c["name"] for c in w0b["cases"]}
    missing = [n for n in W0B_REQUIRED if n not in names]
    return f"W0b: required torch tests did not run: {missing}" if missing else None


def a9_problem(t7, gpu_model: str) -> str | None:
    """The A9 document must be a passing pod ``laned_t7/1`` on ``gpu_model``
    at ``A9_WINDOWS`` x ``A9_MODELS``, every row re-derived."""
    from .lane_d_t7 import POD_SOURCE, SCHEMA, build_evidence

    if not isinstance(t7, dict) or t7.get("schema") != SCHEMA:
        return f"A9: not a {SCHEMA} document"
    if t7.get("source") != POD_SOURCE:
        return "A9: not pod evidence"
    try:
        redo = build_evidence(t7["rows"], short_window=t7["short_window"], tool=t7["tool"],
                              torch_version=t7["torch"], source=t7["source"])
    except (KeyError, TypeError, ValueError) as e:
        return f"A9: rows do not re-derive: {e}"
    if not redo["passed"]:
        return "A9: the T7 document does not pass (4096 must fit, short-window identical)"
    for r in redo["rows"]:
        if r["gpu"] != gpu_model:
            return f"A9: row gpu {r['gpu']!r} != {gpu_model!r}"
        if r["n_windows"] < A9_WINDOWS or r["n_models"] < A9_MODELS:
            return (f"A9: variant {r['variant']} measured at {r['n_windows']} windows x "
                    f"{r['n_models']} models < {A9_WINDOWS} x {A9_MODELS}")
    return None


# ── evidence ────────────────────────────────────────────────────────────────


def compare(golden_dir, reference_dir, candidate_dir, *, lane_d: Mapping, seed: str,
            w0b_junit, a9_t7_path) -> dict:
    """The ``laned_env_qual/1`` evidence of the candidate run vs the reference."""
    meta, hists, tgts = load_golden(golden_dir)
    ref, ref_q = load_run(reference_dir)
    cand, cand_q = load_run(candidate_dir)
    for r, who in ((ref, "reference"), (cand, "candidate")):
        if r["golden_sha256"] != meta["sha256"]:
            raise QualError(f"{who} run forecast another golden set")
    if [m["weights_sha256"] for m in ref["models"]] != [m["weights_sha256"] for m in cand["models"]]:
        raise QualError("the runs used different checkpoints")
    if ref["contexts"] != cand["contexts"] or sorted(ref_q) != sorted(cand_q):
        raise QualError("the runs cover different (model, context) cells")
    t7 = json.loads(Path(a9_t7_path).read_text(encoding="utf-8"))
    doc = {
        "schema": EVIDENCE_SCHEMA, "env": cand["env"], "seed": str(seed),
        "lane_d": {k: lane_d[k] for k in ("alpha", "bootstrap_B", "d_margin", "d_margin_start")},
        "golden": {"sha256": meta["sha256"], "horizon": meta["horizon"], "pools": meta["pools"],
                   "halves": [w["half"] for w in meta["windows"]],
                   "clusters": [w["source"] for w in meta["windows"]]},
        "models": [m["name"] for m in cand["models"]],
        "checkpoints": cand["models"], "contexts": cand["contexts"],
        "reference": {k: ref[k] for k in ("env", "host", "lock", "lock_sha256", "git_head",
                                          "quantiles_sha256")},
        "candidate": {k: cand[k] for k in ("env", "host", "lock", "lock_sha256", "git_head",
                                           "quantiles_sha256")},
        "scores": {"reference": score_run(meta, hists, tgts, ref_q),
                   "candidate": score_run(meta, hists, tgts, cand_q)},
        "w0b": parse_junit(w0b_junit),
        "a9": t7,
    }
    doc["verdict"] = derive_verdict(doc)
    return doc


def derive_verdict(doc: Mapping) -> dict:
    """Every pass/fail of the evidence, recomputed from its contents."""
    sref, scand = doc["scores"]["reference"], doc["scores"]["candidate"]
    worst = 0.0
    for key in sorted(sref):
        for metric in ("crps", "mase"):
            worst = max(worst, rel_diff(sref[key][metric], scand[key][metric]))
    crowns = {who: replay_crowns(doc["scores"][who], doc["models"], doc["contexts"],
                                 doc["golden"]["clusters"], doc["lane_d"], doc["seed"])
              for who in ("reference", "candidate")}
    same = all(
        [m["cleared"] for m in crowns["reference"][c][b]]
        == [m["cleared"] for m in crowns["candidate"][c][b]]
        for c in crowns["reference"] for b in crowns["reference"][c])
    lcb_gap = max(abs(a["lcb"] - b["lcb"]) for c in crowns["reference"]
                  for bar in crowns["reference"][c]
                  for a, b in zip(crowns["reference"][c][bar], crowns["candidate"][c][bar],
                                  strict=True))
    w0b = w0b_problem(doc["w0b"])
    a9 = a9_problem(doc["a9"], doc["candidate"]["host"]["gpu_model"])
    ok = worst <= REL_TOL and same and w0b is None and a9 is None
    return {"max_rel_diff": worst, "rel_tol": REL_TOL, "crowns": crowns["candidate"],
            "crowns_identical": same, "max_lcb_gap": lcb_gap, "w0b": w0b or "passed",
            "a9": a9 or "passed", "passed": bool(ok)}


def evidence_problem(doc, env, *, envs) -> str | None:
    """Why ``doc`` does not qualify ``env`` (a :class:`ScoringEnv`), or
    ``None``. ``envs`` is the committed list (the reference must be in it,
    qualified). Recomputes every verdict; trusts no flag in ``doc``."""
    from ..shared.config.lane_d import host_problem

    if not isinstance(doc, dict) or doc.get("schema") != EVIDENCE_SCHEMA:
        return f"evidence schema is not {EVIDENCE_SCHEMA}"
    if doc.get("env") != env.name or doc["candidate"]["env"] != env.name:
        return f"evidence is for {doc.get('env')!r}, not {env.name!r}"
    if not env.lock or doc["candidate"]["lock"] != env.lock \
            or doc["candidate"]["lock_sha256"] != env.lock_sha256:
        return "evidence was measured with another lock than the entry pins"
    stand_in = dataclasses.replace(env, status="qualified", from_block=0, until_block=0)
    why = host_problem(doc["candidate"]["host"], [stand_in], 1)
    if why:
        return f"candidate host: {why}"
    ref_env = {e.name: e for e in envs}.get(REFERENCE_ENV)
    if ref_env is None or ref_env.status != "qualified" or doc["reference"]["env"] != REFERENCE_ENV:
        return f"the reference run is not {REFERENCE_ENV}"
    if doc["reference"]["lock_sha256"] != ref_env.lock_sha256:
        return "the reference run used another release lock"
    ref_host = dataclasses.replace(ref_env, from_block=0, until_block=0)
    why = host_problem(doc["reference"]["host"], [ref_host], 1)
    if why:
        return f"reference host: {why}"
    halves = doc["golden"]["halves"]
    n = len(halves)
    if len(doc["golden"]["clusters"]) != n:
        return "golden clusters / halves differ in length"
    for half, need in HALF_MIN.items():
        if halves.count(half) < need:
            return f"golden half {half!r}: {halves.count(half)} windows < {need}"
    if len(doc["models"]) < MIN_MODELS or len(set(doc["models"])) != len(doc["models"]):
        return f"golden set needs >= {MIN_MODELS} distinct models"
    if len({m["weights_sha256"] for m in doc["checkpoints"]}) != len(doc["checkpoints"]):
        return "golden models are not distinct checkpoints"
    if not set(CONTEXTS) <= set(doc["contexts"]):
        return f"golden contexts {doc['contexts']} lack {sorted(set(CONTEXTS))}"
    want = {f"{m}@{int(c)}" for m in doc["models"] for c in doc["contexts"]}
    for who in ("reference", "candidate"):
        s = doc["scores"][who]
        if set(s) != want or any(len(s[k][m]) != n for k in s for m in ("crps", "mase")):
            return f"{who} scores do not cover every (model, context, window)"
        if not all(math.isfinite(x) for k in s for m in ("crps", "mase") for x in s[k][m]):
            return f"{who} scores are not finite"
    v = derive_verdict(doc)
    if v["max_rel_diff"] > REL_TOL:
        return f"per-window score rel. diff {v['max_rel_diff']:.3g} > {REL_TOL}"
    if not v["crowns_identical"]:
        return "the replayed cohort's crown decisions differ from the reference"
    if v["w0b"] != "passed":
        return v["w0b"]
    if v["a9"] != "passed":
        return v["a9"]
    return None


# ── CLI ─────────────────────────────────────────────────────────────────────


def _kv(text: str) -> tuple[str, str]:
    k, sep, v = text.partition("=")
    if not sep or not k or not v:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {text!r}")
    return k, v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cascade.eval.lane_d_env_qual", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("golden", help="build the golden set from pool_d/1 dirs")
    g.add_argument("--half", type=_kv, action="append", required=True,
                   help="HALF=POOL_DIR (live=..., public=...)")
    g.add_argument("--per-half", type=int, default=500)
    g.add_argument("--horizon", type=int, default=64)
    g.add_argument("--seed", required=True)
    g.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("forecast", help="POD: forecast the golden set in this venv")
    f.add_argument("--golden", type=Path, required=True)
    f.add_argument("--checkpoint", type=_kv, action="append", required=True,
                   help="NAME=DIR, model 0 = KING_D; >= 3")
    f.add_argument("--env", required=True)
    f.add_argument("--contexts", default=",".join(map(str, CONTEXTS)))
    f.add_argument("--device", default="cuda")
    f.add_argument("--git-head", default="")
    f.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare", help="write the evidence file")
    c.add_argument("--golden", type=Path, required=True)
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--candidate", type=Path, required=True)
    c.add_argument("--w0b-junit", type=Path, required=True)
    c.add_argument("--a9", type=Path, required=True, help="laned_t7/1 pod document")
    c.add_argument("--chain-config", type=Path, default=REPO / "chain.toml")
    c.add_argument("--seed", required=True)
    c.add_argument("--out", type=Path, required=True)
    k = sub.add_parser("check", help="re-derive an evidence file against its entry")
    k.add_argument("--env", required=True)
    k.add_argument("--evidence", type=Path, default=None)
    a = p.parse_args(argv)
    try:
        if a.cmd == "golden":
            from ..validator.lane_d import load_pool

            pools, halves = {}, {}
            for half, d in a.half:
                pool = load_pool(d, a.horizon)
                pools[half] = {"snapshot": pool["snapshot"], "sha256": pool["sha256"]}
                halves[half] = pool["windows"]
            ws = pick_golden(halves, horizon=a.horizon, per_half=a.per_half, seed=a.seed)
            meta = write_golden(ws, a.out, horizon=a.horizon, seed=a.seed, pools=pools)
            print(json.dumps({"golden": meta["sha256"], "windows": len(ws)}))
        elif a.cmd == "forecast":
            run = forecast_run(a.golden, dict(a.checkpoint), env=a.env, out_dir=a.out,
                               contexts=[int(x) for x in a.contexts.split(",")],
                               device=a.device, git_head=a.git_head)
            print(json.dumps({"run": run["quantiles_sha256"], "host": run["host"]}))
        elif a.cmd == "compare":
            from ..shared.config import load_chain_config

            ld = load_chain_config(a.chain_config).lane_d
            lane_d = {k: getattr(ld, k) for k in ("alpha", "bootstrap_B", "d_margin",
                                                  "d_margin_start", "window_ratio_floor")}
            doc = compare(a.golden, a.reference, a.candidate, lane_d=lane_d, seed=a.seed,
                          w0b_junit=a.w0b_junit, a9_t7_path=a.a9)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
            v = doc["verdict"]
            print(json.dumps({k: v[k] for k in ("max_rel_diff", "crowns_identical",
                                                "max_lcb_gap", "w0b", "a9", "passed")}))
            return 0 if v["passed"] else 1
        else:
            from ..shared.config.lane_d import load_scoring_envs

            envs = load_scoring_envs()
            env = {e.name: e for e in envs}.get(a.env)
            if env is None:
                raise QualError(f"env {a.env!r} not listed")
            path = a.evidence or REPO / (env.evidence or f"evidence/lane_d_env_{env.name}.json")
            why = evidence_problem(json.loads(Path(path).read_text(encoding="utf-8")), env,
                                   envs=envs)
            print(why or "PASS")
            return 1 if why else 0
    except (QualError, OSError, KeyError, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
