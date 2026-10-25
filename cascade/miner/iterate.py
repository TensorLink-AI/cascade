"""``cascade iterate`` — one offline improvement step, built for agent loops.

    verify → score the candidate over --seeds → score the baseline (the king by
    default) the same way → compare → append to the history log

Nothing is spent and nothing is sent: it is ``cascade score`` run on both sides
under identical settings, so the comparison is like for like. Scores are cached
in ``--work-dir`` by (generator content, scoring settings, seed), so the
baseline is trained once per settings change and re-running an unchanged
candidate is instant. Every run appends one line to
``<work-dir>/history.jsonl`` and reports the best candidate seen so far.

The result is directional, not the verdict: validators score on a private,
rotating pool and duel on an LCB with a dethrone margin (DEC-CA-0012/0039).
Use ``--pool-dir`` with real held-out data for a meaningful signal; the
default synthetic sample only catches gross regressions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path

from .output import NETWORK, OK, REJECTED, USAGE, reporter

DEFAULT_WORK_DIR = Path(".cascade-work")


def add_iterate(sub: argparse._SubParsersAction) -> None:
    from .cli import add_score_options

    p = sub.add_parser(
        "iterate",
        help="Verify, score your generator and the king (or --baseline) under identical "
             "local settings, and compare. Offline, never spends; cached and logged.",
    )
    p.add_argument("repo_dir", type=Path, help="Your candidate generator repo.")
    p.add_argument("--baseline", default="king", metavar="king|none|DIR|repo@digest",
                   help="What to compare against (default: the reigning king, fetched once "
                        "and cached; 'none' scores the candidate only).")
    p.add_argument("--seeds", default="0",
                   help="Comma-separated round seeds to average over (default: 0). More seeds "
                        "= less noise, proportionally more time.")
    p.add_argument("--min-improvement", type=float, default=0.0,
                   help="Relative improvement over the baseline mean geomean needed to report "
                        "beats_baseline (default 0.0; the live bar is stricter).")
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
                   help="Cache, fetched baselines and history.jsonl (default: ./.cascade-work).")
    p.add_argument("--chain-toml", type=Path, default=None,
                   help="Complete local configuration; can select a released smaller model for cheaper experiments.")
    p.add_argument("--network", default="finney", help="Network for fetching the king.")
    add_score_options(p)
    p.set_defaults(func=cmd_iterate)


def parse_seeds(text: str) -> list[int]:
    seeds = [int(s) for s in str(text).replace(" ", "").split(",") if s != ""]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must be distinct integers, e.g. 0,1,2")
    return seeds


def content_digest(repo_dir: Path) -> str:
    """sha256 of exactly what ``cascade submit`` would upload."""
    from .cli import zip_repo_bytes

    return hashlib.sha256(zip_repo_bytes(repo_dir)).hexdigest()


def settings_key(args, seed: int) -> dict:
    pool_dir = args.pool_dir
    pool_stamp = args._pool_stamp if hasattr(args, "_pool_stamp") else pool_content_stamp(pool_dir)
    return json.loads(json.dumps({"pool_dir": str(pool_dir) if pool_dir else None, "pool_stamp": pool_stamp,
            "pool_ref": args.pool_ref or None, "train_hours": args.train_hours,
            "n_windows": args.n_windows, "device": args.device,
            "warm_start": args.warm_start, "seed": seed,
            "context": getattr(args, "_cache_context", None), "cache_schema": 2}))


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pool_content_stamp(pool_dir):
    if pool_dir is None or not Path(pool_dir).is_dir():
        return None
    return sorted((str(p.relative_to(pool_dir)), file_digest(p))
                  for p in Path(pool_dir).rglob("*") if p.is_file())


def cache_context(cfg, warm_start):
    from dataclasses import asdict

    from ..shared.manifest import contract_digest

    local = Path(str(warm_start)) if warm_start else None
    implementation = hashlib.sha256()
    package = Path(__file__).resolve().parents[1]
    for folder in ("miner", "trainer", "eval", "validator", "interface"):
        for source in sorted((package / folder).rglob("*.py")):
            implementation.update(str(source.relative_to(package)).encode())
            implementation.update(source.read_bytes())
    return {"contract_digest": contract_digest(cfg.screen_contract()),
            "implementation": implementation.hexdigest(),
            "eval": asdict(cfg.eval), "generator": asdict(cfg.generator),
            "scoring": asdict(cfg.scoring),
            "local_init": sorted((str(p.relative_to(local)), file_digest(p))
                                 for p in local.rglob("*") if p.is_file())
                          if local and local.is_dir() else None}


class ScoreCache:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.rows = json.loads(path.read_text())
        except (OSError, ValueError):
            self.rows = {}

    @staticmethod
    def key(digest: str, settings: dict) -> str:
        return hashlib.sha256(json.dumps({"generator": digest, **settings},
                                         sort_keys=True).encode()).hexdigest()

    def get(self, key):
        return self.rows.get(key)

    def put(self, key, row):
        self.rows[key] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.rows, sort_keys=True, indent=1))
        tmp.replace(self.path)


def _safe(ref: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in ref)


def resolve_baseline(args, cfg, out):
    """``(exit_code, dir_or_None, label)`` for ``--baseline``."""
    spec = str(args.baseline).strip()
    if spec.lower() == "none":
        return OK, None, None
    local = Path(spec)
    if spec.lower() != "king" and local.is_dir():
        return OK, local, f"dir:{local}"
    import shutil

    from .cli import fetch_generator

    king = spec.lower() == "king"
    # A raw ref is immutable, so its dir is reused; the king moves, so it is
    # fetched to a staging dir and filed under whatever ref it resolved to.
    dest_dir = args.work_dir / "baselines" / ("_incoming" if king else _safe(spec))
    if not king and dest_dir.is_dir():
        return OK, dest_dir, f"ref {spec}"
    if king:
        shutil.rmtree(dest_dir, ignore_errors=True)
    rc, dest, ref, label = fetch_generator("king" if king else spec, cfg, args.network,
                                           dest_dir, out)
    if rc != OK:
        if king:
            out.warn("could not fetch the king — pass --baseline DIR (e.g. a fork you fetched "
                     "earlier) or --baseline none to score the candidate only")
        return rc, None, None
    if king:
        final = args.work_dir / "baselines" / _safe(ref)
        if final.exists():
            shutil.rmtree(dest, ignore_errors=True)
        else:
            Path(dest).rename(final)
        dest = final
    return OK, Path(dest), f"{label} {ref}"


def score_all(repo: Path, digest: str, args, cfg, seeds, cache: ScoreCache, out, role: str):
    from .cli import score_result_dict
    from .score import score_generator

    rows = []
    for seed in seeds:
        key = cache.key(digest, settings_key(args, seed))
        hit = cache.get(key)
        if hit is not None:
            out.info(f"  {role} seed {seed}: geomean={hit['geomean']:.5f} (cached)")
            rows.append({**hit, "seed": seed, "cached": True})
            continue
        out.info(f"  {role} seed {seed}: training…")
        r = score_generator(repo, cfg, pool_dir=args.pool_dir, pool_ref=args.pool_ref,
                            train_hours=args.train_hours, n_windows=args.n_windows,
                            device=args.device, seed=seed, warm_start=args.warm_start,
                            cache_dir=args.work_dir / "score")
        row = score_result_dict(r)
        cache.put(key, row)
        out.info(f"  {role} seed {seed}: geomean={r.geomean:.5f} ({r.train_seconds:.0f}s)")
        rows.append({**row, "seed": seed, "cached": False})
    return rows


def summarise(rows):
    values = [r["geomean"] for r in rows]
    return {"mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "per_seed": {str(r["seed"]): r["geomean"] for r in rows}}


def compare(candidate, baseline, min_improvement):
    """Lower geomean is better; ``relative`` > 0 means the candidate is better."""
    rel = (baseline["mean"] - candidate["mean"]) / baseline["mean"] if baseline["mean"] else 0.0
    seeds = sorted(set(candidate["per_seed"]) & set(baseline["per_seed"]))
    wins = sum(candidate["per_seed"][s] < baseline["per_seed"][s] for s in seeds)
    return {"relative_improvement": rel, "seed_wins": wins, "seeds_compared": len(seeds),
            "beats_baseline": rel > min_improvement and wins * 2 > len(seeds)}


def read_history(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def cmd_iterate(args: argparse.Namespace) -> int:
    from ..shared.config import load_chain_config
    from .verify import verify_repo

    out = reporter(args)
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as e:
        return out.fail("bad_seeds", str(e), USAGE)
    cfg = load_chain_config(args.chain_toml)

    from .live_status import LiveStateUnavailable
    from .training_context import UnresolvedLaunch, configured_context, live_context

    args = copy.copy(args)
    try:
        if str(args.warm_start).lower() == "live":
            cfg, args.warm_start, _ = live_context(cfg)
            out.info(f"Pinned live contract and init for all seeds: {args.warm_start or 'random init'}")
        else:
            cfg = configured_context(cfg, args.at_block)
    except LiveStateUnavailable as exc:
        return out.fail("live_warm_start_unavailable", str(exc), NETWORK)
    except UnresolvedLaunch as exc:
        return out.fail("launch_unresolved", str(exc), USAGE)
    # No bound contract: the candidate must pass on either side of the v2
    # launch, exactly as `cascade verify` / `deploy` / `submit` judge it.
    unbound = str(args.warm_start).lower() != "live" and args.at_block is None
    report = verify_repo(args.repo_dir, cfg, quick=True, across_launch=unbound)
    out.result(verify=report.to_dict())
    if not report.ok:
        return out.fail("verify_failed", "candidate fails verify — fix these first:\n"
                        + report.render())
    args._cache_context = cache_context(cfg, args.warm_start)
    args._pool_stamp = pool_content_stamp(args.pool_dir)
    digest = content_digest(args.repo_dir)
    out.info(f"candidate {args.repo_dir} ({digest[:12]}) verified")

    rc, baseline_dir, baseline_label = resolve_baseline(args, cfg, out)
    if rc != OK:
        return rc
    cache = ScoreCache(args.work_dir / "scores.json")
    try:
        cand_rows = score_all(args.repo_dir, digest, args, cfg, seeds, cache, out, "candidate")
        base_rows = (score_all(baseline_dir, content_digest(baseline_dir), args, cfg, seeds,
                               cache, out, "baseline") if baseline_dir is not None else None)
    except ImportError as e:
        return out.fail("missing_train_extra",
                        f"error: `cascade iterate` needs the [train] extra (torch): {e}", USAGE)
    except Exception as e:  # noqa: BLE001 — surface any train/eval failure cleanly
        return out.fail("scoring_failed", f"scoring failed: {type(e).__name__}: {e}", REJECTED)

    candidate = summarise(cand_rows)
    result = {"candidate": {"path": str(args.repo_dir), "content_digest": digest, **candidate},
              "seeds": seeds, "pool": cand_rows[0]["pool"], "init": cand_rows[0]["init"],
              "lower_is_better": True, "directional_only": True}
    if base_rows is not None:
        baseline = summarise(base_rows)
        result["baseline"] = {"label": baseline_label, **baseline}
        result["comparison"] = compare(candidate, baseline, args.min_improvement)

    history_path = args.work_dir / "history.jsonl"
    entry = {"time": int(time.time()), "content_digest": digest, "path": str(args.repo_dir),
             "mean": candidate["mean"], "seeds": seeds,
             "settings": settings_key(args, None),
             "baseline_mean": result.get("baseline", {}).get("mean"),
             "relative_improvement": result.get("comparison", {}).get("relative_improvement")}
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    comparable = [h for h in read_history(history_path) if h.get("settings") == entry["settings"]]
    best = min(comparable, key=lambda h: h["mean"])
    result["history"] = {"path": str(history_path), "runs": len(comparable),
                         "best": {k: best[k] for k in ("content_digest", "path", "mean", "time")}}
    out.result(**result)

    out.info(f"\ncandidate: mean geomean {candidate['mean']:.5f} ± {candidate['stdev']:.5f} "
             f"over seeds {seeds}  (lower is better)")
    if base_rows is not None:
        c = result["comparison"]
        out.info(f"baseline:  mean geomean {result['baseline']['mean']:.5f}  [{baseline_label}]")
        out.info(f"→ {c['relative_improvement']:+.2%} vs baseline, better on "
                 f"{c['seed_wins']}/{c['seeds_compared']} seeds: "
                 + ("BEATS the baseline locally" if c["beats_baseline"] else "does not beat it"))
    out.info(f"best of {len(comparable)} comparable runs: {best['mean']:.5f} "
             f"({best['content_digest'][:12]}, {best['path']})")
    out.info(f"pool: {result['pool']} — directional only; the live duel uses a private pool "
             "and a dethrone margin.")
    return OK
