"""``cascade lane-d test`` — score a lane D checkpoint offline the way the
validators do (directional; nothing is submitted, signed or rented).

The preview is the validators' own code, imported, never re-implemented:

* windows: a local ``pool_d/1`` snapshot read by
  :func:`cascade.validator.lane_d.load_pool` (or your own series, cut into
  windows of the same shape); the multichannel pick
  (``_pick_channels``), the §1.3 forward eligibility at ``--as-of``
  (``_admitted_versions``, optional: a local snapshot rarely carries the
  live clocks), the ``max_source_share`` cap
  (:func:`~cascade.validator.lane_d_frozen.cap_source_share`) and the seeded
  per-bin draw (``_drawn`` / :func:`~cascade.validator.lane_d_draw.draw_windows`)
  over ``max_windows_per_settlement``;
* forecasts: :func:`cascade.eval.lane_d_scorer.forecast_quantiles` at the
  :func:`~cascade.eval.lane_d_scorer.effective_context` (your declared
  context under the armed set; the house at its trained 4096);
* scores: :func:`~cascade.eval.lane_d_scorer.score_window` (per-window CRPS
  over the 9-level grid and MASE of the median);
* verdict: :func:`cascade.validator.lane_d.judge_cohort` (the rev6.18 A3
  combined rule: clamped per-window geomean ratios, the max-T two-part
  cluster bootstrap LCB) against the crown bar
  :meth:`~cascade.shared.config.lane_d.LaneDConfig.crown_bar` at
  ``--tenure``.

Live-only by default (``public_weight = 0``, the launch overlay's value);
``--public-weight W`` with a local public set previews the later combined
rule. The §1.3b bench gate, the coldkey and queue caps, and the live pool
itself stay live-only.

This module is loaded at call time by ``cascade.miner.lane_d`` (importlib):
the audit's static import closure reaches ``cascade.miner.cli`` and must not
reach a lane D producer module (W9 invariant 12).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .output import OK, REJECTED, USAGE, reporter

DEFAULT_SHOW = 5
LOCAL_SERIES_SNAPSHOT = "local-series"


class PreviewError(ValueError):
    """A typed refusal of the preview: ``code`` and a human message."""

    def __init__(self, code: str, message: str, exit_code: int = REJECTED):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.exit_code = exit_code


# ── the command ─────────────────────────────────────────────────────────────


def add_lane_d_test(lsub) -> None:
    t = lsub.add_parser(
        "test", help="Score a local checkpoint offline the way validators do (directional; "
                     "submits nothing).")
    t.add_argument("candidate", type=Path,
                   help="Your checkpoint directory (config.json + weights.safetensors; an "
                        "exported entry or the trained checkpoint).")
    t.add_argument("--king", type=Path, required=True,
                   help="The bar's checkpoint directory: the current KING_D, or before the "
                        "first lane D crown the house (the lane A king). Kings are public "
                        "(HF mirror tensorlink-dev/cascade-lane-d-kings).")
    t.add_argument("--context", type=int, required=True,
                   help="Your declared context (one of [lane_d] contexts).")
    t.add_argument("--king-context", type=int, default=None,
                   help="KING_D's declared context; default the house rule (its trained "
                        "4096 when armed, else the lowest armed).")
    t.add_argument("--armed-contexts", default=None,
                   help="Comma-separated armed contexts to assume (default: every [lane_d] "
                        "context). The validators arm each per settlement.")
    src = t.add_argument_group("windows (one of)")
    src.add_argument("--pool-dir", type=Path, action="append", default=[],
                     help="A pool_d/1 snapshot directory (metadata.json, hist.npz, tgt.npz); "
                          "repeat for older snapshots, newest last.")
    src.add_argument("--series", type=Path, default=None,
                     help="Your own series: an .npz of 1-D (L,) or 2-D (C, L) arrays; the key "
                          "prefix before '/' is the source (the bootstrap cluster).")
    t.add_argument("--series-freq", default="H",
                   help="Pandas frequency of --series (sets the MASE seasonal period).")
    t.add_argument("--windows-per-series", type=int, default=1,
                   help="Non-overlapping windows cut from the end of each --series array.")
    t.add_argument("--as-of", type=int, default=None,
                   help="Apply the §1.3 forward-only eligibility as if your entry and the king "
                        "were revealed at this unix time (needs the snapshot's clocks).")
    t.add_argument("--tenure", type=int, default=0,
                   help="KING_D's tenure in lane D settlements (sets the crown bar: "
                        "d_margin_start at 0 falling to d_margin).")
    t.add_argument("--public-weight", type=float, default=0.0,
                   help="Weight of the public half, 0 (launch: live-only) to < 1 (template "
                        "0.5). > 0 needs --public-pool-dir or --public-series.")
    t.add_argument("--public-pool-dir", type=Path, action="append", default=[],
                   help="A pool_d/1 directory scored as the public half.")
    t.add_argument("--public-series", type=Path, default=None,
                   help="An .npz scored as the public half (as --series).")
    t.add_argument("--block-hash", default=None,
                   help="Seed of the draw and the channel pick (default: derived from --seed).")
    t.add_argument("--seed", type=int, default=0, help="Local seed (draw, cap and bootstrap).")
    t.add_argument("--device", default="cpu", help="Torch device for the forecasts (e.g. cuda).")
    t.add_argument("--show-windows", type=int, default=DEFAULT_SHOW,
                   help="Best and worst windows to print (all are in --json).")
    t.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    t.set_defaults(func=cmd_lane_d_test, command="lane-d test")


def cmd_lane_d_test(args) -> int:
    from ..shared.config import load_chain_config

    out = reporter(args)
    try:
        data = run_preview(args, load_chain_config(args.chain_toml))
    except PreviewError as e:
        return out.fail(e.code, e.message, e.exit_code)
    except ImportError as e:
        return out.fail("torch_unavailable", f"lane-d test forecasts need the [train] extra "
                                             f"(torch): {e}", USAGE)
    except (ValueError, OSError) as e:
        return out.fail("lane_d_test_failed", str(e), REJECTED)
    for line in data.pop("report_lines"):
        out.info(line)
    out.result(**data)
    return OK


# ── inputs ──────────────────────────────────────────────────────────────────


def _contexts(raw: str | None, cfg) -> tuple[int, ...]:
    if raw is None:
        return tuple(int(c) for c in cfg.contexts)
    try:
        got = tuple(sorted({int(x) for x in str(raw).split(",") if x.strip()}))
    except ValueError:
        raise PreviewError("bad_context", f"--armed-contexts {raw!r} is not a list of "
                                          "integers", USAGE) from None
    bad = [c for c in got if c not in cfg.contexts]
    if not got or bad:
        raise PreviewError("bad_context", f"--armed-contexts {list(got)} must be a non-empty "
                                          f"subset of [lane_d] contexts {list(cfg.contexts)}",
                           USAGE)
    return got


def _checkpoint(d: Path, what: str) -> Path:
    d = Path(d)
    for name in ("config.json", "weights.safetensors"):
        if not (d / name).is_file():
            raise PreviewError("checkpoint_missing", f"{what} {d} has no {name}", USAGE)
    return d


def _patch_size(d: Path) -> int | None:
    try:
        return int(json.loads((d / "config.json").read_text(encoding="utf-8"))["moe"]["patch_size"])
    except (KeyError, TypeError, ValueError, OSError):
        return None


def series_pool(path: Path, *, horizon: int, freq: str, per_series: int) -> dict:
    """Your own series as one pool in :func:`~cascade.validator.lane_d.load_pool`'s
    shape: each array's last ``per_series`` non-overlapping ``horizon``-point
    spans are targets, everything before a span its history. Target clocks
    are the point indices (no live clocks: use it without ``--as-of``)."""
    from ..validator.lane_d import window_id
    from ..validator.lane_d_public import seasonal_period

    if int(per_series) < 1:
        raise PreviewError("bad_series", "--windows-per-series must be >= 1", USAGE)
    p = Path(path)
    sp = max(1, int(seasonal_period(freq)))
    windows = []
    with np.load(p, allow_pickle=False) as z:
        for name in sorted(z.files):
            a = np.asarray(z[name], dtype=np.float64)
            if a.ndim == 2 and a.shape[0] == 1:
                a = a.reshape(-1)
            if a.ndim not in (1, 2):
                raise PreviewError("bad_series", f"{p}: {name} has shape {a.shape}; want (L,) "
                                                 "or (C, L)", USAGE)
            n = a.shape[-1]
            source = name.split("/", 1)[0] if "/" in name else name
            for k in range(int(per_series)):
                cut = n - (k + 1) * int(horizon)
                if cut < 1:
                    break
                ts = list(range(cut, cut + int(horizon)))
                windows.append({
                    "window_id": window_id(name, ts, horizon), "series_id": name,
                    "source": source, "feed": source, "freq": str(freq), "frontier": None,
                    "seasonal_period": sp, "target_ts": ts, "first_seen_ts": list(ts),
                    "end_ts": ts[-1], "history": a[..., :cut], "target": a[..., cut:cut + int(horizon)],
                })
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"snapshot": LOCAL_SERIES_SNAPSHOT, "sha256": sha, "feeds": {},
            "first_seen_roots": [], "windows": windows}


def _pools(dirs, series, *, horizon: int, freq: str, per_series: int, what: str) -> list[dict]:
    from ..validator.lane_d import load_pool

    if bool(dirs) == (series is not None):
        raise PreviewError("windows_missing", f"give exactly one of --{what}pool-dir or "
                                              f"--{what}series", USAGE)
    if series is not None:
        return [series_pool(series, horizon=horizon, freq=freq, per_series=per_series)]
    return [load_pool(d, horizon) for d in dirs]


def _triples(pools: list[dict], ld, h: int, as_of: int | None, patch: int | None
             ) -> tuple[list[tuple], dict]:
    """``(window, pool, verdict)`` per usable window, earliest snapshot first
    (the validator's one-version rule), plus the §1.3 refusal counts."""
    from ..validator.lane_d import _admitted_versions, _usable
    from ..validator.lane_d_forward import WindowVerdict

    if as_of is not None:
        admitted, refused = _admitted_versions(pools, int(as_of), int(as_of), ld)
    else:
        admitted, refused = {}, {}
        for pool in pools:
            for w in pool["windows"]:
                wid = w["window_id"]
                if wid and wid not in admitted:
                    admitted[wid] = (w, pool, WindowVerdict(window_id=wid, eligible=True,
                                                            reason=None, bound=0.0))
    out, short = [], 0
    for wid in sorted(admitted):
        w, pool, v = admitted[wid]
        if not _usable(w, h):
            continue
        if patch and np.shape(w["history"])[-1] < patch:
            short += 1
            continue
        out.append((w, pool, v))
    if short:
        refused = {**refused, "history_shorter_than_one_patch": short}
    return out, refused


def _forecast(ckpt: Path, ws: list[dict], *, context: int, horizon: int, armed, device: str
              ) -> dict:
    """``{window_id: (h, 9) | (k, h, 9)}`` — the eval host's forecast shape."""
    from ..eval import lane_d_scorer as scorer

    hists = [w["history"] for w in ws]
    q = scorer.forecast_quantiles(ckpt, hists, context=int(context), horizon=int(horizon),
                                  contexts=armed, device=device)
    rows = scorer.forecast_rows(hists)
    off = np.concatenate([[0], np.cumsum(rows)]).astype(np.int64)
    return {w["window_id"]: (q[off[i]] if rows[i] == 1 else q[off[i]:off[i + 1]])
            for i, w in enumerate(ws)}


def _scores(fc: dict, ws: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    from ..eval.lane_d_scorer import score_window

    pairs = [score_window(fc[w["window_id"]], w["target"], w["history"], w["seasonal_period"])
             for w in ws]
    return np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])


def _num(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _clean(obj):
    """JSON-safe (``--json`` refuses NaN): non-finite floats become null."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    if isinstance(obj, (float, np.floating)):
        return _num(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _pct(x) -> str:
    x = _num(x)
    return "n/a" if x is None else f"{x * 100:+.2f}%"


# ── the preview ─────────────────────────────────────────────────────────────


def run_preview(args, cfg) -> dict:
    from ..eval.bootstrap import clamped_log_ratios
    from ..eval.lane_d_scorer import effective_context
    from ..validator.lane_d import _drawn, _pick_channels, _subscore, judge_cohort
    from ..validator.lane_d_frozen import cap_source_share
    from ..validator.lane_d_inputs import HOUSE_TRAINED_CONTEXT

    base = cfg.lane_d
    w_pub = float(args.public_weight)
    if not 0.0 <= w_pub < 1.0:
        raise PreviewError("bad_public_weight", f"--public-weight {w_pub} must be in [0, 1)",
                           USAGE)
    ld = dataclasses.replace(base, public_weight=w_pub)
    h = int(ld.horizon)
    if int(args.context) not in ld.contexts:
        raise PreviewError("bad_context", f"--context {args.context} is not in [lane_d] "
                                          f"contexts {list(ld.contexts)}", USAGE)
    armed = _contexts(args.armed_contexts, ld)
    cand = _checkpoint(args.candidate, "candidate")
    king = _checkpoint(args.king, "--king")
    cand_ctx = effective_context(int(args.context), armed)
    if cand_ctx is None:
        raise PreviewError("context_below_armed", f"no armed context <= your declared "
                                                  f"{args.context} (armed {list(armed)}): a "
                                                  "validator would skip the entry "
                                                  "(context_below_armed)")
    if args.king_context is None:
        king_ctx = effective_context(HOUSE_TRAINED_CONTEXT, armed) if any(
            c <= HOUSE_TRAINED_CONTEXT for c in armed) else None
        king_ctx = min(armed) if king_ctx is None else king_ctx
    else:
        king_ctx = effective_context(int(args.king_context), armed)
        if king_ctx is None:
            raise PreviewError("context_below_armed", f"the king's context "
                                                      f"{args.king_context} has no armed "
                                                      f"context at or below it")
    seed = int(args.seed)
    block_hash = args.block_hash or hashlib.sha256(f"lane-d-local:{seed}".encode()).hexdigest()
    max_channels = getattr(getattr(cfg, "generator", None), "max_channels", None)
    patch = _patch_size(cand)

    pools = _pools(args.pool_dir, args.series, horizon=h, freq=args.series_freq,
                   per_series=args.windows_per_series, what="")
    picks = _pick_channels(pools, block_hash, None if max_channels is None else int(max_channels))
    triples, refused = _triples(pools, ld, h, args.as_of, patch)
    cap_seed = f"lane-d/1:cap:local:{seed}"
    kept = {w["window_id"] for w in cap_source_share(
        [{"window_id": w["window_id"], "source": w["source"]} for w, _, _ in triples],
        max_share=float(ld.max_source_share), seed=cap_seed)}
    capped = [t for t in triples if t[0]["window_id"] in kept]
    picked, draw, meets = _drawn(capped, ld, block_hash, int(ld.max_windows_per_settlement))
    ws = [w for w, _, _ in picked]
    if not ws:
        raise PreviewError("no_windows", f"no scorable {h}-step window (eligible "
                                         f"{len(triples)}, refused {refused or 'none'}); "
                                         "check the horizon, the clocks with --as-of, or give "
                                         "longer series")

    k_fc = _forecast(king, ws, context=king_ctx, horizon=h, armed=armed, device=args.device)
    c_fc = _forecast(cand, ws, context=cand_ctx, horizon=h, armed=armed, device=args.device)
    k_sc, c_sc = _scores(k_fc, ws), _scores(c_fc, ws)
    clusters = [w["source"] for w in ws]

    public = None
    pub_ws: list[dict] = []
    if w_pub > 0.0:
        ppools = _pools(args.public_pool_dir, args.public_series, horizon=h,
                        freq=args.series_freq, per_series=args.windows_per_series,
                        what="public-")
        _pick_channels(ppools, block_hash, None if max_channels is None else int(max_channels))
        pub_ws = [w for w, _, _ in _triples(ppools, ld, h, None, patch)[0]]
        if not pub_ws:
            raise PreviewError("no_public_windows", "the public set has no scorable window")
        kp = _scores(_forecast(king, pub_ws, context=king_ctx, horizon=h, armed=armed,
                               device=args.device), pub_ws)
        cp = _scores(_forecast(cand, pub_ws, context=cand_ctx, horizon=h, armed=armed,
                               device=args.device), pub_ws)
        public = {"anchor": kp, "members": [cp],
                  "clusters": [f"{w['source']}|{w.get('family') or w.get('freq') or 'default'}"
                               for w in pub_ws],
                  "sources": [w["source"] for w in pub_ws]}

    newest = str(pools[-1]["sha256"])
    bar = ld.crown_bar(int(args.tenure))
    (verdict,) = judge_cohort(k_sc, [c_sc], clusters=clusters, cfg=ld,
                              seed=f"lane-d/1:local:{seed}:{newest}", public=public, bar=bar)
    x = clamped_log_ratios(*k_sc, *c_sc, floor=float(ld.window_ratio_floor))
    rows = [{"window_id": w["window_id"], "source": w["source"],
             "king_crps": float(k_sc[0][i]), "king_mase": float(k_sc[1][i]),
             "crps": float(c_sc[0][i]), "mase": float(c_sc[1][i]),
             "log_ratio": float(x[i]),
             "channels": len(picks[w["window_id"]].channels) if w["window_id"] in picks else 1}
            for i, w in enumerate(ws)]
    idx = list(range(len(ws)))
    agg = {"king": {"mean_crps": float(np.mean(k_sc[0])), "mean_mase": float(np.mean(k_sc[1])),
                    **_subscore(k_sc, idx)},
           "candidate": {"mean_crps": float(np.mean(c_sc[0])),
                         "mean_mase": float(np.mean(c_sc[1])), **_subscore(c_sc, idx)}}
    sources = sorted(set(clusters))
    floors = {"min_windows": int(ld.min_windows), "min_clusters": int(ld.min_clusters),
              "windows": len(ws), "sources": len(sources), "met": bool(meets)}
    would = bool(verdict["cleared"] and meets)
    data = {
        "directional_only": True, "candidate": str(cand), "king": str(args.king),
        "contexts": {"declared": int(args.context), "effective": int(cand_ctx),
                     "king": int(king_ctx), "armed": list(armed)},
        "public_weight": w_pub,
        "crown_bar": {"tenure": int(args.tenure), "margin": float(bar),
                      "start": float(ld.d_margin_start), "end": float(ld.d_margin),
                      "warmup_settlements": int(ld.d_margin_warmup_settlements)},
        "windows": {"eligible": len(triples), "after_source_cap": len(capped),
                    "judged": len(ws), "public": len(pub_ws), "refused": refused,
                    "fresh": None if draw is None else int(draw["fresh_windows"]),
                    "embargo": None if draw is None else int(draw["embargo_windows"]),
                    "sources": sources, "block_hash": block_hash},
        "floors": floors, "aggregate": agg, "verdict": verdict, "per_window": rows,
        "would_clear": would, "live_reward_guaranteed": False,
        "limitations": [
            "Local windows are not the validators' private live pool; the live verdict can differ.",
            "The §1.3b bench gate (one public suite per settlement, tau_public) is not run here.",
            "Queue, coldkey caps, eval delay and the licence/ingest checks stay live-only "
            "(run `cascade lane-d verify` for the ingest guards).",
        ],
    }
    data = _clean(data)
    data["report_lines"] = describe(data, show=int(args.show_windows))
    return data


def describe(d: dict, *, show: int = DEFAULT_SHOW) -> list[str]:
    """Plain-English lines for a preview result (reads it, never re-judges)."""
    v, cb, fl, ag, ctx = d["verdict"], d["crown_bar"], d["floors"], d["aggregate"], d["contexts"]
    mode = ("live-only (public_weight 0, the launch rule)" if not d["public_weight"]
            else f"combined: {1 - d['public_weight']:.2f} live + {d['public_weight']:.2f} public")
    lines = [f"Lane D offline preview — the validators' judge_cohort, {mode}.",
             f"  contexts: yours declared {ctx['declared']} -> scored at {ctx['effective']}; "
             f"king at {ctx['king']} (armed assumed {ctx['armed']})",
             f"  windows: {fl['windows']} judged from {fl['sources']} sources "
             f"(eligible {d['windows']['eligible']}, after the source cap "
             f"{d['windows']['after_source_cap']})"]
    for who in ("king", "candidate"):
        a = ag[who]
        lines.append(f"  {who:9s}: mean CRPS {a['mean_crps']:.4f}  mean MASE {a['mean_mase']:.4f}"
                     f"  geomean {a['geomean']:.4f}" if a["geomean"] is not None else
                     f"  {who}: no scores")
    per = sorted(d["per_window"], key=lambda r: r["log_ratio"])
    n = max(0, int(show))
    if n and per:
        lines.append(f"  worst {min(n, len(per))} windows for you (log-ratio < 0 = the king wins):")
        lines += [f"    {r['window_id']}: CRPS {r['crps']:.4f} vs {r['king_crps']:.4f}, "
                  f"MASE {r['mase']:.4f} vs {r['king_mase']:.4f}, log-ratio {r['log_ratio']:+.3f}"
                  for r in per[:n]]
        lines.append(f"  best {min(n, len(per))} windows for you:")
        lines += [f"    {r['window_id']}: CRPS {r['crps']:.4f} vs {r['king_crps']:.4f}, "
                  f"MASE {r['mase']:.4f} vs {r['king_mase']:.4f}, log-ratio {r['log_ratio']:+.3f}"
                  for r in per[::-1][:n]]
    for half, rec in (v.get("halves") or {}).items():
        lines.append(f"  {half} half: point gain {_pct(rec.get('gain'))} on {rec.get('n')} windows")
    lines.append(f"  combined gain {_pct(v.get('gain'))}, LCB {_pct(v.get('lcb'))} vs the crown "
                 f"bar {_pct(cb['margin'])} (tenure {cb['tenure']}; {_pct(cb['start'])} at tenure 0 "
                 f"falling to {_pct(cb['end'])} after {cb['warmup_settlements']} settlements)")
    if not fl["met"]:
        lines.append(f"  floors NOT met: a verdict needs {fl['min_windows']} windows from "
                     f"{fl['min_clusters']} sources; a validator would wait (pending_floor).")
    if d["would_clear"]:
        verdict = "WOULD CLEAR the bar (the bench gate still runs live)"
    elif v.get("cleared"):
        verdict = "clears the statistic but the floors are not met here"
    else:
        why = []
        if (_num(v.get("lcb")) or -1.0) < cb["margin"]:
            why.append("the LCB is below the bar")
        if any((_num(r.get("gain")) or 0.0) <= 0.0 for r in (v.get("halves") or {}).values()):
            why.append("a half's point gain is not positive")
        verdict = "would NOT clear — " + ("; ".join(why) or "see the verdict")
    lines.append(f"  verdict: {verdict}.")
    lines.append("  Before the first lane D crown the bar is the house (the lane A king): pass it "
                 "as --king.")
    return lines


__all__ = ["PreviewError", "add_lane_d_test", "cmd_lane_d_test", "describe", "run_preview",
           "series_pool"]
