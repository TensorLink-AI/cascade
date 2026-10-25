#!/usr/bin/env python3
"""DEC-CA-0059 "measure before arming horizons": per-layer Jaccard of the
lane-B horizon-target expert selections on one king checkpoint.

For each horizon (default 64 / 256 / 720 — mainnet ``[eval] scored_horizons``)
the king routes the MASKED forecast patches of a horizon-``h`` decode of the
same context slice (``cascade.trainer.expert_lane.probe_horizon_routing``:
inputs only, no target values), the ESFT rule selects experts per MoE layer
(``[lane_b] esft_p / min_experts / max_experts``), and every pair of horizons
is compared with ``selection_jaccard_layers``. Near-1.0 on every layer means
the horizon slots pick the same experts and would only compete with domain
legs for rows (DEC-CA-0059 Open).

Contexts come from a pool snapshot directory (``--pool-dir``, the layout
``cascade.validator.calibration.load_pool_series`` reads — the first
``--windows`` series, any domain, their last ``context_length`` points), or
from seeded random walks (``--synthetic``) for a smoke run.

Usage:
  lane_b_horizon_jaccard.py KING_DIR --pool-dir POOL [--horizons 64 256 720]
  lane_b_horizon_jaccard.py KING_DIR --synthetic 64 [--json]

Read-only: loads one checkpoint on CPU, writes nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
from pathlib import Path


def _contexts(args) -> list:
    import numpy as np

    if args.pool_dir is not None:
        from cascade.validator.calibration import load_pool_series

        series, _ids, _md = load_pool_series(Path(args.pool_dir))
        out = []
        for s in series[: args.windows]:
            a = np.atleast_2d(np.asarray(s, dtype=np.float64))[0]
            out.append(a)
        return out
    rng = np.random.default_rng(args.seed)
    return [np.cumsum(rng.normal(size=args.length)) for _ in range(args.synthetic)]


def measure(king_dir: Path, contexts: list, horizons: list[int], lane_b_cfg) -> dict:
    """``{"selections": {h: {layer: ids}}, "jaccard": {"h1/h2": [per layer]},
    "tokens": {h: n}}`` for ``horizons`` on ``king_dir``."""
    from cascade.trainer.expert_lane import probe_horizon_routing, selection_jaccard_layers

    sels, tokens = {}, {}
    for h in horizons:
        probe = probe_horizon_routing(Path(king_dir), contexts, lane_b_cfg, int(h))
        sels[int(h)] = probe.select(lane_b_cfg)
        tokens[int(h)] = probe.tokens
    jac = {}
    for a, b in itertools.combinations(sorted(sels), 2):
        per = selection_jaccard_layers(sels[a], sels[b])
        jac[f"{a}/{b}"] = [round(per[li], 6) for li in sorted(per)]
    return {"selections": {str(h): {str(k): v for k, v in s.items()} for h, s in sels.items()},
            "jaccard": jac, "tokens": {str(h): n for h, n in tokens.items()}}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("king_dir", type=Path, help="A cascade_moe king checkpoint directory.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pool-dir", type=Path, default=None, help="Pool snapshot directory.")
    src.add_argument("--synthetic", type=int, default=None, metavar="N",
                     help="N seeded random-walk contexts instead of a pool.")
    p.add_argument("--horizons", type=int, nargs="+", default=[64, 256, 720])
    p.add_argument("--windows", type=int, default=256, help="Pool series used (default 256).")
    p.add_argument("--length", type=int, default=4096, help="Synthetic context length.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chain-toml", type=Path, default=None,
                   help="Read [lane_b] esft_p/min_experts/max_experts from this chain.toml.")
    p.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    args = p.parse_args(argv)

    from cascade.shared.config import load_chain_config, parse_lane_b

    lb = (load_chain_config(args.chain_toml).lane_b if args.chain_toml is not None
          else parse_lane_b({}))
    lb = dataclasses.replace(lb, affinity_windows=max(lb.affinity_windows, args.windows))
    res = measure(args.king_dir, _contexts(args), args.horizons, lb)
    if args.json:
        print(json.dumps(res, sort_keys=True))
        return 0
    print(f"king {args.king_dir}  masked-patch tokens per horizon: {res['tokens']}")
    for pair, per in res["jaccard"].items():
        print(f"  {pair:>9}  per-layer Jaccard {per}  (min {min(per, default=1.0):.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
