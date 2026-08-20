"""Precompute promotion error vectors — the per-window error profile of every
candidate in the trainer's promotion log, written to
``<work_root>/promotion_error_vectors.json`` for select_members' trajectory-
diversity policy (error decorrelation; see DEC-CA-0013 evolution).

Each candidate checkpoint is fetched from the registry and evaluated with the
round screener on the private pool slice for ``--block`` (default: the current
epoch start), producing one positive error scalar per window:
``sqrt(wql_w * mase_w)`` — the per-window analogue of the round geomean, so
correlations are measured on exactly the error the throne is judged on.

Usage (from /root/cascade, .env loaded):
    .venv/bin/python scripts/compute_promotion_error_vectors.py \
        --work-root _train_work [--block 8823600] [--base-seed <round-id>]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade.eval.crps import wql_per_window  # noqa: E402
from cascade.eval.scoring import stack_components  # noqa: E402
from cascade.shared.config import load_chain_config  # noqa: E402
from cascade.shared.hippius import HubConfig, fetch_from_hub  # noqa: E402
from cascade.shared.manifest import parse_trained_pointer  # noqa: E402
from cascade.trainer.loop import ResolvedGenerator  # noqa: E402
from cascade.trainer.main import _build_screen_fn  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", type=Path, default=Path("_train_work"))
    ap.add_argument("--chain-toml", type=Path, default=Path("chain.toml"))
    ap.add_argument("--block", type=int, default=None,
                    help="Epoch-start block keying the pool slice (default: now).")
    ap.add_argument("--base-seed", type=int, default=0,
                    help="Round id keying windows_for_round (default 0: fixed slice).")
    args = ap.parse_args()

    cfg = load_chain_config(args.chain_toml)
    hub = HubConfig.from_storage(cfg.storage)
    state = json.loads((args.work_root / "trainer_promotion.json").read_text())
    candidates = state.get("candidates", [])
    if not candidates:
        print("no candidates in the promotion log; nothing to do")
        return 0

    block = args.block
    if block is None:
        from cascade.shared.chain import ChainClient
        client = ChainClient(network=cfg.chain.network)
        blk = client.current_block()
        epoch = cfg.round.epoch_blocks if hasattr(cfg.round, "epoch_blocks") else 3600
        block = (blk // epoch) * epoch

    screen, _, _ = _build_screen_fn(cfg, cache_dir=args.work_root)
    out_path = args.work_root / "promotion_error_vectors.json"
    vectors: dict[str, list[float]] = {}
    if out_path.is_file():
        try:
            vectors = json.loads(out_path.read_text())
        except Exception:
            vectors = {}

    for c in candidates:
        cid = c["checkpoint_id"]
        if cid in vectors:
            print(f"cached: {cid[:80]}")
            continue
        ref = parse_trained_pointer(cid)
        if ref is None:
            print(f"skip (unparseable pointer): {cid[:80]}")
            continue
        dest = args.work_root / "promotion_vectors_cache" / cid.split("/")[-1].replace(":", "_")
        fetch_from_hub(ref, dest, hub)
        gen = ResolvedGenerator(c.get("hotkey", ""), 0, cid)
        raw = list(screen(dest, gen, args.base_seed, block))
        qloss, abs_t, mase = stack_components(raw)
        # Per-window sqrt(WQL * MASE) via the round's own wql_per_window.
        # Zero-|y| windows are masked exactly as the round metric masks them;
        # the mask is target-dependent, hence identical across candidates, so
        # vectors stay window-aligned after filtering.
        wql, valid = wql_per_window(qloss, abs_t)
        vec = [math.sqrt(max(float(w), 1e-12) * max(float(m), 1e-9))
               for w, m, ok in zip(wql, mase, valid) if ok]
        vectors[cid] = vec
        out_path.write_text(json.dumps(vectors))
        print(f"scored: {cid[:80]} n_windows={len(vec)}")

    print(f"DONE — {len(vectors)} vector(s) in {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
