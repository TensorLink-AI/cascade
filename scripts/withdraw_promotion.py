"""Owner-run, trainer STOPPED: withdraw a promotion generation validators
rejected ("fired before the reign clock ripened") and return the trainer's
promotion state to the previous generation, anchored on the VALIDATORS'
reign clock, so the (fixed) trainer re-fires when every attesting validator
judges the reign ripe.

Generalises the 2026-09-25 withdraw_gen12.py (gen-12 split) for the
2026-10-17 gen-15 split.

    # plan only (default) — prints every change, writes nothing
    .venv/bin/python scripts/withdraw_promotion.py --generation 15
    # state edits (snapshots first: <file>.bak-pre-gen<N>-withdraw-<ts>)
    .venv/bin/python scripts/withdraw_promotion.py --generation 15 --apply
    # + bucket locator index -> latest_generation N-1 (separate, explicit)
    .venv/bin/python scripts/withdraw_promotion.py --generation 15 --apply --bucket
    # (or the index step alone, after an earlier --apply)
    .venv/bin/python scripts/withdraw_promotion.py --generation 15 --bucket-only

State edits (``--apply``):
  _train_work/trainer_promotion.json  generation N -> N-1, members = the signed
                                      gen-(N-1) record's members, candidates
                                      cleared (the boundary replay re-ingests the
                                      reign's bench reports), no pending record,
                                      reign clock = the validator's anchor
  _train_work/era_state.json          generations["N"] removed
  warm_start_init.json                pointer rewritten for generation N-1
Bucket (``--bucket``): promotions/index.json -> {"latest_generation": N-1}.
Validators locate the latest record through the index only; the trainer
overwrites gen-N.json when it re-fires. The gen-N object stays (no delete).

The validator anchor defaults to THIS box's validator
``cascade_state.json`` ``reign_start_block`` (read-only), and must belong to the
same king the trainer is timing; ``--reign-start`` overrides it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", type=int, required=True)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--reign-start", type=int, default=None,
                    help="validator reign anchor (default: <root>/cascade_state.json)")
    ap.add_argument("--apply", action="store_true", help="write the state edits")
    ap.add_argument("--bucket", action="store_true",
                    help="also rewrite promotions/index.json (with --apply)")
    ap.add_argument("--bucket-only", action="store_true",
                    help="only the index step, after a prior --apply")
    a = ap.parse_args()
    root = a.root.resolve()
    sys.path.insert(0, str(root))

    from cascade.shared.config import load_chain_config
    from cascade.shared.env import load_env_files

    load_env_files()
    from cascade.shared.era import era_index_of
    from cascade.shared.hippius import StorageError, open_manifest_store
    from cascade.shared.promotion import (
        load_promotion_record,
        promotion_index_key,
        promotion_record_key,
        verify_promotion_record_signature,
    )
    from cascade.trainer.promotion import TrainerPromotion

    n = int(a.generation)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    work = root / "_train_work"
    state_p, era_p = work / "trainer_promotion.json", work / "era_state.json"
    pointer_p = root / "warm_start_init.json"

    def write_index(store, key, StorageError) -> None:
        text = json.dumps({"latest_generation": n - 1}, sort_keys=True)
        try:
            store.put_text(key, text, content_type="application/json", acl="public-read")
        except StorageError:
            store.put_text(key, text, content_type="application/json")
        import urllib.request

        url = f"https://s3.hippius.com/cascade-manifests/{key}?t={int(time.time())}"
        with urllib.request.urlopen(url, timeout=30) as r:
            print("anonymous read of", key, "->", r.read().decode())

    if a.bucket_only:
        st = json.loads(state_p.read_text())
        if int(st.get("generation", 0)) != n - 1:
            print(f"ABORT: trainer generation {st.get('generation')} != {n - 1}; run --apply first")
            return 2
        write_index(open_manifest_store(load_chain_config(str(root / "chain.toml")).storage),
                    promotion_index_key(), StorageError)
        return 0

    # ── guards ────────────────────────────────────────────────────────────
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    if any(ln.lstrip().split(" ")[0].endswith("python") and "cascade-trainer" in ln
           for ln in ps.splitlines()):
        print("ABORT: cascade-trainer is RUNNING — stop it first")
        return 2
    cfg = load_chain_config(str(root / "chain.toml"))
    state = json.loads(state_p.read_text())
    era = json.loads(era_p.read_text())
    if int(state.get("generation", 0)) != n:
        print(f"ABORT: trainer generation is {state.get('generation')}, not {n}")
        return 2
    if state.get("pending_record"):
        print("ABORT: engine holds an unpublished record")
        return 2
    store = open_manifest_store(cfg.storage)
    prev = load_promotion_record(store.get_text(promotion_record_key(n - 1)))
    cur = load_promotion_record(store.get_text(promotion_record_key(n)))
    hk = cfg.manifest.trainer_hotkey
    for rec, g in ((prev, n - 1), (cur, n)):
        if rec.generation != g or not verify_promotion_record_signature(rec, hk):
            print(f"ABORT: promotions/gen-{g}.json missing/unsigned/wrong generation")
            return 2
    eff = int(cur.effective_era or 0)
    for k in ("current", "next"):
        e = era.get(k) or {}
        if int(e.get("generation", 0) or 0) >= n or (eff and int(e.get("index", 0) or 0) >= eff):
            print(f"ABORT: era_state.{k} (index {e.get('index')}, gen {e.get('generation')}) "
                  f"already runs/reaches gen {n} (effective era {eff})")
            return 2
    anchor = a.reign_start
    if anchor is None:
        vs = json.loads((root / "cascade_state.json").read_text())
        if vs.get("king_hotkey") != state.get("king_hotkey"):
            print(f"ABORT: validator king {str(vs.get('king_hotkey'))[:12]} != trainer king "
                  f"{str(state.get('king_hotkey'))[:12]}; pass --reign-start")
            return 2
        if int(vs.get("generation", 0)) != n - 1:
            print(f"ABORT: validator generation {vs.get('generation')} != {n - 1}")
            return 2
        anchor = int(vs["reign_start_block"])
    threshold_blocks = int(cfg.scoring.cascade_reign_blocks or 0)
    print(f"gen {n} fired {cur.fired_block} (era {era_index_of(cfg.round, cur.fired_block)}), "
          f"effective era {eff}")
    print(f"withdraw -> gen {n - 1} ({len(prev.members)} members); reign anchor {anchor}"
          f"{f'; validator ripe from ~{anchor + threshold_blocks}' if threshold_blocks else ''}")
    print(f"era_state.generations: {sorted(era.get('generations', {}))} -> drop '{n}'")
    if not a.apply:
        print("plan only — rerun with --apply (and --bucket for the index)")
        return 0

    # ── snapshots ─────────────────────────────────────────────────────────
    for f in (state_p, era_p, pointer_p):
        shutil.copy2(f, f.with_name(f.name + f".bak-pre-gen{n}-withdraw-{ts}"))
    print("snapshots taken", ts)

    eng = TrainerPromotion.load(
        reign_threshold=cfg.scoring.cascade_reign_days, k_max=cfg.scoring.cascade_top_k,
        quality_epsilon=cfg.scoring.cascade_quality_epsilon, state_path=state_p,
        pointer_path=pointer_p, round_cfg=cfg.round,
        error_vectors_path=work / "promotion_error_vectors.json")
    eng.withdraw_to(prev, reign_start_block=anchor)
    after = json.loads(state_p.read_text())
    print("trainer_promotion.json: generation", after["generation"], "members",
          len(after["members"]), "reign_start_block", after["reign_start_block"],
          "candidates", len(after["candidates"]))
    era.get("generations", {}).pop(str(n), None)
    era_p.write_text(json.dumps(era, sort_keys=True))
    print("era_state.json generations:", sorted(era.get("generations", {})))
    print("warm_start_init.json generation:", json.loads(pointer_p.read_text()).get("generation"))

    if a.bucket:
        write_index(store, promotion_index_key(), StorageError)
    else:
        print("bucket index NOT touched — run again with --bucket-only to rewrite it")
    print("done — restart the trainer on the merged fix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
