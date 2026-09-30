"""Example `cascade mine --proposer cmd` strategy: coordinate search over mixture weights.

Copy this file and replace ``propose()`` with your own idea. The loop handles
everything else (verify, score, accept, best/, resume, UI):

    cascade mine --proposer cmd --propose-cmd "python /work/my_strategy.py"

Contract (see cascade/miner/optimize.py ``CommandProposer``):
  * cwd is the candidate dir, already a copy of the current best generator;
  * stdin is JSON: {iteration, candidate_dir, workdir, history, best, king};
  * edit files in place, write one line to $CASCADE_NOTE_FILE, exit 0.
    A non-zero exit skips this iteration without scoring it.

This one walks the weights of the first mixture it finds (e.g. the king's
``family_weights``), one family at a time, trying x2 then x0.5. It uses the
history to skip moves already tried from the same parent and to keep
pushing a move that just won.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def find_mixture(cfg: dict) -> str | None:
    for k, v in cfg.items():
        if isinstance(v, dict) and len(v) >= 2 and all(
                isinstance(x, int | float) and not isinstance(x, bool) and x >= 0
                for x in v.values()):
            return k
    return None


def propose(cfg: dict, ctx: dict) -> str:
    key = find_mixture(cfg)
    if key is None:
        raise SystemExit("no mixture dict in config.json; write your own propose()")
    weights = cfg[key]
    best = ctx.get("best") or {}
    history = ctx.get("history", [])
    # Moves already tried from the current best (same parent) are not retried.
    tried = {h.get("note", "").split("cmd: ", 1)[-1]
             for h in history if h.get("parent") == best.get("iteration")}
    # If the move that produced the current best was a win, push it once more.
    last = best.get("note", "").split("cmd: ", 1)[-1]
    order = [last] if last.startswith(f"{key}.") else []
    order += [f"{key}.{fam} x{f}" for fam in sorted(weights) for f in ("2", "0.5")]
    move = next((m for m in order if m not in tried), None)
    if move is None:
        raise SystemExit("every coordinate move tried from this best; nothing left")
    fam, factor = move[len(key) + 1:].rsplit(" x", 1)
    total = sum(weights.values())
    weights[fam] = max(weights[fam], total * 1e-3) * float(factor)
    s = sum(weights.values())
    cfg[key] = {k: v * total / s for k, v in weights.items()}      # keep the total
    return move


def main() -> int:
    ctx = json.load(sys.stdin)
    path = Path("config.json")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    note = propose(cfg, ctx)
    path.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    Path(os.environ["CASCADE_NOTE_FILE"]).write_text(note + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
