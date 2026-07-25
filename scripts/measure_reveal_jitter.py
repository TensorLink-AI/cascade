#!/usr/bin/env python
"""Measure timelock reveal jitter on a live network — sizes `reveal_margin_blocks`.

`cascade deploy` times its reveal at ``epoch boundary − [round]
reveal_margin_blocks`` (docs/MINER.md §5a). Round eligibility gates on the
REVEAL block strictly before the boundary, so a reveal landing later than
targeted can silently cost a miner the round. The margin must therefore exceed
the worst observed lateness (with headroom), while staying short enough that a
copier can't fetch + re-commit + land their own reveal inside it. This script
measures that lateness empirically:

for each requested delay D, it submits a probe commitment with
``blocks_until_reveal=D`` at block S, polls until the hotkey's reveal appears,
and reports ``reveal_block − (S + D)`` — positive = late, the number the margin
must absorb.

Probe payloads are plain strings (``probe:reveal-jitter:…``) that
``parse_commit`` rejects, so they never enter a round, never burn the
one-submission budget, and cost only the extrinsic fee. Use a THROWAWAY
testnet hotkey: the chain keeps ~10 recent reveals per hotkey, and your real
submission should not share a hotkey with probe noise.

Usage::

    python scripts/measure_reveal_jitter.py \\
        --chain-toml chain.testnet.toml --network test \\
        --wallet-name probe --wallet-hotkey probe1 \\
        --delays 1,2,5,10,25

Needs the ``[chain]`` extra (bittensor) and the wallet password if encrypted.
Exit 0 always (a measurement, not a gate); read the summary against your
configured margin.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade.shared.chain import ChainClient  # noqa: E402
from cascade.shared.config import load_chain_config  # noqa: E402


def summarize(records: list[tuple[int, int]], margin: int) -> str:
    """Render measured (delay, lateness_blocks) pairs against ``margin``.

    Pure — ``lateness`` is ``reveal_block − (submit_block + delay)``; positive
    means the reveal landed late. The recommendation is the contract the deploy
    default relies on: margin > worst lateness, with ~2× headroom."""
    lines = ["delay  lateness(blocks)"]
    lines += [f"{d:>5}  {late:+d}" for d, late in records]
    worst = max((late for _, late in records), default=0)
    lines.append(f"worst lateness: {worst:+d} block(s); configured reveal_margin_blocks={margin}")
    if worst >= margin:
        lines.append(f"⚠ margin does NOT absorb observed jitter — raise [round] "
                     f"reveal_margin_blocks above {worst} (with headroom).")
    elif 2 * worst >= margin:
        lines.append("margin absorbs observed jitter but with <2× headroom; consider raising it.")
    else:
        lines.append("margin comfortably absorbs observed jitter (≥2× headroom).")
    return "\n".join(lines)


def _dump_pending_commitment(client: ChainClient, netuid: int, hotkey: str, when: str) -> None:
    """Print what the PENDING commitment store (``Commitments.CommitmentOf``)
    exposes for ``hotkey`` right now — best-effort, read-only.

    This settles the commit-age-floor feasibility question empirically: the
    floor ("a commitment must predate ``boundary − margin`` to enter the
    round") is implementable iff the commit block is readable here while the
    commitment is pending, and re-derivable by auditors via archive state at a
    past block. Run this before AND after a probe's reveal to learn whether the
    record (and its ``block`` field) survives the reveal or is cleaned up."""
    try:
        substrate = client.subtensor().substrate
        rec = substrate.query(
            module="Commitments", storage_function="CommitmentOf", params=[netuid, hotkey]
        )
        value = getattr(rec, "value", rec)
        print(f"pending-store [{when}]: {value!r}"[:600])
    except Exception as e:  # noqa: BLE001 — a diagnostic must never abort the probe
        print(f"pending-store [{when}]: unreadable ({type(e).__name__}: {e})")


def _await_reveal(client: ChainClient, hotkey: str, after_block: int, timeout_s: int) -> int | None:
    """Poll until ``hotkey`` shows a reveal at/after ``after_block``; its block."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for c in client.poll_commitments():
            if c.hotkey == hotkey and c.reveal_block >= after_block:
                return c.reveal_block
        time.sleep(6)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain-toml", type=Path, default=None)
    ap.add_argument("--network", default="test")
    ap.add_argument("--wallet-name", required=True)
    ap.add_argument("--wallet-hotkey", required=True)
    ap.add_argument("--wallet-path", default=None)
    ap.add_argument("--delays", default="1,2,5,10,25",
                    help="Comma-separated blocks_until_reveal values to probe.")
    ap.add_argument("--timeout-s", type=int, default=1800,
                    help="Max seconds to wait for each probe's reveal.")
    ap.add_argument("--inspect-pending", action="store_true",
                    help="Also dump the raw Commitments.CommitmentOf record before and "
                    "after each reveal — answers whether the COMMIT block is readable "
                    "on chain (pending and/or post-reveal), i.e. whether a commit-age "
                    "eligibility floor is implementable.")
    args = ap.parse_args(argv)

    cfg = load_chain_config(args.chain_toml)
    client = ChainClient.from_config(
        cfg, network=args.network,
        wallet_name=args.wallet_name, wallet_hotkey=args.wallet_hotkey,
        wallet_path=args.wallet_path,
    )
    hotkey = client.wallet().hotkey.ss58_address
    print(f"probing netuid {cfg.netuid} on {args.network} as {hotkey[:12]}…")

    records: list[tuple[int, int]] = []
    for i, d in enumerate(int(x) for x in args.delays.split(",")):
        submit_block = client.current_block()
        payload = f"probe:reveal-jitter:{submit_block}:{i}"  # parse_commit-invalid on purpose
        client.commit_submission(payload, blocks_until_reveal=d)
        print(f"probe {i}: committed at block {submit_block}, target reveal {submit_block + d} "
              f"(delay {d}) — waiting…")
        if args.inspect_pending:
            _dump_pending_commitment(client, cfg.netuid, hotkey, f"probe {i}, pre-reveal")
        revealed = _await_reveal(client, hotkey, submit_block, args.timeout_s)
        if revealed is None:
            print(f"probe {i}: NOT revealed within {args.timeout_s}s — treat as a "
                  "reliability red flag, not just jitter.")
            continue
        late = revealed - (submit_block + d)
        print(f"probe {i}: revealed at {revealed} (lateness {late:+d} block(s))")
        if args.inspect_pending:
            _dump_pending_commitment(client, cfg.netuid, hotkey, f"probe {i}, post-reveal")
        records.append((d, late))

    print()
    print(summarize(records, cfg.round.reveal_margin_blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
