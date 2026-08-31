"""``cascade`` console-script: ``verify``, ``deploy``, ``fetch``, ``score``, and ``round``.

* ``cascade verify <repo_dir>`` — run every check the trainer runs before it
  trains on your generator, including the determinism check. Returns non-zero
  if anything would reject. ``--skip-runtime`` runs the static checks only.

* ``cascade score <repo_dir>`` — train the fixed model on your generator's data
  at the cheap heat budget and score it on a local/sample pool, entirely offline
  (no chain, no TAO, no ~12h round). The fast iteration loop; needs the
  ``[train]`` extra. Trains from random init — live rounds train from the
  promoted cascade warm-start once a generation is live, so compare against
  ``cascade score ./king`` on the same pool, not against live heat numbers.
  See ``cascade/miner/score.py``.

* ``cascade deploy <repo_dir> --hub-repo <namespace/name>`` — verify the local
  generator, push it to your Hippius Hub repo, and commit
  ``metro-v1:gen:hippius:<repo>@<digest>`` via ``set_reveal_commitment``. The OCI
  digest content-addresses your submission, so ``repo@digest`` both locates and
  pins it (no separate git SHA). The timelock reveal defaults to TIMED: the
  payload decrypts ``[round] reveal_margin_blocks`` before the next epoch
  boundary, so the submission stays hidden for its whole window and cannot be
  copied into its own round (``--reveal-now`` / ``--blocks-until-reveal`` /
  ``--next-epoch`` override). Pair with ``--hub-namespace`` (a fresh
  non-guessable repo per submission) so the content is as undiscoverable as the
  pointer — see docs/MINER.md "Protecting your submission". Requires the ``[chain]`` extra (bittensor) + a
  wallet, and the ``[hippius]`` extra + Hub credentials in the environment.
  ``--hf-repo <namespace/name>`` is a HuggingFace fallback (``repo@hf:<sha>``) used
  ONLY if the Hub push fails — the Hub is always tried first, so a healthy Hippius
  always wins (you cannot bypass it while it's up). The chain commit and the
  trainer's fetch/audit treat an ``hf:`` ref exactly like a Hub one.

* ``cascade reveal-status <hotkey|uid>`` — check whether a timelock reveal has
  landed and which round it is eligible for; with ``--expect-boundary`` (deploy
  prints it) a reveal that jittered past its target is reported as a LOUD miss
  instead of failing silently. ``--watch`` polls until it lands. Read-only, no
  wallet.

* ``cascade fetch king`` (or a uid / hotkey / ``repo@digest``) — download a
  competitor's on-chain generator to a local dir so you can inspect or fork it.
  Generators are content-addressed and public by design (the whole eval is
  re-derivable), so the reigning king's data process is open to study — that is
  the competition: beat the visible best, don't hide. Read-only; no wallet
  needed, only the ``[chain]``/``[hippius]`` extras + Hub credentials.

* ``cascade round`` — a live terminal dashboard counting down to the next
  round: current block, epoch progress, and the submission deadline (the next
  epoch boundary — commit strictly before it to enter that round). Also shows
  where the round roughly is (``heat ▸ duel ▸ validation ▸ settled`` —
  estimated from the configured budgets, confirmed settled via the public
  receipt index) and a live feed of revealed on-chain submissions (a commit
  landing while you watch is flagged ``● new``). Once the round's heat settles
  it also shows the heat standings inline (``--hotkey`` marks your row), and a
  warm-started round shows which promoted cascade init it trains from.
  Ticks every second, re-syncing to the chain every ``--refresh`` seconds;
  ``--once`` prints a single snapshot (also the automatic behaviour when
  piped). Read-only; needs the ``[chain]`` extra, no wallet.

* ``cascade heat`` — the heat standings for a round: every entrant's rank, its
  score relative to the best entrant, its raw CRPS/MASE on the round's eval-pool
  slice, and whether it advanced. Published by the trainer the moment the heat
  settles — hours before the round's receipt, and the ONLY place they appear for
  a round later rejected at a gate. ``--hotkey`` marks your row, ``--round``
  reads an archived round, ``--history`` lists what has been published.
  Read-only: no wallet, no chain call, no credentials.

* ``cascade duel`` — the full verdict for a settled round, from the public
  receipt index: dethrone margin (LCB vs required), both geomeans, win rate,
  bootstrap quantiles, per-domain win rates, and per-validator agreement
  (rejected validator rows are shown with their reason). ``--round`` reads an
  archived round, ``--history`` lists every settled round's outcome.
  Read-only: no wallet, no chain call, no credentials.

* ``cascade fund <intake_url> --ref <repo@digest>`` — fund your revealed
  submission's training leg with YOUR Lium API key (DEC-CA-0036). The key is
  read from the environment (``$LIUM_API_KEY`` by default; never pass it on
  the command line — argv is world-readable) and travels only as the
  ``X-Lium-Api-Key`` header of one authenticated POST to the operator's
  intake; the request is signed by your hotkey, so nobody can fund (or
  withdraw) on your behalf. ``--withdraw`` exits a still-queued entry and
  makes the operator forget your key. Needs the ``[chain]`` extra (wallet
  signing only — no chain connection is made).

* ``cascade submit <repo_dir> <intake_url>`` — the DIRECT path (DEC-CA-0036):
  verify locally, ZIP deterministically, POST the code straight to the
  operator's intake (with your Lium key in the same request when
  ``$LIUM_API_KEY`` is set — one request submits AND funds), then chain-commit
  the returned ``vault/direct@sha256:…`` ref with the usual timed reveal.
  Nothing is miner-hosted and your code stays PRIVATE unless it takes the
  throne — champions publish to ``champions/`` per the operator's policy
  (crown / delay / dethrone); losers never do. ``cascade fetch king``
  resolves a published champion anonymously.

Exit codes: 0 = success, 1 = checked but rejected, 2 = bad CLI usage, 3 =
chain/network failure, 4 = registry upload/fetch failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..interface.validation import format_commit, parse_commit
from ..shared.config import effective_epoch_blocks, load_chain_config
from .verify import verify_repo


def _add_verify(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("verify", help="Run all pre-submission checks on a local generator repo.")
    p.add_argument("repo_dir", type=Path, help="Path to your prepared HF generator repo.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument(
        "--skip-runtime",
        action="store_true",
        help="Skip the determinism (corpus build) check; static checks only.",
    )
    p.set_defaults(func=_cmd_verify)


def _add_deploy(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("deploy", help="Upload your generator to Hippius and commit it on-chain.")
    p.add_argument("repo_dir", type=Path, help="Path to your prepared generator repo (local dir).")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--network", default="finney", help="Bittensor network (finney/test/local).")
    p.add_argument("--wallet-name", required=True, help="Bittensor wallet (coldkey) name.")
    p.add_argument("--wallet-hotkey", required=True, help="Bittensor wallet hotkey name.")
    p.add_argument("--wallet-path", default=None, help="Optional non-default wallet root.")
    p.add_argument(
        "--blocks-until-reveal",
        type=int,
        default=None,
        help="Explicit timelock reveal delay in blocks. Default: TIMED REVEAL — the "
        "payload decrypts just before the next epoch boundary ([round] "
        "reveal_margin_blocks early), so your submission stays hidden for its whole "
        "window and competitors cannot copy it into the same round.",
    )
    p.add_argument(
        "--reveal-now",
        action="store_true",
        help="Reveal immediately (blocks_until_reveal=1) instead of the timed default. "
        "Your pointer is public for the rest of the window — copyable into this round.",
    )
    p.add_argument(
        "--next-epoch",
        action="store_true",
        help="Time the reveal for the FOLLOWING epoch boundary instead of the imminent "
        "one — a guaranteed-hidden window when you'd otherwise commit inside the "
        "reveal margin, at the cost of sitting out the imminent round.",
    )
    p.add_argument("--skip-verify", action="store_true", help="Skip the local verify before upload.")
    p.add_argument(
        "--hub-repo",
        default=None,
        help="Your Hippius Hub repo id (namespace/name) to push the generator to.",
    )
    p.add_argument(
        "--hub-namespace",
        default=None,
        help="Push to a FRESH, non-guessable repo under this Hub namespace "
        "(gen-<random hex>) instead of a fixed --hub-repo name. Recommended: a "
        "predictable repo name lets competitors watch your namespace and copy the "
        "generator content before the on-chain pointer ever reveals.",
    )
    p.add_argument(
        "--hf-repo",
        default=None,
        help="A HuggingFace model repo (namespace/name) used ONLY as a fallback if the "
        "Hippius Hub push fails — the Hub is always tried first, so a healthy Hippius "
        "always wins. Requires --hub-repo. Needs HF_TOKEN. The resulting repo@hf:<sha> "
        "ref trains + audits like a Hub one.",
    )
    p.add_argument(
        "--ref",
        default=None,
        help="Skip the upload and commit this already-uploaded ref (repo@digest) directly.",
    )
    p.set_defaults(func=_cmd_deploy)


def _add_score(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "score",
        help="Train the fixed model on your generator at the heat budget and score it "
        "locally (offline, minutes) — the fast iteration loop.",
    )
    p.add_argument("repo_dir", type=Path, help="Path to your generator repo.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--pool-dir", type=Path, default=None,
                   help="Local dir of .npy/.npz held-out series to score on (recommended: your "
                        "own real data). Falls back to --pool, then an offline synthetic sample.")
    p.add_argument("--pool", default="", dest="pool_ref",
                   help="A Hippius Hub pool ref (repo@digest) to score on instead of --pool-dir.")
    p.add_argument("--train-hours", type=float, default=None,
                   help="Training budget (default: [round] heat_train_hours — the cheap screen).")
    p.add_argument("--n-windows", type=int, default=None,
                   help="Eval windows to score on (default: [round] heat_n_windows).")
    p.add_argument("--device", default="cpu", help="Torch device (cuda recommended).")
    p.add_argument("--seed", type=int, default=0, help="Round seed (fixes generation + training).")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip the pre-score determinism/guard check.")
    p.set_defaults(func=_cmd_score)


def _cmd_score(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)
    if not args.skip_verify:
        report = verify_repo(args.repo_dir, cfg, skip_runtime=False)
        if not report.ok:
            print("verify failed — fix before scoring:", file=sys.stderr)
            print(report.render(), file=sys.stderr)
            return 1
    try:
        r = _run_score(args, cfg)
    except ImportError as e:
        print(f"error: `cascade score` needs the [train] extra (torch): {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — surface any train/eval failure cleanly
        print(f"scoring failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(
        f"\nscore: geomean={r.geomean:.5f}  (lower is better)\n"
        f"  pool:    {r.pool_label}  ({r.n_windows} windows)\n"
        f"  corpus:  {r.n_series} series, digest {r.corpus_digest[:12]}…\n"
        f"  trained: {r.train_seconds:.0f}s\n"
        f"\ncompare against the king:  cascade fetch king --out ./king && "
        f"cascade score ./king --pool-dir <same pool>"
    )
    return 0


def _run_score(args: argparse.Namespace, cfg):
    from .score import score_generator

    return score_generator(
        args.repo_dir, cfg, pool_dir=args.pool_dir, pool_ref=args.pool_ref,
        train_hours=args.train_hours, n_windows=args.n_windows, device=args.device,
        seed=args.seed,
    )


def _add_fetch(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "fetch",
        help="Download a competitor's on-chain generator (king / uid / hotkey / repo@digest).",
    )
    p.add_argument(
        "target",
        help="'king' (the highest-incentive UID), a miner UID (int), a hotkey (ss58), "
        "or a raw Hippius ref (repo@digest, which skips the chain lookup).",
    )
    p.add_argument("--out", type=Path, default=None,
                   help="Directory to download into (default: ./fetched-<name>).")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--network", default="finney", help="Bittensor network (finney/test/local).")
    p.add_argument("--verify", action="store_true",
                   help="Run `cascade verify` on the fetched generator after downloading.")
    p.set_defaults(func=_cmd_fetch)


def _resolve_fetch_ref(target: str, cfg, network: str) -> tuple[str, str]:
    """Resolve a fetch target to ``(ref, label)``.

    A ``repo@digest`` is returned as-is (no chain needed). Otherwise the chain is
    queried: ``king`` → the highest-incentive UID; an integer → that UID; anything
    else → a hotkey (ss58). Raises ``ValueError`` if the target can't be resolved
    to a committed generator.
    """
    from ..shared.hippius import is_hub_ref

    if is_hub_ref(target):
        return target, target.split("@")[0].replace("/", "-")

    from ..shared.chain import ChainClient

    client = ChainClient.from_config(cfg, network=network)
    commitments = client.poll_commitments()
    by_uid = {c.uid: c for c in commitments}
    by_hotkey = {c.hotkey: c for c in commitments}

    if target.lower() == "king":
        king_hk = client.highest_incentive_hotkey()
        if king_hk is None:
            raise ValueError("no king on the metagraph (vacant throne / empty subnet)")
        commit = by_hotkey.get(king_hk)
        if commit is None:
            raise ValueError(f"king {king_hk[:12]}… has no committed generator this round")
        label = f"king-uid{commit.uid}"
    elif target.isdigit():
        commit = by_uid.get(int(target))
        if commit is None:
            raise ValueError(f"uid {target} has no committed generator")
        label = f"uid{target}"
    else:
        commit = by_hotkey.get(target)
        if commit is None:
            raise ValueError(f"hotkey {target} has no committed generator")
        label = f"{target[:10]}"

    ref = commit.payload.split("hippius:")[-1].strip()
    if not is_hub_ref(ref):
        raise ValueError(f"commitment for {label} is not a valid generator ref: {commit.payload!r}")
    return ref, label


def _cmd_fetch(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)
    from ..shared.chain import ChainError

    try:
        ref, label = _resolve_fetch_ref(args.target, cfg, args.network)
    except ChainError as e:
        print(f"chain error: {e}", file=sys.stderr)
        return 3
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    out = args.out or Path(f"./fetched-{label}")
    print(f"fetching {ref}\n  → {out}")
    import os

    from ..funding.store import CHAMPION_BASE_ENV, is_vault_ref
    from ..shared.hippius import HubConfig, StorageError, fetch_from_hub

    if is_vault_ref(ref) and not os.environ.get(CHAMPION_BASE_ENV):
        # A direct (vault) submission resolves publicly ONLY through the
        # published champions/ objects — point the fetch at the same
        # anonymous endpoint the dashboards read.
        endpoint = str(getattr(cfg.storage, "s3_endpoint", "") or "").rstrip("/")
        bucket = str(getattr(cfg.storage, "manifest_bucket", "") or "")
        if endpoint and bucket:
            os.environ[CHAMPION_BASE_ENV] = f"{endpoint}/{bucket}"
    try:
        dest = fetch_from_hub(ref, out, HubConfig.from_storage(cfg.storage))
    except StorageError as e:
        if is_vault_ref(ref):
            print("fetch failed: this is a direct (private) submission and its "
                  "code has not been published to champions/ yet — under the "
                  "operator's champion_publish policy it goes public on crown, "
                  "after a reign delay, or at dethronement.", file=sys.stderr)
            print(f"  detail: {e}", file=sys.stderr)
            return 4
        print(f"registry fetch failed: {e}", file=sys.stderr)
        return 4
    files = sorted(p.name for p in dest.iterdir()) if dest.is_dir() else []
    print(f"fetched {label}: {ref}\n  {len(files)} top-level entries: {', '.join(files[:12])}")

    if args.verify:
        report = verify_repo(dest, cfg, skip_runtime=False)
        print(report.render())
        return 0 if report.ok else 1
    print(f"\ninspect it, or fork + improve it:  cascade verify {out}")
    return 0


def _add_round(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "round",
        help="Live round dashboard: deadline countdown, current stage "
        "(heat/duel/validation/settled), and revealed submissions.",
    )
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--network", default="finney", help="Bittensor network (finney/test/local).")
    p.add_argument("--once", action="store_true",
                   help="Print a single snapshot instead of the live countdown.")
    p.add_argument("--refresh", type=float, default=30.0,
                   help="Seconds between chain re-syncs in watch mode (default: 30).")
    p.add_argument("--hotkey", default=None,
                   help="Your hotkey (ss58) or UID — marks your row '← you' in the "
                        "heat standings and always shows it, however far down you placed.")
    p.set_defaults(func=_cmd_round)


def _cmd_round(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)
    from ..shared.chain import ChainClient, ChainError
    from .dashboard import (
        RoundTimeline,
        fetch_public_heat,
        fetch_public_receipt_index,
        fetch_public_round_status,
        run_dashboard,
    )

    try:
        client = ChainClient.from_config(cfg, network=args.network)
        return run_dashboard(
            client, cfg.round, args.network, once=args.once, refresh=args.refresh,
            timeline=RoundTimeline.from_chain_config(cfg),
            index_fetch=lambda: fetch_public_receipt_index(cfg.storage),
            status_fetch=lambda: fetch_public_round_status(cfg.storage),
            heat_fetch=lambda: fetch_public_heat(cfg.storage),
            me=args.hotkey,
            scoring=cfg.scoring,
        )
    except ChainError as e:
        print(f"chain error: {e}", file=sys.stderr)
        return 3


def _add_heat(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "heat",
        help="Heat standings: where every entrant placed in the cheap screen "
        "(published as soon as the heat settles, before the duel finishes).",
    )
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--hotkey", default=None,
                   help="Your hotkey (ss58) or UID — marks your row '← you'.")
    p.add_argument("--round", dest="round_id", default=None,
                   help="A specific round id (default: the latest published heat).")
    p.add_argument("--history", action="store_true",
                   help="List the published heats (heats/index.json) instead of one round.")
    p.add_argument("--limit", type=int, default=20,
                   help="Rounds listed by --history (default: 20).")
    p.set_defaults(func=_cmd_heat)


def _cmd_heat(args: argparse.Namespace) -> int:
    """Print the public heat standings — no wallet, no chain call, no credentials.

    The trainer publishes the standings when the heat settles, so this is a
    miner's feedback on the round it just entered while the duel is still
    training (the receipt lands hours later, and never at all if the round is
    rejected at a gate).
    """
    cfg = load_chain_config(args.chain_toml)
    from .dashboard import (
        fetch_public_heat,
        fetch_public_heat_index,
        fetch_public_heat_round,
        render_heat,
        render_heat_index,
    )

    if args.history:
        print(render_heat_index(fetch_public_heat_index(cfg.storage), limit=args.limit))
        return 0
    doc = (fetch_public_heat_round(cfg.storage, args.round_id) if args.round_id
           else fetch_public_heat(cfg.storage))
    if doc is None:
        target = f"round {args.round_id}" if args.round_id else "the latest heat"
        print(f"no published heat standings for {target} — the trainer writes them "
              "when a round's heat settles; try 'cascade heat --history'",
              file=sys.stderr)
        return 1
    print(render_heat(doc, me=args.hotkey))
    return 0


def _add_duel(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "duel",
        help="Duel verdict for a settled round: dethrone margin, both geomeans, "
        "per-domain win rates — the full breakdown behind DETHRONED/king-held.",
    )
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--round", dest="round_id", default=None,
                   help="A specific round id (default: the latest settled round).")
    p.add_argument("--history", action="store_true",
                   help="One line per settled round instead of one round's detail.")
    p.add_argument("--limit", type=int, default=20,
                   help="Rounds listed by --history (default: 20).")
    p.set_defaults(func=_cmd_duel)


def _cmd_duel(args: argparse.Namespace) -> int:
    """Print a settled round's full duel verdict — no wallet, no chain call, no
    credentials. Everything comes from the public receipt index the validators
    publish a few minutes after each duel; `cascade round` shows the same data
    as a one-line outcome."""
    cfg = load_chain_config(args.chain_toml)
    from .dashboard import (
        duel_round_rows,
        fetch_public_receipt_index,
        render_duel,
        render_duel_index,
    )

    doc = fetch_public_receipt_index(cfg.storage)
    if doc is None:
        print("could not fetch the public receipt index (receipts/index.json) — "
              "offline, or the round has not settled yet; try 'cascade round'",
              file=sys.stderr)
        return 1
    if args.history:
        print(render_duel_index(doc, limit=args.limit))
        return 0
    rows = duel_round_rows(doc, args.round_id)
    if not rows:
        target = f"round {args.round_id}" if args.round_id else "any settled round"
        print(f"no receipt-index rows for {target} — receipts land a few minutes "
              "after the duel manifest; try 'cascade duel --history'", file=sys.stderr)
        return 1
    print(render_duel(rows))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)
    report = verify_repo(args.repo_dir, cfg, skip_runtime=args.skip_runtime)
    print(report.render())
    return 0 if report.ok else 1


def _add_reveal_status(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "reveal-status",
        help="Check whether a hotkey's timelock reveal has landed and which round "
        "it is eligible for — catches a reveal that missed its target boundary.",
    )
    p.add_argument("hotkey", help="The miner hotkey (ss58) to check, or a UID (int).")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--network", default="finney", help="Bittensor network (finney/test/local).")
    p.add_argument(
        "--expect-boundary",
        type=int,
        default=None,
        help="The epoch boundary the deploy targeted (deploy prints it). With this "
        "set, a reveal landing at/after it is reported as a LOUD miss.",
    )
    p.add_argument("--watch", action="store_true",
                   help="Poll until the reveal lands (or --timeout-s expires).")
    p.add_argument("--timeout-s", type=int, default=3600,
                   help="Max seconds to watch for (default 1h).")
    p.set_defaults(func=_cmd_reveal_status)


def _reveal_verdict(
    reveal_block: int,
    current_block: int,
    epoch_blocks: int,
    margin_blocks: int,
    expect_boundary: int | None = None,
) -> tuple[bool, str]:
    """Judge a landed reveal: ``(missed, human-readable report)``.

    A reveal is eligible for the round locking at the first epoch boundary
    STRICTLY AFTER it (the trainer's cutoff rule); ``missed`` is True only when
    ``expect_boundary`` was given and the reveal landed at/after it. Pure —
    unit-testable without a chain."""
    eligible_boundary = (reveal_block // epoch_blocks + 1) * epoch_blocks
    lead = eligible_boundary - reveal_block
    lines = [f"revealed at block {reveal_block} — eligible for the round locking at "
             f"block {eligible_boundary} ({lead} blocks of pre-boundary exposure)"]
    if lead > margin_blocks:
        lines.append(f"note: exposure exceeds the {margin_blocks}-block reveal margin — "
                     "the ref was copyable for longer than the timed default allows.")
    if current_block >= eligible_boundary:
        lines.append("that round's field has locked; the submission is in it (or was, "
                     "if since replaced).")
    else:
        lines.append(f"field locks in {eligible_boundary - current_block} block(s).")
    missed = expect_boundary is not None and reveal_block >= expect_boundary
    if missed:
        lines.insert(0, f"⚠ MISSED the targeted boundary {expect_boundary}: the reveal "
                        f"landed {reveal_block - expect_boundary} block(s) at/after it.")
        lines.append("consequences: the submission auto-rolls into the NEXT round (no "
                     "re-commit needed) but its ref is public until then — a copy can "
                     "only tie it, never take its slot (earliest reveal wins), yet a "
                     "derived/tweaked fork is now possible. It has NOT consumed the "
                     "one-submission budget (that burns only after a heat screens it). "
                     "To re-hide "
                     "improved content instead, re-deploy: the latest reveal per hotkey "
                     "wins.")
    return missed, "\n".join(lines)


def _pending_timelock(client, hotkey: str) -> tuple[int, int] | None:
    """``(commit_block, reveal_round)`` of an un-revealed timelock commit, if any.

    The plain commitment store (``Commitments::CommitmentOf``) holds the
    encrypted payload until drand reveals it; once revealed the record is
    consumed. Without this check, reveal-status shows the hotkey's PREVIOUS
    reveal while a fresh timelock is pending — telling a miner their new
    submission doesn't exist (observed during the 2026-07-15 live test)."""
    try:
        sub = client.subtensor()
        q = sub.substrate.query(module="Commitments", storage_function="CommitmentOf",
                                params=[client.netuid, hotkey])
        v = getattr(q, "value", None) or {}
        for f in ((v.get("info") or {}).get("fields") or []):
            tl = f.get("TimelockEncrypted") if isinstance(f, dict) else None
            if tl is not None:
                return int(v.get("block") or 0), int(tl.get("reveal_round") or 0)
    except Exception:  # noqa: BLE001 — advisory; never break the status report
        return None
    return None


def _cmd_reveal_status(args: argparse.Namespace) -> int:
    import time

    cfg = load_chain_config(args.chain_toml)
    from ..shared.chain import ChainClient, ChainError

    client = ChainClient.from_config(cfg, network=args.network)
    deadline = time.monotonic() + args.timeout_s

    try:
        while True:
            commitments = client.poll_commitments()
            if args.hotkey.isdigit():
                match = next((c for c in commitments if c.uid == int(args.hotkey)), None)
            else:
                match = next((c for c in commitments if c.hotkey == args.hotkey), None)
            pending = None if args.hotkey.isdigit() else _pending_timelock(client, args.hotkey)
            if pending is not None and (match is None or pending[0] > match.commit_block):
                blk, rnd = pending
                print(f"PENDING timelock commit (committed at block {blk}, drand round "
                      f"{rnd}) — payload still encrypted on-chain; nothing copyable yet."
                      + (f" Latest REVEALED entry below is the PREVIOUS submission "
                         f"(block {match.commit_block})." if match else ""))
                if args.watch and time.monotonic() < deadline:
                    time.sleep(12)
                    continue
                if match is None:
                    return 0
            if match is not None:
                missed, report = _reveal_verdict(
                    match.commit_block, client.current_block(),
                    effective_epoch_blocks(cfg.round, match.commit_block),
                    cfg.round.reveal_margin_blocks,
                    args.expect_boundary,
                )
                print(report)
                return 1 if missed else 0
            if not args.watch or time.monotonic() >= deadline:
                print("no revealed commitment for that hotkey — still timelock-hidden, "
                      "never committed, or not registered on the netuid."
                      + ("" if args.watch else " (--watch polls until it lands.)"))
                return 0 if not args.watch else 1
            time.sleep(12)
    except ChainError as e:
        print(f"chain error: {e}", file=sys.stderr)
        return 3


def _upload_generator(args: argparse.Namespace, cfg) -> tuple[int, str | None]:
    """Upload the generator and return ``(exit_code, ref)``. Hippius is priority one:
    the Hub (``--hub-repo``, required) is ALWAYS tried first, so a healthy Hippius
    always wins. Only if the Hub push fails does it fall back to a HuggingFace mirror
    (``--hf-repo``), so a miner can still submit through a Hub outage. Returns
    ``(0, ref)`` on success, else ``(4, None)``."""
    from ..shared.hippius import (
        HubConfig,
        StorageError,
        upload_dir_to_hf,
        upload_dir_to_hub,
    )

    try:
        up = upload_dir_to_hub(args.repo_dir, args.hub_repo, HubConfig.from_storage(cfg.storage))
        print(f"pushed to Hippius Hub: {up.ref.immutable_ref} ({up.size_bytes} bytes)")
        return 0, up.ref.immutable_ref
    except StorageError as e:
        hub_err = str(e)  # bind now — the `as` name is cleared at except-block exit
        if not args.hf_repo:
            print(f"registry upload failed: {e}", file=sys.stderr)
            return 4, None
        print(f"Hippius Hub upload failed ({e});\n"
              f"  falling back to HuggingFace mirror {args.hf_repo}", file=sys.stderr)
        print("warning: HuggingFace repos are PUBLIC and enumerable — anyone watching "
              "your HF account sees this generator's content now, before the on-chain "
              "pointer reveals. Prefer retrying the Hub for a competitive submission.",
              file=sys.stderr)

    try:
        up = upload_dir_to_hf(args.repo_dir, args.hf_repo)
    except StorageError as e:
        print(f"HuggingFace mirror upload failed: {e} (Hub also failed: {hub_err})",
              file=sys.stderr)
        return 4, None
    print(f"mirrored to HuggingFace: {up.ref.immutable_ref} ({up.size_bytes} bytes)")
    return 0, up.ref.immutable_ref


def _fresh_hub_repo(namespace: str) -> str:
    """A non-guessable, single-use Hub repo id under ``namespace``.

    Content is public-by-ref once fetched, but an unpredictable repo name keeps
    the generator undiscoverable while its on-chain pointer is still
    timelock-hidden — a predictable name lets competitors poll the namespace
    and copy the content before the reveal."""
    import secrets

    return f"{namespace}/gen-{secrets.token_hex(6)}"


def _resolve_blocks_until_reveal(args: argparse.Namespace, cfg, current_block: int) -> int:
    """The reveal delay for this deploy: an explicit ``--blocks-until-reveal``
    wins, ``--reveal-now`` forces 1, and the default is the TIMED reveal —
    ``next epoch boundary − [round] reveal_margin_blocks`` (see
    :func:`cascade.shared.chain.blocks_until_boundary_reveal`), floored to
    reveal-now when already inside the margin. Flag validation (mutual
    exclusion) happens in ``_cmd_deploy`` before any chain connection."""
    from ..shared.chain import blocks_until_boundary_reveal

    if args.blocks_until_reveal is not None:
        return int(args.blocks_until_reveal)
    if args.reveal_now:
        return 1
    epoch_blocks = effective_epoch_blocks(cfg.round, current_block)
    delay = blocks_until_boundary_reveal(
        current_block,
        epoch_blocks,
        cfg.round.reveal_margin_blocks,
        next_epoch=args.next_epoch,
    )
    target = current_block + delay
    boundary = (target // epoch_blocks + 1) * epoch_blocks
    print(
        f"timed reveal: payload decrypts ~block {target} "
        f"({delay} blocks from now, {boundary - target} blocks before the epoch "
        f"boundary at {boundary}) — hidden until the field locks. "
        f"Override with --reveal-now / --blocks-until-reveal / --next-epoch."
    )
    return delay


def _cmd_deploy(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)

    if args.reveal_now and (args.blocks_until_reveal is not None or args.next_epoch):
        print("error: --reveal-now conflicts with --blocks-until-reveal / --next-epoch.",
              file=sys.stderr)
        return 2
    if args.next_epoch and args.blocks_until_reveal is not None:
        print("error: --next-epoch conflicts with an explicit --blocks-until-reveal.",
              file=sys.stderr)
        return 2
    if args.hub_repo and args.hub_namespace:
        print("error: pass --hub-repo OR --hub-namespace, not both.", file=sys.stderr)
        return 2
    if args.hub_namespace:
        args.hub_repo = _fresh_hub_repo(args.hub_namespace)
        print(f"fresh submission repo: {args.hub_repo}")

    ref = args.ref
    if ref is None:
        if not args.hub_repo:
            print("error: --hub-repo or --hub-namespace (Hippius Hub) is required to "
                  "upload — the Hub is always tried first. --hf-repo is only a fallback "
                  "for when the Hub push fails; pass it alongside one of them. Or use "
                  "--ref to commit an already-uploaded ref.", file=sys.stderr)
            return 2
        # Verify locally (cheaper than burning a chain commit), then upload.
        if not args.skip_verify:
            report = verify_repo(args.repo_dir, cfg, skip_runtime=False)
            if not report.ok:
                print("local verify failed — refusing to deploy:", file=sys.stderr)
                print(report.render(), file=sys.stderr)
                return 1

        rc, ref = _upload_generator(args, cfg)
        if rc != 0:
            return rc

    try:
        payload = format_commit(ref)
    except ValueError as e:
        print(f"refusing to deploy: {e}", file=sys.stderr)
        return 2
    assert parse_commit(payload) is not None  # format_commit guarantees this

    from ..shared.chain import ChainClient, ChainError

    try:
        client = ChainClient.from_config(
            cfg,
            network=args.network,
            wallet_name=args.wallet_name,
            wallet_hotkey=args.wallet_hotkey,
            wallet_path=args.wallet_path,
        )
        current_block = client.current_block()
        blocks_until_reveal = _resolve_blocks_until_reveal(args, cfg, current_block)
        client.commit_submission(payload, blocks_until_reveal=blocks_until_reveal)
    except ChainError as e:
        print(f"chain error: {e}", file=sys.stderr)
        return 3
    except ValueError as e:
        # blocks_until_boundary_reveal rejects inconsistent [round] config
        # (e.g. reveal_margin_blocks >= epoch_blocks).
        print(f"bad [round] reveal config: {e}", file=sys.stderr)
        return 2

    print(f"committed: {payload}")
    if args.blocks_until_reveal is None and not args.reveal_now:
        # A timed reveal that jitters past its boundary silently misses the
        # round — hand the miner the exact command that catches it loudly.
        target = current_block + blocks_until_reveal
        _eb = effective_epoch_blocks(cfg.round, current_block)
        boundary = (target // _eb + 1) * _eb
        try:
            hotkey = client.wallet().hotkey.ss58_address
        except Exception:  # noqa: BLE001 — a hint must never fail the deploy
            hotkey = "<your-hotkey-ss58>"
        print(f"confirm the reveal lands in time (from ~block {target}):\n"
              f"  cascade reveal-status {hotkey} --network {args.network} "
              f"--expect-boundary {boundary} --watch")
    return 0


def _add_submit(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "submit",
        help="Verify, ZIP, and submit your generator DIRECTLY to the operator's "
             "intake — private until it takes the throne — then commit the "
             "returned vault ref on-chain.",
    )
    p.add_argument("repo_dir", type=Path, help="Path to your prepared generator repo.")
    p.add_argument("intake_url", help="Operator intake base URL (https://…).")
    p.add_argument("--chain-toml", type=Path, default=None)
    p.add_argument("--network", default="finney")
    p.add_argument("--wallet-name", required=True)
    p.add_argument("--wallet-hotkey", required=True)
    p.add_argument("--wallet-path", default=None)
    p.add_argument("--lium-key-env", default="LIUM_API_KEY",
                   help="Env var holding your Lium key: when set, the SAME request "
                        "funds your entry (auto-queues once the reveal lands).")
    p.add_argument("--no-fund", action="store_true",
                   help="Submit code only; fund later with `cascade fund`.")
    p.add_argument("--skip-runtime", action="store_true",
                   help="Skip the determinism check during pre-submit verify.")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip local verification entirely (the trainer still verifies).")
    p.add_argument("--no-commit", action="store_true",
                   help="Upload only; print the commit payload without touching the chain.")
    p.add_argument("--blocks-until-reveal", type=int, default=None,
                   help="Explicit timelock reveal delay (default: TIMED reveal, as deploy).")
    p.add_argument("--reveal-now", action="store_true")
    p.add_argument("--next-epoch", action="store_true")
    p.set_defaults(func=_cmd_submit)


def zip_repo_bytes(repo_dir: Path) -> bytes:
    """Deterministically ZIP the SUBMITTABLE part of a repo tree.

    Two properties matter:

    * Determinism (sorted paths, zeroed timestamps): re-zipping an unchanged
      tree yields identical bytes, so the sha256 the miner signs — and the
      vault ref the chain commit pins — is a property of the CODE.
    * The SAME file filter as the Hub upload path (``hippius.ALLOW_PATTERNS``,
      plus dropping dotted dirs and ``__pycache__``): a raw ``rglob('*')``
      would pack ``.git`` packfiles — commit history, committer emails, any
      secret ever committed — into the operator store and, on a throne, into
      PUBLIC ``champions/`` (review 2026-08-29). Matching the Hub filter also
      keeps the two channels' trees identical, so the dedup screen's tree
      tier compares like with like.
    """
    import fnmatch
    import io
    import zipfile

    from ..shared.hippius import ALLOW_PATTERNS

    d = Path(repo_dir)
    if not d.is_dir():
        raise ValueError(f"not a directory: {d}")

    def _wanted(p: Path) -> bool:
        rel = p.relative_to(d)
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            return False
        rel_posix = rel.as_posix()
        return any(fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(p.name, pat)
                   for pat in ALLOW_PATTERNS)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(x for x in d.rglob("*") if x.is_file() and _wanted(x)):
            info = zipfile.ZipInfo(p.relative_to(d).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, p.read_bytes())
    return buf.getvalue()


def _cmd_submit(args: argparse.Namespace) -> int:
    import hashlib
    import os
    import urllib.error
    import urllib.request

    base = args.intake_url.rstrip("/")
    if not _intake_transport_ok(base):
        print("refusing to submit over plain http to a non-local intake; use https",
              file=sys.stderr)
        return 2
    cfg = load_chain_config(args.chain_toml)
    if not args.skip_verify:
        report = verify_repo(args.repo_dir, cfg, skip_runtime=args.skip_runtime)
        print(report.render())
        if not report.ok:
            print("refusing to submit: verification failed (fix, or --skip-verify "
                  "to send anyway — the trainer will reject the same faults)",
                  file=sys.stderr)
            return 1

    try:
        import bittensor  # signing only — the chain connects later, if committing
        wallet = bittensor.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey,
                                  path=args.wallet_path)
        hotkey_ss58 = wallet.hotkey.ss58_address
        sign_fn = wallet.hotkey.sign
    except Exception as e:  # noqa: BLE001 — wallet errors are usage errors here
        print(f"could not load wallet for signing: {e}", file=sys.stderr)
        return 2

    try:
        body = zip_repo_bytes(args.repo_dir)
    except (ValueError, OSError) as e:
        print(f"could not package the repo: {e}", file=sys.stderr)
        return 2
    digest = f"sha256:{hashlib.sha256(body).hexdigest()}"
    import time as _time

    from ..funding.intake import canonical_fund_message

    ts = str(int(_time.time()))
    headers = {
        "X-Miner-Hotkey": hotkey_ss58,
        "X-Content-Digest": digest,
        "X-Timestamp": ts,
        "X-Signature": sign_fn(canonical_fund_message("submit", hotkey_ss58, digest, ts)).hex(),
        "Content-Type": "application/zip",
    }
    if not args.no_fund:
        api_key = (os.environ.get(args.lium_key_env) or "").strip()
        if api_key:
            headers["X-Lium-Api-Key"] = api_key
        else:
            print(f"note: ${args.lium_key_env} not set — submitting unfunded "
                  f"(fund later with `cascade fund`)")

    req = urllib.request.Request(f"{base}/v1/submit", data=body, method="POST",
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp_body = _decode_json_body(resp.read())
            status = resp.status
    except urllib.error.HTTPError as e:
        resp_body, status = _decode_json_body(e.read()), e.code
    except (urllib.error.URLError, OSError) as e:
        print(f"intake unreachable: {e}", file=sys.stderr)
        return 3
    if not (200 <= status < 300):
        print(f"submit rejected ({status}): {resp_body.get('code', '?')} — "
              f"{resp_body.get('message', '')}", file=sys.stderr)
        return 1
    ref = resp_body["ref"]
    payload = resp_body["commit_payload"]
    print(f"stored privately: {ref} (funding={resp_body.get('funding', 'none')})")
    if args.no_commit:
        print(f"commit it yourself when ready:\n  payload: {payload}")
        return 0

    from ..shared.chain import ChainClient, ChainError

    try:
        client = ChainClient.from_config(
            cfg, network=args.network, wallet_name=args.wallet_name,
            wallet_hotkey=args.wallet_hotkey, wallet_path=args.wallet_path,
        )
        current_block = client.current_block()
        blocks_until_reveal = _resolve_blocks_until_reveal(args, cfg, current_block)
        client.commit_submission(payload, blocks_until_reveal=blocks_until_reveal)
    except ChainError as e:
        print(f"chain error: {e} — your code IS stored; commit later with:\n"
              f"  payload: {payload}", file=sys.stderr)
        return 3
    except ValueError as e:
        print(f"bad [round] reveal config: {e}", file=sys.stderr)
        return 2
    print(f"committed: {payload}")
    print("your code stays PRIVATE unless it takes the throne (champion_publish "
          "policy); once revealed on-chain, a funded entry queues automatically.")
    return 0


def _add_fund(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "fund",
        help="Fund your revealed submission's training leg with YOUR Lium API key.",
    )
    p.add_argument("intake_url",
                   help="Operator intake base URL (https://…; TLS protects the key in transit).")
    p.add_argument("--ref", required=True,
                   help="The revealed generator ref this funds (repo@digest — what you deployed).")
    p.add_argument("--wallet-name", required=True, help="Bittensor wallet (coldkey) name.")
    p.add_argument("--wallet-hotkey", required=True, help="Bittensor wallet hotkey name.")
    p.add_argument("--wallet-path", default=None, help="Optional non-default wallet root.")
    p.add_argument("--lium-key-env", default="LIUM_API_KEY",
                   help="Environment variable holding your Lium API key (default "
                        "LIUM_API_KEY). The key is NEVER accepted on the command line.")
    p.add_argument("--withdraw", action="store_true",
                   help="Withdraw a still-queued entry (the operator forgets your key).")
    p.set_defaults(func=_cmd_fund)


def build_fund_headers(action: str, hotkey_ss58: str, ref: str, api_key: str,
                       sign_fn, *, now=None) -> dict[str, str]:
    """The signed header set for one intake request (pure; testable).

    ``sign_fn(message: bytes) -> bytes`` is the hotkey's sr25519 signer. The
    canonical message binds action + hotkey + ref + timestamp, so a captured
    fund request cannot be replayed as a withdraw (or vice versa), and none of
    it can be replayed at all past the intake's freshness window.
    """
    import time as _time

    from ..funding.intake import canonical_fund_message

    ts = str(int((now or _time.time)()))
    headers = {
        "X-Miner-Hotkey": hotkey_ss58,
        "X-Commit-Ref": ref,
        "X-Timestamp": ts,
        "X-Signature": sign_fn(canonical_fund_message(action, hotkey_ss58, ref, ts)).hex(),
    }
    if action == "fund":
        headers["X-Lium-Api-Key"] = api_key
    return headers


def _intake_transport_ok(url: str) -> bool:
    """True when the intake URL may carry the key: https, or genuinely local.

    The hostname is PARSED and compared exactly — a substring check would
    pass ``http://localhost.evil.example`` and ship the key in cleartext to a
    non-local box (audit 2026-08-29).
    """
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.scheme == "https":
        return True
    if parts.scheme == "http":
        return (parts.hostname or "") in ("127.0.0.1", "localhost", "::1")
    return False


def _decode_json_body(raw: bytes) -> dict:
    """Best-effort JSON body; a proxy's HTML error page must not traceback."""
    import json as _json

    try:
        body = _json.loads(raw or b"{}")
        return body if isinstance(body, dict) else {"message": str(body)}
    except ValueError:
        text = raw.decode("utf-8", errors="replace").strip()
        return {"code": "non_json_response", "message": text[:200]}


def _cmd_fund(args: argparse.Namespace) -> int:
    import os
    import urllib.error
    import urllib.request

    action = "withdraw" if args.withdraw else "fund"
    api_key = ""
    if action == "fund":
        api_key = (os.environ.get(args.lium_key_env) or "").strip()
        if not api_key:
            print(f"no Lium API key in ${args.lium_key_env} — export it first "
                  f"(never pass the key itself as an argument)", file=sys.stderr)
            return 2
    base = args.intake_url.rstrip("/")
    if not _intake_transport_ok(base):
        print("refusing to send your Lium key over plain http to a non-local intake; "
              "use https", file=sys.stderr)
        return 2

    try:
        import bittensor  # the [chain] extra; signing only — no connection
        wallet = bittensor.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey,
                                  path=args.wallet_path)
        hotkey_ss58 = wallet.hotkey.ss58_address
        sign_fn = wallet.hotkey.sign
    except Exception as e:  # noqa: BLE001 — wallet errors are usage errors here
        print(f"could not load wallet for signing: {e}", file=sys.stderr)
        return 2

    headers = build_fund_headers(action, hotkey_ss58, args.ref, api_key, sign_fn)
    req = urllib.request.Request(f"{base}/v1/{action}", method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = _decode_json_body(resp.read())
            status = resp.status
    except urllib.error.HTTPError as e:
        body, status = _decode_json_body(e.read()), e.code
    except (urllib.error.URLError, OSError) as e:
        print(f"intake unreachable: {e}", file=sys.stderr)
        return 3

    if 200 <= status < 300:
        print(f"{action}: {body.get('status', 'ok')} (hotkey {hotkey_ss58})")
        if action == "fund":
            print("your entry queues by reveal block; watch it via the intake's "
                  "/v1/queue or `cascade round`. An infra failure on your pod "
                  "re-queues you without burning the entry.")
        return 0
    print(f"{action} rejected ({status}): {body.get('code', '?')} — "
          f"{body.get('message', '')}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    from ..shared.env import load_env_files
    load_env_files()
    parser = argparse.ArgumentParser(prog="cascade", description="cascade subnet miner CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_verify(sub)
    _add_deploy(sub)
    _add_fetch(sub)
    _add_score(sub)
    _add_reveal_status(sub)
    _add_round(sub)
    _add_heat(sub)
    _add_duel(sub)
    _add_fund(sub)
    _add_submit(sub)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
