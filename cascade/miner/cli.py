"""``cascade`` — the miner CLI for both lanes. docs/MINER.md is the walkthrough;
AGENTS.md is the contract for coding agents that mine.

Generator lane (DEC-CA-0043):

* ``verify <repo>`` — every check the trainer runs before it trains on your
  generator, including determinism (``--skip-runtime``: static checks only).
* ``score <repo>`` — train the fixed model on your data at a cheap local budget
  and score it offline (``[train]`` extra); ``--warm-start live`` trains the
  way the live round does. See ``cascade/miner/score.py``.
* ``iterate <repo>`` — verify, score over seeds, score the king (or any
  ``--baseline``) once, cached, and compare. Never spends. See
  ``cascade/miner/iterate.py``.
* ``submit <repo> <intake>`` — verify, ZIP deterministically, POST privately to
  the operator's intake (funded with ``$LIUM_API_KEY`` when set), then commit
  the returned vault ref with a timed reveal. Code stays private unless it
  takes the throne.
* ``deploy <repo>`` — the public path: push to your Hippius Hub repo (HF only as
  a fallback) and commit ``repo@digest`` with a timed reveal (DEC-CA-0051
  encoding). ``--expert-domain`` commits a lane-B entry (DEC-CA-0050) — up to
  ``[lane_b] max_domains_per_leg`` comma-separated targets (DEC-CA-0059).
* ``fund <intake> --ref R`` — fund a revealed submission's leg with your Lium
  key (env only, never argv); ``--withdraw`` exits a queued entry.
* ``fetch king|UID|HOTKEY|repo@digest`` — download a public generator.
* ``reveal-status HOTKEY|UID`` — has the timelock reveal landed, and in time?
* ``round`` / ``queue`` / ``duel`` — the live countdown, the funded roster and
  settled verdicts, from public data (no wallet).

Direct-MoE lane (DEC-CA-0052): ``cascade direct status|deposit-key|fund``, see
``cascade/miner/direct.py``.

Fine-tuning lane (DEC-CA-0056; MarketCast-only: ``mkt:<suite>`` targets):
``cascade lane-c check|commit|fund|status``, see ``cascade/miner/lane_c.py``. Domain-expert lane funding (DEC-CA-0059 item
8): ``cascade lane-b fund|status``, see ``cascade/miner/lane_b.py``.

Every command takes ``--json`` (or ``CASCADE_JSON=1``): one JSON document on
stdout, human lines on stderr (``cascade/miner/output.py``). Paid actions
(``fund``, ``submit`` with a key, ``direct fund``) refuse without ``--yes``.

Exit codes: 0 success, 1 checked but rejected, 2 bad usage, 3 chain/network
failure, 4 registry upload/fetch failure, 5 paid action needs ``--yes`` (or is
over ``--max-spend``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..interface.validation import format_commit, parse_commit
from ..shared.config import effective_epoch_blocks, load_chain_config
from .output import (
    NEEDS_CONFIRM,
    NETWORK,
    OK,
    REGISTRY,
    REJECTED,
    USAGE,
    Reporter,
    json_default,
    reporter,
)
from .verify import verify_repo


def _add_verify(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("verify", help="Run all pre-submission checks on a local generator repo.")
    p.add_argument("repo_dir", type=Path, help="Path to your prepared HF generator repo.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    context = p.add_mutually_exclusive_group()
    context.add_argument("--live", action="store_true", help="Verify against fresh deployed contract and CPU-tier limits.")
    context.add_argument("--at-block", type=int, default=None,
                   help="Resolve the offline verification contract at this era-start block.")
    runtime = p.add_mutually_exclusive_group()
    runtime.add_argument("--quick", action="store_true",
                         help="Static checks plus two small deterministic corpora; 30s timeout. Not full admission verification.")
    runtime.add_argument(
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
    p.add_argument(
        "--expert-domain",
        default=None,
        metavar="DOMAIN",
        help="Lane B (v2, DEC-CA-0050): commit the generator as a DOMAIN-EXPERT entry "
        "(metro-v1:expert:<DOMAIN>:hippius:<ref>) instead of a lane-A entry. DOMAIN must "
        "be one of chain.toml [lane_b] domains — or a comma-separated list of up to "
        "[lane_b] max_domains_per_leg targets (DEC-CA-0059: domains and/or horizon "
        "targets horizon:<h> / h<h>, h a scored horizon), trained as ONE leg and judged "
        "and paid per target. A hotkey holds ONE commitment: this replaces your lane-A "
        "entry until you commit again.",
    )
    p.set_defaults(func=_cmd_deploy)


def _add_score(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "score",
        help="Train the fixed model on your generator at a cheap budget and score it "
        "locally (offline, minutes) — the fast iteration loop.",
    )
    p.add_argument("repo_dir", type=Path, help="Path to your generator repo.")
    p.add_argument("--chain-toml", type=Path, default=None,
                   help="Complete local configuration; can select a released smaller model for cheaper experiments.")
    add_score_options(p)
    p.add_argument("--seed", type=int, default=0, help="Round seed (fixes generation + training).")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip the pre-score determinism/guard check.")
    p.set_defaults(func=_cmd_score)


def add_score_options(p: argparse.ArgumentParser) -> None:
    """The local-scoring knobs shared by ``score`` and ``iterate``."""
    p.epilog = (
        "Improve generators cheaply with score or iterate on a smaller model once released, "
        "using a complete --chain-toml configuration. The planned 2M, 7M, 28M, 56M and ~120M "
        "variants are currently blocked placeholders. Compare candidate and baseline with "
        "the same model, budget, seeds, pool and compatible starting checkpoint. Recheck "
        "promising ideas with --warm-start live: it selects the live competition model and "
        "init. A small-model improvement does not establish a live win.")
    p.add_argument("--pool-dir", type=Path, default=None,
                   help="Local dir of .npy/.npz held-out series to score on (recommended: your "
                        "own real data). Falls back to --pool, then an offline synthetic sample.")
    p.add_argument("--pool", default="", dest="pool_ref",
                   help="A Hippius Hub pool ref (repo@digest) to score on instead of --pool-dir.")
    p.add_argument("--train-hours", type=float, default=None,
                   help="Training budget (default: [round] heat_train_hours — the cheap local budget).")
    p.add_argument("--n-windows", type=int, default=None,
                   help="Eval windows to score on (default: [round] heat_n_windows).")
    p.add_argument("--device", default="cpu", help="Torch device (cuda recommended).")
    p.add_argument("--at-block", type=int, default=None,
                   help="Explicit offline era block for selecting the effective contract; live resolves this automatically.")
    p.add_argument("--warm-start", default=None, metavar="live|repo@digest|DIR",
                   help="Train from a promoted cascade init instead of random init, with the "
                        "live warm-started recipe: 'live' (the current era's published init; "
                        "matching effective contract is required), a Hub ref / trained pointer, or a "
                        "local checkpoint dir. Default: random init.")


def score_result_dict(r) -> dict:
    return {"geomean": r.geomean, "n_windows": r.n_windows, "corpus_digest": r.corpus_digest,
            "n_series": r.n_series, "train_seconds": r.train_seconds,
            "pool": r.pool_label, "init": r.init_label,
            "contract_digest": getattr(r, "contract_digest", ""),
            "train_tokens": getattr(r, "train_tokens", 0)}


def _cmd_score(args: argparse.Namespace) -> int:
    from .live_status import LiveStateUnavailable

    out = reporter(args)
    cfg = load_chain_config(args.chain_toml)
    if str(args.warm_start).lower() == "live":
        import copy

        from .training_context import live_context
        try:
            cfg, pinned, _ = live_context(cfg)
        except LiveStateUnavailable as exc:
            return out.fail("live_warm_start_unavailable", str(exc), NETWORK)
        args = copy.copy(args)
        args.warm_start = pinned
    if not args.skip_verify:
        # No bound contract: the pass must hold on either side of the v2 launch
        # (the same rule as `cascade verify`, so a scored generator is admissible).
        unbound = str(args.warm_start).lower() != "live" and args.at_block is None
        report = verify_repo(args.repo_dir, cfg, quick=True, across_launch=unbound)
        if not report.ok:
            out.result(verify=report.to_dict())
            return out.fail("verify_failed", "verify failed — fix before scoring:\n" + report.render())
    from .training_context import UnresolvedLaunch

    try:
        r = _run_score(args, cfg)
    except LiveStateUnavailable as e:
        return out.fail("live_warm_start_unavailable", str(e), NETWORK)
    except UnresolvedLaunch as e:
        return out.fail("launch_unresolved", str(e), USAGE)
    except ImportError as e:
        return out.fail("missing_train_extra",
                        f"error: `cascade score` needs the [train] extra (torch): {e}", USAGE)
    except Exception as e:  # noqa: BLE001 — surface any train/eval failure cleanly
        return out.fail("scoring_failed", f"scoring failed: {type(e).__name__}: {e}")
    out.result(**score_result_dict(r), lower_is_better=True)
    same = "--pool-dir <same pool>" + (f" --warm-start {args.warm_start}" if args.warm_start else "")
    out.info(
        f"\nscore: geomean={r.geomean:.5f}  (lower is better)\n"
        f"  pool:    {r.pool_label}  ({r.n_windows} windows)\n"
        f"  corpus:  {r.n_series} series, digest {r.corpus_digest[:12]}…\n"
        f"  init:    {r.init_label}\n"
        f"  trained: {r.train_seconds:.0f}s\n"
        f"\ncompare against the king:  cascade iterate {args.repo_dir} {same}"
    )
    return OK


def _run_score(args: argparse.Namespace, cfg):
    from .score import score_generator

    return score_generator(
        args.repo_dir, cfg, pool_dir=args.pool_dir, pool_ref=args.pool_ref,
        train_hours=args.train_hours, n_windows=args.n_windows, device=args.device,
        seed=args.seed, warm_start=args.warm_start, at_block=args.at_block,
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


def fetch_generator(target: str, cfg, network: str, out_dir: Path | None, out: Reporter):
    """Resolve and download ``target``; ``(exit_code, dest, ref, label)``."""
    import os

    from ..funding.store import CHAMPION_BASE_ENV, is_vault_ref
    from ..shared.chain import ChainError
    from ..shared.hippius import HubConfig, StorageError, fetch_from_hub

    try:
        ref, label = _resolve_fetch_ref(target, cfg, network)
    except ChainError as e:
        return out.fail("chain_error", f"chain error: {e}", NETWORK), None, None, None
    except ValueError as e:
        return out.fail("unresolved_target", f"error: {e}", USAGE), None, None, None

    dest_dir = out_dir or Path(f"./fetched-{label}")
    out.info(f"fetching {ref}\n  → {dest_dir}")
    if is_vault_ref(ref) and not os.environ.get(CHAMPION_BASE_ENV):
        # A direct (vault) submission resolves publicly ONLY through the
        # published champions/ objects — point the fetch at the same
        # anonymous endpoint the dashboards read.
        endpoint = str(getattr(cfg.storage, "s3_endpoint", "") or "").rstrip("/")
        bucket = str(getattr(cfg.storage, "manifest_bucket", "") or "")
        if endpoint and bucket:
            os.environ[CHAMPION_BASE_ENV] = f"{endpoint}/{bucket}"
    try:
        dest = fetch_from_hub(ref, dest_dir, HubConfig.from_storage(cfg.storage))
    except StorageError as e:
        if is_vault_ref(ref):
            return out.fail(
                "champion_unpublished",
                "fetch failed: this is a direct (private) submission and its "
                "code has not been published to champions/ yet — under the "
                "operator's champion_publish policy it goes public on crown, "
                f"after a reign delay, or at dethronement.\n  detail: {e}",
                REGISTRY), None, ref, label
        return out.fail("registry_fetch_failed", f"registry fetch failed: {e}",
                        REGISTRY), None, ref, label
    return OK, Path(dest), ref, label


def _cmd_fetch(args: argparse.Namespace) -> int:
    out = reporter(args)
    cfg = load_chain_config(args.chain_toml)
    rc, dest, ref, label = fetch_generator(args.target, cfg, args.network, args.out, out)
    if rc != OK:
        return rc
    files = sorted(p.name for p in dest.iterdir()) if dest.is_dir() else []
    out.result(ref=ref, label=label, path=str(dest), files=files)
    out.info(f"fetched {label}: {ref}\n  {len(files)} top-level entries: {', '.join(files[:12])}")

    if args.verify:
        report = verify_repo(dest, cfg, skip_runtime=False, across_launch=True)
        out.result(verify=report.to_dict())
        out.info(report.render())
        return OK if report.ok else out.fail("verify_failed", "fetched generator fails verify")
    out.info(f"\ninspect it, or fork + improve it:  cascade verify {dest}")
    return OK


def _add_round(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "round",
        help="Live round dashboard: boundary countdown, the latest verdict, "
        "and revealed submissions.",
    )
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--network", default="finney", help="Bittensor network (finney/test/local).")
    p.add_argument("--once", action="store_true",
                   help="Print a single snapshot instead of the live countdown.")
    p.add_argument("--refresh", type=float, default=30.0,
                   help="Seconds between chain re-syncs in watch mode (default: 30).")
    p.set_defaults(func=_cmd_round)


def _live_cfg(cfg, client):
    """``cfg`` with the fleet's resolved rollover applied (DEC-CA-0045).

    A miner's chain.toml keeps the DEC-CA-0043 keys at 0; once the validators
    lock in, the grid this CLI times reveals and countdowns on would be the
    pre-switch one. One chain read of the validators' notes fixes that; any
    failure (no chain, feature off, typed-in keys) returns ``cfg`` as loaded.
    """
    from ..shared.activation import startup_activation

    try:
        return startup_activation(cfg, client, store_path=None)
    except Exception:  # noqa: BLE001 — a CLI convenience must never fail a command
        return cfg


def _cmd_round(args: argparse.Namespace) -> int:
    out = reporter(args)
    cfg = load_chain_config(args.chain_toml)
    from ..shared.chain import ChainClient, ChainError
    from .dashboard import (
        fetch_public_receipt_index,
        latest_settled_before,
        round_status,
        run_dashboard,
    )

    try:
        client = ChainClient.from_config(cfg, network=args.network)
        cfg = _live_cfg(cfg, client)
        index = {}

        def _index():
            index["doc"] = fetch_public_receipt_index(cfg.storage)
            return index["doc"]

        rc = run_dashboard(
            client, cfg.round, args.network, once=args.once or out.json_mode,
            refresh=args.refresh, out=sys.stderr if out.json_mode else None,
            index_fetch=_index, chain_cfg=cfg,
        )
        if out.json_mode:
            st = round_status(client.current_block(), cfg.round)
            out.result(block=st.block, epoch=st.epoch, epoch_blocks=st.epoch_blocks,
                       next_boundary=st.next_epoch_start, blocks_remaining=st.blocks_remaining,
                       seconds_remaining=round(st.seconds_remaining),
                       last_settled=latest_settled_before(index.get("doc"), st.epoch_start + st.epoch_blocks))
        return rc
    except ChainError as e:
        return out.fail("chain_error", f"chain error: {e}", NETWORK)


def _add_queue(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "queue",
        help="Funded-leg transparency: the roster published at each settlement — "
        "the legs it judged (with on-chain reveal blocks), the legs still "
        "training, the queue waiting for a lane in reveal order, and failed "
        "entries. Add --intake to also show the live queue.",
    )
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--round", default=None,
                   help="Settlement round id (default: the latest published roster).")
    p.add_argument("--intake", default=None,
                   help="Intake base URL — also fetch the live /v1/queue.")
    p.add_argument("--hotkey", default=None,
                   help="Your hotkey (ss58) — marks your rows '← you'.")
    p.set_defaults(func=_cmd_queue)


def _cmd_queue(args: argparse.Namespace) -> int:
    out = reporter(args)
    cfg = load_chain_config(args.chain_toml)
    from .dashboard import fetch_public_json

    key = (f"funded/round-{args.round}.json" if args.round else "funded/latest.json")
    doc = fetch_public_json(cfg.storage, key)
    me = args.hotkey or ""
    out.result(roster=doc)

    def _tag(hk: str) -> str:
        return "  ← you" if me and hk == me else ""

    def _label_of(row: dict) -> str:
        # Presentational: printed BESIDE the hotkey (never instead of it), and
        # re-sanitised on the way out — the document is public JSON.
        from ..funding.queue import normalize_label

        try:
            lbl = normalize_label(row.get("label") or "")
        except ValueError:
            return ""
        return f' "{lbl}"' if lbl else ""

    def _gpu_of(row: dict) -> str:
        # DEC-CA-0060: the miner-chosen GPU type ("" = cheapest fitting type)
        gpu = str(row.get("gpu") or "")
        return f"  gpu={gpu}" if gpu else ""

    if doc is None:
        out.info(f"no historical funded roster available at {key}; this does not describe the current queue")
    else:
        out.info(f"round {doc.get('round_id')} — historical funded roster (settlement snapshot, not live status)")
        out.info(f"  settled ({len(doc.get('seated') or [])}, reveal-block seniority):")
        for e in doc.get("seated") or []:
            out.info(f"    {e.get('hotkey')}{_label_of(e)}  reveal={e.get('reveal_block')}"
                     f"{_tag(str(e.get('hotkey')))}")
        for e in doc.get("in_flight") or []:
            out.info(f"  admitted (historical): {e.get('hotkey')}{_label_of(e)}  "
                     f"reveal={e.get('reveal_block')}  target was {e.get('target_boundary')}"
                     f"{_gpu_of(e)}{_tag(str(e.get('hotkey')))}")
        for e in doc.get("waiting") or []:
            out.info(f"  waiting: {e.get('hotkey')}{_label_of(e)}  "
                     f"reveal={e.get('reveal_block')}{_gpu_of(e)}{_tag(str(e.get('hotkey')))}")
        for e in doc.get("terminal") or []:
            out.info(f"  terminal: {e.get('hotkey')}  [{e.get('error_class')}]"
                     f"{_tag(str(e.get('hotkey')))}")
        out.info("  (verify independently: reveal blocks are on-chain; "
                 "`cascade-audit round <id>` cross-checks this roster against "
                 "the signed manifest)")
    if args.intake:
        from .submission_status import read_queue, render_submission, submission_view

        if not _intake_transport_ok(args.intake):
            return out.fail("insecure_intake", "refusing plain http to a non-loopback intake", USAGE)
        try:
            live = read_queue(args.intake, "a", timeout=10)
        except (OSError, ValueError) as e:
            # URLError covers HTTPError; ValueError covers a proxy's HTML body.
            return out.fail("intake_unreachable",
                            f"intake unreachable or not answering JSON: {e}", NETWORK)
        out.result(live_queue=live)
        out.info(f"live queue (depth {live.get('queued_depth')}):")
        for e in live.get("entries") or []:
            err = f" [{e.get('last_error_class')}]" if e.get("last_error_class") else ""
            out.info(f"  {e.get('hotkey')}{_label_of(e)}  {e.get('status')}  "
                     f"reveal={e.get('reveal_block')} attempts={e.get('attempts')}"
                     f"{_gpu_of(e)}{err}{_tag(str(e.get('hotkey')))}")
        if me:
            view = submission_view({}, hotkey=me, lane="a", queue=live,
                                   source=args.intake.rstrip("/") + "/v1/queue")
            out.result(submission=view)
            render_submission(out, view)
    elif me:
        out.info(f"For current activity: cascade status --hotkey {me} --intake YOUR_INTAKE --watch")
    return OK


def _add_duel(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "duel",
        help="Duel verdict for a settled round: dethrone margin, both geomeans, "
        "per-domain scores — the full breakdown behind DETHRONED/king-held. "
        "--hotkey adds YOUR per-domain scores vs the king (and, with --history, "
        "your per-domain trend across rounds).",
    )
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--round", dest="round_id", default=None,
                   help="A specific round id (default: the latest settled round).")
    p.add_argument("--history", action="store_true",
                   help="One line per settled round instead of one round's detail "
                   "(with --hotkey: your gap vs the king per domain, per round).")
    p.add_argument("--hotkey", default=None,
                   help="Your hotkey (ss58) — adds a 'your domains' block: which "
                   "domains you beat the king in this round and by how much.")
    p.add_argument("--limit", type=int, default=20,
                   help="Rounds listed by --history (default: 20).")
    p.set_defaults(func=_cmd_duel)


def _cmd_duel(args: argparse.Namespace) -> int:
    """Print a settled round's full duel verdict — no wallet, no chain call, no
    credentials. Everything comes from the public receipt index the validators
    publish a few minutes after each duel; `cascade round` shows the same data
    as a one-line outcome."""
    out = reporter(args)
    cfg = load_chain_config(args.chain_toml)
    from .dashboard import (
        duel_round_rows,
        fetch_public_receipt_index,
        render_duel,
        render_duel_domain_history,
        render_duel_index,
    )

    doc = fetch_public_receipt_index(cfg.storage)
    if doc is None:
        return out.fail("receipts_unavailable",
                        "could not fetch the public receipt index (receipts/index.json) — "
                        "offline, or the round has not settled yet; try 'cascade round'")
    me = getattr(args, "hotkey", None) or None
    if args.history:
        rounds = (doc.get("rounds") or [])[-args.limit:] if isinstance(doc, dict) else []
        out.result(rounds=rounds)
        if me:
            out.info(render_duel_domain_history(doc, me, limit=args.limit))
        else:
            out.info(render_duel_index(doc, limit=args.limit))
        return OK
    rows = duel_round_rows(doc, args.round_id)
    if not rows:
        target = f"round {args.round_id}" if args.round_id else "any settled round"
        return out.fail("round_not_found",
                        f"no receipt-index rows for {target} — receipts land a few minutes "
                        "after the duel manifest; try 'cascade duel --history'")
    out.result(rows=rows)
    out.info(render_duel(rows, me=me))
    return OK


def _cmd_verify(args: argparse.Namespace) -> int:
    out = reporter(args)
    cfg = load_chain_config(args.chain_toml)
    if getattr(args, "live", False):
        from .live_status import LiveStateUnavailable
        from .training_context import live_context

        try:
            cfg, _, _ = live_context(cfg)
        except LiveStateUnavailable as exc:
            return out.fail("live_status_unavailable", str(exc), NETWORK)
    elif getattr(args, "at_block", None) is not None:
        from .training_context import UnresolvedLaunch, configured_context

        try:
            cfg = configured_context(cfg, args.at_block)
        except UnresolvedLaunch as exc:
            return out.fail("launch_unresolved", str(exc), USAGE)
    # No bound contract: a pass must hold on either side of the v2 launch.
    unbound = not getattr(args, "live", False) and getattr(args, "at_block", None) is None
    report = verify_repo(args.repo_dir, cfg, skip_runtime=args.skip_runtime, quick=args.quick,
                         across_launch=unbound)
    out.result(**report.to_dict())
    out.info(report.render())
    if not report.ok:
        out.error = {"code": "verify_failed",
                     "message": "; ".join(f"[{s}] {r.reason}" for s, r in report.failures)}
    return OK if report.ok else REJECTED


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

    Rolling intake: funding/reveal makes an entry eligible for admission;
    capacity and checks decide when it runs. The report names the first settlement
    boundary STRICTLY AFTER the reveal. ``missed`` is True only when
    ``expect_boundary`` was given and the reveal landed at/after it. Pure —
    unit-testable without a chain."""
    eligible_boundary = (reveal_block // epoch_blocks + 1) * epoch_blocks
    lead = eligible_boundary - reveal_block
    lines = [f"revealed at block {reveal_block} — next settlement at block "
             f"{eligible_boundary}; funding/reveal makes an entry eligible, not GPU-reserved "
             f"({lead} blocks of pre-boundary exposure)"]
    if lead > margin_blocks:
        lines.append(f"note: exposure exceeds the {margin_blocks}-block reveal margin — "
                     "the ref was copyable for longer than the timed default allows.")
    if current_block >= eligible_boundary:
        lines.append("that settlement boundary has passed; it does not prove training or settlement. "
                     "Track the exact entry with `cascade status --hotkey ... --ref ...`.")
    else:
        lines.append(f"next settlement in {eligible_boundary - current_block} block(s).")
    missed = expect_boundary is not None and reveal_block >= expect_boundary
    if missed:
        lines.insert(0, f"⚠ MISSED the targeted boundary {expect_boundary}: the reveal "
                        f"landed {reveal_block - expect_boundary} block(s) at/after it.")
        lines.append("consequences: a valid funded/revealed entry remains eligible for later "
                     "admission without re-committing; the NEXT round is not guaranteed. "
                     "Missing this boundary alone does not consume the one-submission budget; "
                     "actual consumption depends on admission, training faults and verdicts. "
                     "The ref is public after reveal. Check status before changing the commitment: "
                     "a new commitment can supersede queued work.")
    return missed, "\n".join(lines)


def _pending_timelock(client, hotkey: str) -> tuple[int, int] | None:
    """``(commit_block, reveal_round)`` of an un-revealed timelock commit, if any.

    The plain commitment store (``Commitments::CommitmentOf``) holds the
    encrypted payload until drand reveals it; once revealed the record is
    consumed. Without this check, reveal-status shows the hotkey's PREVIOUS
    reveal while a fresh timelock is pending — telling a miner their new
    submission doesn't exist."""
    try:
        v = client.pending_commitment(hotkey) or {}
        for f in ((v.get("info") or {}).get("fields") or []):
            tl = f.get("TimelockEncrypted") if isinstance(f, dict) else None
            if tl is not None:
                return int(v.get("block") or 0), int(tl.get("reveal_round") or 0)
    except Exception:  # noqa: BLE001 — advisory; never break the status report
        return None
    return None


def _cmd_reveal_status(args: argparse.Namespace) -> int:
    import time

    out = reporter(args)
    cfg = load_chain_config(args.chain_toml)
    from ..shared.chain import ChainClient, ChainError

    client = ChainClient.from_config(cfg, network=args.network)
    cfg = _live_cfg(cfg, client)
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
                out.result(state="pending", pending_commit_block=blk, drand_round=rnd)
                out.info(f"PENDING timelock commit (committed at block {blk}, drand round "
                         f"{rnd}) — payload still encrypted on-chain; nothing copyable yet."
                         + (f" Latest REVEALED entry below is the PREVIOUS submission "
                            f"(block {match.commit_block})." if match else ""))
                if args.watch and time.monotonic() < deadline:
                    time.sleep(12)
                    continue
                if match is None:
                    return OK
            if match is not None:
                epoch_blocks = effective_epoch_blocks(cfg.round, match.commit_block)
                missed, report = _reveal_verdict(
                    match.commit_block, client.current_block(), epoch_blocks,
                    cfg.round.reveal_margin_blocks, args.expect_boundary,
                )
                out.result(state="revealed", reveal_block=match.commit_block,
                           payload=match.payload, missed=missed,
                           next_settlement=(match.commit_block // epoch_blocks + 1) * epoch_blocks)
                out.info(report)
                if missed:
                    out.error = {"code": "reveal_missed_boundary",
                                 "message": f"reveal landed at/after boundary {args.expect_boundary}"}
                return REJECTED if missed else OK
            if not args.watch or time.monotonic() >= deadline:
                out.result(state="none")
                out.info("no revealed commitment for that hotkey — still timelock-hidden, "
                         "never committed, or not registered on the netuid."
                         + ("" if args.watch else " (--watch polls until it lands.)"))
                return OK if not args.watch else REJECTED
            time.sleep(12)
    except ChainError as e:
        return out.fail("chain_error", f"chain error: {e}", NETWORK)


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

    out = reporter(args)
    try:
        up = upload_dir_to_hub(args.repo_dir, args.hub_repo, HubConfig.from_storage(cfg.storage))
        out.info(f"pushed to Hippius Hub: {up.ref.immutable_ref} ({up.size_bytes} bytes)")
        return OK, up.ref.immutable_ref
    except StorageError as e:
        hub_err = str(e)  # bind now — the `as` name is cleared at except-block exit
        if not args.hf_repo:
            return out.fail("registry_upload_failed", f"registry upload failed: {e}", REGISTRY), None
        out.warn(f"Hippius Hub upload failed ({e});\n"
                 f"  falling back to HuggingFace mirror {args.hf_repo}")
        out.warn("warning: HuggingFace repos are PUBLIC and enumerable — anyone watching "
                 "your HF account sees this generator's content now, before the on-chain "
                 "pointer reveals. Prefer retrying the Hub for a competitive submission.")

    try:
        up = upload_dir_to_hf(args.repo_dir, args.hf_repo)
    except StorageError as e:
        return out.fail("registry_upload_failed",
                        f"HuggingFace mirror upload failed: {e} (Hub also failed: {hub_err})",
                        REGISTRY), None
    out.info(f"mirrored to HuggingFace: {up.ref.immutable_ref} ({up.size_bytes} bytes)")
    return OK, up.ref.immutable_ref


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
    reporter(args).info(
        f"timed reveal: payload decrypts ~block {target} "
        f"({delay} blocks from now, {boundary - target} blocks before the epoch "
        f"boundary at {boundary}) — hidden until just before that settlement. "
        f"Override with --reveal-now / --blocks-until-reveal / --next-epoch."
    )
    return delay


def expert_domain_error(cfg, domain: str) -> str | None:
    """Why ``domain`` cannot be committed as a lane-B expert entry (None = ok):
    every comma-separated target must be one of the ``[lane_b]`` targets
    (domains, plus ``horizon:<h>`` per scored horizon — DEC-CA-0059), at most
    ``max_domains_per_leg`` of them, no repeats — the trainer silently skips
    any other commitment, so committing one would only burn the hotkey's
    commitment."""
    _targets, why = expert_targets(cfg, domain)
    return why


def expert_targets(cfg, spec: str) -> tuple[tuple[str, ...], str | None]:
    """``(declared targets, None)`` for a ``--expert-domain`` value, or
    ``((), why)``. Accepts ``h<h>`` for ``horizon:<h>``."""
    lb = cfg.lane_b
    domains = tuple(getattr(lb, "domains", ()) or ())
    if not domains:
        return (), "this chain.toml configures no [lane_b] domains (lane B is not set up)"
    targets = tuple(getattr(lb, "targets", None) or domains)
    cap = int(getattr(lb, "max_domains_per_leg", 1) or 1)
    import re

    want = []
    for raw in str(spec).split(","):
        t = raw.strip()
        if re.fullmatch(r"h[0-9]+", t):
            t = f"horizon:{int(t[1:])}"
        if t not in targets:
            return (), (f"unknown lane-B domain {raw.strip()!r} (not a [lane_b] target); "
                        "choose from: "
                        + ", ".join(targets))
        want.append(t)
    if len(set(want)) != len(want):
        return (), f"lane-B target repeated in {spec!r}"
    if len(want) > cap:
        return (), (f"{len(want)} lane-B targets > [lane_b] max_domains_per_leg = {cap}")
    return tuple(want), None


def _reveal_flag_error(args) -> str | None:
    if args.reveal_now and (args.blocks_until_reveal is not None or args.next_epoch):
        return "error: --reveal-now conflicts with --blocks-until-reveal / --next-epoch."
    if args.next_epoch and args.blocks_until_reveal is not None:
        return "error: --next-epoch conflicts with an explicit --blocks-until-reveal."
    return None


def _cmd_deploy(args: argparse.Namespace) -> int:
    out = reporter(args)
    cfg = load_chain_config(args.chain_toml)

    why = _reveal_flag_error(args)
    if why:
        return out.fail("conflicting_flags", why, USAGE)
    if args.hub_repo and args.hub_namespace:
        return out.fail("conflicting_flags", "error: pass --hub-repo OR --hub-namespace, not both.", USAGE)
    expert_domain = getattr(args, "expert_domain", None)
    if expert_domain:
        # validated BEFORE any upload or chain write: a bad domain costs nothing
        why = expert_domain_error(cfg, expert_domain)
        if why:
            return out.fail("bad_expert_domain", f"error: {why}", USAGE)
    if args.hub_namespace:
        args.hub_repo = _fresh_hub_repo(args.hub_namespace)
        out.info(f"fresh submission repo: {args.hub_repo}")

    ref = args.ref
    if ref is None:
        if not args.hub_repo:
            return out.fail(
                "missing_hub_repo",
                "error: --hub-repo or --hub-namespace (Hippius Hub) is required to "
                "upload — the Hub is always tried first. --hf-repo is only a fallback "
                "for when the Hub push fails; pass it alongside one of them. Or use "
                "--ref to commit an already-uploaded ref.", USAGE)
        # Verify locally (cheaper than burning a chain commit), then upload.
        if not args.skip_verify:
            report = verify_repo(args.repo_dir, cfg, skip_runtime=False, across_launch=True)
            if not report.ok:
                out.result(verify=report.to_dict())
                return out.fail("verify_failed",
                                "local verify failed — refusing to deploy:\n" + report.render())

        rc, ref = _upload_generator(args, cfg)
        if rc != 0:
            return rc

    try:
        if expert_domain:
            from ..shared.expert_manifest import format_expert_targets

            # DEC-CA-0059: one target ⇒ today's payload byte for byte
            declared, _ = expert_targets(cfg, expert_domain)
            payload = format_expert_targets(
                declared, ref, cfg.lane_b.targets,
                max_targets=cfg.lane_b.max_domains_per_leg)
        else:
            payload = format_commit(ref)
            assert parse_commit(payload) is not None  # format_commit guarantees this
    except ValueError as e:
        return out.fail("bad_ref", f"refusing to deploy: {e}", USAGE)

    from ..shared.chain import ChainClient, ChainError

    try:
        client = ChainClient.from_config(
            cfg,
            network=args.network,
            wallet_name=args.wallet_name,
            wallet_hotkey=args.wallet_hotkey,
            wallet_path=args.wallet_path,
        )
        cfg = _live_cfg(cfg, client)
        current_block = client.current_block()
        blocks_until_reveal = _resolve_blocks_until_reveal(args, cfg, current_block)
        client.commit_submission(payload, blocks_until_reveal=blocks_until_reveal)
    except ChainError as e:
        return out.fail("chain_error", f"chain error: {e}", NETWORK, ref=ref)
    except ValueError as e:
        # blocks_until_boundary_reveal rejects inconsistent [round] config
        # (e.g. reveal_margin_blocks >= epoch_blocks).
        return out.fail("bad_reveal_config", f"bad [round] reveal config: {e}", USAGE)

    out.result(ref=ref, payload=payload, commit_block=current_block,
               reveal_block=current_block + blocks_until_reveal)
    out.info(f"committed: {payload}")
    if expert_domain:
        # DEC-CA-0059 item 8: every lane is miner-funded
        out.info(f"once it reveals, fund the leg on your Lium key:\n"
                 f"  cascade lane-b fund <intake> --targets {expert_domain} --ref {ref} "
                 "--yes ...")
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
        out.result(expect_boundary=boundary)
        out.info(f"confirm the reveal lands in time (from ~block {target}):\n"
                 f"  cascade reveal-status {hotkey} --network {args.network} "
                 f"--expect-boundary {boundary} --watch")
    return OK


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
    from ..funding.providers import add_provider_args

    add_provider_args(p)
    p.add_argument("--lium-key-env", default="LIUM_API_KEY",
                   help="Env var holding your Lium key: when set, the SAME request "
                        "funds your entry (auto-queues once the reveal lands).")
    p.add_argument("--no-fund", action="store_true",
                   help="Submit code only; fund later with `cascade fund`.")
    p.add_argument("--yes", action="store_true",
                   help="Confirm funding: with a Lium key set, this submission rents a "
                        "training pod billed to that key. Required to send a key.")
    p.add_argument("--label", default="",
                   help="Display name for this submission (≤32 chars, letters/digits/"
                        "._-) shown beside your hotkey in the public queue and roster. "
                        "Cosmetic only — never identity.")
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
      PUBLIC ``champions/``. Matching the Hub filter also
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


def needs_confirmation(args, out: Reporter, what: str) -> int | None:
    """``NEEDS_CONFIRM`` (after saying why) unless ``--yes`` was passed."""
    if getattr(args, "yes", False):
        return None
    return out.fail("confirmation_required",
                    f"refusing: {what}. Nothing was sent; re-run with --yes to confirm.",
                    NEEDS_CONFIRM)


def load_signing_wallet(args, out: Reporter):
    """``(hotkey_ss58, sign_fn)`` or ``None`` after reporting why (signing only)."""
    try:
        # open_wallet drops a None --wallet-path (11.x's Wallet would choke on path=None).
        from ..shared.chain import open_wallet

        wallet = open_wallet(args.wallet_name, args.wallet_hotkey, args.wallet_path)
        return wallet.hotkey.ss58_address, wallet.hotkey.sign
    except Exception as e:  # noqa: BLE001 — wallet errors are usage errors here
        out.fail("wallet_unavailable", f"could not load wallet for signing: {e}", USAGE)
        return None


def post_intake(req, out: Reporter, *, timeout: float) -> tuple[int, dict] | None:
    """``(status, json_body)`` for one intake request; ``None`` if unreachable."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _decode_json_body(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, _decode_json_body(e.read())
    except (urllib.error.URLError, OSError) as e:
        out.fail("intake_unreachable", f"intake unreachable: {e}", NETWORK)
        return None


def _cmd_submit(args: argparse.Namespace) -> int:
    import hashlib
    import urllib.request

    out = reporter(args)
    base = args.intake_url.rstrip("/")
    if not _intake_transport_ok(base):
        return out.fail("insecure_intake",
                        "refusing to submit over plain http to a non-local intake; use https", USAGE)
    why = _reveal_flag_error(args)
    if why:
        return out.fail("conflicting_flags", why, USAGE)
    label = _cli_label(args)
    if label is None:
        return USAGE
    api_key = ""
    if not args.no_fund:
        from ..funding.providers import credentials_from_env

        try:
            api_key = credentials_from_env(args)
        except ValueError as exc:
            return out.fail("bad_provider_credentials", str(exc), USAGE)
        if not api_key:
            out.info(f"note: ${args.lium_key_env} not set — submitting unfunded "
                     f"(fund later with `cascade fund`)")
        else:
            refused = needs_confirmation(
                args, out, "Provider credentials are set, so this submission also funds a "
                           "training leg billed to your selected provider account (pass --no-fund to submit code only)")
            if refused is not None:
                return refused
    cfg = load_chain_config(args.chain_toml)
    if api_key:
        from .live_status import LiveStateUnavailable
        from .training_context import live_context

        try:
            cfg, _, _ = live_context(cfg)
        except LiveStateUnavailable as exc:
            return out.fail("live_status_unavailable", str(exc), NETWORK)
    if not args.skip_verify:
        report = verify_repo(args.repo_dir, cfg, skip_runtime=args.skip_runtime,
                             across_launch=not api_key)
        out.info(report.render())
        if not report.ok:
            out.result(verify=report.to_dict())
            return out.fail("verify_failed",
                            "refusing to submit: verification failed (fix, or --skip-verify "
                            "to send anyway — the trainer will reject the same faults)")

    wallet = load_signing_wallet(args, out)
    if wallet is None:
        return USAGE
    hotkey_ss58, sign_fn = wallet

    try:
        body = zip_repo_bytes(args.repo_dir)
    except (ValueError, OSError) as e:
        return out.fail("package_failed", f"could not package the repo: {e}", USAGE)
    digest = f"sha256:{hashlib.sha256(body).hexdigest()}"
    import time as _time

    from ..funding.intake import canonical_fund_message

    ts = str(int(_time.time()))
    # v2 canonical message: the signature binds the key header (its sha256)
    # so an on-path replay cannot attach or swap a key.
    key_digest = hashlib.sha256(api_key.encode()).hexdigest() if api_key else "-"
    headers = {
        "X-Miner-Hotkey": hotkey_ss58,
        "X-Content-Digest": digest,
        "X-Timestamp": ts,
        "X-Signature": sign_fn(canonical_fund_message(
            "submit", hotkey_ss58, digest, ts, key_digest)).hex(),
        "Content-Type": "application/zip",
    }
    if api_key:
        from ..funding.providers import credential_headers

        headers.update(credential_headers(api_key))
    if label:
        from ..funding.intake import LABEL_HEADER
        headers[LABEL_HEADER] = label

    req = urllib.request.Request(f"{base}/v1/submit", data=body, method="POST",
                                 headers=headers)
    answer = post_intake(req, out, timeout=120)
    if answer is None:
        return NETWORK
    status, resp_body = answer
    if not (200 <= status < 300):
        return out.fail(str(resp_body.get("code") or "submit_rejected"),
                        f"submit rejected ({status}): {resp_body.get('code', '?')} — "
                        f"{resp_body.get('message', '')}", http_status=status)
    ref = resp_body.get("ref")
    payload = resp_body.get("commit_payload")
    if not ref or not payload:
        # A middlebox can 200 with a non-JSON body; don't traceback on it.
        return out.fail("bad_intake_response",
                        f"intake answered {status} but the body is not a submit "
                        f"response: {resp_body.get('message', resp_body)!s:.200}", NETWORK)
    funding = resp_body.get("funding", "none")
    out.result(ref=ref, payload=payload, funding=funding, content_digest=digest,
               hotkey=hotkey_ss58, committed=False)
    from .submission_status import tracking_result

    if funding not in ("none", "unfunded"):
        tracking_result(out, args, hotkey=hotkey_ss58, ref=ref)
    else:
        out.result(queued=False, next_action="Commit the stored ref, then fund it; no training leg is queued.")
    out.info(f"stored privately: {ref} (funding={funding})")
    if args.no_commit:
        out.info(f"commit it yourself when ready:\n  payload: {payload}")
        return OK

    from ..shared.chain import ChainClient, ChainError

    try:
        client = ChainClient.from_config(
            cfg, network=args.network, wallet_name=args.wallet_name,
            wallet_hotkey=args.wallet_hotkey, wallet_path=args.wallet_path,
        )
        cfg = _live_cfg(cfg, client)
        current_block = client.current_block()
        blocks_until_reveal = _resolve_blocks_until_reveal(args, cfg, current_block)
        client.commit_submission(payload, blocks_until_reveal=blocks_until_reveal)
    except ChainError as e:
        return out.fail("chain_error", f"chain error: {e} — your code IS stored; commit later with:\n"
                                       f"  payload: {payload}", NETWORK)
    except ValueError as e:
        return out.fail("bad_reveal_config", f"bad [round] reveal config: {e}", USAGE)
    out.result(committed=True, commit_block=current_block,
               reveal_block=current_block + blocks_until_reveal)
    out.info(f"committed: {payload}")
    out.info("your code stays PRIVATE unless it takes the throne (champion_publish "
             "policy); once revealed on-chain, a funded entry queues automatically.")
    return OK


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
    from ..funding.providers import add_provider_args

    add_provider_args(p)
    p.add_argument("--lium-key-env", default="LIUM_API_KEY",
                   help="Environment variable holding your Lium API key (default "
                        "LIUM_API_KEY). The key is NEVER accepted on the command line.")
    p.add_argument("--withdraw", action="store_true",
                   help="Withdraw a still-queued entry (the operator forgets your key).")
    p.add_argument("--yes", action="store_true",
                   help="Confirm funding: the leg rents a pod billed to your selected provider key. "
                        "Required to fund (not to --withdraw).")
    p.add_argument("--label", default="",
                   help="Display name for this entry (≤32 chars, letters/digits/._-) "
                        "shown beside your hotkey in the public queue and roster. "
                        "Cosmetic only — never identity; a re-fund with a "
                        "new --label renames, without one keeps the old name.")
    add_gpu_args(p)
    p.set_defaults(func=_cmd_fund)


def _add_credentials(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("credentials", help="Store or replace provider keys without queuing another submission.")
    p.add_argument("intake_url")
    p.add_argument("--ref", required=True, help="Your matching revealed commitment ref.")
    p.add_argument("--wallet-name", required=True)
    p.add_argument("--wallet-hotkey", required=True)
    p.add_argument("--wallet-path", default=None)
    p.add_argument("--lium-key-env", default="LIUM_API_KEY")
    from ..funding.providers import add_provider_args

    add_provider_args(p)
    p.add_argument("--yes", action="store_true", help="Authorize storing these keys for future rentals under your account.")
    p.set_defaults(func=_cmd_credentials)


def _cmd_credentials(args: argparse.Namespace) -> int:
    import urllib.request

    from ..funding.providers import decode_credentials
    from .funding import build_intake_headers, prepare_funding_request

    out = reporter(args)
    prepared = prepare_funding_request(args, out, funding=True,
                                      confirmation="store provider keys to fund future GPU rentals")
    if isinstance(prepared, int):
        return prepared
    headers = build_intake_headers("credentials", prepared.hotkey, args.ref,
                                   prepared.api_key, prepared.sign_fn, fund_action="credentials")
    req = urllib.request.Request(f"{prepared.base}/v1/credentials", method="POST", headers=headers)
    answer = post_intake(req, out, timeout=30)
    if answer is None:
        return NETWORK
    status, body = answer
    if not 200 <= status < 300:
        return out.fail(str(body.get("code") or "credentials_rejected"),
                        f"credential update rejected ({status})", http_status=status)
    out.result(status="stored", providers=[c.provider for c in decode_credentials(prepared.api_key)])
    out.info("Provider keys stored; no submission queued or GPU rented by this command.")
    return OK


def add_gpu_args(p: argparse.ArgumentParser) -> None:
    """DEC-CA-0060 ``--gpu`` / ``--allow-fallback`` / ``--max-price`` on a fund
    command (lane A ``fund``, ``lane-b fund`` and ``lane-c fund``)."""
    p.add_argument("--gpu", default="",
                   help="GPU type your key rents (one of chain.toml [round] funded_pod_skus, "
                        "e.g. H100). Same token budget and wall as every leg — a faster "
                        "type costs more per hour, never buys more compute. Default: the "
                        "cheapest fitting type under the operator's price caps. The "
                        "operator caps do not apply to a chosen type; --max-price does. "
                        "A re-fund without --gpu resets to the default.")
    p.add_argument("--allow-fallback", action="store_true",
                   help="With --gpu: rent the cheapest fitting type when the chosen one "
                        "has no capacity (default: wait for it; the entry requeues "
                        "unburned like any capacity miss).")
    p.add_argument("--max-price", type=float, default=None,
                   help="With --gpu: your own cap in USD per GPU-hour on the chosen type "
                        "(default: none — you accept the listed price).")


def cli_gpu_choice(args: argparse.Namespace) -> tuple[str, bool, float | None] | None:
    """``(gpu, allow_fallback, max_price)`` from the fund flags, or ``None``
    after printing why they were refused."""
    import math

    gpu = (getattr(args, "gpu", "") or "").strip()
    fallback = bool(getattr(args, "allow_fallback", False))
    price = getattr(args, "max_price", None)
    if not gpu and (fallback or price is not None):
        reporter(args).fail("gpu_options_without_gpu",
                            "--allow-fallback / --max-price apply to a chosen GPU type — "
                            "add --gpu", USAGE)
        return None
    if price is not None and (not math.isfinite(price) or price < 0):
        reporter(args).fail("bad_gpu_max_price", "--max-price must be >= 0 USD/h", USAGE)
        return None
    if any(c in gpu for c in ";=:|"):
        reporter(args).fail("unknown_gpu", f"--gpu {gpu!r} is not a GPU type", USAGE)
        return None
    return gpu, fallback, price


def gpu_headers(gpu: str, allow_fallback: bool,
                max_price: float | None) -> tuple[dict[str, str], str]:
    """``(headers, signed_segment)`` for a GPU choice (DEC-CA-0060); both
    empty without ``gpu`` — the request then signs the v2 message unchanged."""
    from ..funding.gpu_choice import (
        GPU_FALLBACK_HEADER,
        GPU_HEADER,
        GPU_MAX_PRICE_HEADER,
        gpu_signed_segment,
    )

    if not gpu:
        return {}, ""
    headers = {GPU_HEADER: gpu, GPU_FALLBACK_HEADER: "1" if allow_fallback else "0"}
    if max_price is not None:
        headers[GPU_MAX_PRICE_HEADER] = repr(float(max_price))
    return headers, gpu_signed_segment(headers[GPU_HEADER], headers[GPU_FALLBACK_HEADER],
                                       headers.get(GPU_MAX_PRICE_HEADER, ""))


def report_gpu_reply(out, body: dict) -> None:
    """Print the intake's GPU echo and its wall warning (advisory)."""
    if body.get("gpu"):
        cap = float(body.get("max_price_per_hour") or 0)
        fallback = bool(body.get("allow_fallback"))
        out.result(gpu=body.get("gpu"), allow_fallback=fallback, max_price_per_hour=cap)
        out.info(f"GPU: {body.get('gpu')} (your cap: {f'${cap:.2f}/h' if cap else 'none'}; "
                 + ("falls back to the cheapest type" if fallback else "waits for this type")
                 + ")")
    if body.get("warning"):
        out.result(warning=body.get("warning"))
        out.info(f"WARNING: {body.get('warning')}")


def build_fund_headers(action: str, hotkey_ss58: str, ref: str, api_key: str,
                       sign_fn, *, now=None, label: str = "",
                       key_digest: str | None = None, gpu: str = "",
                       allow_fallback: bool = False,
                       max_price: float | None = None) -> dict[str, str]:
    """The signed header set for one intake request (pure; testable).

    ``sign_fn(message: bytes) -> bytes`` is the hotkey's sr25519 signer. The
    canonical message binds action + hotkey + ref + timestamp, so a captured
    fund request cannot be replayed as a withdraw (or vice versa), and none of
    it can be replayed at all past the intake's freshness window. ``label``
    (already normalised, optional) rides as its own header outside the
    signed message — it is a display name, not identity. ``key_digest``
    overrides the Lium key binding (direct credential deposits bind
    ``provider:key`` and send no Lium header). ``gpu`` / ``allow_fallback``
    / ``max_price`` (DEC-CA-0060, fund only) add the GPU headers and sign the
    v3 message over them; without ``gpu`` the headers and signature are
    exactly the v2 ones.
    """
    from .funding import build_intake_headers

    return build_intake_headers(action, hotkey_ss58, ref, api_key, sign_fn,
                                now=now, label=label, key_digest=key_digest, gpu=gpu,
                                allow_fallback=allow_fallback, max_price=max_price)


def _cli_label(args: argparse.Namespace) -> str | None:
    """The normalised ``--label`` ("" = none), or ``None`` after printing why
    it was refused — the same rule the intake applies, so a bad label fails
    here instead of costing a signed round-trip."""
    from ..funding.queue import normalize_label

    try:
        return normalize_label(getattr(args, "label", "") or "")
    except ValueError as e:
        reporter(args).fail("bad_label", f"--label refused: {e}", USAGE)
        return None


def _intake_transport_ok(url: str) -> bool:
    """True when the intake URL may carry the key: https, or genuinely local.

    The hostname is PARSED and compared exactly — a substring check would
    pass ``http://localhost.evil.example`` and ship the key in cleartext to a
    non-local box.
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
    import urllib.request

    from .funding import prepare_funding_request

    out = reporter(args)
    action = "withdraw" if args.withdraw else "fund"
    prepared = prepare_funding_request(
        args, out, funding=action == "fund",
        confirmation="funding rents a training pod billed to your selected provider key",
        key_hint="key itself as an argument")
    if isinstance(prepared, int):
        return prepared
    headers = build_fund_headers(
        action, prepared.hotkey, args.ref, prepared.api_key, prepared.sign_fn,
        label=prepared.label, gpu=prepared.choice[0], allow_fallback=prepared.choice[1],
        max_price=prepared.choice[2])
    req = urllib.request.Request(f"{prepared.base}/v1/{action}", method="POST", headers=headers)
    answer = post_intake(req, out, timeout=30)
    if answer is None:
        return NETWORK
    status, body = answer

    if 200 <= status < 300:
        out.result(action=action, status=body.get("status", "ok"), hotkey=prepared.hotkey, ref=args.ref)
        out.info(f"{action}: {body.get('status', 'ok')} (hotkey {prepared.hotkey})")
        report_gpu_reply(out, body)
        if action == "fund":
            from .submission_status import tracking_result

            tracking_result(out, args, hotkey=prepared.hotkey, ref=args.ref)
            out.info("your entry queues by reveal block; watch it via the intake's "
                     "/v1/queue or `cascade round`. An infra failure on your pod "
                     "re-queues you without burning the entry.")
        return OK
    return out.fail(str(body.get("code") or f"{action}_rejected"),
                    f"{action} rejected ({status}): {body.get('code', '?')} — "
                    f"{body.get('message', '')}", http_status=status)


def _add_json_flag(parser: argparse.ArgumentParser, *, root: bool = True) -> None:
    """``--json`` before or after any (sub)command name."""
    parser.add_argument("--json", action="store_true",
                        default=False if root else argparse.SUPPRESS,
                        help="Print one JSON result document on stdout (human lines go to "
                             "stderr). Also: CASCADE_JSON=1.")
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                _add_json_flag(child, root=False)


def build_parser() -> argparse.ArgumentParser:
    from .direct import add_direct
    from .discovery_check import add_discovery_check
    from .iterate import add_iterate
    from .lane_b import add_lane_b
    from .lane_c import add_lane_c
    from .live_status import add_status
    from .preflight import add_preflight
    from .starter import add_init

    parser = argparse.ArgumentParser(
        prog="cascade", description="cascade subnet miner CLI (generator and direct-MoE lanes).",
        epilog="--json on any command prints one machine-readable result. Exit codes: 0 ok, "
               "1 rejected, 2 usage, 3 network, 4 registry, 5 needs --yes. See AGENTS.md.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_verify(sub)
    add_iterate(sub)
    _add_score(sub)
    _add_submit(sub)
    _add_deploy(sub)
    _add_fund(sub)
    _add_credentials(sub)
    _add_fetch(sub)
    _add_reveal_status(sub)
    _add_round(sub)
    add_status(sub)
    add_preflight(sub)
    add_init(sub)
    add_discovery_check(sub)
    _add_queue(sub)
    _add_duel(sub)
    add_direct(sub)
    add_lane_b(sub)
    add_lane_c(sub)
    _add_json_flag(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    from ..shared.env import load_env_files
    load_env_files()
    raw = list(sys.argv[1:] if argv is None else argv)
    json_mode = "--json" in raw or json_default()
    try:
        args = build_parser().parse_args(raw)
    except SystemExit as e:
        if e.code in (0, None) or not json_mode:
            raise
        out = Reporter(raw[0] if raw else "", json_mode=True)
        out.error = {"code": "usage", "message": "invalid arguments (see stderr)"}
        return out.finish(USAGE)
    out = Reporter(getattr(args, "command", None) or args.cmd, json_mode=json_mode)
    args._reporter = out
    try:
        rc = int(args.func(args))
    except KeyboardInterrupt:
        rc = out.fail("interrupted", "interrupted", REJECTED)
    except Exception as e:  # noqa: BLE001 — JSON callers get a document, humans a traceback
        if not json_mode:
            raise
        rc = out.fail("internal_error", f"{type(e).__name__}: {e}", REJECTED)
    return out.finish(rc)


if __name__ == "__main__":
    raise SystemExit(main())
