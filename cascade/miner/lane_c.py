"""``cascade lane-c`` — miner commands for the fine-tuning lane (DEC-CA-0056).

Lane C is a container for fine-tuning COMPETITIONS, and only a competition
with a forward-only eval is admitted (MarketCast-only lane C, owner
2026-10-20). Today the only one is MarketCast (DEC-CA-0064): a ``--target
mkt:<suite>`` entry is a tree with ``marketcast.json`` at its root (the
declarative fine-tune recipe for the pinned Timestral-28M base) plus an
OPTIONAL generator (``generator.py`` / a packaged tree) whose series are mixed
in at ``synthetic_fraction``. The DEC-CA-0056 generic targets (``domain:`` /
``freq:`` / ``horizon:``) are refused everywhere, and while MarketCast is off
every ``mkt:`` target is refused with a clear message. Lane-C entries never
consume your lane-A lifetime submission, but a hotkey holds ONE chain
commitment: committing a lane-C entry replaces your lane-A / lane-B
commitment until you commit again.

* ``check <repo_dir> --target T`` — validate ``marketcast.json`` locally (the
  trainer's own rule). Nothing is sent.
* ``commit --target T --ref R`` — validate the recipe (``--repo-dir`` for a
  local copy, else the ref is fetched), then chain-commit
  ``metro-v1:ft:<T>:hippius:<R>`` with a timed reveal.
* ``fund <intake> --target T --ref R --yes`` — fund ONE fine-tune leg with
  your Lium key (env only, never argv): ``POST /v1/lane-c/fund``, signed by
  your hotkey over the key digest, the target and the ref.
  ``--withdraw`` takes a still-queued entry back out.
* ``status <intake> [--hotkey H]`` — the public lane-C queue
  (``GET /v1/lane-c/queue``), optionally just your entry.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from .output import NETWORK, OK, REGISTRY, REJECTED, USAGE, reporter


def _wallet_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--wallet-name", required=True, help="Bittensor wallet (coldkey) name.")
    p.add_argument("--wallet-hotkey", required=True, help="Bittensor wallet hotkey name.")
    p.add_argument("--wallet-path", default=None, help="Optional non-default wallet root.")


def add_lane_c(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("lane-c", help="Lane C (fine-tuning competitions — today MarketCast): "
                                      "check, commit, fund and track a recipe.")
    lsub = p.add_subparsers(dest="lane_c_cmd", required=True)
    from .lane_tests import add_lane_test
    add_lane_test(lsub, "c")
    from .marketcast_data import add_data_commands
    add_data_commands(lsub)
    u = lsub.add_parser("upload", help="Validate and upload a recipe tree; no chain commitment or GPU funding.")
    u.add_argument("repo_dir", type=Path)
    u.add_argument("--target", required=True)
    u.add_argument("--hub-repo", required=True, help="Your registry namespace/repository.")
    u.add_argument("--chain-toml", type=Path, default=None)
    u.set_defaults(func=cmd_upload, command="lane-c upload")

    c = lsub.add_parser("check", help="Validate a recipe tree's marketcast.json locally.")
    c.add_argument("repo_dir", type=Path, help="Your recipe tree (with marketcast.json).")
    c.add_argument("--target", required=True,
                   help="The target you will commit (must equal the recipe's).")
    c.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    c.set_defaults(func=cmd_check, command="lane-c check")

    m = lsub.add_parser("commit", help="Validate the recipe, then commit it on chain.")
    m.add_argument("--target", required=True,
                   help="Declared target: one of chain.toml [lane_c] targets "
                        "(mkt:<suite>).")
    m.add_argument("--ref", required=True,
                   help="The uploaded recipe tree (repo@digest — e.g. from `cascade deploy "
                        "--ref` / your Hub push).")
    m.add_argument("--repo-dir", type=Path, default=None,
                   help="Local copy of the tree to validate (default: fetch --ref).")
    m.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    m.add_argument("--network", default="finney", help="Bittensor network (finney/test/local).")
    m.add_argument("--blocks-until-reveal", type=int, default=None,
                   help="Explicit timelock reveal delay in blocks (default: timed reveal "
                        "just before the next epoch boundary).")
    m.add_argument("--reveal-now", action="store_true",
                   help="Reveal immediately (blocks_until_reveal=1).")
    m.add_argument("--next-epoch", action="store_true",
                   help="Time the reveal for the FOLLOWING epoch boundary.")
    _wallet_args(m)
    m.set_defaults(func=cmd_commit, command="lane-c commit")

    f = lsub.add_parser("fund", help="Fund one fine-tune leg with YOUR provider key (paid).")
    f.add_argument("intake_url", help="Operator intake base URL (https://…).")
    f.add_argument("--target", required=True, help="The committed target.")
    f.add_argument("--ref", required=True, help="The committed recipe tree ref (repo@digest).")
    from ..funding.providers import add_provider_args

    add_provider_args(f)
    f.add_argument("--lium-key-env", default="LIUM_API_KEY",
                   help="Env var holding your Lium API key (never on the command line).")
    f.add_argument("--label", default="", help="Display name for the public queue (cosmetic).")
    f.add_argument("--withdraw", action="store_true",
                   help="Withdraw a still-queued lane-C entry instead of funding.")
    f.add_argument("--yes", action="store_true",
                   help="Confirm funding: the leg rents a pod billed to your selected provider key.")
    f.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    from .cli import add_gpu_args

    add_gpu_args(f)
    _wallet_args(f)
    f.set_defaults(func=cmd_fund, command="lane-c fund")

    s = lsub.add_parser("status", help="The public lane-C queue (optionally your entry).")
    s.add_argument("intake_url", help="Operator intake base URL.")
    s.add_argument("--hotkey", default=None, help="Show only this hotkey's entry.")
    s.set_defaults(func=cmd_status, command="lane-c status")


# ── validation (pure; testable) ──────────────────────────────────────────────


def recipe_error(cfg, tree_dir: Path, target: str | None) -> str | None:
    """Why ``tree_dir`` cannot be committed as a lane-C entry for ``target``
    (None = ok): the target must be a ``[lane_c]`` target (a forward-only
    competition's — today ``mkt:<suite>``) and its recipe must pass the
    trainer's own rule with the same target."""
    import time as _time

    from ..shared.config.lane_c import check_target
    from ..trainer.finetune import RecipeError
    from ..trainer.marketcast_runner import read_mkt_recipe

    lc = getattr(cfg, "lane_c", None)
    targets = tuple(getattr(lc, "targets", ()) or ())
    if not targets:
        return "this chain.toml configures no [lane_c] targets (lane C is not set up)"
    if target is None:
        return "pass --target (the competition target you will commit, e.g. mkt:returns)"
    why = mkt_inactive(cfg, target)
    if why:
        return why
    try:
        target = check_target(target, mkt_suites=_mkt_suites(cfg))
    except ValueError as e:
        return str(e)
    if target not in targets:
        return f"unknown lane-C target {target!r}; choose one of: " + ", ".join(targets)
    try:
        read_mkt_recipe(tree_dir, cfg, target=target, submission_time=_time.time())
    except RecipeError as e:
        return str(e)
    return None


def _mkt_suites(cfg) -> tuple[str, ...]:
    return tuple(getattr(getattr(cfg, "marketcast", None), "target_suites", ()) or ())


def mkt_inactive(cfg, target: str) -> str | None:
    """The refusal for an ``mkt:<suite>`` target while MarketCast is off."""
    from ..shared.config.lane_c import is_mkt_target

    if is_mkt_target(target) and not _mkt_suites(cfg):
        return (f"{target!r}: MarketCast (DEC-CA-0064) is not active on this subnet "
                "([marketcast] from_block = 0) — mkt:<suite> targets are refused")
    return None


def build_lane_c_headers(action: str, hotkey_ss58: str, target: str, ref: str, api_key: str,
                         sign_fn, *, now=None, label: str = "", gpu: str = "",
                         allow_fallback: bool = False,
                         max_price: float | None = None) -> dict[str, str]:
    """The signed header set of a lane-C intake request (pure; testable):
    the canonical message binds the action, hotkey, ``"<target>|<ref>"``, the
    timestamp and the key digest — and, with ``gpu`` (DEC-CA-0060, fund
    only), the GPU headers through the v3 message."""
    from ..funding.intake import LANE_C_TARGET_HEADER, lane_c_signed_ref
    from .funding import build_intake_headers

    return build_intake_headers(action, hotkey_ss58, ref, api_key, sign_fn,
                                now=now, label=label, gpu=gpu,
                                allow_fallback=allow_fallback, max_price=max_price,
                                fund_action="lane-c-fund",
                                target_headers={LANE_C_TARGET_HEADER: target},
                                signed_ref=lane_c_signed_ref(target, ref))


# ── commands ─────────────────────────────────────────────────────────────────


def cmd_upload(args: argparse.Namespace) -> int:
    from ..shared.config import load_chain_config
    from ..shared.hippius import HubConfig, upload_dir_to_hub

    out = reporter(args)
    cfg = load_chain_config(args.chain_toml)
    error = recipe_error(cfg, args.repo_dir, args.target)
    if error:
        return out.fail("bad_recipe", error, REJECTED)
    if has_local_generator(args.repo_dir):
        from .verify import verify_repo
        report = verify_repo(args.repo_dir, cfg)
        if not report.ok:
            return out.fail("verify_failed", report.render(), REJECTED)
    try:
        result = upload_dir_to_hub(args.repo_dir, args.hub_repo, HubConfig.from_storage(cfg.storage))
    except Exception:  # noqa: BLE001 — storage clients can include credentials in their exception text
        return out.fail("upload_failed", "Recipe upload failed; check registry authentication and connectivity.", REGISTRY)
    out.result(ref=result.ref.immutable_ref, size_bytes=result.size_bytes, committed=False, funded=False)
    out.info(f"Uploaded recipe: {result.ref.immutable_ref}; no commitment or funding was sent.")
    return OK


def has_local_generator(repo_dir):
    from ..trainer.marketcast_runner import has_generator_tree
    return has_generator_tree(repo_dir)


def cmd_check(args: argparse.Namespace) -> int:
    from ..shared.config import load_chain_config

    out = reporter(args)
    cfg = load_chain_config(args.chain_toml)
    why = recipe_error(cfg, Path(args.repo_dir), args.target)
    if why:
        return out.fail("bad_recipe", f"recipe refused: {why}", REJECTED)
    out.result(ok=True, repo_dir=str(args.repo_dir), target=args.target)
    out.info("recipe OK — the trainer will accept it")
    return OK


def _fetch_tree(cfg, ref: str, dest: Path) -> None:
    from ..shared.hippius import HubConfig, fetch_from_hub

    fetch_from_hub(ref, dest, HubConfig.from_storage(cfg.storage))


def cmd_commit(args: argparse.Namespace) -> int:
    from ..shared.config import load_chain_config
    from ..shared.lane_c_manifest import format_finetune_commit
    from .cli import _live_cfg, _resolve_blocks_until_reveal, _reveal_flag_error

    out = reporter(args)
    cfg = load_chain_config(args.chain_toml)
    why = _reveal_flag_error(args)
    if why:
        return out.fail("conflicting_flags", why, USAGE)
    why = mkt_inactive(cfg, args.target)
    if why:
        return out.fail("bad_commit", f"refusing to commit: {why}", USAGE)
    try:
        payload = format_finetune_commit(args.target, args.ref, mkt_suites=_mkt_suites(cfg))
    except ValueError as e:
        return out.fail("bad_commit", f"refusing to commit: {e}", USAGE)
    if args.repo_dir is not None:
        why = recipe_error(cfg, Path(args.repo_dir), args.target)
    else:
        with tempfile.TemporaryDirectory(prefix="cascade-lane-c-") as tmp:
            try:
                _fetch_tree(cfg, args.ref, Path(tmp) / "tree")
            except Exception as e:  # noqa: BLE001
                return out.fail("fetch_failed", f"could not fetch {args.ref} to validate its "
                                f"recipe ({e}); pass --repo-dir with a local copy", REGISTRY)
            why = recipe_error(cfg, Path(tmp) / "tree", args.target)
    if why:
        return out.fail("bad_recipe", f"recipe refused — nothing committed: {why}", REJECTED)

    from ..shared.chain import ChainClient, ChainError

    try:
        client = ChainClient.from_config(cfg, network=args.network,
                                         wallet_name=args.wallet_name,
                                         wallet_hotkey=args.wallet_hotkey,
                                         wallet_path=args.wallet_path)
        cfg = _live_cfg(cfg, client)
        current_block = client.current_block()
        from ..shared.lane_c import competition_closed

        why = competition_closed(cfg.lane_c, args.target, current_block)
        if why:
            # concurrent competitions (owner 2026-10-20): nothing new past the end
            return out.fail("competition_ended", f"refusing to commit: {why}", USAGE)
        blocks = _resolve_blocks_until_reveal(args, cfg, current_block)
        client.commit_submission(payload, blocks_until_reveal=blocks)
    except ChainError as e:
        return out.fail("chain_error", f"chain error: {e}", NETWORK)
    except ValueError as e:
        return out.fail("bad_reveal_config", f"bad [round] reveal config: {e}", USAGE)
    out.result(payload=payload, target=args.target, ref=args.ref, commit_block=current_block,
               reveal_block=current_block + blocks)
    out.info(f"committed: {payload}\nonce it reveals, fund the leg:\n"
             f"  cascade lane-c fund <intake> --target {args.target} --ref {args.ref} --yes ...")
    return OK


def cmd_fund(args: argparse.Namespace) -> int:
    import urllib.request

    from .cli import post_intake, report_gpu_reply
    from .funding import prepare_funding_request

    out = reporter(args)
    action = "lane-c-withdraw" if args.withdraw else "lane-c-fund"
    try:
        from ..shared.config import load_chain_config
        from ..shared.config.lane_c import check_target

        cfg = load_chain_config(getattr(args, "chain_toml", None))
        why = mkt_inactive(cfg, args.target)
        if why:
            return out.fail("bad_target", f"--target refused: {why}", USAGE)
        target = check_target(args.target, mkt_suites=_mkt_suites(cfg))
    except ValueError as e:
        return out.fail("bad_target", f"--target refused: {e}", USAGE)
    prepared = prepare_funding_request(
        args, out, funding=action == "lane-c-fund",
        confirmation="funding rents a fine-tune pod billed to your selected provider key")
    if isinstance(prepared, int):
        return prepared
    headers = build_lane_c_headers(
        action, prepared.hotkey, target, args.ref, prepared.api_key, prepared.sign_fn,
        label=prepared.label, gpu=prepared.choice[0], allow_fallback=prepared.choice[1],
        max_price=prepared.choice[2])
    path = "/v1/lane-c/withdraw" if args.withdraw else "/v1/lane-c/fund"
    req = urllib.request.Request(f"{prepared.base}{path}", method="POST", headers=headers)
    answer = post_intake(req, out, timeout=30)
    if answer is None:
        return NETWORK
    status, body = answer
    if 200 <= status < 300:
        out.result(action=action, status=body.get("status", "ok"), hotkey=prepared.hotkey,
                   target=target, ref=args.ref)
        out.info(f"{action}: {body.get('status', 'ok')} (hotkey {prepared.hotkey}, target {target})")
        report_gpu_reply(out, body)
        if action == "lane-c-fund":
            from .submission_status import tracking_result

            tracking_result(out, args, hotkey=prepared.hotkey, ref=args.ref, lane="c", targets=(target,))
        return OK
    return out.fail(str(body.get("code") or f"{action}_rejected"),
                    f"{action} rejected ({status}): {body.get('code', '?')} — "
                    f"{body.get('message', '')}", http_status=status)


def cmd_status(args: argparse.Namespace) -> int:
    from .submission_status import lane_queue_status

    return lane_queue_status(args, "c")


__all__ = ["add_lane_c", "build_lane_c_headers", "recipe_error"]
