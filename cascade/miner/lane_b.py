"""``cascade lane-b`` — fund and track a domain-expert leg (DEC-CA-0059 item 8).

A lane-B entry is committed with ``cascade deploy --expert-domain <targets>``
(``metro-v1:expert:<domain|codes>:hippius:<ref>``). Every lane is miner-
funded: once the commitment reveals, fund ONE leg with your Lium key and the
trainer normally rents it on YOUR key. A capacity miss waits unburned;
operator-caused make-good work can use configured operator capacity without
charging the miner for a second rental.

* ``fund <intake> --targets T[,T2[,T3]] --ref R --yes`` — ``POST
  /v1/lane-b/fund``, signed by your hotkey over the key digest, the targets
  (exactly as listed) and the ref. ``--withdraw`` takes a still-queued entry
  back out. ``--gpu`` / ``--allow-fallback`` / ``--max-price`` as ``cascade
  fund`` (DEC-CA-0060).
* ``status <intake> [--hotkey H]`` — the public lane-B queue
  (``GET /v1/lane-b/queue``), optionally just your entry.

The targets are validated locally against ``chain.toml [lane_b]`` (the rule
``cascade deploy --expert-domain`` uses) before anything is sent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .output import NETWORK, OK, USAGE, reporter


def add_lane_b(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("lane-b", help="Lane B (domain experts): fund and track a leg.")
    lsub = p.add_subparsers(dest="lane_b_cmd", required=True)
    from .lane_tests import add_lane_test
    add_lane_test(lsub, "b")

    f = lsub.add_parser("fund", help="Fund one domain-expert leg with YOUR provider key (paid).")
    f.add_argument("intake_url", help="Operator intake base URL (https://…).")
    f.add_argument("--targets", required=True,
                   help="The committed targets, comma-separated (as for `cascade deploy "
                        "--expert-domain`, e.g. energy or energy,h720).")
    f.add_argument("--ref", required=True, help="The committed generator ref (repo@digest).")
    from ..funding.providers import add_provider_args

    add_provider_args(f)
    f.add_argument("--lium-key-env", default="LIUM_API_KEY",
                   help="Env var holding your Lium API key (never on the command line).")
    f.add_argument("--label", default="", help="Display name for the public queue (cosmetic).")
    f.add_argument("--withdraw", action="store_true",
                   help="Withdraw a still-queued lane-B entry instead of funding.")
    f.add_argument("--yes", action="store_true",
                   help="Confirm funding: the leg rents a pod billed to your selected provider key.")
    f.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    from .cli import add_gpu_args

    add_gpu_args(f)
    f.add_argument("--wallet-name", required=True, help="Bittensor wallet (coldkey) name.")
    f.add_argument("--wallet-hotkey", required=True, help="Bittensor wallet hotkey name.")
    f.add_argument("--wallet-path", default=None, help="Optional non-default wallet root.")
    f.set_defaults(func=cmd_fund, command="lane-b fund")

    s = lsub.add_parser("status", help="The public lane-B queue (optionally your entry).")
    s.add_argument("intake_url", help="Operator intake base URL.")
    s.add_argument("--hotkey", default=None, help="Show only this hotkey's entry.")
    s.set_defaults(func=cmd_status, command="lane-b status")


def lane_b_fund_targets(cfg, spec: str) -> tuple[tuple[str, ...], str | None]:
    """``(targets in the [lane_b] slot order, None)`` or ``((), why)`` — the
    ``--expert-domain`` rule (``h<h>`` = ``horizon:<h>``), so a fund names
    exactly what the commitment declared."""
    from .cli import expert_targets

    declared, why = expert_targets(cfg, spec)
    if why:
        return (), why
    slots = tuple(getattr(cfg.lane_b, "targets", None) or cfg.lane_b.domains)
    return tuple(t for t in slots if t in declared), None


def build_lane_b_headers(action: str, hotkey_ss58: str, targets, ref: str, api_key: str,
                         sign_fn, *, now=None, label: str = "", gpu: str = "",
                         allow_fallback: bool = False,
                         max_price: float | None = None) -> dict[str, str]:
    """The signed header set of a lane-B intake request (pure; testable):
    the canonical message binds the action, hotkey, ``"<targets>|<ref>"``
    (targets exactly as sent), the timestamp and the key digest — and, with
    ``gpu`` (fund only), the GPU headers through the v3 message."""
    from ..funding.intake import LANE_B_TARGETS_HEADER, lane_b_signed_ref
    from .funding import build_intake_headers

    return build_intake_headers(action, hotkey_ss58, ref, api_key, sign_fn,
                                now=now, label=label, gpu=gpu,
                                allow_fallback=allow_fallback, max_price=max_price,
                                fund_action="lane-b-fund",
                                target_headers={LANE_B_TARGETS_HEADER: ",".join(targets)},
                                signed_ref=lane_b_signed_ref(targets, ref))


def cmd_fund(args: argparse.Namespace) -> int:
    import urllib.request

    from ..shared.config import load_chain_config
    from .cli import post_intake, report_gpu_reply
    from .funding import prepare_funding_request

    out = reporter(args)
    action = "lane-b-withdraw" if args.withdraw else "lane-b-fund"
    cfg = load_chain_config(args.chain_toml)
    targets, why = lane_b_fund_targets(cfg, args.targets)
    if why:
        return out.fail("bad_targets", f"--targets refused: {why}", USAGE)
    prepared = prepare_funding_request(
        args, out, funding=action == "lane-b-fund",
        confirmation="funding rents a domain-expert pod billed to your selected provider key")
    if isinstance(prepared, int):
        return prepared
    headers = build_lane_b_headers(
        action, prepared.hotkey, targets, args.ref, prepared.api_key, prepared.sign_fn,
        label=prepared.label, gpu=prepared.choice[0], allow_fallback=prepared.choice[1],
        max_price=prepared.choice[2])
    path = "/v1/lane-b/withdraw" if args.withdraw else "/v1/lane-b/fund"
    req = urllib.request.Request(f"{prepared.base}{path}", method="POST", headers=headers)
    answer = post_intake(req, out, timeout=30)
    if answer is None:
        return NETWORK
    status, body = answer
    if 200 <= status < 300:
        out.result(action=action, status=body.get("status", "ok"), hotkey=prepared.hotkey,
                   targets=list(targets), ref=args.ref)
        out.info(f"{action}: {body.get('status', 'ok')} (hotkey {prepared.hotkey}, targets "
                 f"{','.join(targets)})")
        report_gpu_reply(out, body)
        if action == "lane-b-fund":
            from .submission_status import tracking_result

            tracking_result(out, args, hotkey=prepared.hotkey, ref=args.ref, lane="b", targets=targets)
        return OK
    return out.fail(str(body.get("code") or f"{action}_rejected"),
                    f"{action} rejected ({status}): {body.get('code', '?')} — "
                    f"{body.get('message', '')}", http_status=status)


def cmd_status(args: argparse.Namespace) -> int:
    from .submission_status import lane_queue_status

    return lane_queue_status(args, "b")


__all__ = ["add_lane_b", "build_lane_b_headers", "lane_b_fund_targets"]
