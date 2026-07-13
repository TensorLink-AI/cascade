"""``cascade-audit`` console script — verify a published round receipt.

Commands:

* ``cascade-audit latest`` — audit the newest receipt (``receipts/latest.json``).
* ``cascade-audit round <id>`` — audit one round by id.

``--tier`` picks the depth (0 = seconds/CPU, 1 = +corpus re-derivation,
2 = +re-training, experimental). ``--json`` emits machine-readable results.
Exit status is nonzero iff any check FAILs — CI-usable. Checks that need chain
history a lite node can't serve WARN explicitly rather than passing silently.

Runs credential-free where the storage allows it: the receipt is fetched with
an unsigned S3 request first (falling back to ``HIPPIUS_S3_*`` credentials if
present), or read from a local file via ``--receipt``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ..shared.config import load_chain_config
from ..shared.receipt import RoundReceipt, load_receipt
from .checks import FAIL, WARN, CheckResult, run_tier0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cascade-audit",
        description="Verify a cascade round receipt against a re-derivation "
                    "(see docs/AUDIT.md).",
    )
    sub = p.add_subparsers(dest="command", required=True)
    round_p = sub.add_parser("round", help="Audit one round by id.")
    round_p.add_argument("round_id", help="Round id (the base seed / manifest round_id).")
    sub.add_parser("latest", help="Audit the newest published receipt.")
    for sp in (round_p, sub.choices["latest"]):
        sp.add_argument("--tier", type=int, choices=(0, 1, 2), default=0,
                        help="Audit depth: 0 = seconds/CPU (default); 1 = re-derive "
                             "corpora (minutes, CPU); 2 = re-train (GPU, [train] extra, "
                             "EXPERIMENTAL).")
        sp.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
        sp.add_argument("--config", type=Path, default=None,
                        help="chain.toml path (default: the repo's).")
        sp.add_argument("--receipt", type=Path, default=None,
                        help="Read the receipt from a local JSON file instead of S3.")
        sp.add_argument("--network", default="finney",
                        help="Bittensor network for the optional chain checks.")
        sp.add_argument("--no-chain", action="store_true",
                        help="Skip chain lookups (their checks WARN).")
        sp.add_argument("--workdir", type=Path, default=Path("./_audit_work"),
                        help="Scratch dir for tier 1/2 fetches and re-derivations.")
        sp.add_argument("--full-stream", action="store_true",
                        help="Tier 1: re-stream the FULL training budget for stream_cpu "
                             "corpora (costs about the round's own generation time).")
        sp.add_argument("--device", default="cuda", help="Tier 2 eval device.")
        sp.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


# ── receipt fetch (credential-free first) ─────────────────────────────────────


def _unsigned_s3_text(cfg, key: str) -> str:
    """GET one S3 object anonymously (public-read buckets need no credentials)."""
    import boto3  # type: ignore
    from botocore import UNSIGNED  # type: ignore
    from botocore.config import Config  # type: ignore

    client = boto3.client(
        "s3", endpoint_url=cfg.storage.s3_endpoint, region_name=cfg.storage.s3_region,
        config=Config(signature_version=UNSIGNED, s3={"addressing_style": "path"}),
    )
    resp = client.get_object(Bucket=cfg.storage.manifest_bucket, Key=key)
    return resp["Body"].read().decode("utf-8")


def fetch_receipt_text(cfg, round_id: str | None) -> str:
    """The receipt JSON for ``round_id`` (None ⇒ latest), anonymous-first.

    When ``[manifest] validator_hotkey`` is pinned, that validator's
    ``receipts/<hotkey>/…`` prefix is tried first, then the legacy shared
    ``receipts/…`` keys (receipts published before per-validator prefixes).
    """
    from ..shared.hippius import (
        open_manifest_store,
        receipt_latest_key,
        receipt_round_key,
    )

    def _key(hotkey: str) -> str:
        return (receipt_latest_key(hotkey) if round_id is None
                else receipt_round_key(round_id, hotkey))

    pinned = str(getattr(cfg.manifest, "validator_hotkey", "") or "")
    keys = [_key(pinned), _key("")] if pinned else [_key("")]
    store = None
    errors: list[str] = []
    for key in keys:
        try:
            return _unsigned_s3_text(cfg, key)
        except ImportError as e:
            raise SystemExit(
                f"boto3 unavailable ({e}); install the [hippius] extra or pass --receipt FILE"
            ) from e
        except Exception as anon_err:  # noqa: BLE001 — fall back to credentials if present
            errors.append(f"{key}: anonymous ({anon_err})")
        # HF-backed store when [storage] hf_backup_repo is set, so an auditor can
        # still fetch the receipt during a Hippius S3 outage.
        if store is None:
            store = open_manifest_store(cfg.storage)
        try:
            return store.get_text(key)
        except Exception as cred_err:  # noqa: BLE001
            errors.append(f"{key}: credentialed ({cred_err})")
    raise SystemExit(
        f"could not fetch a receipt from s3://{cfg.storage.manifest_bucket}: "
        + "; ".join(errors) + ". Pass --receipt FILE to audit a local copy."
    )


def _chain_client(cfg, network: str):
    """A read-only chain client, or None (checks then WARN) if unavailable."""
    from ..shared.chain import ChainClient

    try:
        client = ChainClient(netuid=cfg.netuid, network=network)
        client.current_block()  # force the connection now, not mid-check
        return client
    except Exception as e:  # noqa: BLE001
        print(f"note: chain unavailable ({e}); chain-dependent checks will WARN",
              file=sys.stderr)
        return None


# ── output ────────────────────────────────────────────────────────────────────


_STATUS_ORDER = {"FAIL": 0, "WARN": 1, "SKIP": 2, "PASS": 3}


def render_table(receipt: RoundReceipt, tier: int, results: list[CheckResult]) -> str:
    lines = [
        f"round {receipt.round_id}  status={receipt.status}  tier={tier}",
        "",
    ]
    width = max(len(r.name) for r in results)
    for r in results:
        lines.append(f"  [{r.status:>4}] {r.name:<{width}}  {r.detail}")
    counts = {s: sum(1 for r in results if r.status == s) for s in _STATUS_ORDER}
    lines.append("")
    lines.append("  " + "  ".join(f"{s}={counts[s]}" for s in _STATUS_ORDER if counts[s]))
    return "\n".join(lines)


def render_json(receipt: RoundReceipt, tier: int, results: list[CheckResult]) -> str:
    return json.dumps({
        "round_id": receipt.round_id,
        "status": receipt.status,
        "tier": tier,
        "ok": not any(r.status == FAIL for r in results),
        "checks": [{"name": r.name, "status": r.status, "detail": r.detail}
                   for r in results],
    }, indent=2, sort_keys=True)


def audit_receipt(
    receipt: RoundReceipt, cfg, *, tier: int, client=None,
    workdir: Path = Path("./_audit_work"), full_stream: bool = False,
    device: str = "cuda",
) -> list[CheckResult]:
    """Run every check up to ``tier``. Pure orchestration; each check is a small
    function in :mod:`cascade.audit.checks` / :mod:`cascade.audit.rederive`."""
    results = run_tier0(receipt, cfg, client)
    if tier >= 1:
        from .rederive import run_tier1

        results += run_tier1(receipt, cfg, workdir=workdir, full_stream=full_stream)
    if tier >= 2:
        from .rederive import run_tier2

        results += run_tier2(receipt, cfg, workdir=workdir, device=device)
    return results


def main(argv: list[str] | None = None) -> int:
    from ..shared.env import load_env_files
    load_env_files()
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_chain_config(args.config)

    round_id = getattr(args, "round_id", None)
    if args.receipt is not None:
        text = args.receipt.read_text(encoding="utf-8")
    else:
        text = fetch_receipt_text(cfg, round_id)
    try:
        receipt = load_receipt(text)
    except (ValueError, KeyError) as e:
        print(f"unparseable receipt: {e}", file=sys.stderr)
        return 2
    if round_id is not None and receipt.round_id != str(round_id):
        print(f"receipt round_id {receipt.round_id} != requested {round_id}", file=sys.stderr)
        return 2

    client = None if args.no_chain else _chain_client(cfg, args.network)
    results = audit_receipt(
        receipt, cfg, tier=args.tier, client=client,
        workdir=args.workdir, full_stream=args.full_stream, device=args.device,
    )

    print(render_json(receipt, args.tier, results) if args.json
          else render_table(receipt, args.tier, results))
    if any(r.status == FAIL for r in results):
        return 1
    if not args.json and any(r.status == WARN for r in results):
        print("\n  (WARN = could not fully verify; see docs/AUDIT.md)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
