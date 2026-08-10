#!/usr/bin/env python
"""Publish a round's public-benchmark report for the dashboards.

The stakeholder scoreboard probes ``benchmarks/round-<round_id>.json`` for the
GIFT-Eval / BOOM / TIME numbers of each round's checkpoints. Where production
already publishes those docs, this script is not needed; where it does not,
this closes the loop: it takes the benchmark sidecar's ``report.json`` for one
or more of a round's checkpoints and publishes the combined round doc
(public-read) to the manifest bucket.

Each entry is stamped with its ``role`` (king/challenger) and ``size`` (arch
preset). The size stamp is REQUIRED: once rounds train at more than one ladder
size, an unstamped entry cannot be attributed to a rung and the page drops it
from the per-size views. The six scores are extracted with the shared
convention (:func:`cascade.eval.benchmarks.extract_bench_scores`); an
incomplete battery is refused, never published.

Round docs are immutable once published — the page fetches each exactly once,
uncache-busted — so an existing doc is refused unless ``--force``.

Example (after the sidecar has scored both checkpoints)::

    python scripts/publish_round_bench.py --round-id 14400 \
        --entry role=king,size=toto2-4m,report=/out/king-report.json \
        --entry role=challenger,size=toto2-4m,report=/out/chal-report.json \
        --wallet-name trainer --wallet-hotkey default

Signing is optional but recommended: with ``--wallet-*`` the doc carries the
hotkey and an sr25519 signature over its canonical body (sorted-key JSON
without the signature fields), mirroring the receipt convention.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

from cascade.eval.benchmarks import extract_bench_scores
from cascade.shared.config import load_chain_config
from cascade.shared.hippius import ObjectNotFound, S3Config, S3Store, StorageError

ROUND_BENCH_SCHEMA = 1
SIX_KEYS = ("gifteval_crps", "gifteval_mase", "boom_crps", "boom_mase",
            "time_crps", "time_mase")
ROLES = ("king", "challenger")


def _canonical_body(doc: dict) -> bytes:
    body = {k: v for k, v in doc.items() if k not in ("signer_hotkey", "signature")}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def _geomean(scores: dict) -> float:
    return math.exp(sum(math.log(scores[k]) for k in SIX_KEYS) / len(SIX_KEYS))


def _parse_entry(spec: str) -> dict:
    """``role=king,size=toto2-4m,report=/out/r.json[,checkpoint=<ptr>]``."""
    fields: dict[str, str] = {}
    for part in spec.split(","):
        k, sep, v = part.partition("=")
        if not sep or k.strip() not in ("role", "size", "report", "checkpoint"):
            sys.exit(f"bad --entry field {part!r} (want role=/size=/report=[/checkpoint=])")
        fields[k.strip()] = v.strip()
    missing = [k for k in ("role", "size", "report") if not fields.get(k)]
    if missing:
        sys.exit(f"--entry {spec!r} is missing {'/'.join(missing)}")
    if fields["role"] not in ROLES:
        sys.exit(f"--entry role must be one of {ROLES}, got {fields['role']!r}")
    return fields


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Publish a round's benchmark report(s) as benchmarks/round-<id>.json.")
    ap.add_argument("--round-id", required=True, help="The round these reports belong to.")
    ap.add_argument("--entry", action="append", required=True, metavar="SPEC",
                    help="role=<king|challenger>,size=<preset>,report=<sidecar report.json>"
                         "[,checkpoint=<registry pointer>]. Repeatable, one per checkpoint.")
    ap.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    ap.add_argument("--wallet-name", default=None, help="Sign with this bittensor wallet.")
    ap.add_argument("--wallet-hotkey", default=None)
    ap.add_argument("--wallet-path", default=None)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an already-published round doc (they are immutable "
                         "by convention — only for correcting a bad publish).")
    ap.add_argument("--dry-run", action="store_true", help="Print the doc and exit.")
    args = ap.parse_args()

    entries: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for spec in args.entry:
        f = _parse_entry(spec)
        if (f["role"], f["size"]) in seen:
            sys.exit(f"duplicate entry for role={f['role']} size={f['size']} — "
                     "a round carries at most one of each per size")
        seen.add((f["role"], f["size"]))
        report = json.loads(Path(f["report"]).read_text(encoding="utf-8"))
        scores = extract_bench_scores(report)
        if scores is None:
            print(f"refusing to publish: {f['report']} is missing at least one of the "
                  "six GIFT-Eval/BOOM/TIME crps+mase scores", file=sys.stderr)
            return 1
        entry = {
            "role": f["role"],
            "size": f["size"],
            "scores": {k: scores[k] for k in SIX_KEYS},
            "geomean": _geomean(scores),
            "data_revisions": report.get("data_revisions"),
        }
        if f.get("checkpoint"):
            entry["checkpoint"] = f["checkpoint"]
        entries.append(entry)

    doc: dict = {
        "schema": ROUND_BENCH_SCHEMA,
        "kind": "round_bench",
        "round_id": str(args.round_id),
        "entries": entries,
        "as_of": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    if args.wallet_name:
        # Mirror the receipt convention: hex sr25519 signature over the
        # canonical (sorted-key, signature-free) JSON body.
        from cascade.shared.chain import ChainClient

        cfg = load_chain_config(args.chain_toml)
        client = ChainClient.from_config(
            cfg, wallet_name=args.wallet_name,
            wallet_hotkey=args.wallet_hotkey, wallet_path=args.wallet_path)
        hotkey = client.wallet().hotkey
        doc["signer_hotkey"] = hotkey.ss58_address
        doc["signature"] = hotkey.sign(_canonical_body(doc)).hex()

    key = f"benchmarks/round-{doc['round_id']}.json"
    text = json.dumps(doc, indent=1)
    if args.dry_run:
        print(f"[dry-run] would publish {len(text)} B to {key}:")
        print(text)
        return 0

    cfg = load_chain_config(args.chain_toml)
    bucket = cfg.storage.manifest_bucket
    s3cfg = S3Config.from_storage(cfg.storage, bucket=bucket)
    store = S3Store(s3cfg)
    if not args.force:
        try:
            store.get_text(key)
        except ObjectNotFound:
            pass
        except StorageError:
            pass  # unreadable ≠ present; the put below will surface real trouble
        else:
            print(f"refusing to overwrite existing {key} — round docs are immutable "
                  "(pass --force only to correct a bad publish)", file=sys.stderr)
            return 1
    try:
        store.put_text(key, text, content_type="application/json", acl="public-read")
    except StorageError:
        # backends that reject canned object ACLs still get the object
        store.put_text(key, text, content_type="application/json")
    print(f"published → {s3cfg.endpoint.rstrip('/')}/{bucket}/{key}"
          f"  ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
