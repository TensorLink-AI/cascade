#!/usr/bin/env python
"""Publish an OFFICIAL-checkpoint benchmark as a dashboard reference doc.

The stakeholder scoreboard frames Cascade's progress against the real
Datadog Toto2 release: same architecture, same size, different data. Both
sides of that comparison must come from the SAME pipeline — the benchmark
sidecar (``benchmarks/``) with its pinned data — so nothing on the page is
a transcribed leaderboard number.

This script takes the sidecar's ``report.json`` for an official checkpoint
and publishes it as ``benchmarks/reference-<preset>.json`` (public-read) in
the manifest bucket, where ``stakeholders.html`` probes for it per ladder
rung. The six scores are extracted with the same convention as everything
else (:func:`cascade.eval.benchmarks.extract_bench_scores`); an incomplete
battery is refused, never published.

Workflow (once per official size)::

    # 1. wrap the official weights in the cascade checkpoint layout
    #    (weights.safetensors + config.json + model.py + forecast_wrapper.py)
    # 2. score it with the sidecar, exactly like a round checkpoint:
    docker run --rm --gpus all -v /path/ckpt:/ckpt:ro -v /out:/out \
        cascade-bench:<tag> /ckpt /out/report.json --device cuda
    # 3. publish the reference doc (HIPPIUS_S3_ACCESS_KEY/_SECRET_KEY set):
    python scripts/publish_reference_bench.py /out/report.json \
        --preset toto2-4m --source "Datadog/Toto-2.0-4m@hf:<revision>" \
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
from cascade.shared.hippius import S3Config, S3Store, StorageError

REFERENCE_SCHEMA = 1
SIX_KEYS = ("gifteval_crps", "gifteval_mase", "boom_crps", "boom_mase",
            "time_crps", "time_mase")


def _canonical_body(doc: dict) -> bytes:
    body = {k: v for k, v in doc.items() if k not in ("signer_hotkey", "signature")}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def _geomean(scores: dict) -> float:
    return math.exp(sum(math.log(scores[k]) for k in SIX_KEYS) / len(SIX_KEYS))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Publish an official-checkpoint bench report as a reference doc.")
    ap.add_argument("report", type=Path, help="Sidecar report.json for the official checkpoint.")
    ap.add_argument("--preset", required=True,
                    help="Ladder preset this references, e.g. toto2-4m.")
    ap.add_argument("--source", required=True,
                    help="Exact provenance of the scored weights, e.g. "
                         "'Datadog/Toto-2.0-4m@hf:<revision>'.")
    ap.add_argument("--note", default="",
                    help="Optional free-text note (e.g. wrapper commit).")
    ap.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    ap.add_argument("--wallet-name", default=None, help="Sign with this bittensor wallet.")
    ap.add_argument("--wallet-hotkey", default=None)
    ap.add_argument("--wallet-path", default=None)
    ap.add_argument("--dry-run", action="store_true", help="Print the doc and exit.")
    args = ap.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    scores = extract_bench_scores(report)
    if scores is None:
        print("refusing to publish: report is missing at least one of the six "
              "GIFT-Eval/BOOM/TIME crps+mase scores", file=sys.stderr)
        return 1

    doc: dict = {
        "schema": REFERENCE_SCHEMA,
        "kind": "reference_bench",
        "preset": args.preset,
        "source": args.source,
        "scores": {k: scores[k] for k in SIX_KEYS},
        "geomean": _geomean(scores),
        "data_revisions": report.get("data_revisions"),
        "note": args.note,
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

    key = f"benchmarks/reference-{args.preset}.json"
    text = json.dumps(doc, indent=1)
    if args.dry_run:
        print(f"[dry-run] would publish {len(text)} B to {key}:")
        print(text)
        return 0

    cfg = load_chain_config(args.chain_toml)
    bucket = cfg.storage.manifest_bucket
    s3cfg = S3Config.from_storage(cfg.storage, bucket=bucket)
    store = S3Store(s3cfg)
    try:
        store.put_text(key, text, content_type="application/json", acl="public-read")
    except StorageError:
        # backends that reject canned object ACLs still get the object
        store.put_text(key, text, content_type="application/json")
    print(f"published → {s3cfg.endpoint.rstrip('/')}/{bucket}/{key}"
          f"  (geomean {doc['geomean']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
