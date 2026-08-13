"""Salvage r15's bench report — sign and publish it from raw pod-side reports.

The r15 (round 9773102965479587330) post-publish bench thread died with the
2026-08-12 22:25 trainer restart (approved: the r16 heat was running on 2
misassigned lanes) after its dataset download wedged on unauthenticated HF
throttling. The batteries were re-run manually on the held final pod with
complete data; this script rebuilds the round's signed BenchReport from those
raw ``benchmark_report.json`` files and publishes it to the manifest bucket —
byte-for-byte the object ``run_post_publish_bench`` would have published.

The next trainer restart backfills the promotion candidates from the published
report (same recovery path as the 2026-08-11 deploy backfill).

Usage (from /root/cascade, .env loaded — uses the trainer wallet to sign):
    .venv/bin/python scripts/salvage_r15_bench_report.py \
        --king king-benchmark_report.json --challenger challenger-benchmark_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade.eval.benchmarks import extract_bench_scores  # noqa: E402
from cascade.shared.bench_report import (  # noqa: E402
    BenchEntry,
    BenchReport,
    bench_report_key,
    dump_bench_report,
    load_bench_report,
    publish_bench_report,
    sign_bench_report,
    verify_bench_report_signature,
)
from cascade.shared.config import load_chain_config  # noqa: E402
from cascade.shared.hippius import manifest_round_key, open_manifest_store  # noqa: E402
from cascade.shared.manifest import BenchScores, load_manifest  # noqa: E402

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-id", default="9773102965479587330",
                    help="round whose bench report to rebuild (default: r15)")
    ap.add_argument("--king", type=Path, default=None,
                    help="raw benchmark_report.json for the king checkpoint")
    ap.add_argument("--challenger", type=Path, default=None,
                    help="raw benchmark_report.json for the challenger checkpoint")
    ap.add_argument("--wallet-name", default="main_one")
    ap.add_argument("--wallet-hotkey", default="sn_validator_hk")
    ap.add_argument("--chain-toml", type=Path, default=Path("chain.toml"))
    ap.add_argument("--yes", action="store_true", help="actually publish")
    args = ap.parse_args()

    cfg = load_chain_config(args.chain_toml)
    store = open_manifest_store(cfg.storage)
    rid = str(args.round_id)
    manifest = load_manifest(store.get_text(manifest_round_key(rid)))

    raw = {}
    for role, p in (("king", args.king), ("challenger", args.challenger)):
        if p is None:
            continue
        raw[role] = json.loads(p.read_text(encoding="utf-8"))

    entries = []
    for m_entry in manifest.entries:
        rep = raw.get(m_entry.role)
        scores = extract_bench_scores(rep) if rep is not None else None
        if scores is None:
            print(f"{m_entry.role}: no complete score set — omitted", file=sys.stderr)
            continue
        entries.append(BenchEntry(
            role=m_entry.role, size=m_entry.size or "toto2-4m",
            miner_hotkey=m_entry.miner_hotkey, miner_uid=m_entry.miner_uid,
            trained_pointer=m_entry.trained_pointer, scores=BenchScores(**scores),
        ))
        print(f"{m_entry.role}: {scores}")
    if not entries:
        print("no entry produced a complete score set; nothing to publish",
              file=sys.stderr)
        return 1

    report = BenchReport(round_id=rid, created_block=manifest.created_block,
                         entries=tuple(entries))
    import bittensor as bt
    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    report = sign_bench_report(report, wallet)
    assert verify_bench_report_signature(report, cfg.manifest.trainer_hotkey), \
        "signed report does not verify against the pinned trainer_hotkey"

    text = dump_bench_report(report)
    print(f"report: round={rid} entries={len(entries)} "
          f"key={bench_report_key(rid)} bytes={len(text)} signature=VERIFIED")
    if not args.yes:
        print("dry-run (pass --yes to publish)")
        return 0
    key = publish_bench_report(store, text, rid)
    print(f"published → {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
