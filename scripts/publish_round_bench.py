#!/usr/bin/env python
"""Publish a round's public-benchmark report as a canonical ``BenchReport``.

Backfill for rounds where the trainer's own post-publish bench sidecar failed
(``benchmarks/round-<round_id>.json`` absent): takes the external sidecar
``report.json`` for one or more of the round's duel checkpoints and publishes
the trainer-signed round doc — byte-compatible with what
``TrainerRunner._post_publish_bench`` publishes in production.

The doc at that key has exactly ONE consumer-facing schema:
:mod:`cascade.shared.bench_report`'s ``BenchReport`` — the validator's Cascade
reign log drains it with the strict ``load_bench_report`` parser (joining
entries to the manifest on ``trained_pointer``), the trainer's promotion
backfill replays it, and the stakeholder scoreboard reads the same docs
production publishes. An earlier version of this script wrote a bespoke
scoreboard-shaped doc to the SAME key; that doc is unparseable by the drain
and, being immutable-once-published, would have blocked the correct publish
behind ``--force``. Do not reintroduce a second schema here.

Entry identity (``miner_hotkey`` / ``miner_uid`` / ``trained_pointer``) and
``created_block`` are derived from the round's signed manifest — the caller
only names the ``role`` (and ``size`` when a round trained more than one) and
points at the sidecar report; an incomplete six-score battery is refused,
never published. Signing follows the trainer convention (`sign_bench_report`
with the trainer wallet's hotkey); an UNSIGNED report is accepted only with
``--allow-unsigned`` because validators verify against the pinned
``[manifest] trainer_hotkey`` and silently ignore anything else.

Example (after the external sweeps scored both checkpoints)::

    python scripts/publish_round_bench.py --round-id 3338583695172902217 \
        --entry role=king,report=/out/king-report.json \
        --entry role=challenger,report=/out/chal-report.json \
        --wallet-name main_one --wallet-hotkey sn_validator_hk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cascade.eval.benchmarks import extract_bench_scores
from cascade.shared.bench_report import (
    BenchEntry,
    BenchReport,
    bench_report_key,
    dump_bench_report,
    load_bench_report,
    publish_bench_report,
    sign_bench_report,
)
from cascade.shared.config import load_chain_config
from cascade.shared.hippius import manifest_round_key, open_manifest_store
from cascade.shared.manifest import BenchScores, load_manifest

ROLES = ("king", "challenger")


def _parse_entry(spec: str) -> dict:
    """``role=king,report=/out/r.json[,size=<preset>][,checkpoint=<ptr>]``."""
    fields: dict[str, str] = {}
    for part in spec.split(","):
        k, sep, v = part.partition("=")
        if not sep or k.strip() not in ("role", "size", "report", "checkpoint"):
            sys.exit(f"bad --entry field {part!r} (want role=/report=[/size=/checkpoint=])")
        fields[k.strip()] = v.strip()
    missing = [k for k in ("role", "report") if not fields.get(k)]
    if missing:
        sys.exit(f"--entry {spec!r} is missing {'/'.join(missing)}")
    if fields["role"] not in ROLES:
        sys.exit(f"--entry role must be one of {ROLES}, got {fields['role']!r}")
    return fields


def _manifest_entry(manifest, f: dict):
    """The manifest entry a spec names: by explicit checkpoint pointer when
    given, else by role (+ size when the round trained several)."""
    if f.get("checkpoint"):
        for e in manifest.entries:
            if e.trained_pointer == f["checkpoint"]:
                return e
        sys.exit(f"checkpoint {f['checkpoint']!r} is not an entry of round "
                 f"{manifest.round_id} — the report must belong to this round")
    matches = [e for e in manifest.entries
               if e.role == f["role"] and (not f.get("size") or (e.size or "") == f["size"])]
    if not matches:
        sys.exit(f"round {manifest.round_id} has no {f['role']} entry"
                 + (f" at size {f['size']}" if f.get("size") else ""))
    if len(matches) > 1:
        sys.exit(f"round {manifest.round_id} has {len(matches)} {f['role']} entries — "
                 "disambiguate with size= or checkpoint=")
    return matches[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Publish a round's benchmark report(s) as the canonical "
                    "trainer-signed BenchReport at benchmarks/round-<id>.json.")
    ap.add_argument("--round-id", required=True, help="The round these reports belong to.")
    ap.add_argument("--entry", action="append", required=True, metavar="SPEC",
                    help="role=<king|challenger>,report=<sidecar report.json>"
                         "[,size=<preset>][,checkpoint=<registry pointer>]. "
                         "Repeatable, one per checkpoint.")
    ap.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    ap.add_argument("--wallet-name", default=None, help="Sign with this bittensor wallet "
                    "(the trainer wallet — validators verify against its hotkey).")
    ap.add_argument("--wallet-hotkey", default=None)
    ap.add_argument("--wallet-path", default=None)
    ap.add_argument("--allow-unsigned", action="store_true",
                    help="Publish without a signature (validators will IGNORE the "
                         "report — only for testing).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an already-published round doc (they are immutable "
                         "by convention — only for correcting a bad publish).")
    ap.add_argument("--dry-run", action="store_true", help="Print the doc and exit.")
    args = ap.parse_args()

    if not args.wallet_name and not args.allow_unsigned:
        sys.exit("refusing to publish unsigned: pass --wallet-name (trainer wallet) "
                 "or --allow-unsigned if you really mean it")

    cfg = load_chain_config(args.chain_toml)
    store = open_manifest_store(cfg.storage)
    manifest = load_manifest(store.get_text(manifest_round_key(str(args.round_id))))
    if manifest.round_id != str(args.round_id):
        sys.exit(f"manifest round_id {manifest.round_id!r} != --round-id {args.round_id!r}")

    entries: list[BenchEntry] = []
    seen: set[str] = set()
    for spec in args.entry:
        f = _parse_entry(spec)
        m_entry = _manifest_entry(manifest, f)
        if m_entry.trained_pointer in seen:
            sys.exit(f"duplicate entry for checkpoint {m_entry.trained_pointer}")
        seen.add(m_entry.trained_pointer)
        import json as _json
        report = _json.loads(Path(f["report"]).read_text(encoding="utf-8"))
        scores = extract_bench_scores(report)
        if scores is None:
            print(f"refusing to publish: {f['report']} is missing at least one of the "
                  "six GIFT-Eval/BOOM/TIME crps+mase scores", file=sys.stderr)
            return 1
        entries.append(BenchEntry(
            role=m_entry.role,
            size=f.get("size") or m_entry.size or cfg.throne_contracts()[0].arch_preset,
            miner_hotkey=m_entry.miner_hotkey,
            miner_uid=m_entry.miner_uid,
            trained_pointer=m_entry.trained_pointer,
            scores=BenchScores(**scores),
        ))

    doc = BenchReport(round_id=manifest.round_id,
                      created_block=manifest.created_block,
                      entries=tuple(entries))
    if args.wallet_name:
        import bittensor as bt

        wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey,
                           path=args.wallet_path) if args.wallet_path else \
            bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
        doc = sign_bench_report(doc, wallet)

    text = dump_bench_report(doc)
    if args.dry_run:
        print(f"[dry-run] would publish {len(text)} B to "
              f"{bench_report_key(manifest.round_id)}:")
        print(text)
        return 0

    if not args.force:
        try:
            load_bench_report(store.get_text(bench_report_key(manifest.round_id)))
        except Exception:  # noqa: BLE001 — absent/unparseable ⇒ safe to publish
            pass
        else:
            print(f"refusing to overwrite existing {bench_report_key(manifest.round_id)} — "
                  "round docs are immutable (pass --force only to correct a bad publish)",
                  file=sys.stderr)
            return 1
    key = publish_bench_report(store, text, manifest.round_id)
    print(f"published → s3://{cfg.storage.manifest_bucket}/{key}  "
          f"({len(entries)} entries, signed={doc.signature is not None})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
