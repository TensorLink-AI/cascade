"""``cascade-audit`` console script — verify a published round receipt.

Commands:

* ``cascade-audit latest`` — audit the newest receipt (``receipts/latest.json``).
* ``cascade-audit round <id>`` — audit one round by id.
* ``cascade-audit bench <id>`` — the main validator's lane-A bench report of a
  round (DEC-CA-0063); ``--reproduce`` re-sweeps sampled items from the
  public checkpoints and compares within a tolerance (report-only).
* ``cascade-audit direct-moe`` — replay the direct-MoE trainer's signed journal;
  ``--reproduce`` re-scores every decision's published checkpoints and fails
  any recorded score or verdict that does not reproduce (DEC-CA-0052).
* ``cascade-audit marketcast <id>`` — the DEC-CA-0064 MarketCast replay of one
  receipt with every loss recomputed from the published snapshot
  (``--mkt-snapshots DIR``) and quantile sidecars; ``--reproduce`` re-fetches
  a sample of the snapshot's bars from Hyperliquid's public candle API and,
  with torch + the ``timestral`` extra installed, re-forecasts a sample of
  windows from the public checkpoints. It reports; it never changes
  consensus.

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
import contextlib
import json
import logging
import sys
from pathlib import Path

from ..shared.config import load_chain_config
from ..shared.receipt import RoundReceipt, load_receipt
from .checks import FAIL, PASS, WARN, CheckResult, run_tier0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cascade-audit",
        description="Verify a cascade round receipt against a re-derivation "
                    "(see docs/AUDIT.md).",
    )
    sub = p.add_subparsers(dest="command", required=True)
    direct_p = sub.add_parser(
        "direct-moe", help="Replay the direct-MoE journal; --reproduce re-scores its decisions.")
    direct_p.add_argument("receipt", type=Path, nargs="?",
                          help="Local journal/receipt JSON (default: fetch --journal-bucket).")
    direct_p.add_argument("--trainer-hotkey", default=None)
    direct_p.add_argument("--journal-bucket", default=None,
                          help="Hippius bucket the trainer publishes its journal to.")
    direct_p.add_argument("--endpoint", default="https://s3.hippius.com")
    direct_p.add_argument("--reproduce", action="store_true",
                          help="Re-score each decision's checkpoints (needs the [train] extra).")
    direct_p.add_argument("--tolerance", type=float, default=1e-3,
                          help="Relative tolerance per recorded score (cross-hardware noise).")
    direct_p.add_argument("--sidecar", type=Path, default=None,
                          help="JSON PublicSidecar config for the bound public-suite revisions.")
    direct_p.add_argument("--device", default=None,
                          help="Scoring device (default: the one each decision records).")
    direct_p.add_argument("--workdir", type=Path, default=Path("./_audit_work/direct"))
    direct_p.add_argument("--json", action="store_true")
    bench_p = sub.add_parser(
        "bench", help="Check the main validator's lane-A bench report of a round (DEC-CA-0063); "
                      "--reproduce re-sweeps its public checkpoints (report-only).")
    bench_p.add_argument("round_id", help="Settlement round id.")
    bench_p.add_argument("--config", type=Path, default=None,
                         help="chain.toml path (default: the repo's).")
    bench_p.add_argument("--reproduce", action="store_true",
                         help="Re-sweep each entry's checkpoint (needs torch + the sidecar).")
    bench_p.add_argument("--sample", type=int, default=20,
                         help="Items compared per suite per entry (deterministic sample; "
                              "default 20).")
    bench_p.add_argument("--tolerance", type=float, default=0.05,
                         help="Relative tolerance per item score (GPU + sampling noise; "
                              "default 0.05).")
    bench_p.add_argument("--device", default="cuda",
                         help="--reproduce sweep device (default cuda).")
    bench_p.add_argument("--workdir", type=Path, default=Path("./_audit_work/bench"),
                         help="Scratch dir for fetched checkpoints (default ./_audit_work/bench).")
    bench_p.add_argument("--json", action="store_true")
    mkt_p = sub.add_parser(
        "marketcast", help="DEC-CA-0064 MarketCast replay of one receipt; --reproduce "
                           "re-fetches bars and re-forecasts a sample.")
    mkt_p.add_argument("round_id", nargs="?", default=None,
                       help="Round id of the receipt (default: the latest).")
    mkt_p.add_argument("--receipt", type=Path, default=None,
                       help="Read the receipt from a local JSON file instead of S3.")
    mkt_p.add_argument("--validator", default="",
                       help="Validator hotkey (ss58) whose receipt to fetch (default: the "
                            "shared latest pointer, or index discovery for a round id).")
    mkt_p.add_argument("--config", type=Path, default=None,
                       help="chain.toml path (default: the repo's).")
    mkt_p.add_argument("--mkt-snapshots", type=Path, default=None,
                       help="Directory of verified MarketCast snapshot dirs (the revealed "
                            "pool_mkt snapshots); needed to recompute losses.")
    mkt_p.add_argument("--reproduce", action="store_true",
                       help="Re-fetch sampled bars (Hyperliquid) and re-forecast sampled "
                            "windows (torch + the timestral extra).")
    mkt_p.add_argument("--sample", type=int, default=16,
                       help="Windows / bars sampled per entry for --reproduce (default 16).")
    mkt_p.add_argument("--tolerance", type=float, default=2e-3,
                       help="Relative quantile tolerance for --reproduce (cross-hardware; "
                            "default 2e-3).")
    mkt_p.add_argument("--bar-tolerance-bp", type=float, default=1.0,
                       help="--reproduce: re-fetched bar vs snapshot bar tolerance in basis "
                            "points (default 1.0).")
    mkt_p.add_argument("--device", default="cpu",
                       help="--reproduce forecast device (default cpu).")
    mkt_p.add_argument("--workdir", type=Path, default=Path("./_audit_work/marketcast"))
    mkt_p.add_argument("--json", action="store_true")
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
        sp.add_argument("--validator", default="",
                        help="Validator hotkey (ss58) whose receipt to fetch "
                             "(receipts/<hotkey>/…). Default: the shared latest "
                             "pointer, or index discovery for a round id.")
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


def _fetch_bytes(cfg, key: str) -> bytes:
    """GET one bucket object's bytes, anonymous-first (the MarketCast quantile
    sidecars are binary)."""
    from ..shared.hippius import open_manifest_store

    try:
        import boto3  # type: ignore
        from botocore import UNSIGNED  # type: ignore
        from botocore.config import Config  # type: ignore

        client = boto3.client(
            "s3", endpoint_url=cfg.storage.s3_endpoint, region_name=cfg.storage.s3_region,
            config=Config(signature_version=UNSIGNED, s3={"addressing_style": "path"}))
        return client.get_object(Bucket=cfg.storage.manifest_bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001 — fall back to credentials if present
        return open_manifest_store(cfg.storage).get_bytes(key)


def _fetch_text(cfg, key: str) -> str:
    """GET one bucket object, anonymous-first with a credentialed fallback."""
    from ..shared.hippius import open_manifest_store

    try:
        return _unsigned_s3_text(cfg, key)
    except ImportError as e:
        raise SystemExit(
            f"boto3 unavailable ({e}); install the [hippius] extra or pass --receipt FILE"
        ) from e
    except Exception as anon_err:  # noqa: BLE001 — fall back to credentials if present
        # HF-backed store when [storage] hf_backup_repo is set, so an auditor can
        # still fetch the receipt during a Hippius S3 outage.
        store = open_manifest_store(cfg.storage)
        try:
            return store.get_text(key)
        except Exception as cred_err:  # noqa: BLE001
            raise RuntimeError(f"anonymous ({anon_err}); credentialed ({cred_err})") from cred_err


def _resolve_via_index(cfg, round_id: str) -> str | None:
    """Find a round's receipt through the public rolling index.

    Receipts live under per-validator prefixes (``receipts/<hotkey>/round-…``),
    which an S3 GET can't discover by round id alone. Each index entry carries
    a ``receipt_key`` pointer back to its signed receipt; newest publication
    first, first fetch that succeeds wins.
    """
    from ..shared.hippius import RECEIPT_INDEX_KEY

    try:
        doc = json.loads(_fetch_text(cfg, RECEIPT_INDEX_KEY))
    except (RuntimeError, ValueError):
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("rounds"), list):
        return None
    entries = [r for r in doc["rounds"]
               if isinstance(r, dict) and str(r.get("round_id")) == round_id
               and r.get("receipt_key")]
    entries.sort(key=lambda r: str(r.get("published_at") or ""), reverse=True)
    for r in entries:
        try:
            return _fetch_text(cfg, str(r["receipt_key"]))
        except RuntimeError:
            continue
    return None


def fetch_receipt_text(cfg, round_id: str | None, validator_hotkey: str = "") -> str:
    """The receipt JSON for ``round_id`` (None ⇒ latest), anonymous-first.

    ``validator_hotkey`` addresses one validator's receipts directly; without
    it, ``latest`` reads the shared pointer and a round id falls back to
    discovery through ``receipts/index.json`` (the legacy un-namespaced key is
    tried first, so pre-namespacing rounds still resolve).
    """
    from ..shared.hippius import receipt_latest_key, receipt_round_key

    key = (receipt_latest_key(validator_hotkey) if round_id is None
           else receipt_round_key(round_id, validator_hotkey))
    try:
        return _fetch_text(cfg, key)
    except RuntimeError as err:
        if round_id is not None and not validator_hotkey:
            text = _resolve_via_index(cfg, str(round_id))
            if text is not None:
                return text
        raise SystemExit(
            f"could not fetch s3://{cfg.storage.manifest_bucket}/{key}: {err}. "
            "Pass --receipt FILE to audit a local copy, or --validator HOTKEY "
            "to address one validator's receipts."
        ) from err


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


class _BucketReader:
    """``get_text`` over :func:`_fetch_text` (anonymous-first), for the
    DEC-CA-0057 overlay fetch."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def get_text(self, key: str) -> str:
        return _fetch_text(self.cfg, key)

    def get_bytes(self, key: str) -> bytes:
        return _fetch_bytes(self.cfg, key)


def apply_receipt_overlay(receipt: RoundReceipt, cfg, *, store=None):
    """DEC-CA-0057: ``(config, check)`` — the config re-derived with the
    overlay the receipt stamps (fetched by seq, checked against the digest
    and the loaded ``[netconfig] owner_hotkey``), and the ``netconfig`` check
    row. ``(cfg, None)`` for a receipt without the stamp."""
    from ..shared.netconfig import NetConfigError, apply_receipt_netconfig

    seq = int(getattr(receipt, "netconfig_seq", 0) or 0)
    if not seq:
        return cfg, None
    d = str(getattr(receipt, "netconfig_digest", "") or "")
    try:
        out = apply_receipt_netconfig(cfg, receipt, store=store or _BucketReader(cfg))
    except NetConfigError as e:
        return cfg, CheckResult("netconfig", FAIL, f"overlay seq {seq} ({d[:12]}…): {e}")
    except Exception as e:  # noqa: BLE001 — storage: could not verify
        return cfg, CheckResult("netconfig", WARN, f"overlay seq {seq} ({d[:12]}…) not "
                                f"fetched ({e}); checks below use the loaded config")
    if not cfg.netconfig.owner_hotkey:
        return out, CheckResult("netconfig", WARN, f"overlay seq {seq} ({d[:12]}…) applied by "
                                "digest; signature NOT verified ([netconfig] owner_hotkey is "
                                "empty in the loaded config)")
    return out, CheckResult("netconfig", PASS, f"overlay seq {seq} ({d[:12]}…) verified "
                            f"against {cfg.netconfig.owner_hotkey[:12]}… and applied")


def fetch_bench_inputs(cfg, receipt: RoundReceipt, store) -> dict | None:
    """The bench-gate replay input (DEC-CA-0054 / DEC-CA-0061 rule v2):
    ``{trained_pointer: (item rows, suite-level scores)}`` from the round's
    trainer-signed bench report and the rows its entries pin by sha256 —
    what every validator read. A report that is absent or fails its signature
    is ``{}`` (the validators saw no report: uncomputable); rows absent or
    failing their sha256 are ``(None, None)``. ``None`` when the report could
    not be read or verified at all (the check then WARNs)."""
    import dataclasses

    from ..shared.bench_report import (
        bench_report_key,
        load_bench_report,
        load_bench_rows,
        verify_bench_report_signature,
    )
    from ..shared.hippius import ObjectNotFound

    try:
        text = store.get_text(bench_report_key(receipt.round_id))
    except ObjectNotFound:
        return {}
    except (Exception, SystemExit):  # noqa: BLE001 — could not read: not replayable
        return None
    try:
        report = load_bench_report(text)
    except (ValueError, KeyError, TypeError):
        return {}
    try:
        if not verify_bench_report_signature(report, cfg.manifest.trainer_hotkey):
            return {}
    except Exception:  # noqa: BLE001 — no SDK: cannot verify, never trust
        return None
    out: dict = {}
    for e in report.entries:
        rows = None
        if e.rows_key:
            try:
                rows = load_bench_rows(store.get_text(e.rows_key), e.rows_sha256)
            except ObjectNotFound:
                rows = None
            except ValueError:
                rows = None                       # forged / malformed: uncomputable
            except (Exception, SystemExit):  # noqa: BLE001
                return None
        out[e.trained_pointer] = ((rows, dataclasses.asdict(e.scores)) if rows is not None
                                  else (None, None))
    return out


def fetch_king_check_report(cfg, round_id: str, store) -> dict | None:
    """The DEC-CA-0055 king check's bench input: the trainer-signed bench
    report of ``round_id`` (the check reads the PREVIOUS settlement's) as
    ``{"report": BenchReport}``; ``{"report": None}`` when it is absent,
    malformed or fails the trainer signature (no report a validator could
    have read). ``None`` when it could not be read or verified at all."""
    from ..shared.bench_report import (
        bench_report_key,
        load_bench_report,
        verify_bench_report_signature,
    )
    from ..shared.hippius import ObjectNotFound

    try:
        text = store.get_text(bench_report_key(str(round_id)))
    except ObjectNotFound:
        return {"report": None}
    except (Exception, SystemExit):  # noqa: BLE001 — could not read: not replayable
        return None
    try:
        report = load_bench_report(text)
    except (ValueError, KeyError, TypeError):
        return {"report": None}
    try:
        if not verify_bench_report_signature(report, cfg.manifest.trainer_hotkey):
            return {"report": None}
    except Exception:  # noqa: BLE001 — no SDK: cannot verify, never trust
        return None
    return {"report": report}


def fetch_validator_bench(cfg, receipt: RoundReceipt, store) -> dict | None:
    """DEC-CA-0063: the main validator's bench report the receipt binds
    (``receipt.bench_report``): ``{"text_sha", "report", "verified"}`` —
    ``text_sha`` None when the object is absent; ``verified`` = it parses and
    carries the bound signer's (``[netconfig] owner_hotkey``, else the
    receipt's ``signer``) signature. ``None`` when the receipt binds none or
    it could not be read / verified at all."""
    import hashlib

    from ..shared.bench_report import load_bench_report, verify_bench_report_signature
    from ..shared.hippius import ObjectNotFound

    br = getattr(receipt, "bench_report", None)
    if not isinstance(br, dict) or not br.get("key"):
        return None
    try:
        text = store.get_text(str(br["key"]))
    except ObjectNotFound:
        return {"text_sha": None, "report": None, "verified": False}
    except (Exception, SystemExit):  # noqa: BLE001 — could not read: not replayable
        return None
    out = {"text_sha": hashlib.sha256(text.encode("utf-8")).hexdigest(), "report": None,
           "verified": False}
    try:
        report = load_bench_report(text)
    except (ValueError, KeyError, TypeError):
        return out
    signer = str(cfg.netconfig.owner_hotkey or "") or str(br.get("signer", "") or "")
    try:
        ok = verify_bench_report_signature(report, signer)
    except Exception:  # noqa: BLE001 — no SDK: cannot verify, never trust
        return None
    return {**out, "report": report if ok else None, "verified": bool(ok)}


def validator_bench_rows(report, store) -> dict | None:
    """``{trained_pointer: (item rows, scores)}`` from a verified validator
    report — the shape :func:`fetch_bench_inputs` returns. ``None`` when a
    rows object could not be read."""
    import dataclasses

    from ..shared.bench_report import load_bench_rows
    from ..shared.hippius import ObjectNotFound

    out: dict = {}
    for e in report.entries:
        rows = None
        if e.rows_key:
            try:
                rows = load_bench_rows(store.get_text(e.rows_key), e.rows_sha256)
            except (ObjectNotFound, ValueError):
                rows = None
            except (Exception, SystemExit):  # noqa: BLE001
                return None
        out[e.trained_pointer] = ((rows, dataclasses.asdict(e.scores)) if rows is not None
                                  else (None, None))
    revs = {e.trained_pointer: dict(e.data_revisions) for e in report.entries
            if e.data_revisions}
    if revs:
        # DEC-CA-0063: the pinned data revisions the gate's mismatch rule reads
        out["_revisions"] = revs
    faults = {str(f["trained_pointer"]): str(f.get("reason", "") or "checkpoint fault")
              for f in getattr(report, "checkpoint_fault", ()) or ()
              if str(f.get("role", "")) == "challenger"}
    if faults:
        # DEC-CA-0063 item 10: a challenger's signed checkpoint fault fails the gate
        out["_faults"] = faults
    return out


def fetch_marketcast_reports(receipt: RoundReceipt, store) -> dict:
    """DEC-CA-0064: ``{key: text}`` of every MarketCast report the receipt's
    ``mkt:`` lane-C rows bind (absent / unreadable ones are left out — the
    check WARNs on them)."""
    out: dict = {}
    for r in getattr(receipt, "lane_c", None) or ():
        key = str((r.get("mkt") or {}).get("report_key", "") or "")
        if not key or key in out:
            continue
        try:
            out[key] = store.get_text(key)
        except (Exception, SystemExit):  # noqa: BLE001 — not replayable here
            continue
    return out


def fetch_marketcast_inputs(receipt: RoundReceipt, reports: dict, store, *,
                            snapshot_dir: Path | None = None) -> dict:
    """DEC-CA-0064: the rest of what :func:`~cascade.audit.checks.
    check_marketcast` recomputes from — the trainer-signed lane-C manifests
    that carried each entry (``train_time``), the quantile sidecars the
    reports bind, and the verified snapshots found under ``snapshot_dir``
    (one snapshot directory per child). Anything unreadable is left out (the
    check WARNs on it)."""
    from ..shared.lane_c_manifest import lane_c_manifest_key
    from ..shared.marketcast_report import load_mkt_report

    manifests: dict = {}
    for r in getattr(receipt, "lane_c", None) or ():
        er = str((r.get("mkt") or {}).get("entry_round", "") or "")
        if er and er not in manifests:
            with contextlib.suppress(Exception, SystemExit):
                manifests[er] = store.get_text(lane_c_manifest_key(er))
    quantiles: dict = {}
    for text in reports.values():
        try:
            rep = load_mkt_report(text)
        except (ValueError, KeyError, TypeError):
            continue
        for e in rep.entries:
            # the entry's sidecar and (red-team F4) its near-dup reference's
            for qk in (e.quantiles_key, str(dict(e.near_dup or {}).get("quantiles_key", ""))):
                if qk and qk not in quantiles:
                    with contextlib.suppress(Exception, SystemExit):
                        quantiles[qk] = store.get_bytes(qk)
    snapshots: dict = {}
    if snapshot_dir is not None and Path(snapshot_dir).is_dir():
        from ..pool.marketcast import SnapshotError, load_snapshot

        for d in sorted(Path(snapshot_dir).iterdir()):
            try:
                snap = load_snapshot(d, verify_signature=False)
            except SnapshotError:
                continue
            snapshots[snap.digest] = snap
    return {"lane_c_manifests": manifests, "quantiles": quantiles, "snapshots": snapshots}


def audit_receipt(
    receipt: RoundReceipt, cfg, *, tier: int, client=None,
    workdir: Path = Path("./_audit_work"), full_stream: bool = False,
    device: str = "cuda", netconfig_store=None, bench_store=None,
    mkt_snapshot_dir: Path | None = None,
) -> list[CheckResult]:
    """Run every check up to ``tier``. Pure orchestration; each check is a small
    function in :mod:`cascade.audit.checks` / :mod:`cascade.audit.rederive`."""
    from ..shared.activation import apply_receipt_activation

    # DEC-CA-0045: a receipt judged under a rollover the validators decided
    # on chain replays under THAT block, not the config this auditor loaded.
    cfg = apply_receipt_activation(cfg, receipt)
    # DEC-CA-0057: then the owner-signed overlay it was judged under, if any.
    cfg, nc_check = apply_receipt_overlay(receipt, cfg, store=netconfig_store)
    from ..shared.calendar_launch import receipt_calendar_config

    if cfg.v2.launch_at_utc and not cfg.v2.from_block:
        if client is None:
            return [CheckResult("calendar-launch", WARN,
                "dated launch needs a chain connection or an explicitly resolved [v2] from_block; "
                "receipt rules were not replayed")]
        try:
            cfg = receipt_calendar_config(cfg, receipt, client)
        except Exception as exc:
            return [CheckResult("calendar-launch", WARN, f"launch resolution unavailable: {exc}")]
    roster = None
    with contextlib.suppress(Exception):  # absent on any pre-funded round; SKIP handles it
        roster = json.loads(_fetch_text(
            cfg, f"funded/round-{receipt.round_id}.json"))
    # DEC-CA-0054 / DEC-CA-0061: the bench-gate replay reads the signed rows,
    # fetched only for a receipt whose verdict records a bench-gate outcome.
    bench = None
    v = receipt.verdict
    reader = bench_store or _BucketReader(cfg)
    # DEC-CA-0063: a receipt that binds the main validator's report replays
    # the gate and the king check from exactly those bytes
    vbench = fetch_validator_bench(cfg, receipt, reader)
    gate_ran = v is not None and (v.bench_gate_passed is not None
                                  or v.bench_gate_detail is not None)
    if gate_ran:
        if getattr(receipt, "bench_report", None):
            if vbench is not None and vbench["text_sha"] == receipt.bench_report.get("sha256"):
                bench = ({} if vbench["report"] is None
                         else validator_bench_rows(vbench["report"], reader))
        else:
            bench = fetch_bench_inputs(cfg, receipt, reader)
    # DEC-CA-0055 king check: its bench half reads the PREVIOUS settlement's
    # signed report (fetched only when a check that reached it is recorded).
    king_bench = None
    kc = getattr(receipt, "king_check", None)
    if isinstance(kc, dict) and kc.get("bench_source") == "validator":
        if vbench is not None and vbench["text_sha"] == (receipt.bench_report or {}).get("sha256"):
            king_bench = {"report": vbench["report"]}
    elif isinstance(kc, dict) and kc.get("bench_round"):
        king_bench = fetch_king_check_report(cfg, str(kc["bench_round"]), reader)
    if (isinstance(kc, dict) and isinstance(kc.get("bench"), dict)
            and kc["bench"].get("gate") is not None and king_bench is not None
            and king_bench.get("report") is not None):
        # DEC-CA-0061: the rule-v2 bench half replays from the king's and the
        # init's sha256-pinned task rows the report names
        king_bench = {**king_bench,
                      "rows": validator_bench_rows(king_bench["report"], reader) or {}}
    mkt_reports = (fetch_marketcast_reports(receipt, reader)
                   if cfg.marketcast.active(int(receipt.epoch_start_block)) else None)
    mkt_inputs = (fetch_marketcast_inputs(receipt, mkt_reports, reader,
                                          snapshot_dir=mkt_snapshot_dir)
                  if mkt_reports else None)
    results = run_tier0(receipt, cfg, client, funded_roster=roster,
                        **({"marketcast_reports": mkt_reports} if mkt_reports else {}),
                        **({"marketcast_inputs": mkt_inputs} if mkt_inputs else {}),
                        **({"bench": bench} if bench is not None else {}),
                        **({"king_bench": king_bench} if king_bench is not None else {}),
                        **({"validator_bench": vbench} if vbench is not None else {}))
    if nc_check is not None:
        results.insert(0, nc_check)
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
    if args.command == "direct-moe":
        return _direct_moe(args)
    if args.command == "marketcast":
        return _marketcast(args)
    if args.command == "bench":
        return _bench(args)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_chain_config(args.config)

    round_id = getattr(args, "round_id", None)
    if args.receipt is not None:
        text = args.receipt.read_text(encoding="utf-8")
    else:
        text = fetch_receipt_text(cfg, round_id, getattr(args, "validator", "") or "")
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


def audit_validator_bench(cfg, round_id: str, store, *, reproduce: bool = False,
                          sweep_fn=None, per_suite: int = 20,
                          tolerance: float = 0.05) -> list[CheckResult]:
    """``cascade-audit bench``: the main validator's report of ``round_id`` —
    signature against ``[netconfig] owner_hotkey``, every entry's rows
    against their signed sha256, and (``reproduce``) a re-sweep of sampled
    items from the public checkpoints (:mod:`cascade.audit.bench_reproduce`;
    report-only, never consensus). ``sweep_fn`` None with ``reproduce`` ⇒
    WARN (torch unavailable)."""
    from ..shared.bench_report import (
        load_bench_report,
        load_bench_rows,
        validator_bench_report_key,
        verify_bench_report_signature,
    )
    from ..shared.hippius import ObjectNotFound

    key = validator_bench_report_key(str(round_id))
    try:
        text = store.get_text(key)
    except ObjectNotFound:
        return [CheckResult("bench-report", WARN, f"{key} is not published")]
    except (Exception, SystemExit) as e:  # noqa: BLE001
        return [CheckResult("bench-report", WARN, f"{key} unreadable: {e}")]
    try:
        report = load_bench_report(text)
    except (ValueError, KeyError, TypeError) as e:
        return [CheckResult("bench-report", FAIL, f"{key} unparseable: {e}")]
    owner = str(cfg.netconfig.owner_hotkey or "")
    try:
        ok = bool(owner) and verify_bench_report_signature(report, owner)
    except Exception as e:  # noqa: BLE001 — no SDK: cannot verify
        return [CheckResult("bench-report", WARN, f"signature not verifiable here: {e}")]
    results = [CheckResult("bench-report", PASS if ok else FAIL,
                           f"sha256 {hashlib_sha(text)[:16]}…, {len(report.entries)} entr"
                           f"{'y' if len(report.entries) == 1 else 'ies'}, "
                           f"{len(report.infra_failed)} infra-failed; "
                           + ("signed by [netconfig] owner_hotkey" if ok
                              else f"NOT signed by owner_hotkey {owner!r}"))]
    for e in report.entries:
        try:
            load_bench_rows(store.get_text(e.rows_key), e.rows_sha256)
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            results.append(CheckResult(f"bench-rows:{e.role}", FAIL,
                                       f"{e.rows_key}: {exc}"))
    if reproduce:
        if sweep_fn is None:
            results.append(CheckResult("bench-reproduce", WARN,
                                       "torch / the benchmark sidecar unavailable here"))
        else:
            from .bench_reproduce import reproduce_report

            results += reproduce_report(report, store, sweep_fn=sweep_fn,
                                        per_suite=per_suite, tolerance=tolerance)
    return results


def hashlib_sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bench(args) -> int:
    cfg = load_chain_config(args.config)
    sweep_fn = None
    if args.reproduce:
        from .bench_reproduce import default_sweep_fn

        sweep_fn = default_sweep_fn(cfg, device=args.device, workdir=args.workdir)
    results = audit_validator_bench(cfg, str(args.round_id), _BucketReader(cfg),
                                    reproduce=args.reproduce, sweep_fn=sweep_fn,
                                    per_suite=args.sample, tolerance=args.tolerance)
    ok = not any(r.status == FAIL for r in results)
    if args.json:
        print(json.dumps({"ok": ok, "checks": [{"name": r.name, "status": r.status,
                                                "detail": r.detail} for r in results]},
                         indent=2, sort_keys=True))
    else:
        width = max(len(r.name) for r in results)
        print("\n".join(f"  [{r.status:>4}] {r.name:<{width}}  {r.detail}" for r in results))
    return 0 if ok else 1


def _direct_moe(args) -> int:
    from ..validator.direct_moe import verify_direct_receipt

    journal_config = {"bucket": args.journal_bucket, "endpoint": args.endpoint}
    if args.receipt is not None:
        receipt = json.loads(args.receipt.read_text())
    elif args.journal_bucket:
        from ..direct.journal import fetch_journal

        receipt = fetch_journal(journal_config)
    else:
        print("pass a journal file or --journal-bucket", file=sys.stderr)
        return 2
    try:
        state = verify_direct_receipt(receipt, expected_trainer=args.trainer_hotkey)
    except (KeyError, ValueError) as e:
        results = [CheckResult("replay", FAIL, str(e))]
        state = None
    else:
        results = [CheckResult("replay", PASS, "signature and every recorded decision replay"
                               if args.trainer_hotkey else "replay only (unsigned/offline)")]
    if args.reproduce and state is not None and "jobs" in state:
        from functools import partial

        from ..direct.journal import fetch_reveal
        from ..direct.storage import anonymous_objects
        from .direct_reproduce import AuditObjects, reproduce_journal

        def remote(uri):
            if not uri.startswith("hippius://"):
                raise ValueError(f"object not publicly fetchable: {uri}")
            return anonymous_objects(uri, endpoint=args.endpoint)

        sidecar = None
        if args.sidecar is not None:
            from ..direct.evaluation import PublicSidecar

            sidecar = PublicSidecar(**json.loads(args.sidecar.read_text()))
        reveal = (partial(fetch_reveal, journal_config) if args.journal_bucket
                  else lambda _digest: None)
        results += reproduce_journal(
            state, objects=AuditObjects(args.workdir / "objects", remote=remote),
            workdir=args.workdir, reveal=reveal, sidecar=sidecar, device=args.device,
            tolerance=args.tolerance)
    ok = not any(r.status == FAIL for r in results)
    if args.json:
        print(json.dumps({"ok": ok, "checks": [{"name": r.name, "status": r.status, "detail": r.detail}
                                               for r in results]}, indent=2, sort_keys=True))
    else:
        width = max(len(r.name) for r in results)
        print("\n".join(f"  [{r.status:>4}] {r.name:<{width}}  {r.detail}" for r in results))
    return 0 if ok else 1


def _marketcast(args) -> int:  # pragma: no cover — network / storage glue
    """``cascade-audit marketcast``: the full MarketCast replay of one receipt
    (+ ``--reproduce``). Exit 1 iff a check FAILs."""
    from ..shared.activation import apply_receipt_activation
    from .checks import check_marketcast

    cfg = load_chain_config(args.config)
    text = (args.receipt.read_text(encoding="utf-8") if args.receipt is not None
            else fetch_receipt_text(cfg, args.round_id, args.validator or ""))
    receipt = load_receipt(text)
    cfg = apply_receipt_activation(cfg, receipt)
    reader = _BucketReader(cfg)
    reports = fetch_marketcast_reports(receipt, reader)
    inputs = fetch_marketcast_inputs(receipt, reports, reader, snapshot_dir=args.mkt_snapshots)
    results = [check_marketcast(receipt, cfg, reports, **inputs)]
    if args.reproduce:
        from ..pool.marketcast_feeds import http_post_json
        from .marketcast_reproduce import reproduce_marketcast, timestral_forecaster_fn

        results += reproduce_marketcast(
            receipt, cfg.marketcast, reports=reports, quantiles=inputs["quantiles"],
            snapshots=inputs["snapshots"], post=http_post_json,
            forecaster_fn=timestral_forecaster_fn(cfg, workdir=args.workdir,
                                                  device=args.device),
            sample=int(args.sample), rtol=float(args.tolerance),
            bar_tol_bp=float(args.bar_tolerance_bp))
    ok = not any(r.status == FAIL for r in results)
    if args.json:
        print(json.dumps({"ok": ok, "checks": [{"name": r.name, "status": r.status,
                                                "detail": r.detail} for r in results]},
                         indent=2, sort_keys=True))
    else:
        width = max(len(r.name) for r in results)
        print("\n".join(f"  [{r.status:>4}] {r.name:<{width}}  {r.detail}" for r in results))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
