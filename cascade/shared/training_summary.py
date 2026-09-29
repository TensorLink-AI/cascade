"""Round training summary — what each duel leg actually TRAINED, public-read.

The dashboard's Training tab denominates a lineage's progress in optimiser
steps and tokens. Neither is carried anywhere a browser can read: the signed
manifest entry is a checkpoint receipt (no metrics, deliberately — the receipt
protocol stays as-is), and the per-leg numbers live only in the end-of-run
``summary``/``done`` records of the training log in the *logs* bucket, which
pods write with write-only credentials and nobody serves publicly. So the
trainer republishes the few numbers that matter as one small unsigned
telemetry document per round, ``training/round-<id>.json`` in the manifest
bucket (public-read, beside ``benchmarks/round-<id>.json``), written right
after the manifest goes out.

Per leg: ``steps``, ``tokens_seen`` (the CONTRACT-denominated budget count the
run stopped on — series-points under DEC-CA-0042, so ``B×L`` per batch),
``channel_tokens`` (every value entry the model processed, ``B×C×L`` — the
number that grows ``C×`` on a multivariate corpus), ``tokens_frac`` /
``deadline_hit`` (a wall-stopped leg trained less than the contract),
``train_seconds``, ``gpu_name``, and the shadow channel telemetry's
``max_channels_seen`` when the corpus was multichannel. Legs trained in the
orchestrator's own process fold in from the in-memory summary row; legs
trained on a pod are read back from the log the worker flushed at the end of
its run. A leg with neither publishes its identity only (``measured: false``)
so the page can fall back to the contracted budget and say so.

Telemetry only (DEC-CA-0010 shape): unsigned, presentational, no consumer in
any scoring, promotion or audit path; a missing or partial document costs
nothing but a dashed tile.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Iterable

from .hippius import StorageError

TRAINING_SUMMARY_VERSION = 1
TRAINING_SUMMARY_PREFIX = "training/"

# Run-summary keys copied verbatim from a leg's ``summary``/``done`` record
# (``TrainResult.metrics`` + the trainer's summary row). Anything not listed
# stays in the log — the document is the dashboard's read, not a log mirror.
LEG_FIELDS = (
    "steps", "tokens_seen", "channel_tokens", "tokens_frac", "deadline_hit",
    "train_seconds", "gpu_name", "n_series", "total_points",
    "lr_schedule", "optim_state_resumed",
)
CONTRACT_FIELDS = (
    "batch_size", "context_length", "budget_denomination", "batch_denomination",
    "target_train_hours", "ref_throughput_tokens_per_s", "max_train_seconds",
)


def training_summary_key(round_id: str) -> str:
    return f"{TRAINING_SUMMARY_PREFIX}round-{round_id}.json"


def leg_log_role(role: str, size: str) -> str:
    """The training-log role a FINAL leg logs under (``<role>-<size>``) — the
    naming ``TrainerRunner._train_one`` derives for non-heat runs, mirrored
    here so the read-back and the write agree."""
    return f"{role}-{size}"


def parse_log_records(text: str) -> list[dict]:
    """Tolerant JSONL parse: one dict per well-formed line, junk lines skipped."""
    out: list[dict] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def leg_from_records(records: Iterable[dict]) -> dict | None:
    """Fold a leg's records into one telemetry dict, or ``None`` when no run
    summary is among them (a leg that died before its ``done`` row).

    Accepts the training log's ``summary`` / ``done`` events and the trainer's
    in-process summary row alike (any record carrying ``steps`` +
    ``tokens_seen``). Later records win, so a ``summary`` row (which repeats
    the ``done`` metrics plus the stream counts) completes an earlier ``done``.
    """
    out: dict = {}
    seen = False
    for r in records:
        if not isinstance(r, dict):
            continue
        ev = r.get("event")
        if ev not in ("summary", "done") and not ("steps" in r and "tokens_seen" in r):
            continue
        seen = True
        for k in LEG_FIELDS:
            if k in r and r[k] is not None:
                out[k] = r[k]
        ct = r.get("channel_telemetry")
        if isinstance(ct, dict):
            for k in ("max_channels_seen", "n_multichannel_series"):
                if ct.get(k) is not None:
                    out[k] = ct[k]
    return out if seen else None


def contract_block(contract: object) -> dict:
    """The contract terms the page needs to price a leg, read leniently off a
    ``TrainingContractConfig`` (a fake in tests may carry a subset)."""
    out: dict = {}
    for k in CONTRACT_FIELDS:
        v = getattr(contract, k, None)
        if v is not None:
            out[k] = v
    tb = getattr(contract, "train_tokens", None)
    if tb is not None:
        with contextlib.suppress(TypeError, ValueError):
            out["token_budget"] = int(tb)
    return out


def collect_round_legs(
    entries: Iterable[object],
    primary: str,
    *,
    cached: dict | None = None,
    read_log: Callable[[str], str | None] | None = None,
) -> list[dict]:
    """One telemetry row per manifest entry, in manifest order.

    ``cached`` maps a leg's log role to the trainer's in-process summary row
    (legs trained locally); ``read_log`` fetches a log role's JSONL text from
    the logs bucket (legs trained on a pod), returning ``None`` when absent.
    A leg found in neither is still listed — identity + ``measured: false`` —
    so the page distinguishes "unmeasured" from "not in the manifest".
    """
    cached = cached or {}
    legs: list[dict] = []
    for e in entries:
        role = str(getattr(e, "role", "") or "")
        size = str(getattr(e, "size", "") or "") or primary
        base = {
            "role": role, "size": size,
            "miner_hotkey": str(getattr(e, "miner_hotkey", "") or ""),
            "miner_uid": getattr(e, "miner_uid", None),
            "trained_pointer": str(getattr(e, "trained_pointer", "") or ""),
        }
        lr = leg_log_role(role, size)
        leg, source = None, None
        rec = cached.get(lr)
        if rec is not None:
            leg, source = leg_from_records([rec]), "trainer"
        if leg is None and read_log is not None:
            try:
                text = read_log(lr)
            except Exception:  # noqa: BLE001 — a log read must never fail the publish
                text = None
            if text:
                leg, source = leg_from_records(parse_log_records(text)), "log"
        row = dict(base)
        if leg:
            row.update(leg)
            row["measured"] = True
            row["source"] = source
        else:
            row["measured"] = False
        legs.append(row)
    return legs


def build_training_summary(round_id: str, created_block: int, contract: dict,
                           legs: list[dict], *, warm_start_ckpt: str = "",
                           warm_start_size: str = "") -> dict:
    """``warm_start_ckpt`` is the init every leg of the round trained from
    (the manifest's, "" = random init): the link the dashboard follows from a
    checkpoint back to the promoted member it continued, so a lineage's total
    training sums leg by leg instead of being assumed."""
    return {
        "kind": "training_summary",
        "summary_version": TRAINING_SUMMARY_VERSION,
        "telemetry_only": True,
        "round_id": str(round_id),
        "created_block": int(created_block),
        "warm_start_ckpt": str(warm_start_ckpt or ""),
        "warm_start_size": str(warm_start_size or ""),
        "contract": dict(contract),
        "legs": list(legs),
    }


def dump_training_summary(doc: dict) -> str:
    return json.dumps(doc, indent=2, sort_keys=True)


def publish_training_summary(store: object, text: str, round_id: str) -> str:
    """Write the round's summary to the manifest-bucket store, PUBLIC-READ (the
    dashboard fetches it anonymously); a backend without canned ACLs publishes
    private rather than not at all — the bench-report convention."""
    key = training_summary_key(round_id)
    try:
        store.put_text(key, text, content_type="application/json", acl="public-read")
    except StorageError:
        store.put_text(key, text, content_type="application/json")
    return key
