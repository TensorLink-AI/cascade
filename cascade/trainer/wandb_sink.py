"""Optional Weights & Biases logging for the reference trainer.

cascade already streams per-step training metrics (loss, lr, throughput) to
Hippius S3 as a JSONL blob flushed at the end of each run — durable, but only
visible *after* the ~3h run finishes. This module mirrors the **same** records
into a live wandb run so miners can watch their generator train *as it occurs*.

It is **observability only** — wandb numbers never feed scoring or weights, and
the held-out eval-pool scores are deliberately *not* sent here (that stays the
validator's private, rotating concern). What lands
in wandb is exactly what already lands in the S3 log: training-corpus metrics.

The integration is best-effort and fully decoupled:

* the ``wandb`` package is a lazy, optional import (the ``[wandb]`` extra); if it
  is missing or ``WANDB_API_KEY`` is unset, logging silently no-ops,
* every wandb call is wrapped so a logging failure can **never** abort a training
  run — the same contract the S3 :class:`~cascade.shared.hippius.LogSink` honours,
* one run is opened per ``(round, competitor, size, stage)`` and tagged with the
  miner hotkey/uid, so a miner filters the (public) wandb project to *their* runs.

Point ``[wandb] project``/``entity`` at a **public** wandb project for miners to
watch; credentials come from the environment (``WANDB_API_KEY``), never
``chain.toml``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass

log = logging.getLogger("cascade.trainer.wandb")

# wandb.log rejects non-scalar values for charting; we forward only JSON scalars
# (the trainer's records are flat dicts of numbers plus a string ``event`` tag,
# which wandb stores fine as a categorical column).
_WANDB_API_KEY_ENVS = ("WANDB_API_KEY",)

# wandb ids are URL fragments: keep them to ``[A-Za-z0-9_-]``.
_WANDB_ID_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


# The competition lanes (DEC-CA-0050 / DEC-CA-0056): A = the king's pretraining
# lineage, B = domain-expert legs, C = king fine-tune legs. W&B-only labels:
# the receipt role (``remote.receipt_role``) is unaffected.
LANES = ("A", "B", "C")
# The lane-C recipe fields mirrored into the run config (DEC-CA-0056).
RECIPE_FIELDS = ("lr_scale", "steps_fraction", "warmup_fraction", "schedule")


def lane_of(*, finetune: bool = False, expert: bool = False) -> str:
    """The lane a training leg belongs to: C (fine-tune) > B (expert) > A."""
    return "C" if finetune else ("B" if expert else "A")


def lane_run_fields(lane: str = "A", target: str = "", recipe: object = None) -> dict:
    """The run-config fields for one leg's lane: ``{"lane", "target"}`` plus,
    for a lane-C leg, the recipe fields (``lr_scale``, ``steps_fraction``,
    ``warmup_fraction``, ``schedule``) read off ``recipe`` (a ``LegRecipe``
    or a dict). Unknown lanes read as A."""
    lane = str(lane or "A").upper()
    if lane not in LANES:
        lane = "A"
    out: dict = {"lane": lane, "target": str(target or "")}
    if lane == "C" and recipe is not None:
        get = recipe.get if isinstance(recipe, dict) else (lambda k: getattr(recipe, k, None))
        for k in RECIPE_FIELDS:
            v = get(k)
            if v is not None:
                out[f"recipe_{k}"] = v
    return out


def wandb_run_id(round_id: str, role: str, hotkey: str, *, lane: str = "A",
                 target: str = "") -> str:
    """Deterministic, stable wandb run id for one ``(round, competitor, size/stage)``.

    wandb keys a run by its ``id``, **not** its display ``name``. Without a stable
    id every re-entry of a training — a service-loop retry of a round that failed
    downstream (a king/challenger fault, a publish error, flaky infra), or any
    re-run of the same round — mints a *new* run that merely shares the name, so a
    repeatedly-retried round floods the project with dozens of duplicate,
    near-empty runs (the ``round-N-king…``/``round-N-challenger…`` pile-up). A
    deterministic id + ``resume="allow"`` makes a re-run **resume** the one run
    for that ``(round, competitor, size/stage)`` — the one-run-per-competitor
    invariant this module's docstring promises.

    ``role`` already carries the size/stage tag (``king-toto2-4m``, ``heat-<hk>``)
    and ``hotkey`` the competitor, so the triple uniquely identifies a logical run
    within lane A. A lane-B / lane-C leg logs under the same ``challenger-<size>``
    role, so its ``lane`` and ``target`` join the key: the same hotkey's lane-A
    leg and expert / fine-tune legs in one round never share a run. A lane-A key
    is unchanged from the pre-lane id (an in-flight run still resumes).
    A short digest of the raw key is appended so two runs can never collide even
    if sanitising or truncation would otherwise collapse their slugs.
    """
    raw = f"{round_id}-{role}-{hotkey}"
    lane = str(lane or "A").upper()
    if lane != "A":
        raw += f"-lane{lane}-{target or ''}"
    slug = _WANDB_ID_UNSAFE.sub("-", raw)[:96]
    tag = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{tag}"


def _wandb_module(wcfg: object):
    """The ``wandb`` module, or ``None`` (with a warning) when ``online`` mode
    has no ``WANDB_API_KEY`` or the package is missing. The caller has
    already checked its own enable switch."""
    mode = (getattr(wcfg, "mode", "online") or "online").strip()
    if mode == "online" and not any(os.environ.get(k) for k in _WANDB_API_KEY_ENVS):
        log.warning(
            "[wandb] enabled but no %s set; skipping wandb logging "
            "(set the key or use mode=offline)", _WANDB_API_KEY_ENVS[0],
        )
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        log.warning("[wandb] enabled but the wandb package is not installed "
                    "(install the [wandb] extra); skipping wandb logging")
        return None
    return wandb


@dataclass
class WandbSink:
    """Thin wrapper over a live ``wandb`` run exposing the trainer's logger API.

    :meth:`emit` matches :meth:`cascade.shared.hippius.LogSink.emit` so the two
    sinks fan out from the same per-step records. Every call swallows its own
    exceptions: a flaky wandb backend degrades to "no live dashboard", never a
    failed training round.
    """

    run: object

    def emit(self, record: dict) -> None:
        try:
            # No explicit step: wandb auto-increments, and the record already
            # carries ``step``/``tokens`` as metrics so either can be the x-axis
            # in the UI. Passing an explicit step here would have to be monotonic
            # and would clash with the summary/done events that carry none.
            self.run.log(dict(record))
        except Exception as e:  # noqa: BLE001 — logging must never abort training
            log.debug("wandb log failed (continuing): %s", e)

    def finish(self) -> None:
        try:
            self.run.finish()
        except Exception as e:  # noqa: BLE001
            log.debug("wandb finish failed (continuing): %s", e)


def open_wandb_run(
    wcfg: object,
    *,
    round_id: str,
    role: str,
    hotkey: str,
    uid: int,
    size: str,
    config: dict | None = None,
    lane: str = "A",
    target: str = "",
    recipe: object = None,
) -> WandbSink | None:
    """Start a wandb run for one training, or return ``None`` if unavailable.

    Returns ``None`` (a no-op for the caller) when wandb logging is disabled in
    config, the ``wandb`` package is not installed, or — in ``online`` mode — no
    ``WANDB_API_KEY`` is set. Any wandb error during init is caught and degrades
    to ``None`` so a misconfigured dashboard never blocks a round.

    ``lane`` / ``target`` / ``recipe`` label the leg's competition lane
    (:func:`lane_run_fields`): every run carries a ``lane:A|B|C`` tag, a
    ``target:<t>`` tag for a lane-B / lane-C leg (a DEC-CA-0059 multi-target
    leg's targets comma-joined), and the lane-C recipe in its config.
    """
    if not getattr(wcfg, "enabled", False):
        return None
    wandb = _wandb_module(wcfg)
    if wandb is None:
        return None

    mode = (getattr(wcfg, "mode", "online") or "online").strip()
    entity = (getattr(wcfg, "entity", "") or "").strip() or None
    lane_fields = lane_run_fields(lane, target, recipe)
    lane, target = lane_fields["lane"], lane_fields["target"]
    run_config = {
        "round_id": round_id, "role": role, "size": size,
        "miner_hotkey": hotkey, "miner_uid": uid, **lane_fields, **(config or {}),
    }
    tags = [f"round:{round_id}", f"hotkey:{hotkey}", f"uid:{uid}",
            f"size:{size}", f"role:{role}", f"lane:{lane}"]
    if target:
        tags.append(f"target:{target}")
    if lane == "C" and str(target).startswith("mkt:"):
        tags.append("lane:C/mkt")          # DEC-CA-0064 MarketCast legs
    name = f"round-{round_id}-{role}"
    if lane != "A":
        name += f"-lane{lane}" + (f"-{target}" if target else "")
    try:
        run = wandb.init(
            project=(getattr(wcfg, "project", "cascade") or "cascade"),
            entity=entity,
            # Stable id so a retried/re-run training RESUMES its one run instead
            # of minting a duplicate on every service-loop retry (see
            # wandb_run_id). ``resume="allow"`` = resume if it exists, else create.
            id=wandb_run_id(round_id, role, hotkey, lane=lane, target=target),
            resume="allow",
            name=name,
            group=str(round_id),
            job_type=role,
            # W&B caps a tag at 64 characters
            tags=[t[:64] for t in tags],
            config=run_config,
            mode=mode,
            reinit=True,
        )
    except Exception as e:  # noqa: BLE001 — never let wandb init abort a round
        log.warning("[wandb] init failed (continuing without wandb): %s", e)
        return None

    # Pin the per-step x-axis: the trainer's records carry a numeric ``step``, so
    # plotting the per-step series against it gives a readable loss-vs-step curve
    # out of the box — instead of wandb's internal ``_step`` counter, which the
    # default dashboard uses when no step_metric is declared. Best-effort: axis
    # setup is cosmetic, so a define_metric failure degrades to the default axes,
    # never a failed round (same swallow-exceptions contract as init/log).
    try:
        run.define_metric("step")
        run.define_metric("loss", step_metric="step", summary="min")
        for k in ("lr", "throughput_tokens_per_s", "steps_per_s", "tokens",
                  "tokens_frac", "data_wait_frac"):
            run.define_metric(k, step_metric="step")
    except Exception as e:  # noqa: BLE001 — axis setup must never abort a round
        log.debug("wandb define_metric failed (continuing): %s", e)
    return WandbSink(run)


def _scalars(record: dict) -> dict:
    """The JSON-scalar fields of ``record`` (wandb charts scalars; nested
    values ride the run config instead)."""
    return {k: v for k, v in record.items()
            if v is None or isinstance(v, (bool, int, float, str))}


def log_wandb_event(wcfg: object, *, kind: str, round_id: str, record: dict,
                    gate: str = "enabled", lane: str = "A", key: str = "") -> bool:
    """Log one small event (a king check, a promotion, a validator settlement
    summary) as its own short W&B run: ``job_type = kind``, tagged
    ``event:<kind>``, ``round:<id>`` and ``lane:<lane>``, ``record`` in the run
    config and its scalars logged once. The run id is deterministic in
    ``(kind, round_id, key)``, so a re-logged event resumes its run instead of
    duplicating it.

    ``gate`` names the ``[wandb]`` switch that enables this event
    (``enabled`` for the trainer, ``validator_summary`` for validators).
    Observability only: returns whether the event was logged and NEVER
    raises — nothing a verdict, a weight vector or a promotion depends on
    may wait on, or fail because of, W&B."""
    try:
        if not getattr(wcfg, gate, False):
            return False
        wandb = _wandb_module(wcfg)
        if wandb is None:
            return False
        mode = (getattr(wcfg, "mode", "online") or "online").strip()
        entity = (getattr(wcfg, "entity", "") or "").strip() or None
        rid = str(round_id)
        raw = f"event-{kind}-{rid}-{key}"
        run_id = (_WANDB_ID_UNSAFE.sub("-", raw)[:96] + "-"
                  + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8])
        run = wandb.init(
            project=(getattr(wcfg, "project", "cascade") or "cascade"),
            entity=entity, id=run_id, resume="allow",
            name=f"{kind}-{rid}" + (f"-{key}" if key else ""), group=rid, job_type=kind,
            tags=[f"event:{kind}", f"round:{rid}", f"lane:{str(lane or 'A').upper()}"],
            config={"event": kind, "round_id": rid, **record},
            mode=mode, reinit=True,
        )
        sink = WandbSink(run)
        sink.emit({"event": kind, **_scalars(record)})
        sink.finish()
        return True
    except Exception as e:  # noqa: BLE001 — an event log must never disturb its caller
        log.debug("wandb %s event failed (continuing): %s", kind, e)
        return False


def settlement_summary(row: dict) -> dict:
    """The validator-side W&B summary of one settlement from its
    ``receipts/index.json`` row (:func:`cascade.shared.receipt.summarize_receipt`):
    the lane-A verdict, the lane-B / lane-C verdict counts and paid slots,
    the emission split, the lane-A ladder and the king check. Pure."""
    def _n(rows, pred):
        return sum(1 for r in rows or () if isinstance(r, dict) and pred(r))

    lb, lc = row.get("lane_b_verdicts") or [], row.get("lane_c_verdicts") or []
    lb_pay = (row.get("lane_b_pay") or {}).get("wins") or []
    lc_pay = (row.get("lane_c_pay") or {}).get("wins") or []
    split = row.get("emission_split") or {}
    ladder = row.get("ladder") or {}
    kc = row.get("king_check") or {}
    out = {
        "status": row.get("status"),
        "epoch_start_block": row.get("epoch_start_block"),
        "lane_a_lcb": row.get("lcb"), "lane_a_margin": row.get("margin"),
        "lane_a_challenger_wins": row.get("challenger_wins_round"),
        "lane_a_dethroned": row.get("dethroned"),
        "lane_a_inconclusive": row.get("inconclusive"),
        "king_hotkey": row.get("post_round_king_hotkey") or row.get("king_hotkey"),
        "king_uid": row.get("post_round_king_uid"),
        "lane_b_judged": len(lb), "lane_b_passed": _n(lb, lambda r: r.get("passed")),
        "lane_b_paid_slots": len(lb_pay),
        "lane_c_judged": len(lc), "lane_c_passed": _n(lc, lambda r: r.get("passed")),
        "lane_c_paid_slots": len(lc_pay),
        # each paid win's gain-tiered full-pay period (owner 2026-10-20)
        "lane_b_payouts": {str(w.get("domain")): {"uid": w.get("uid"),
                                                   "multiplier": w.get("multiplier"),
                                                   "tier": w.get("tier"),
                                                   "full_eras": w.get("full_eras")}
                           for w in lb_pay if isinstance(w, dict)},
        "lane_c_payouts": {str(w.get("target")): {"uid": w.get("uid"),
                                                   "multiplier": w.get("multiplier"),
                                                   "tier": w.get("tier"),
                                                   "full_eras": w.get("full_eras")}
                           for w in lc_pay if isinstance(w, dict)},
    }
    for k in ("lane_a", "lane_b", "lane_c", "burn"):
        if k in split:
            out[f"emission_{k}"] = split[k]
    if ladder:
        out["ladder_king_share"] = ladder.get("king_share")
        out["ladder_event"] = ladder.get("event")
        if ladder.get("crown_units"):
            # DEC-CA-0055 margin-tiered start share at a crowning
            out["ladder_crown_lcb"] = ladder.get("crown_lcb")
            out["ladder_crown_units"] = ladder.get("crown_units")
            out["ladder_crown_tier"] = ladder.get("crown_tier")
            out["ladder_crown_start_bp"] = ladder.get("crown_start_bp")
        out["ladder_slashed"] = ladder.get("slashed")
        out["ladder_slash_reason"] = ladder.get("slash_reason")
    if kc:
        out["king_check_result"] = kc.get("result")
        out["king_check_lcb"] = kc.get("lcb")
        out["king_check_margin"] = kc.get("margin")
    return out


def log_validator_settlement(wcfg: object, row: dict, *, validator_hotkey: str = "") -> bool:
    """OPTIONAL validator-side W&B summary of one settlement (``[wandb]
    validator_summary``, off by default). Called after the receipt is
    published, i.e. after the weights are set: it can never shape a verdict,
    and it never raises (:func:`log_wandb_event`)."""
    try:
        rec = settlement_summary(row)
    except Exception as e:  # noqa: BLE001
        log.debug("validator settlement summary unavailable: %s", e)
        return False
    if validator_hotkey:
        rec["validator_hotkey"] = validator_hotkey
    return log_wandb_event(wcfg, kind="settlement", round_id=str(row.get("round_id", "")),
                           record=rec, gate="validator_summary", key=validator_hotkey[:16])
