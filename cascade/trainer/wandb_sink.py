"""Optional Weights & Biases logging for the reference trainer.

cascade already streams per-step training metrics (loss, lr, throughput) to
Hippius S3 as a JSONL blob flushed at the end of each run — durable, but only
visible *after* the ~3h run finishes. This module mirrors the **same** records
into a live wandb run so miners can watch their generator train *as it occurs*.

It is **observability only** — wandb numbers never feed scoring or weights, and
the held-out eval-pool scores are deliberately *not* sent here (that stays the
validator's private, rotating concern; see ``OPEN_QUESTIONS.md`` #6). What lands
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


def wandb_run_id(round_id: str, role: str, hotkey: str) -> str:
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
    and ``hotkey`` the competitor, so the triple uniquely identifies a logical run.
    A short digest of the raw key is appended so two runs can never collide even
    if sanitising or truncation would otherwise collapse their slugs.
    """
    raw = f"{round_id}-{role}-{hotkey}"
    slug = _WANDB_ID_UNSAFE.sub("-", raw)[:96]
    tag = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{tag}"


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
) -> WandbSink | None:
    """Start a wandb run for one training, or return ``None`` if unavailable.

    Returns ``None`` (a no-op for the caller) when wandb logging is disabled in
    config, the ``wandb`` package is not installed, or — in ``online`` mode — no
    ``WANDB_API_KEY`` is set. Any wandb error during init is caught and degrades
    to ``None`` so a misconfigured dashboard never blocks a round.
    """
    if not getattr(wcfg, "enabled", False):
        return None

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

    entity = (getattr(wcfg, "entity", "") or "").strip() or None
    run_config = {
        "round_id": round_id, "role": role, "size": size,
        "miner_hotkey": hotkey, "miner_uid": uid, **(config or {}),
    }
    try:
        run = wandb.init(
            project=(getattr(wcfg, "project", "cascade") or "cascade"),
            entity=entity,
            # Stable id so a retried/re-run training RESUMES its one run instead
            # of minting a duplicate on every service-loop retry (see
            # wandb_run_id). ``resume="allow"`` = resume if it exists, else create.
            id=wandb_run_id(round_id, role, hotkey),
            resume="allow",
            name=f"round-{round_id}-{role}",
            group=str(round_id),
            job_type=role,
            tags=[f"round:{round_id}", f"hotkey:{hotkey}", f"uid:{uid}",
                  f"size:{size}", f"role:{role}"],
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
        for k in ("lr", "throughput_tokens_per_s", "tokens", "tokens_frac", "data_wait_frac"):
            run.define_metric(k, step_metric="step")
    except Exception as e:  # noqa: BLE001 — axis setup must never abort a round
        log.debug("wandb define_metric failed (continuing): %s", e)
    return WandbSink(run)
