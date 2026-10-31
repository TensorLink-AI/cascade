"""Miner-cost protections for funded legs (2026-10-31 incident).

That day four v2 challenger legs trained on their miners' own Lium keys at
0.13 steps/s against a wall budget that assumed 2.86 (the ``funded_sku_wall_
seconds`` "8xH200" entry was never measured on the pinned image / v2 shape):
every leg would have been wall-capped at ~10 % of its steps, and each pod
billed its miner ~3 h before anyone noticed. Five guards, each a pure function
of the runner so the TrainerRunner mixins and their unit fakes share them:

* :class:`ProgressWatchdog` — projects a running leg's steps/s to completion
  and aborts a leg that cannot finish inside its budget ("THROUGHPUT ABORT");
* :func:`measured_wall_refusal` — refuses a rent whose SKU walls were measured
  on another image than the one pinned;
* :func:`doomed_leg_reason` — refuses a rent whose projected end (now + the
  SKU's wall) is past the settlement the leg must land in;
* :func:`provider_ttl_seconds` — the provider-side auto-terminate every rent
  carries, so a dead trainer cannot leave a pod billing;
* :func:`payer_key_listing_error` / :func:`check_payer_key` — the vaulted key
  must still list pods (the call every teardown makes), else
  "PAYER KEY CANNOT STOP POD".

Every refusal is an OPERATOR-side verdict: nothing here ever burns a miner's
attempt.
"""

from __future__ import annotations

import logging
import math
import time

log = logging.getLogger("cascade.trainer")

THROUGHPUT_ABORT = "THROUGHPUT ABORT"
PAYER_KEY_CANNOT_STOP = "PAYER KEY CANNOT STOP POD"


# ── 1. progress watchdog ─────────────────────────────────────────────────────


class ProgressWatchdog:
    """``on_progress`` callback for one detached leg (see
    :func:`cascade.trainer.remote.run_detached`).

    Called on every poll that finds the worker RUNNING, with the worker's
    progress record (``{"step": int, "tokens": int, ...}`` from
    ``CASCADE_PROGRESS_FILE``) or ``None`` when the pod has none (an image
    that predates the file — the watchdog is then inert). Returns an abort
    reason (the caller kills the worker) or ``None``.

    The rate is measured on the ORCHESTRATOR's clock from the first observed
    step, so pod clock skew cannot move it. Judged once ``grace_seconds`` of
    training have been observed; a leg is aborted when its projected
    training wall (``training_steps / rate``) exceeds ``tolerance`` × the
    contract's ``max_train_seconds`` (the worker would stop wall-capped with
    a fraction of its steps) or its projected remaining time exceeds
    ``tolerance`` × the time left before ``end_wall`` (its target settlement).
    No progress at all over the grace window is rate 0 — aborted too.

    ``key_check`` (optional, no args) runs every ``key_check_seconds`` on the
    same poll — the periodic teardown-key check; it never aborts the leg (an
    abort needs the very key that failed to stop the pod).
    """

    def __init__(self, *, label: str, hotkey: str, pod_name: str, training_steps: int,
                 max_train_seconds: float, end_wall: float | None, grace_seconds: float,
                 tolerance: float = 1.5, clock=time.time, key_check=None,
                 key_check_seconds: float = 0.0) -> None:
        self.label, self.hotkey, self.pod_name = label, hotkey, pod_name
        self.training_steps = int(training_steps or 0)
        self.max_train_seconds = float(max_train_seconds or 0.0)
        self.end_wall = None if end_wall is None else float(end_wall)
        self.grace = float(grace_seconds or 0.0)
        self.tolerance = max(1.0, float(tolerance or 1.0))
        self.clock = clock
        self.key_check = key_check
        self.key_check_seconds = float(key_check_seconds or 0.0)
        self._first: tuple[float, int] | None = None
        self._last_key = clock()
        self.verdict: str | None = None

    @staticmethod
    def _step(progress) -> int | None:
        if not isinstance(progress, dict):
            return None
        try:
            step = int(progress.get("step"))
        except (TypeError, ValueError):
            return None
        return step if step >= 0 else None

    def __call__(self, progress) -> str | None:
        now = self.clock()
        if (self.key_check is not None and self.key_check_seconds > 0
                and now - self._last_key >= self.key_check_seconds):
            self._last_key = now
            try:
                self.key_check()
            except Exception as e:  # noqa: BLE001 — a check must never sink the leg
                log.warning("%s: payer key check errored (%s)", self.label, e)
        if self.verdict is not None:
            return self.verdict
        if self.grace <= 0 or self.training_steps <= 0:
            return None
        step = self._step(progress)
        if step is None:
            return None
        if self._first is None:
            self._first = (now, step)
            return None
        t0, s0 = self._first
        elapsed = now - t0
        if elapsed < self.grace:
            return None
        rate = max(0.0, (step - s0) / elapsed) if elapsed > 0 else 0.0
        remaining = max(0, self.training_steps - step)
        total = self.training_steps / rate if rate > 0 else math.inf
        left = remaining / rate if rate > 0 else math.inf
        why = []
        if self.max_train_seconds > 0 and total > self.tolerance * self.max_train_seconds:
            why.append(f"projected training wall {_hours(total)} > {self.tolerance:g} x "
                       f"max_train_seconds {_hours(self.max_train_seconds)}")
        if self.end_wall is not None:
            budget = max(0.0, self.end_wall - now)
            if left > self.tolerance * budget:
                why.append(f"projected {_hours(left)} to finish > {self.tolerance:g} x "
                           f"{_hours(budget)} left to the target settlement")
        if not why:
            return None
        self.verdict = (f"{THROUGHPUT_ABORT}: hotkey={self.hotkey} pod={self.pod_name} "
                        f"rate={rate:.3f} steps/s (step {step}/{self.training_steps} after "
                        f"{elapsed:.0f}s observed) projected={_hours(total)} total — "
                        + "; ".join(why))
        log.error("%s — %s: aborting the leg early, tearing the pod down; requeued as an "
                  "operator fault (no attempt burned)", self.verdict, self.label)
        return self.verdict


def _hours(seconds: float) -> str:
    return "inf h" if not math.isfinite(seconds) else f"{seconds / 3600.0:.2f} h"


def leg_watchdog(runner, *, gen, pod, contract, key_check=None) -> ProgressWatchdog | None:
    """The watchdog for one funded leg, or ``None`` with every guard off."""
    rnd = runner.cfg.round
    grace = float(getattr(rnd, "progress_watchdog_grace_seconds", 0) or 0)
    key_every = float(getattr(rnd, "payer_key_check_seconds", 0) or 0)
    if grace <= 0 and not (key_every > 0 and key_check is not None):
        return None
    local = runner.__dict__.get("_leg_local_obj")
    end = getattr(local, "end_wall", None)
    return ProgressWatchdog(
        label=f"funded leg {gen.hotkey[:12]}", hotkey=gen.hotkey,
        pod_name=str(getattr(pod, "instance_id", "?")),
        training_steps=int(getattr(contract, "training_steps", 0) or 0),
        max_train_seconds=float(getattr(contract, "max_train_seconds", 0) or 0),
        end_wall=None if end is None else float(end), grace_seconds=grace,
        tolerance=float(getattr(rnd, "progress_watchdog_tolerance", 1.5) or 1.5),
        key_check=key_check, key_check_seconds=key_every)


# ── 2. measured-wall gate ────────────────────────────────────────────────────


def measured_wall_refusal(round_cfg) -> str:
    """Why no funded leg / king pod may rent now, or ``""``.

    ``[round] funded_sku_wall_measured_image`` names the image the SKU wall
    table was measured on; every latest-safe-start and per-leg cost cap is
    built on those walls, so renting on another image is renting on a guess
    (2026-10-31: 20x off). Empty knob = gate off."""
    from ..shared.config import image_digest

    measured = str(getattr(round_cfg, "funded_sku_wall_measured_image", "") or "")
    if not measured:
        return ""
    pinned = image_digest(str(getattr(round_cfg, "funded_pod_image", "") or ""))
    if pinned == measured:
        return ""
    return (f"funded_sku_wall_seconds were measured on image {measured[:19]}… but "
            f"funded_pod_image is {(pinned or 'unpinned')[:19]}… — no measured wall for "
            "the pinned image; rent refused (operator: measure the walls on this image "
            "and set [round] funded_sku_wall_measured_image)")


# ── 3. doomed legs ───────────────────────────────────────────────────────────


def doomed_leg_reason(runner, skus) -> str:
    """Why a rent of any of ``skus`` NOW cannot land its leg before the
    settlement it must land in, or ``""``.

    The leg's target settlement is the thread-local ``end_wall`` the rolling
    scheduler set; a type fits when now + its wall + the publish margin is
    before it (``_funded_rent_wait_deadline_for``). The rent loop's
    ``_skus_fitting_now(skus) or skus`` falls back to EVERY type when none
    fits — the leg then trains toward a settlement it can only miss (and,
    when a block-gated contract switch sits at the next era boundary, under
    a contract that is gone by the time it finishes). Unknown target ⇒ ""."""
    local = runner.__dict__.get("_leg_local_obj")
    end = getattr(local, "end_wall", None)
    deadline_for = getattr(runner, "_funded_rent_wait_deadline_for", None)
    if end is None or deadline_for is None or not skus:
        return ""
    now_fn = getattr(runner, "_rent_wait_now", None) or time.time
    now = float(now_fn())
    try:
        fits = [s for s in skus if now < float(deadline_for(s))]
    except Exception:  # noqa: BLE001 — a broken estimate must never block a leg
        return ""
    if fits:
        return ""
    try:
        wall = min(float(runner._leg_wall_seconds(s)) for s in skus)
    except Exception:  # noqa: BLE001
        wall = float("nan")
    margin = float(getattr(runner, "FUNDED_PUBLISH_MARGIN_SECONDS", 0.0) or 0.0)
    return (f"doomed leg: projected end now + {wall:.0f}s wall + {margin:.0f}s margin is "
            f"{(now + wall + margin - float(end)) / 3600.0:+.2f} h past its target "
            f"settlement on every type ({', '.join(skus)}) — not dispatched, requeued "
            "unburned")


# ── 4. provider-side TTL ─────────────────────────────────────────────────────


def provider_ttl_seconds(runner, *, kind: str = "leg", contract=None) -> float:
    """Auto-terminate (seconds from create) for a rent of ``kind``.

    ``leg`` (a payer pod or an operator-rented leg pod): the contract's
    ``max_train_seconds`` (the worker's hard stop) + the boot ceiling + the
    bench margin + the completion bench hold when the pod is kept for it +
    ``[round] provider_ttl_margin_seconds``. 0 = no TTL (margin 0). Never
    shorter than the pod can legitimately live — it is a backstop against a
    dead trainer, not a leg budget."""
    rnd = runner.cfg.round
    margin = float(getattr(rnd, "provider_ttl_margin_seconds", 0) or 0)
    if margin <= 0:
        return 0.0
    if contract is None:
        contract = getattr(runner.__dict__.get("_leg_local_obj"), "funded_contract", None)
    try:
        cap = (float(contract.max_train_seconds) if contract is not None else
               float(max(int(c.max_train_seconds) for c in runner.cfg.throne_contracts())))
    except Exception:  # noqa: BLE001 — offline fakes: no contract, no TTL
        return 0.0
    life = cap + float(getattr(rnd, "funded_ready_timeout_seconds", 900.0) or 0.0)
    life += float(getattr(rnd, "funded_bench_margin_seconds", 0) or 0)
    keep = getattr(runner, "_keep_funded_pod_for_bench", None)
    try:
        if kind == "leg" and keep is not None and keep():
            life += 3600.0 * float(getattr(runner.cfg.eval, "bench_hold_max_hours", 2.0) or 0.0)
    except Exception:  # noqa: BLE001
        pass
    return life + margin


def generation_ttl_seconds(chain_cfg, window_seconds: float) -> float:
    """Auto-terminate for a CPU generation VM (``cgen-*``): its window + an
    hour for boot/bootstrap/upload + the margin; 0 = no TTL."""
    margin = float(getattr(getattr(chain_cfg, "round", None), "provider_ttl_margin_seconds", 0)
                   or 0)
    if margin <= 0 or window_seconds <= 0:
        return 0.0
    return float(window_seconds) + 3600.0 + margin


# ── 5. teardown-key check ────────────────────────────────────────────────────


def _default_lister(provider: str, api_key: str, prefix: str) -> list:
    from ..provision.funded import provider_for_key

    return list(provider_for_key(provider, api_key).list_tagged(prefix))


def payer_key_listing_error(raw_key: str, prefix: str, *, lister=None) -> str:
    """Why ``raw_key`` (a vaulted payer key — one Lium key, or an encoded
    multi-provider bundle) cannot list pods named ``prefix…``, or ``""``.
    Listing is the call every verified teardown starts with: a key that cannot
    list cannot stop the pod it rents. Errors are key-scrubbed."""
    from ..funding.providers import decode_credentials

    try:
        creds = decode_credentials(raw_key)
    except ValueError as e:
        return str(e)
    return credentials_listing_error(creds, prefix, lister=lister)


def credentials_listing_error(creds, prefix: str, *, lister=None) -> str:
    """:func:`payer_key_listing_error` over decoded ``PayerCredential`` s."""
    creds = tuple(creds or ())
    if not creds:
        return "no payer key"
    lister = lister or _default_lister
    for c in creds:
        try:
            lister(c.provider, c.api_key, prefix)
        except Exception as e:  # noqa: BLE001 — any listing failure is the verdict
            msg = str(e)
            for k in (cr.api_key for cr in creds):
                if k:
                    msg = msg.replace(k, "<redacted>")
            return f"{c.provider}: {msg[-300:]}"
    return ""


def check_payer_key(runner, *, hotkey: str, pod_name: str, raw_key: str | None = None,
                    pod=None) -> str:
    """Run the teardown-key check, log ERROR ``PAYER KEY CANNOT STOP POD`` on a
    failure, and return the failure (``""`` = fine). ``raw_key`` is the
    vaulted key (pre-rent); ``pod`` a ledgered pod whose leased credential is
    checked (mid-leg)."""
    lister = runner.__dict__.get("_payer_key_lister")
    if raw_key is None and pod is not None:
        vault = runner._payer_vault()
        if vault is None:
            err = "payer vault unavailable"
        else:
            from ..funding.providers import pod_credential

            try:
                cred = pod_credential(vault, pod)
            except Exception as e:  # noqa: BLE001 — an unreadable lease is the verdict
                cred, err = None, f"payer credential unreadable: {e}"
            else:
                err = "payer key no longer in the vault" if cred is None else ""
            if cred is not None:
                err = credentials_listing_error((cred,), pod_name, lister=lister)
    else:
        err = payer_key_listing_error(raw_key or "", pod_name, lister=lister)
    if err:
        log.error("%s: hotkey=%s pod=%s — the vaulted key cannot list pods on the payer's "
                  "account (%s); tell the miner: a pod rented on this key cannot be stopped "
                  "from here", PAYER_KEY_CANNOT_STOP, hotkey, pod_name, err)
    return err
