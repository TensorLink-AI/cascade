"""The trainer side of the miner waste cap (cascade.funding.miner_spend).

Accounting at the real sites — every allocation on a miner's key
(:func:`spend_start`), every teardown / lemon / loss (:func:`spend_end` with
``waste``), every verified harvest (:func:`spend_end` with ``result``) — and
the pre-rent cap (:func:`waste_cap_check`), run before EVERY rent on a
miner's key (fresh leg, retry, lemon re-rent, retrain after a loss).

At the cap the entry pauses (``paused_spend_cap``: never failed, burned,
expired or rejected) with ONE critical alert and a miner-readable reason.
When the waste is infra / our side, an owner approval request of kind
``spend_cap_make_good`` is filed to FINISH the leg on the operator's account
(the normal operator gates still apply); a genuine miner fault gets the pause
only (a re-fund resumes). Bookkeeping errors never block a teardown; an
unreadable ledger fails CLOSED for new miner-key rents.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..funding.miner_spend import (
    INFRA,
    MAKE_GOOD,
    MINER,
    RESULT,
    WASTE,
    MinerSpend,
    SpendUnknown,
)
from ..funding.queue import SPEND_CAP_CLASS
from .faults import _FundedLegSkip

log = logging.getLogger("cascade.trainer.money")


def _ledger(runner) -> MinerSpend | None:
    try:
        q = runner._funded_queue()
    except Exception:  # noqa: BLE001
        return None
    return MinerSpend.beside(q.path) if q is not None and getattr(q, "path", None) else None


def _entry(runner, hotkey: str):
    try:
        q = runner._funded_queue()
        return q.get(hotkey) if q is not None else None
    except Exception:  # noqa: BLE001
        return None


def fallback_rate(runner, gpu_choice=None) -> float:
    """$/h of a miner-key pod whose landed price is unknown: the miner's own
    cap for a chosen type, else the operator's funded price ceiling."""
    own = float(getattr(gpu_choice, "max_price_per_hour", 0.0) or 0.0)
    return own or float(getattr(runner.cfg.round, "funded_max_price_per_hour", 0.0) or 0.0)


def spend_start(runner, hotkey: str, pod_id: str, *, rate: float) -> None:
    """A pod allocated on the miner's key. Best-effort (never raises)."""
    try:
        led, e = _ledger(runner), _entry(runner, hotkey)
        if led is None or e is None:
            return
        led.start(hotkey, ref=e.ref, funded_at=e.funded_at, pod_id=pod_id, rate=rate)
    except Exception as exc:  # noqa: BLE001 — accounting never blocks a rent
        log.warning("miner spend: could not record pod %s of %s (%s)", pod_id, hotkey[:12], exc)


def _any_ledger(runner) -> MinerSpend | None:
    led = _ledger(runner)
    if led is None and getattr(runner, "work_root", None) is not None:
        led = MinerSpend(Path(runner.work_root) / "miner_spend.json")
    return led


def spend_touch(runner, hotkey: str, pod_id: str) -> None:
    """The pod was seen alive (bounds its cost if it later vanishes)."""
    try:
        led = _any_ledger(runner)
        if led is not None and hotkey:
            led.touch(hotkey, pod_id)
    except Exception:  # noqa: BLE001
        pass


def close_vanished(runner) -> None:
    """Close open records of pods no longer on the funded ledger at their
    last-seen-alive time (review S1). Best-effort."""
    try:
        led = _any_ledger(runner)
        if led is not None:
            led.close_missing(p.instance_id for p in runner._load_funded_ledger())
    except Exception as exc:  # noqa: BLE001
        log.debug("miner spend: vanished-pod close skipped (%s)", exc)


def spend_end(runner, hotkey: str, pod_id: str, *, outcome: str, reason: str = "",
              kind: str = INFRA) -> None:
    """Close a pod's record (``result`` / ``waste`` / ``make_good``). Never
    raises — a teardown is never blocked by bookkeeping. An infra ``waste``
    may file the early make-good request (:func:`early_make_good`); a
    ``make_good`` raises the owner's make-good alert (:func:`note_make_good`)."""
    try:
        led = _any_ledger(runner)
        if led is not None and pod_id:
            led.end(hotkey or None, pod_id, outcome=outcome, reason=reason, kind=kind)
    except Exception as exc:  # noqa: BLE001
        log.warning("miner spend: could not close pod %s of %s (%s)", pod_id, hotkey[:12], exc)
        return
    if not hotkey:
        return
    if outcome == WASTE and kind == INFRA:
        early_make_good(runner, hotkey)
    elif outcome == MAKE_GOOD:
        note_make_good(runner, hotkey)


def note_make_good(runner, hotkey: str) -> None:
    """ONE alert (refreshed) pricing what post-rent rejects / lemons on this
    miner's key cost them — the operator owes it back. Never raises."""
    try:
        led, e = _ledger(runner), _entry(runner, hotkey)
        if led is None or e is None:
            return
        s = led.summary(hotkey, ref=e.ref, funded_at=e.funded_at)
        if s.make_good_usd <= 0:
            return
        from .money_guard import alert

        n = sum(1 for p in s.pods if p.get("outcome") == MAKE_GOOD)
        alert(runner, "miner_make_good_owed", hotkey[:16],
              f"{hotkey[:12]}: ${s.make_good_usd:.2f} of the MINER's money went to {n} pod(s) "
              "we rejected after the rent or that never came up (short pod / stale image / "
              "slow host / NCCL / lemon) — operator make-good owed; not counted as their "
              "waste", severity="critical", make_good_usd=s.make_good_usd)
        log.error("miner spend: %s — $%.2f operator make-good owed (%d rejected/lemon pod(s))",
                  hotkey[:12], s.make_good_usd, n)
    except Exception as exc:  # noqa: BLE001
        log.debug("miner spend: make-good alert skipped (%s)", exc)


def early_make_good(runner, hotkey: str) -> None:
    """Money safety II: the FIRST infra loss taking a submission's infra
    waste to ``[round] miner_waste_alert_usd`` files the
    ``spend_cap_make_good`` owner request (and one critical alert) at once —
    not at the next rent, one leg late. Filed once per submission. Never
    raises; the leg itself is not paused here (the waste cap still is)."""
    try:
        threshold = float(getattr(runner.cfg.round, "miner_waste_alert_usd", 0.0) or 0.0)
        if threshold <= 0:
            return
        led, e = _ledger(runner), _entry(runner, hotkey)
        if led is None or e is None:
            return
        s = led.summary(hotkey, ref=e.ref, funded_at=e.funded_at)
        if s.infra_waste_usd < threshold or s.make_good_requested:
            return
        if not led.mark_make_good_requested(hotkey, e.ref):
            return
        from ..funding.operator_approvals import OperatorApprovals
        from .money_guard import alert, projected_operator_leg_cost

        q = runner._funded_queue()
        sku, rate, hours, op_cost = projected_operator_leg_cost(runner)
        OperatorApprovals.for_work_root(runner.work_root).request(
            Path(q.path).name, hotkey, ref=e.ref, attempt_id=e.attempt_id,
            queue_path=str(q.path), uid=None, sku=sku, rate_per_hour=rate,
            expected_hours=hours, projected_cost_usd=op_cost, kind="spend_cap_make_good",
            funded_at=e.funded_at, miner_lost_usd=s.infra_waste_usd,
            reason=(f"the miner lost ${s.infra_waste_usd:.2f} to infra failures (alert "
                    f"threshold ${threshold:.2f})"))
        alert(runner, "miner_waste_alert", hotkey[:16],
              f"{hotkey[:12]}: ${s.infra_waste_usd:.2f} of the miner's money lost to infra "
              f"failures (≥ ${threshold:.2f}) — make-good ≈${op_cost:.2f} on the operator "
              f"account filed: `cascade-trainer approve-operator-leg {hotkey}`",
              severity="critical")
        log.error("miner spend: %s infra waste $%.2f ≥ $%.2f — spend_cap_make_good filed",
                  hotkey[:12], s.infra_waste_usd, threshold)
    except Exception as exc:  # noqa: BLE001
        log.warning("miner spend: early make-good request skipped (%s)", exc)


def _pause(runner, gen, msg: str) -> None:
    runner._record_funded_failure(gen.hotkey, msg, miner_fault=False, error_class=SPEND_CAP_CLASS,
                                  burn=False)
    raise _FundedLegSkip(gen.hotkey)


def waste_cap_check(runner, gen) -> None:
    """Before ANY rent on a miner's key: pause the entry when the waste so far
    reached ``[round] miner_waste_cap_usd`` (0 = off). King legs and
    operator-billed legs are not miner-key rents and are not checked."""
    cap = float(getattr(runner.cfg.round, "miner_waste_cap_usd", 0.0) or 0.0)
    if cap <= 0:
        return
    led, e = _ledger(runner), _entry(runner, gen.hotkey)
    if led is None or e is None:
        return
    close_vanished(runner)
    try:
        s = led.summary(gen.hotkey, ref=e.ref, funded_at=e.funded_at)
    except SpendUnknown as exc:
        from .money_guard import alert

        alert(runner, "miner_spend_unknown", gen.hotkey[:16], f"{exc} — miner-key rents of "
              f"{gen.hotkey[:12]} paused (fail closed)", severity="critical")
        _pause(runner, gen, "paused: the operator's spend record for this submission is "
               "unreadable — nothing rented, nothing burned; the operator will review")
    if s.waste_usd < cap:
        return
    trip(runner, gen, e, s, cap)


def trip(runner, gen, entry, s, cap: float) -> None:
    """At the cap: pause, alert once, and (infra waste) file the make-good."""
    from ..funding.operator_approvals import OperatorApprovals
    from .money_guard import alert, projected_operator_leg_cost

    led = _ledger(runner)
    if led is not None:
        led.mark_capped(gen.hotkey, entry.ref)
    infra = s.infra_waste_usd >= cap
    sku, rate, hours, op_cost = projected_operator_leg_cost(runner)
    pods = ", ".join(f"{p['pod_id'][-14:]} ${float(p.get('rate') or 0):.2f}/h "
                     f"{p.get('outcome') or 'open'}" for p in s.pods[-6:])
    if infra:
        q = runner._funded_queue()
        OperatorApprovals.for_work_root(runner.work_root).request(
            Path(q.path).name, gen.hotkey, ref=gen.ref, attempt_id=entry.attempt_id,
            queue_path=str(q.path), uid=getattr(gen, "uid", None), sku=sku,
            rate_per_hour=rate, expected_hours=hours, projected_cost_usd=op_cost,
            kind="spend_cap_make_good", funded_at=entry.funded_at, miner_lost_usd=s.waste_usd,
            reason=f"the miner lost ${s.waste_usd:.2f} to infra failures (cap ${cap:.2f})")
    alert(runner, "miner_waste_cap", gen.hotkey[:16],
          f"{gen.hotkey[:12]} (uid {getattr(gen, 'uid', '?')}): spent ${s.spend_usd:.2f}, wasted "
          f"${s.waste_usd:.2f} on pods that produced nothing (cap ${cap:.2f}) — paused"
          + (f"; make-good on the operator account ≈ ${op_cost:.2f} awaits "
             f"`cascade-trainer approve-operator-leg {gen.hotkey}`" if infra else
             "; miner-side causes: a re-fund resumes it") + f". Pods: {pods}",
          severity="critical")
    log.error("miner waste cap: %s paused — $%.2f wasted of $%.2f spent (cap $%.2f)%s",
              gen.hotkey[:12], s.waste_usd, s.spend_usd, cap,
              " — operator make-good requested" if infra else "")
    _pause(runner, gen, (f"paused after ${s.waste_usd:.0f} of your funds were lost to infra "
                         "failures; the operator will review — your submission is kept"
                         if infra else
                         f"paused: ${s.waste_usd:.0f} of your funds lost to failed attempts; "
                         "re-fund or contact the operator — your submission is kept"))


def budget_check(runner, gen, gpu_choice=None) -> None:
    """Before ANY rent on a miner's key: the miner's own TOTAL budget
    (``max_total_spend_usd``) must cover all spend so far (success included)
    plus this leg's projection (rate × (expected wall + ready timeout)).
    Otherwise nothing is rented and the entry parks ``paused_budget`` —
    never failed, burned, expired or rejected. A running leg is never killed
    for a budget overrun (Lium key budgets are the hard backstop)."""
    from ..funding.queue import BUDGET_CLASS

    e = _entry(runner, gen.hotkey)
    budget = getattr(e, "max_total_spend_usd", None) if e is not None else None
    if budget is None:
        return
    led = _ledger(runner)
    close_vanished(runner)
    try:
        spent = led.summary(gen.hotkey, ref=e.ref, funded_at=e.funded_at).spend_usd if led else 0.0
    except SpendUnknown:
        _pause_as(runner, gen, BUDGET_CLASS, "paused: the operator's spend record for this "
                  "submission is unreadable — nothing rented, nothing burned")
    rnd = runner.cfg.round
    try:
        wall = float(runner._leg_wall_seconds())
    except Exception:  # noqa: BLE001 — unknown wall: a full day (strict)
        wall = 24 * 3600.0
    hours = (wall + float(getattr(rnd, "funded_ready_timeout_seconds", 0.0) or 0.0)) / 3600.0
    need = fallback_rate(runner, gpu_choice) * hours
    if spent + need <= float(budget):
        return
    from .money_guard import alert

    alert(runner, "miner_budget", gen.hotkey[:16],
          f"{gen.hotkey[:12]} (uid {getattr(gen, 'uid', '?')}): budget ${budget:.2f}, spent "
          f"${spent:.2f}, next leg ≈${need:.2f} — paused, nothing rented")
    _pause_as(runner, gen, BUDGET_CLASS,
              f"budget ${budget:.0f} {'reached' if spent >= budget else 'insufficient'} "
              f"(${spent:.0f} spent, this leg needs ≈${need:.0f}); raise it or re-fund — "
              "your submission is kept")


def _pause_as(runner, gen, cls: str, msg: str) -> None:
    runner._record_funded_failure(gen.hotkey, msg, miner_fault=False, error_class=cls, burn=False)
    raise _FundedLegSkip(gen.hotkey)


def capped(runner, gen) -> bool:
    """This submission tripped the cap (and was not reset / re-funded)."""
    led, e = _ledger(runner), _entry(runner, gen.hotkey)
    if led is None or e is None:
        return False
    try:
        return led.summary(gen.hotkey, ref=e.ref, funded_at=e.funded_at).capped
    except SpendUnknown:
        return True


__all__ = ["INFRA", "MAKE_GOOD", "MINER", "RESULT", "WASTE", "budget_check", "capped",
           "early_make_good", "fallback_rate", "note_make_good",
           "spend_end",
           "spend_start", "trip", "waste_cap_check"]
