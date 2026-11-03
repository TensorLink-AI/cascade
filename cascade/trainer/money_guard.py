"""Money safety for every rent (2026-11-02 incident, owner: "It's too expensive
and important to waste our funds and miners' funds on training").

What went wrong that day, and the gate here that now stops each one:

* An expired payer key / make-good rented an 8xH100 on the OPERATOR's Lium
  account without anyone deciding to pay for it → :func:`operator_leg_gate`:
  every miner leg that would be billed to the operator parks
  ``awaiting_operator_approval`` until the owner approves THAT leg once
  (:mod:`cascade.funding.operator_approvals`).
* The operator balance hit zero, Lium answered 402 and terminated every pod
  on the account (validator eval pods included) → :func:`operator_rent_preflight`:
  balance ≥ projected cost × safety + a reserve for the eval pods, a hard
  daily operator spend cap, the per-leg cap — checked BEFORE the rent.
* The 402 then hot-looped one leg every ~2.5 min → :class:`OperatorBreaker`:
  a balance refusal on any operator rent pauses ALL operator rents for a
  cooldown (:func:`guard_operator_provider` wraps every operator provider).
* A miner's leg could start on a key whose balance cannot finish it →
  :func:`payer_balance_refusal` (Lium ``GET /users/me`` with the payer key).

Every block emits ONE operator_health alert (an active condition the
watchdog dedups and delivers; :func:`health_conditions`), never a per-poll
log loop. Trainer-only: nothing here changes what validators compute.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from pathlib import Path

from ..funding.queue import (
    APPROVAL_CLASS,
    AWAITING_APPROVAL_STATUS,
    DENIED_CLASS,
    OPERATOR_BLOCKED_CLASS,
    PAUSED_STATUS,
)
from ..shared.atomic import write_text_atomic
from .faults import _FundedLegSkip

log = logging.getLogger("cascade.trainer.money")

BREAKER_FILE = "operator_breaker.json"
SPEND_FILE = "operator_spend.json"
# An alert nobody re-raised for this long is dropped (its cause was never
# re-checked — e.g. no leg tried to rent since): the watchdog resolves it.
ALERT_STALE_SECONDS = 6 * 3600.0

_LOCK = threading.Lock()


# ── one-shot alerts (surfaced through the operator_health watchdog) ──────────


def alert(runner, code: str, scope: str, message: str, *, severity: str = "warning",
          **data) -> None:
    """Raise (or refresh) an active operator alert. The watchdog emits ONE
    line per activation and resolves it when :func:`clear_alert` runs (or it
    goes stale); repeated raises only refresh it."""
    now = time.time()
    with _LOCK:
        alerts = runner.__dict__.setdefault("_money_alerts", {})
        key = (code, scope)
        prev = alerts.get(key)
        alerts[key] = {"code": code, "scope": scope, "message": message[:300],
                       "severity": severity,
                       "since": prev["since"] if prev else now, "last": now, **data}
    if prev is None:
        log.info("money guard: %s %s raised (%s)", code, scope, message[:200])


def clear_alert(runner, code: str, scope: str) -> None:
    with _LOCK:
        alerts = runner.__dict__.get("_money_alerts") or {}
        gone = alerts.pop((code, scope), None)
    if gone is not None:
        log.info("money guard: %s %s cleared", code, scope)


def _alert_conditions(runner, now: float) -> list[dict]:
    with _LOCK:
        alerts = runner.__dict__.get("_money_alerts") or {}
        for key in [k for k, v in alerts.items() if now - v["last"] > ALERT_STALE_SECONDS]:
            alerts.pop(key, None)
        return [{k: v for k, v in a.items() if k not in ("since", "last")}
                for a in alerts.values()]


def health_conditions(runner, queues: dict, *, now: float | None = None) -> list[dict]:
    """Money-safety conditions for the operator_health watchdog: raised
    alerts, the open breaker, legs awaiting approval and paused entries."""
    t = time.time() if now is None else float(now)
    out = _alert_conditions(runner, t)
    with contextlib.suppress(Exception):
        from .box_health import BoxHealth

        halt = BoxHealth.for_runner(runner).halt()
        if halt:
            out.append({"code": "rents_halted", "scope": "box", "severity": "critical",
                        "class": halt.get("cls"), "since": halt.get("since"),
                        "message": (f"ALL new GPU rents halted: {halt.get('reason', '')[:200]} — "
                                    "clears on a passing health check or `cascade-trainer "
                                    "resume-rents`")})
    with contextlib.suppress(Exception):
        rec = OperatorBreaker.for_runner(runner).open_record(t)
        if rec:
            out.append({"code": "operator_breaker_open", "scope": "operator",
                        "until": rec.get("open_until"),
                        "message": f"all operator rents paused: {rec.get('reason', '')[:200]}"})
    with contextlib.suppress(Exception):
        from ..funding.operator_approvals import OperatorApprovals

        for r in OperatorApprovals.for_work_root(runner.work_root).pending():
            lost = float(r.get("miner_lost_usd") or 0.0)
            out.append({
                "code": "operator_leg_awaiting_approval", "scope": str(r.get("hotkey", ""))[:16],
                "uid": r.get("uid"), "sku": r.get("sku"),
                "projected_cost_usd": r.get("projected_cost_usd"),
                "severity": "critical" if lost else "warning",
                "message": (f"{r.get('kind', '')}: ~${r.get('projected_cost_usd', 0):.2f} on the "
                            f"operator account"
                            + (f"; the miner already lost ≤ ${lost:.2f} on a finished run"
                               if lost else "")
                            + f" — `cascade-trainer approve-operator-leg {r.get('hotkey')}` "
                            "or `deny-operator-leg`")})
    for lane, entries in (queues or {}).items():
        for e in entries:
            if e.status == PAUSED_STATUS:
                out.append({"code": "entry_paused", "scope": f"{lane}:{e.hotkey[:16]}",
                            "operator_requeues": int(getattr(e, "operator_requeues", 0)),
                            "message": (f"paused after operator-side faults (nothing burned) — "
                                        f"`cascade-trainer resume-operator-leg {e.hotkey}`")})
    return out


# ── the operator-account circuit breaker ────────────────────────────────────


class OperatorBreaker:
    """``<work_root>/operator_breaker.json``: while open, NO operator rent of
    any kind is attempted. Persisted, so a restart cannot reset it."""

    def __init__(self, path: Path | str, *, cooldown_seconds: float = 3600.0) -> None:
        self.path = Path(path)
        self.cooldown_seconds = float(cooldown_seconds)

    @classmethod
    def for_runner(cls, runner) -> OperatorBreaker:
        rnd = getattr(getattr(runner, "cfg", None), "round", None)
        return cls(Path(runner.work_root) / BREAKER_FILE, cooldown_seconds=float(
            getattr(rnd, "operator_breaker_cooldown_seconds", 3600) or 0))

    def open_record(self, now: float | None = None) -> dict | None:
        t = time.time() if now is None else float(now)
        try:
            rec = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            # Unreadable breaker state: fail CLOSED (treated as open) — a
            # corrupt file must never re-open the money tap.
            return {"open_until": t + 60, "reason": "breaker state unreadable"}
        return rec if float(rec.get("open_until", 0)) > t else None

    def trip(self, reason: str, *, now: float | None = None) -> dict:
        t = time.time() if now is None else float(now)
        rec = {"open_until": t + self.cooldown_seconds, "tripped_at": t, "reason": reason[:300]}
        with _LOCK:
            prev = self.open_record(t)
            if prev and float(prev.get("open_until", 0)) >= rec["open_until"]:
                return prev
            write_text_atomic(self.path, json.dumps(rec, sort_keys=True))
        if prev is None:
            # the watchdog raises the one operator_breaker_open alert from the file
            log.error("money guard: operator breaker OPEN — %s; ALL operator rents paused "
                      "for %.0f s", reason[:200], self.cooldown_seconds)
        return rec


def is_balance_refusal(text: str) -> bool:
    """A provider's balance/payment refusal (402, BALANCE_REQUIRED, "insufficient
    balance" …) — the class that trips the breaker."""
    from ..funding.faults import is_funding_failure

    low = (text or "").lower()
    return is_funding_failure(low) or "balance_required" in low


class _BreakerGuardedProvider:
    """An operator provider whose ``launch`` refuses while the breaker is open
    and trips it on a balance refusal. Everything else is delegated."""

    def __init__(self, provider, breaker: OperatorBreaker, *, enforce: bool = True) -> None:
        self._provider = provider
        self._breaker = breaker
        # False for the king's JIT pod and king backups: they are never held
        # by a breaker someone else tripped — only their own 402 trips it.
        self._enforce = enforce

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def launch(self, spec, *args, **kwargs):
        from ..provision.core import ProvisionError

        rec = self._breaker.open_record() if self._enforce else None
        if rec:
            raise ProvisionError(f"operator rents paused: circuit breaker open until "
                                 f"{time.strftime('%H:%M:%SZ', time.gmtime(rec['open_until']))} "
                                 f"({rec.get('reason', '')})")
        try:
            return self._provider.launch(spec, *args, **kwargs)
        except Exception as e:
            if is_balance_refusal(str(e)):
                self._breaker.trip(f"operator rent refused for balance: {str(e)[:200]}")
            raise


def guard_operator_provider(runner, provider, *, enforce: bool = True):
    """Wrap an OPERATOR-account provider with the circuit breaker. A real
    balance refusal (402 / BALANCE_REQUIRED) from ANY launch trips it;
    ``enforce=False`` (king pod / king backups) is never refused by it."""
    if isinstance(provider, _BreakerGuardedProvider):
        provider = provider._provider
    return _BreakerGuardedProvider(provider, OperatorBreaker.for_runner(runner), enforce=enforce)


# ── projected cost, daily spend ledger, balance ─────────────────────────────


def best_offer_rate(runner, skus) -> float | None:
    """The cheapest operator-account offer's $/h for ``skus`` right now (the
    same listing and price caps a rent uses), or None when unknown."""
    seam = getattr(runner, "_operator_offer_rate", None)
    if callable(seam):
        return seam(tuple(skus))
    try:
        from ..provision.core import executor_price_per_hour
        from ..provision.operator_pool import operator_price_caps
        from .lane_pool import _lium_provider

        prov = _lium_provider(runner.cfg.round, price_caps=operator_price_caps(runner))
        for _sku, ex in prov.list_offers(tuple(skus)):
            price = executor_price_per_hour(ex)
            if price:
                return float(price)
    except Exception as e:  # noqa: BLE001 — unknown: the ceiling applies
        log.debug("operator offer price unknown (%s)", e)
    return None


def projected_operator_leg_cost(runner, *, live_price: bool = False
                                ) -> tuple[str, float, float, float]:
    """``(sku, $/h, hours, $)`` an operator-account leg is projected to cost:
    the cheapest current offer's $/h when ``live_price`` and known, else the
    operator $/h ceiling, × the leg's expected wall (+ the ready timeout).
    The spend actually BOOKED is the landed price (:func:`note_operator_launch`;
    review R4)."""
    from ..provision.operator_pool import operator_max_price_per_hour

    rnd = runner.cfg.round
    skus = tuple(runner._funded_skus_for_rent() or ()) or ((rnd.funded_pod_sku,)
                                                          if rnd.funded_pod_sku else ())
    sku = skus[0] if skus else ""
    ceiling = float(operator_max_price_per_hour(rnd) or 0.0)
    rate = ceiling
    if live_price and skus:
        best = best_offer_rate(runner, skus)
        if best:
            rate = min(best, ceiling) if ceiling else best
    try:
        wall = float(runner._leg_wall_seconds())
    except Exception:  # noqa: BLE001 — unknown wall: project a full day (caps stay strict)
        wall = 24 * 3600.0
    hours = (wall + float(getattr(rnd, "funded_ready_timeout_seconds", 0.0) or 0.0)) / 3600.0
    return sku, rate, hours, rate * hours


class OperatorSpend:
    """``<work_root>/operator_spend.json``: projected operator spend per UTC day."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @classmethod
    def for_runner(cls, runner) -> OperatorSpend:
        return cls(Path(runner.work_root) / SPEND_FILE)

    def _load(self) -> dict:
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        return doc if isinstance(doc, dict) else {}

    @staticmethod
    def day(now: float) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(now))

    def total(self, now: float | None = None) -> float:
        t = time.time() if now is None else float(now)
        rows = self._load().get(self.day(t), [])
        return float(sum(float(r.get("usd", 0.0)) for r in rows))

    def rows(self, now: float | None = None) -> list[dict]:
        t = time.time() if now is None else float(now)
        return list(self._load().get(self.day(t), []))

    def charge(self, hotkey: str, usd: float, *, kind: str, projected: float | None = None,
               now: float | None = None) -> None:
        t = time.time() if now is None else float(now)
        with _LOCK:
            doc = self._load()
            day = self.day(t)
            doc = {k: v for k, v in doc.items() if k >= self.day(t - 14 * 86400)}
            doc.setdefault(day, []).append({
                "hotkey": hotkey, "usd": round(float(usd), 2), "kind": kind, "at": t,
                "projected_usd": round(float(usd if projected is None else projected), 2)})
            write_text_atomic(self.path, json.dumps(doc, sort_keys=True))


def daily_cap_usd(rnd) -> float:
    """The effective daily operator cap: the lower positive of
    ``operator_daily_spend_cap_usd`` and ``ops_operator_daily_cap_usd``."""
    caps = [float(getattr(rnd, k, 0.0) or 0.0)
            for k in ("operator_daily_spend_cap_usd", "ops_operator_daily_cap_usd")]
    caps = [c for c in caps if c > 0]
    return min(caps) if caps else 0.0


def _balance(reader) -> float | None:
    try:
        return reader()
    except Exception as e:  # noqa: BLE001 — unknown, never zero
        log.warning("operator balance unreadable (%s)", type(e).__name__)
        return None


def _skip(runner, hotkey: str, msg: str, *, cls: str, miner_fault: bool = False) -> None:
    runner._record_funded_failure(hotkey, msg, miner_fault=miner_fault, error_class=cls,
                                  burn=False)
    raise _FundedLegSkip(hotkey)


def operator_rent_preflight(runner, gen, cost_usd: float) -> None:
    """Before an operator-account rent for a MINER leg: breaker closed,
    projected cost within the per-leg and daily caps, operator balance ≥
    cost × safety factor + reserve. Any miss: nothing rented, the leg
    requeued unburned (``operator_blocked``, backed off, not counted against
    the entry) with one alert; a balance shortfall also trips the breaker."""
    rnd = runner.cfg.round
    hk = gen.hotkey
    breaker = OperatorBreaker.for_runner(runner)
    rec = breaker.open_record()
    if rec:
        _skip(runner, hk, f"operator rents paused (circuit breaker: {rec.get('reason', '')}); "
              "requeued, nothing rented, nothing burned", cls=OPERATOR_BLOCKED_CLASS)
    leg_cap = float(getattr(rnd, "ops_operator_leg_cap_usd", 0.0) or 0.0)
    if leg_cap and cost_usd > leg_cap:
        alert(runner, "operator_leg_cap", hk[:16],
              f"projected ${cost_usd:.2f} > ops_operator_leg_cap_usd ${leg_cap:.2f}")
        _skip(runner, hk, f"projected operator cost ${cost_usd:.2f} exceeds the per-leg cap "
              f"${leg_cap:.2f}; nothing rented", cls=OPERATOR_BLOCKED_CLASS)
    clear_alert(runner, "operator_leg_cap", hk[:16])
    cap = daily_cap_usd(rnd)
    spent = OperatorSpend.for_runner(runner).total()
    if cap and spent + cost_usd > cap:
        alert(runner, "operator_daily_cap", "operator",
              f"today ${spent:.2f} + projected ${cost_usd:.2f} > daily cap ${cap:.2f}")
        _skip(runner, hk, f"operator daily spend cap reached (${spent:.2f} spent + "
              f"${cost_usd:.2f} projected > ${cap:.2f}); nothing rented, retried after "
              "the cap resets", cls=OPERATOR_BLOCKED_CLASS)
    clear_alert(runner, "operator_daily_cap", "operator")
    reader = getattr(runner, "_operator_balance_reader", None)
    if reader is None:
        from .operator_health import lium_balance as reader
    bal = _balance(reader)
    need = cost_usd * float(getattr(rnd, "operator_cost_safety_factor", 1.25) or 1.0) + float(
        getattr(rnd, "operator_balance_reserve_usd", 0.0) or 0.0)
    if bal is None:
        alert(runner, "operator_balance_unknown", "operator",
              "operator Lium balance unreadable — operator rents refused (fail closed)")
        _skip(runner, hk, "operator balance unknown — no operator rent without a known "
              "balance; requeued, nothing burned", cls=OPERATOR_BLOCKED_CLASS)
    clear_alert(runner, "operator_balance_unknown", "operator")
    if bal < need:
        # Refuses THIS leg only: a projection never trips the account-wide
        # breaker (that would hold the king's pod) — only a real 402 does.
        alert(runner, "operator_balance_low", hk[:16],
              f"operator balance ${bal:.2f} < ${need:.2f} needed for a ${cost_usd:.2f} leg "
              "plus the eval-pod reserve — leg not rented")
        _skip(runner, hk, f"operator balance below projected cost + reserve (needs "
              f"${need:.2f}); nothing rented, nothing burned", cls=OPERATOR_BLOCKED_CLASS)
    clear_alert(runner, "operator_balance_low", hk[:16])


def payer_balance_refusal(runner, api_key: str, *, hours: float, rate: float | None = None) -> str:
    """Why a miner's own-key leg must not start ("" = go): the payer's Lium
    balance cannot cover ``rate`` × ``hours`` (``[round]
    payer_projected_rate_per_hour``; 0 = off). An unreadable balance (not a
    Lium key, API down) never blocks — the rent's own 402 still classifies."""
    rnd = runner.cfg.round
    r = float(rate if rate is not None else getattr(rnd, "payer_projected_rate_per_hour", 0.0)
              or 0.0)
    if r <= 0 or not api_key:
        return ""
    reader = getattr(runner, "_payer_balance_reader", None)
    if reader is None:
        from .operator_health import lium_balance

        def reader(key):
            return lium_balance(api_key=key)
    bal = None
    for _ in range(3):                  # an API blip is retried; unknown never blocks
        try:
            bal = reader(api_key)
        except Exception:  # noqa: BLE001 — the key is never logged
            bal = None
        if bal is not None:
            break
    if bal is None:
        log.warning("payer balance unknown after retries — the rent proceeds (its own 402 "
                    "still classifies)")
        return ""
    need = r * float(hours)
    if bal >= need:
        return ""
    return (f"top up: need ≈${need:.2f} for this leg (${r:.2f}/h for {hours:.1f} h), balance "
            f"${bal:.2f} — nothing rented, no attempt charged; it retries by itself")


# ── the per-leg owner approval gate ─────────────────────────────────────────


def _leg_local(runner):
    import threading

    return runner.__dict__.setdefault("_leg_local_obj", threading.local())


def preapproved(runner, hotkey: str, *, kind: str = "") -> bool:
    """Whether ``hotkey``'s current leg holds an unlaunched owner approval
    (peek only; an owner pre-approval counts). A miner-funded leg with a
    working key leaves its market wait for operator capacity ONLY then;
    otherwise it keeps waiting on its own key and is never parked for it."""
    try:
        queue = runner._funded_queue()
        if queue is None or not hotkey:
            return False
        from ..funding.operator_approvals import OperatorApprovals

        entry = queue.get(hotkey)
        return bool(entry is not None and OperatorApprovals.for_work_root(
            runner.work_root).peek_approved(Path(queue.path).name, hotkey, entry.ref,
                                            funded_at=entry.funded_at, kind=kind))
    except Exception:  # noqa: BLE001 — unknown ⇒ not approved
        return False


def grandfathered(runner, hotkey: str, *, allocated: bool) -> bool:
    """A leg already running on the operator's bill when this gate shipped
    (or before a restart): operator-billed, in flight and either holding a
    ledgered operator pod (``allocated``) or re-attached by this process. It
    is ADOPTED, never parked — parking it would drop its pod from the sweep's
    keep-set and lose the run (review F1)."""
    try:
        queue = runner._funded_queue()
        e = queue.get(hotkey) if queue is not None else None
    except Exception:  # noqa: BLE001
        return False
    if e is None or e.status != "in_flight" or not e.operator_billed:
        return False
    if allocated:
        return True
    from .admission import is_admitted

    return is_admitted(runner, hotkey)


def operator_leg_gate(runner, gen, *, kind: str, rent: bool) -> None:
    """Gate EVERY miner leg about to be billed to the operator (``kind``:
    ``expired_key`` / ``make_good`` / ``operator_lane`` / ``paid_fallback``)
    and every retrain after a finished run was lost on our side
    (``retrain_after_loss``, on the miner's key). ``rent`` = a new
    operator-account pod (money preflight + spend booking at launch), else
    no new operator spend.

    Returns when the owner's approval of this leg applies (and, for a rent,
    every money preflight passed). Nothing is used up here: the approval is
    marked launched only once the pod is actually rented
    (:func:`note_operator_launch`), so a no-capacity / lemon / rent error
    keeps it for the next attempt. Otherwise the verdict is recorded and
    :class:`_FundedLegSkip` raised: no approval → ``awaiting_operator_approval``
    (parked, unburned, one alert); denied → the miner's own key (or, for an
    expired key, a terminal unburned "re-fund needed"). King legs (no funded
    queue in scope) are not gated."""
    queue = runner._funded_queue()
    if queue is None:
        return
    from ..funding.operator_approvals import OperatorApprovals

    hk = gen.hotkey
    entry = queue.get(hk)
    attempt_id = str(getattr(entry, "attempt_id", "") or "")
    funded_at = getattr(entry, "funded_at", None)
    sku, rate, hours, cost = projected_operator_leg_cost(
        runner, live_price=bool(rent and kind != "retrain_after_loss"))
    lost = 0.0
    if kind == "retrain_after_loss":
        from .box_health import miner_lost_usd

        lost, cost = miner_lost_usd(runner), 0.0    # the miner's key pays the retrain
    elif not rent:
        cost = 0.0          # an operator lane is already paid for; no new spend
    store = OperatorApprovals.for_work_root(runner.work_root)
    qname = Path(queue.path).name
    verdict = store.check(qname, hk, ref=gen.ref, attempt_id=attempt_id, kind=kind,
                          funded_at=funded_at, projected_cost_usd=cost)
    if verdict == "approved":
        if rent and kind != "retrain_after_loss":
            operator_rent_preflight(runner, gen, cost)   # a block keeps the approval
        _leg_local(runner).operator_approval = {
            "queue": qname, "hotkey": hk, "ref": gen.ref, "attempt_id": attempt_id,
            "cost": cost, "hours": hours, "kind": kind,
            "book": bool(rent and kind != "retrain_after_loss")}
        if not rent:
            note_operator_launch(runner, hk)           # a lane dispatch IS the launch
        log.warning("operator-billed leg %s (%s, uid %s): owner approval applies — %s, "
                    "projected ≤ $%.2f (%s, $%.2f/h x %.1f h)", hk[:12], kind,
                    getattr(gen, "uid", "?"),
                    "renting a pod" if rent else "no new operator rent", cost, sku, rate, hours)
        return
    if verdict == "denied":
        store.take_denial(qname, hk)
        if kind in ("expired_key", "retrain_after_loss"):
            _skip(runner, hk, ("payer key expired and the operator declined to pay for this "
                               "leg — fund again with a fresh key (no attempt charged)"
                               if kind == "expired_key" else
                               "the operator declined the retrain after a lost run — fund "
                               "again to retrain (no attempt charged)"),
                  cls="auth", miner_fault=True)
        _skip(runner, hk, "the operator declined to pay for this leg; it retries on the "
              "miner's own key (no attempt charged)", cls=DENIED_CLASS)
    reason = {"expired_key": "payer key expired",
              "make_good": "operator make-good after an operator-side loss",
              "operator_lane": "no marketplace capacity: an operator lane",
              "paid_fallback": "ops_paid_fallback: marketplace sold out",
              "retrain_after_loss": "retrain after a finished run was lost on our side"
              }.get(kind, kind)
    store.request(qname, hk, ref=gen.ref, attempt_id=attempt_id, queue_path=str(queue.path),
                  uid=getattr(gen, "uid", None), sku=sku, rate_per_hour=rate,
                  expected_hours=hours, projected_cost_usd=cost, kind=kind, reason=reason,
                  funded_at=funded_at, miner_lost_usd=lost)
    _skip(runner, hk, f"awaiting operator approval ({reason}; ~${cost:.2f} projected on "
          f"{sku or 'the round SKU'}"
          + (f"; the miner already lost ≤ ${lost:.2f}" if lost else "")
          + ") — parked, nothing rented, no attempt charged", cls=APPROVAL_CLASS)


def operator_leg_allocated(runner, round_id: str, hotkey: str) -> bool:
    """This leg's own operator pod is on the ledger (a restart, a retry after
    a transport drop, a kept pod): it must be ADOPTED by name, never re-gated."""
    try:
        want = runner._operator_leg_pod_prefix(str(round_id), hotkey) + "-0"
        return any(p.instance_id == want for p in runner._load_funded_ledger())
    except Exception:  # noqa: BLE001 — unreadable ledger: not allocated
        return False


def gate_or_adopt(runner, round_id: str, gen, *, kind: str, rent: bool) -> None:
    """:func:`operator_leg_gate`, except for a leg that already runs on the
    operator's bill (review F1): its ledgered operator pod is adopted with no
    approval, preflight or spend booking — if that pod turns out to be gone,
    the fresh rent is gated then (``operator_gate_deferred``). An in-flight
    operator-billed leg re-attached after a restart takes its lane the same
    way (grandfathered)."""
    if rent and operator_leg_allocated(runner, round_id, gen.hotkey):
        _leg_local(runner).operator_gate_deferred = kind
        log.warning("operator-billed leg %s: its operator pod is on the ledger — adopting it "
                    "(no new approval, no new spend)", gen.hotkey[:12])
        return
    if not rent and grandfathered(runner, gen.hotkey, allocated=False):
        log.warning("operator-billed leg %s: already in flight on the operator's bill "
                    "(re-attached) — continuing without a new approval", gen.hotkey[:12])
        return
    operator_leg_gate(runner, gen, kind=kind, rent=rent)


def take_deferred_gate(runner) -> str | None:
    loc = _leg_local(runner)
    kind = getattr(loc, "operator_gate_deferred", None)
    loc.operator_gate_deferred = None
    return kind


def note_operator_launch(runner, hotkey: str | None = None, *,
                         landed_rate: float | None = None) -> None:
    """The approved leg's pod was rented (or its lane dispatched): mark the
    approval launched for THIS attempt and book the operator spend — only
    now, never for a rent that did not happen (review F3)."""
    loc = _leg_local(runner)
    a = getattr(loc, "operator_approval", None)
    if not a:
        return
    loc.operator_approval = None
    if hotkey is not None and a["hotkey"] != hotkey:
        return                        # a stale approval of another leg on this thread
    from ..funding.operator_approvals import OperatorApprovals

    # Book the LANDED $/h × the expected wall when the rent price is known
    # (the projection was an upper bound; review R4).
    booked = (float(landed_rate) * float(a.get("hours") or 0.0) if landed_rate
              else float(a["cost"]))
    OperatorApprovals.for_work_root(runner.work_root).mark_launched(
        a["queue"], a["hotkey"], ref=a["ref"], attempt_id=a["attempt_id"], cost_usd=booked)
    if a.get("book"):
        OperatorSpend.for_runner(runner).charge(a["hotkey"], booked, kind=a["kind"],
                                                projected=float(a["cost"]))


__all__ = ["AWAITING_APPROVAL_STATUS", "OperatorBreaker", "OperatorSpend", "alert",
           "clear_alert", "daily_cap_usd", "guard_operator_provider", "health_conditions",
           "is_balance_refusal", "operator_leg_gate", "operator_rent_preflight",
           "gate_or_adopt", "grandfathered", "note_operator_launch", "operator_leg_allocated",
           "take_deferred_gate", "payer_balance_refusal", "preapproved",
           "projected_operator_leg_cost"]
