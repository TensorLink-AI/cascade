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


def launch_kind(name_prefix: str) -> str:
    """The spend kind of an operator launch, from its deterministic pod name."""
    name = str(name_prefix or "")
    if "-funded-king" in name:
        return KING_BACKUP_KIND if "-replica-" in name else KING_KIND
    if "-funded-" in name and (name.endswith("-op") or "-op-" in name):
        return OPERATOR_LEG_KIND
    if "-funded-" in name and "-replica-" in name:
        return KING_BACKUP_KIND
    return OPERATOR_OTHER_KIND


class _BreakerGuardedProvider:
    """An operator provider whose ``launch`` refuses while the breaker is open
    and trips it on a balance refusal; with a runner (money safety II) every
    launch — the king's included — is also re-checked against the runway
    guard and the daily cap RIGHT BEFORE the create, and every allocation
    (lemons too) is booked into :class:`OperatorSpend` at its landed price.
    Everything else is delegated."""

    def __init__(self, provider, breaker: OperatorBreaker, *, enforce: bool = True,
                 runner=None) -> None:
        self._provider = provider
        self._breaker = breaker
        # False for the king's JIT pod and king backups: they are never held
        # by a breaker someone else tripped — only their own 402 trips it.
        self._enforce = enforce
        self._runner = runner

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def launch(self, spec, *args, **kwargs):
        from ..provision.core import LaunchRefused

        rec = self._breaker.open_record() if self._enforce else None
        if rec:
            raise LaunchRefused(f"operator rents paused: circuit breaker open until "
                                 f"{time.strftime('%H:%M:%SZ', time.gmtime(rec['open_until']))} "
                                 f"({rec.get('reason', '')})")
        runner = self._runner
        kind = launch_kind(getattr(spec, "name_prefix", ""))
        hours = 0.0
        if runner is not None:
            hours = expected_pod_hours(runner, kind)
            rate = ceiling_rate(runner)
            if kind in (KING_KIND, KING_BACKUP_KIND):
                # review M1: the king is judged at the market's best offer, not
                # the ceiling (a $25 x 12 h ceiling projection held it)
                skus = tuple(getattr(spec, "sku_choices", ()) or ()) or (
                    (getattr(spec, "sku", ""),) if getattr(spec, "sku", "") else ())
                best = best_offer_rate(runner, skus) if skus else None
                if best:
                    rate = min(best, rate) if rate else best
            operator_launch_check(runner, kind=kind, projected_usd=rate * hours)
        try:
            ids = self._provider.launch(spec, *args, **kwargs)
        except Exception as e:
            if is_balance_refusal(str(e)):
                self._breaker.trip(f"operator rent refused for balance: {str(e)[:200]}")
            raise
        if runner is not None:
            for pid in ids or ():
                book_operator_allocation(runner, self._provider, str(pid), kind=kind,
                                         hours=hours)
        return ids


class BookedCpuProvider:
    """A CPU-generation provider (``cgen-*`` VMs, operator money) whose every
    launch is re-checked against the runway guard / daily cap and booked
    into :class:`OperatorSpend` (kind ``cpu_vm``), and whose terminate
    settles the booking to the VM's real cost (money safety II). Everything
    else is delegated."""

    def __init__(self, runner, provider) -> None:
        self._runner = runner
        self._provider = provider

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def launch(self, shape, *, name: str, ssh_pubkey: str, image=None, ttl_seconds=None):
        import inspect

        rate = float(getattr(shape, "price_per_hour", 0.0) or 0.0)
        hours = (float(ttl_seconds) if ttl_seconds else 2 * 3600.0) / 3600.0
        operator_launch_check(self._runner, kind=CPU_VM_KIND, projected_usd=rate * hours)
        kw = {"name": name, "ssh_pubkey": ssh_pubkey}
        if image is not None:
            kw["image"] = image
        if ttl_seconds:
            with contextlib.suppress(TypeError, ValueError):
                if "ttl_seconds" in inspect.signature(self._provider.launch).parameters:
                    kw["ttl_seconds"] = ttl_seconds
        vm = self._provider.launch(shape, **kw)
        with contextlib.suppress(Exception):
            OperatorSpend.for_runner(self._runner).charge(
                "", rate * hours, kind=CPU_VM_KIND, pod_id=str(vm), rate=rate, hours=hours)
        return vm

    def terminate(self, vm_id: str):
        settle_operator_pod(self._runner, str(vm_id))
        return self._provider.terminate(vm_id)


def guard_operator_provider(runner, provider, *, enforce: bool = True):
    """Wrap an OPERATOR-account provider with the circuit breaker (and the
    runway / daily-cap re-check + spend booking at every launch). A real
    balance refusal (402 / BALANCE_REQUIRED) from ANY launch trips it;
    ``enforce=False`` (king pod / king backups) is never refused by the
    breaker — the runway guard holds them too (money safety II)."""
    if isinstance(provider, _BreakerGuardedProvider):
        provider = provider._provider
    return _BreakerGuardedProvider(provider, OperatorBreaker.for_runner(runner), enforce=enforce,
                                   runner=runner)


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


KING_KIND, KING_BACKUP_KIND = "king", "king_backup"
OPERATOR_LEG_KIND, OPERATOR_OTHER_KIND, CPU_VM_KIND = "operator_leg", "operator", "cpu_vm"
# Kinds that are NOT a miner leg rented on the operator's account (the
# daily cap's ``miner_legs`` scope leaves them out).
NON_MINER_KINDS = frozenset({KING_KIND, KING_BACKUP_KIND, CPU_VM_KIND, OPERATOR_OTHER_KIND})


def _scoped(rows: list[dict], scope: str) -> list[dict]:
    """The rows a daily-cap ``scope`` counts (see ``DAILY_CAP_SCOPES``)."""
    if scope == "all_incl_king":
        return rows
    if scope == "all":
        return [r for r in rows if r.get("kind") not in (KING_KIND, KING_BACKUP_KIND)]
    return [r for r in rows if r.get("kind") not in NON_MINER_KINDS]


class OperatorSpend:
    """``<work_root>/operator_spend.json``: operator spend, one row per
    allocation (``pod_id``, landed ``rate``, ``hours`` projected), keyed by UTC
    day for retention.

    Booked at the launch itself (every allocation, lemons too), relabelled
    when an approved miner leg's pod turns out usable, SETTLED to its real
    cost when torn down early and TRUED UP at teardown (keeps, long harvests
    and adopted pods included). The daily cap reads a ROLLING 24 h window,
    and a preflight RESERVES its projection atomically (flock + process lock)
    so two legs can never both pass the same remaining headroom (money
    safety II, audit H); a reservation is released at the launch or on a
    no-rent, and expires by itself after ``RESERVATION_TTL_SECONDS``."""

    WINDOW_SECONDS = 24 * 3600.0
    RESERVATION_TTL_SECONDS = 3600.0

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @classmethod
    def for_runner(cls, runner) -> OperatorSpend:
        return cls(Path(runner.work_root) / SPEND_FILE)

    @contextlib.contextmanager
    def _locked(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, open(self.path.with_suffix(self.path.suffix + ".lock"), "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def _load(self) -> dict:
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        return doc if isinstance(doc, dict) else {}

    def _save(self, doc: dict) -> None:
        write_text_atomic(self.path, json.dumps(doc, sort_keys=True))

    @staticmethod
    def day(now: float) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(now))

    def _window(self, doc: dict, t: float, *, exclude: str = "") -> list[dict]:
        out = []
        for rows in doc.values():
            for r in rows:
                if r.get("reservation"):
                    if r.get("id") != exclude and float(r.get("expires") or 0) > t:
                        out.append(r)
                elif float(r.get("at") or 0) >= t - self.WINDOW_SECONDS:
                    out.append(r)
        return out

    def total(self, now: float | None = None, *, scope: str = "all_incl_king",
              exclude: str = "") -> float:
        """Booked operator spend over the last 24 h (+ live reservations, but
        ``exclude``); ``scope="miner_legs"`` leaves out the king, backups, CPU
        VMs and other operator pods."""
        t = time.time() if now is None else float(now)
        rows = _scoped(self._window(self._load(), t, exclude=exclude), scope)
        return float(sum(float(r.get("usd", 0.0)) for r in rows))

    def reserve_if_room(self, hotkey: str, usd: float, *, cap: float, kind: str,
                        scope: str = "miner_legs", now: float | None = None) -> str | None:
        """Atomically: when the window + ``usd`` fits ``cap`` (0 = no cap),
        record a reservation and return its id; else None."""
        import uuid

        t = time.time() if now is None else float(now)
        with self._locked():
            doc = self._load()
            rows = _scoped([r for r in self._window(doc, t)
                            if not (r.get("reservation") and r.get("hotkey") == hotkey)], scope)
            if cap and sum(float(r.get("usd", 0.0)) for r in rows) + float(usd) > cap:
                return None
            # one live reservation per hotkey: a re-gated retry REPLACES it
            for day in list(doc):
                doc[day] = [r for r in doc[day]
                            if not (r.get("reservation") and r.get("hotkey") == hotkey)]
            rid = f"resv-{uuid.uuid4().hex[:12]}"
            doc.setdefault(self.day(t), []).append({
                "reservation": True, "id": rid, "hotkey": hotkey, "kind": kind,
                "usd": round(float(usd), 2), "at": t,
                "expires": t + self.RESERVATION_TTL_SECONDS})
            self._save(doc)
            return rid

    def release(self, rid: str) -> None:
        if not rid:
            return
        with self._locked():
            doc = self._load()
            n = 0
            for day in list(doc):
                keep = [r for r in doc[day] if r.get("id") != rid]
                n += len(doc[day]) - len(keep)
                doc[day] = keep
            if n:
                self._save(doc)

    def row_for(self, pod_id: str) -> dict | None:
        """The newest booking of ``pod_id`` (any day kept), or None."""
        if not pod_id:
            return None
        hit = self._newest(self._load(), pod_id)
        return dict(hit) if hit else None

    def settle(self, pod_id: str, *, now: float | None = None,
               true_up: bool = False) -> float | None:
        """``pod_id`` ended: book what it really cost — rate x the time it
        lived. Early (a lemon, a rejected pod): never more than projected.
        ``true_up`` (its teardown): the real cost either way, so a kept pod, a
        6 h harvest or an adopted pod that outlived its projection is booked
        in full. Returns the settled USD (None when it was never booked)."""
        t = time.time() if now is None else float(now)
        with self._locked():
            doc = self._load()
            hit = self._newest(doc, pod_id, unsettled=True)
            if hit is None:
                return None
            real = max(0.0, t - float(hit.get("at") or t)) / 3600.0 * float(hit.get("rate") or 0)
            hit["usd"] = round(real if true_up else min(float(hit.get("usd") or 0.0), real), 2)
            hit["settled_at"] = t
            self._save(doc)
            return float(hit["usd"])

    def rows(self, now: float | None = None) -> list[dict]:
        """The booked rows of the rolling 24 h window (reservations excluded)."""
        t = time.time() if now is None else float(now)
        return [r for r in self._window(self._load(), t) if not r.get("reservation")]

    @staticmethod
    def _newest(doc: dict, pod_id: str, *, unsettled: bool = False) -> dict | None:
        hit = None
        for _day, rows in sorted(doc.items()):
            for r in rows:
                if r.get("pod_id") == pod_id and not (unsettled and r.get("settled_at")):
                    hit = r
        return hit

    def charge(self, hotkey: str, usd: float, *, kind: str, projected: float | None = None,
               now: float | None = None, pod_id: str = "", rate: float | None = None,
               hours: float | None = None, relabel: bool = False) -> None:
        """Book ``usd`` (one row per allocation: a re-rent under the same
        deterministic name is a NEW row). ``relabel``: the newest unsettled
        booking of ``pod_id`` (the launch booked it) is RELABELLED (hotkey,
        kind, the landed cost) instead — one allocation never counts twice."""
        t = time.time() if now is None else float(now)
        with self._locked():
            doc = self._load()
            day = self.day(t)
            doc = {k: v for k, v in doc.items() if k >= self.day(t - 14 * 86400)}
            hit = self._newest(doc, pod_id, unsettled=True) if relabel and pod_id else None
            if hit is not None:
                hit.update(hotkey=hotkey or hit.get("hotkey", ""), kind=kind,
                           usd=round(float(usd), 2))
                if rate:
                    hit["rate"] = round(float(rate), 4)
                self._save(doc)
                return
            row = {"hotkey": hotkey, "usd": round(float(usd), 2), "kind": kind, "at": t,
                   "projected_usd": round(float(usd if projected is None else projected), 2)}
            if pod_id:
                row["pod_id"] = pod_id
            if rate is not None:
                row["rate"] = round(float(rate), 4)
            if hours is not None:
                row["hours"] = round(float(hours), 3)
            doc.setdefault(day, []).append(row)
            self._save(doc)


def ceiling_rate(runner) -> float:
    """The operator's $/h ceiling (the projection rate when no price is known)."""
    from ..provision.operator_pool import operator_max_price_per_hour

    try:
        return float(operator_max_price_per_hour(runner.cfg.round) or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def era_wall_seconds(rnd) -> float:
    """One era's wall: epoch_blocks x era_settlements x the block time."""
    from ..shared.config import BLOCK_SECONDS

    try:
        return (max(1, int(getattr(rnd, "epoch_blocks", 0) or 0))
                * max(1, int(getattr(rnd, "era_settlements", 0) or 0)) * float(BLOCK_SECONDS))
    except Exception:  # noqa: BLE001
        return 12 * 3600.0


def _leg_life_seconds(runner) -> float:
    rnd = runner.cfg.round
    try:
        wall = float(runner._leg_wall_seconds())
    except Exception:  # noqa: BLE001 — unknown wall: a full day (strict)
        wall = 24 * 3600.0
    return wall + float(getattr(rnd, "funded_ready_timeout_seconds", 0.0) or 0.0)


def expected_pod_hours(runner, kind: str) -> float:
    """How long an operator allocation of ``kind`` is expected to bill: a
    king pod / backup lives for the era, everything else for one leg."""
    if kind in (KING_KIND, KING_BACKUP_KIND):
        return max(era_wall_seconds(runner.cfg.round), _leg_life_seconds(runner)) / 3600.0
    return _leg_life_seconds(runner) / 3600.0


def book_operator_allocation(runner, provider, pod_id: str, *, kind: str, hours: float,
                             hotkey: str = "") -> None:
    """Book one operator allocation at its REAL price (the provider's landed
    $/h, else the ceiling) x its expected hours. Never raises."""
    try:
        rate = None
        price_fn = getattr(provider, "price_of", None)
        if callable(price_fn):
            with contextlib.suppress(Exception):
                rate = price_fn(pod_id)
        rate = float(rate or ceiling_rate(runner) or 0.0)
        OperatorSpend.for_runner(runner).charge(hotkey, rate * hours, kind=kind, pod_id=pod_id,
                                                rate=rate, hours=hours)
    except Exception as e:  # noqa: BLE001 — bookkeeping never blocks a rent
        log.warning("operator spend: allocation %s not booked (%s)", pod_id, e)


def settle_operator_pod(runner, pod_id: str, *, true_up: bool = False) -> None:
    """``pod_id`` (an operator lemon / rejected pod, or ``true_up``: any
    operator pod at its teardown) is being torn down: its booking becomes its
    real cost. Never raises."""
    try:
        OperatorSpend.for_runner(runner).settle(pod_id, true_up=true_up)
    except Exception as e:  # noqa: BLE001
        log.debug("operator spend: settle of %s skipped (%s)", pod_id, e)


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


def operator_rent_preflight(runner, gen, cost_usd: float) -> str | None:
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
    spent = OperatorSpend.for_runner(runner).total(
        scope=str(getattr(rnd, "operator_daily_cap_scope", "miner_legs") or "miner_legs"))
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
    # Money safety II (audit H): the check above and this reservation are one
    # atomic step under the spend ledger's flock — two legs can never both
    # pass the same remaining headroom. Released at the launch (which books
    # the real allocation) or on a no-rent; expires by itself.
    rid = OperatorSpend.for_runner(runner).reserve_if_room(
        hk, cost_usd, cap=cap, kind="reservation",
        scope=str(getattr(rnd, "operator_daily_cap_scope", "miner_legs") or "miner_legs"))
    if rid is None:
        alert(runner, "operator_daily_cap", "operator",
              f"daily cap ${cap:.2f} reached by a concurrent reservation")
        _skip(runner, hk, f"operator daily spend cap reached (a concurrent leg reserved the "
              f"headroom; ${cap:.2f}); nothing rented", cls=OPERATOR_BLOCKED_CLASS)
    return rid


def release_reservation(runner) -> None:
    """The approved leg will not rent now (no capacity, a refusal): free its
    daily-cap reservation. Never raises."""
    a = getattr(_leg_local(runner), "operator_approval", None) or {}
    rid = a.get("reservation") or ""
    if rid:
        with contextlib.suppress(Exception):
            OperatorSpend.for_runner(runner).release(rid)
        a["reservation"] = ""


# ── the operator RUNWAY guard (money safety II) ─────────────────────────────
# 2026-11-02 the balance gate was point-in-time: it ignored what the pods
# already running would still bill (king, backups, legs, eval pods), so the
# account drained to 402 and Lium killed every pod on it. The runway guard
# counts that committed spend before ANY new operator rent.

RUNWAY_FILE = "operator_runway.json"


def _parse_iso(text: str) -> float | None:
    import calendar

    try:
        return float(calendar.timegm(time.strptime(str(text), "%Y-%m-%dT%H:%M:%SZ")))
    except (TypeError, ValueError):
        return None


def committed_spend(runner, *, now: float | None = None) -> dict:
    """``{committed_usd, burn_usd_per_hour, pods: [...]}`` — every live
    operator pod (funded-ledger rows with no payer: king pods, backups,
    operator legs) at its booked rate (else the ceiling) x its remaining
    expected wall, plus each standing pod (``[round]
    operator_standing_pods_usd_per_hour``: eval pods, hand-rented lanes) at
    its rate x the alert horizon (they run at least that long)."""
    t = time.time() if now is None else float(now)
    rnd = runner.cfg.round
    spend = OperatorSpend.for_runner(runner)
    ceiling = ceiling_rate(runner)
    pods, committed, burn = [], 0.0, 0.0
    try:
        ledger = list(runner._load_funded_ledger())
    except Exception:  # noqa: BLE001 — unreadable: nothing extra counted (logged)
        log.warning("runway: funded ledger unreadable — committed spend undercounted")
        ledger = []
    for pod in ledger:
        if getattr(pod, "payer_hotkey", ""):
            continue                                    # a miner's own key pays it
        pid = str(getattr(pod, "instance_id", "") or "")
        row = spend.row_for(pid) or {}
        kind = row.get("kind") or launch_kind(pid)
        rate = float(row.get("rate") or ceiling)
        start = float(row.get("at") or _parse_iso(getattr(pod, "rented_at_iso", "")) or t)
        hours = float(row.get("hours") or expected_pod_hours(runner, kind))
        left = max(0.0, start + hours * 3600.0 - t) / 3600.0
        committed += rate * left
        burn += rate
        pods.append({"pod": pid, "rate": rate, "hours_left": round(left, 2), "kind": kind})
    horizon = float(getattr(rnd, "operator_runway_alert_hours", 0.0) or 0.0)
    for name, rate in tuple(getattr(rnd, "operator_standing_pods_usd_per_hour", ()) or ()):
        committed += float(rate) * horizon
        burn += float(rate)
        pods.append({"pod": name, "rate": float(rate), "hours_left": horizon, "kind": "standing"})
    return {"committed_usd": round(committed, 2), "burn_usd_per_hour": round(burn, 4),
            "pods": pods}


def _read_balance(runner) -> float | None:
    reader = getattr(runner, "_operator_balance_reader", None)
    if reader is None:
        from .operator_health import lium_balance as reader
    return _balance(reader)


def operator_runway(runner, *, balance: float | None, now: float | None = None,
                    extra_usd: float = 0.0) -> dict:
    """The runway verdict. ``held`` = balance - committed (- ``extra_usd``,
    a launch about to happen) < the reserve: no new operator rent of any
    kind. ``runway_hours`` = (balance - committed - reserve) / burn rate
    (inf with nothing burning). An unknown balance is never ``held`` here
    (the miner-leg preflight already fails closed on it)."""
    c = committed_spend(runner, now=now)
    reserve = float(getattr(runner.cfg.round, "operator_balance_reserve_usd", 0.0) or 0.0)
    out = {**c, "balance_usd": balance, "reserve_usd": reserve, "held": False,
           "runway_hours": None}
    if balance is None:
        return out
    free = float(balance) - c["committed_usd"] - reserve
    out["held"] = float(balance) - c["committed_usd"] - float(extra_usd) < reserve
    burn = c["burn_usd_per_hour"]
    out["runway_hours"] = (round(free / burn, 2) if burn > 0
                           else (float("inf") if free >= 0 else 0.0))
    return out


def operator_launch_check(runner, *, kind: str, projected_usd: float) -> None:
    """Right before EVERY operator launch (the king's included): the runway
    guard (balance - committed - this launch < reserve ⇒ refused) and the
    daily cap (over its scope). A refusal raises
    :class:`~cascade.provision.core.OperatorRentsHeld` with one critical alert;
    the state clears by itself once the balance / the day recovers."""
    from ..provision.core import OperatorRentsHeld

    try:
        _operator_launch_check(runner, kind=kind, projected_usd=projected_usd)
    except OperatorRentsHeld:
        raise
    except Exception as e:  # noqa: BLE001 — a bookkeeping fault never blocks a launch
        log.warning("operator launch check unavailable for a %s launch (%s) — the rent's "
                    "other gates decide", kind, e)


def _operator_launch_check(runner, *, kind: str, projected_usd: float) -> None:
    from ..provision.core import OperatorRentsHeld

    rnd = runner.cfg.round
    scope = str(getattr(rnd, "operator_daily_cap_scope", "miner_legs") or "miner_legs")
    cap = daily_cap_usd(rnd)
    # The king pod and its backups COUNT toward the cap ("all_incl_king") but are
    # never refused by it: one king era at the ceiling exceeds any sane daily
    # cap and a refused king stalls every settlement. The runway guard below
    # is what protects the account from the king (backups also keep their
    # own ops_king_backup_cap_usd).
    if cap and kind not in (KING_KIND, KING_BACKUP_KIND) and (
            scope in ("all", "all_incl_king") or kind not in NON_MINER_KINDS):
        mine = (getattr(_leg_local(runner), "operator_approval", None) or {}).get(
            "reservation", "")
        spent = OperatorSpend.for_runner(runner).total(scope=scope, exclude=mine)
        if spent + projected_usd > cap:
            msg = (f"operator daily cap: today ${spent:.2f} + this {kind} launch "
                   f"≈${projected_usd:.2f} > ${cap:.2f} ({scope}) — no new operator rent")
            alert(runner, "operator_daily_cap", "operator", msg, severity="critical")
            raise OperatorRentsHeld(msg)
    bal = _read_balance(runner)
    if bal is None:
        log.warning("runway: operator balance unknown at a %s launch — not held here (the "
                    "miner-leg preflight fails closed on it)", kind)
        return
    king = kind in (KING_KIND, KING_BACKUP_KIND)
    rw = operator_runway(runner, balance=bal, extra_usd=0.0 if king else projected_usd)
    if king and not rw["held"]:
        short = operator_runway(runner, balance=bal, extra_usd=projected_usd)
        if short["held"]:
            # review M1: a king whose own era projection does not fit is
            # ALERTED, never held — only balance - committed < reserve holds it
            alert(runner, "operator_runway_king", "operator",
                  f"king launch ≈${projected_usd:.2f} does not fit the operator balance "
                  f"${bal:.2f} - committed ${rw['committed_usd']:.2f} - reserve "
                  f"${rw['reserve_usd']:.2f} — renting anyway (the king is required); top up",
                  severity="critical")
        return
    if rw["held"]:
        msg = (f"NO NEW OPERATOR RENTS: balance ${bal:.2f} - committed "
               f"${rw['committed_usd']:.2f} - this {kind} launch ≈${projected_usd:.2f} < the "
               f"${rw['reserve_usd']:.2f} reserve (king included) — top up the operator account;"
               " clears by itself")
        alert(runner, "operator_rents_held", "operator", msg, severity="critical")
        _save_runway(runner, rw)
        raise OperatorRentsHeld(msg)


def _save_runway(runner, rw: dict) -> None:
    with contextlib.suppress(Exception):
        doc = {**{k: v for k, v in rw.items() if k != "pods"}, "at": time.time(),
               "runway_hours": (None if rw.get("runway_hours") in (None, float("inf"))
                                else rw["runway_hours"])}
        write_text_atomic(Path(runner.work_root) / RUNWAY_FILE, json.dumps(doc, sort_keys=True))


def runway_tick(runner, balance: float | None) -> dict:
    """The periodic half (the operator-health sampler): one critical alert
    while the runway is under ``operator_runway_alert_hours``; the held
    state alert while balance - committed < reserve; both clear by
    themselves. Returns the verdict."""
    rw = operator_runway(runner, balance=balance)
    alert_h = float(getattr(runner.cfg.round, "operator_runway_alert_hours", 0.0) or 0.0)
    hrs = rw.get("runway_hours")
    if balance is not None and alert_h and hrs is not None and hrs < alert_h:
        alert(runner, "operator_runway_low", "operator",
              f"operator runway {hrs:.1f} h < {alert_h:.0f} h: balance ${balance:.2f}, committed "
              f"${rw['committed_usd']:.2f} by {len(rw['pods'])} pod(s), burn "
              f"${rw['burn_usd_per_hour']:.2f}/h, reserve ${rw['reserve_usd']:.2f}",
              severity="critical")
    else:
        clear_alert(runner, "operator_runway_low", "operator")
    if rw["held"]:
        alert(runner, "operator_rents_held", "operator",
              f"NO NEW OPERATOR RENTS (king included): balance ${balance:.2f} - committed "
              f"${rw['committed_usd']:.2f} < reserve ${rw['reserve_usd']:.2f} — top up; "
              "clears by itself", severity="critical")
    elif balance is not None:
        clear_alert(runner, "operator_rents_held", "operator")
    _save_runway(runner, rw)
    return rw


def would_rent_refusal(runner, gen, *, operator: bool) -> tuple[str, str]:
    """``(why, error_class)`` — "" when a rent for ``gen``'s leg could go
    ahead now. A side-effect-free verdict over the money / box gates the
    rent itself applies (no rent, no approval consumed, no probe, no alert),
    run BEFORE the leg's corpus is generated so no CPU VM is paid for a leg
    that cannot rent (money safety II). ``operator``: the leg is billed to
    the operator's account (breaker, runway, daily cap, balance apply)."""
    from ..funding.queue import OPERATOR_DISK_CLASS
    from .box_health import rent_halt_refusal
    from .cost_guards import doomed_leg_reason, measured_wall_refusal
    from .leg_pins import LegImageMismatch, leg_round

    try:
        # DEC-CA-0067: the leg's era pins (the after view for a post-switch
        # leg; ``runner.cfg.round`` itself otherwise — byte-identical until
        # the switch). An unqualified after release is a refusal here too.
        try:
            rnd = leg_round(runner)
        except LegImageMismatch as e:
            return (f"operator-side fault: {e} — nothing rented", OPERATOR_BLOCKED_CLASS)
        why = measured_wall_refusal(rnd)
        if why:
            return why, OPERATOR_BLOCKED_CLASS
        why = rent_halt_refusal(runner, king=False, cached_only=True)
        if why:
            return f"operator disk: {why} — nothing rented", OPERATOR_DISK_CLASS
        skus = tuple(runner._funded_skus_for_rent() or ())
        why = doomed_leg_reason(runner, skus) if skus else ""
        if why:
            return why, "no_capacity"
        if not operator:
            return "", ""
        rec = OperatorBreaker.for_runner(runner).open_record()
        if rec:
            return (f"operator rents paused (circuit breaker: {rec.get('reason', '')}); nothing "
                    "rented, nothing burned", OPERATOR_BLOCKED_CLASS)
        hours = expected_pod_hours(runner, OPERATOR_LEG_KIND)
        projected = ceiling_rate(runner) * hours
        cap = daily_cap_usd(rnd)
        if cap:
            spent = OperatorSpend.for_runner(runner).total(
                scope=str(getattr(rnd, "operator_daily_cap_scope", "miner_legs") or "miner_legs"))
            if spent + projected > cap:
                return (f"operator daily spend cap reached (${spent:.2f} + ${projected:.2f} > "
                        f"${cap:.2f}); nothing rented", OPERATOR_BLOCKED_CLASS)
        bal = _read_balance(runner)
        # unknown: not refused HERE — the gate's preflight fails closed on it
        rw = operator_runway(runner, balance=bal, extra_usd=projected)
        if bal is not None and rw["held"]:
            return (f"no new operator rents: balance ${bal:.2f} - committed "
                    f"${rw['committed_usd']:.2f} - ${projected:.2f} < reserve "
                    f"${rw['reserve_usd']:.2f}", OPERATOR_BLOCKED_CLASS)
    except Exception as e:  # noqa: BLE001 — unsure ⇒ let the rent's own gates decide
        log.warning("would-rent verdict for %s unavailable (%s) — the rent's gates decide",
                    str(getattr(gen, "hotkey", ""))[:12], e)
    return "", ""


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


def approval_on_file(runner, hotkey: str) -> bool:
    """An owner approval (any kind; unlaunched, or launched for THIS attempt)
    bound to ``hotkey``'s current leg — what lets it rent on the operator's
    account (audit E)."""
    try:
        queue = runner._funded_queue()
        entry = queue.get(hotkey) if queue is not None and hotkey else None
        if entry is None:
            return False
        from ..funding.operator_approvals import OperatorApprovals

        store = OperatorApprovals.for_work_root(runner.work_root)
        verdict = store.check(Path(queue.path).name, hotkey, ref=entry.ref,
                              attempt_id=str(getattr(entry, "attempt_id", "") or ""), kind="",
                              funded_at=entry.funded_at)
        return verdict == "approved"
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


def operator_leg_gate(runner, gen, *, kind: str, rent: bool, lost_usd: float = 0.0) -> None:
    """Gate EVERY miner leg about to be billed to the operator (``kind``:
    ``expired_key`` / ``make_good`` / ``operator_lane`` / ``paid_fallback``).
    A retrain after a finished run was lost on our side is a ``make_good``
    with ``lost_usd`` (money safety II: OPERATOR-billed, never on the
    miner's key). ``rent`` = a new operator-account pod (money preflight +
    spend booking at launch), else no new operator spend.

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
    if entry is not None and entry.status in ("withdrawn", "done", "failed"):
        # money safety II: a withdrawn / terminal submission never gets an
        # operator request (2026-11-03: a withdrawn entry with no key filed
        # an "expired key" operator rent request)
        _skip(runner, hk, "the submission is withdrawn or settled — nothing is rented on the "
              "operator's account", cls="withdrawn")
    attempt_id = str(getattr(entry, "attempt_id", "") or "")
    funded_at = getattr(entry, "funded_at", None)
    sku, rate, hours, cost = projected_operator_leg_cost(runner, live_price=bool(rent))
    lost = float(lost_usd or 0.0)
    if not rent:
        cost = 0.0          # an operator lane is already paid for; no new spend
    store = OperatorApprovals.for_work_root(runner.work_root)
    qname = Path(queue.path).name
    verdict = store.check(qname, hk, ref=gen.ref, attempt_id=attempt_id, kind=kind,
                          funded_at=funded_at, projected_cost_usd=cost)
    if verdict == "approved":
        rid = operator_rent_preflight(runner, gen, cost) if rent else None  # keeps approval
        _leg_local(runner).operator_approval = {
            "queue": qname, "hotkey": hk, "ref": gen.ref, "attempt_id": attempt_id,
            "cost": cost, "hours": hours, "kind": kind, "book": bool(rent),
            "reservation": rid or ""}
        if not rent:
            note_operator_launch(runner, hk)           # a lane dispatch IS the launch
        log.warning("operator-billed leg %s (%s, uid %s): owner approval applies — %s, "
                    "projected ≤ $%.2f (%s, $%.2f/h x %.1f h)", hk[:12], kind,
                    getattr(gen, "uid", "?"),
                    "renting a pod" if rent else "no new operator rent", cost, sku, rate, hours)
        return
    if verdict == "denied":
        store.take_denial(qname, hk)
        if kind == "expired_key" or lost:
            # A declined retrain after OUR loss never falls back to the
            # miner's key (money safety II): the miner decides by re-funding.
            _skip(runner, hk, ("payer key expired and the operator declined to pay for this "
                               "leg — fund again with a fresh key (no attempt charged)"
                               if kind == "expired_key" else
                               "the operator declined the retrain after a lost run — fund "
                               "again to retrain (no attempt charged)"),
                  cls="auth", miner_fault=True)
        _skip(runner, hk, "the operator declined to pay for this leg; it retries on the "
              "miner's own key (no attempt charged)", cls=DENIED_CLASS)
    reason = {"expired_key": "payer key expired",
              "make_good": ("operator make-good: retrain after a finished run was lost on "
                            "our side" if lost else
                            "operator make-good after an operator-side loss"),
              "operator_lane": "no marketplace capacity: an operator lane",
              "paid_fallback": "ops_paid_fallback: marketplace sold out",
              "retrain_after_loss": "retrain after a finished run was lost on our side"
              }.get(kind, kind)
    store.request(qname, hk, ref=gen.ref, attempt_id=attempt_id, queue_path=str(queue.path),
                  uid=getattr(gen, "uid", None), sku=sku, rate_per_hour=rate,
                  expected_hours=hours, projected_cost_usd=cost, kind=kind, reason=reason,
                  funded_at=funded_at, miner_lost_usd=lost,
                  last_error=str(getattr(entry, "last_error", "") or ""),
                  error_class=str(getattr(entry, "last_error_class", "") or ""),
                  evidence=dict(getattr(_leg_local(runner), "loss_evidence", None) or {}))
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


def gate_or_adopt(runner, round_id: str, gen, *, kind: str, rent: bool,
                  lost_usd: float = 0.0) -> None:
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
    operator_leg_gate(runner, gen, kind=kind, rent=rent, lost_usd=lost_usd)


def take_deferred_gate(runner) -> str | None:
    loc = _leg_local(runner)
    kind = getattr(loc, "operator_gate_deferred", None)
    loc.operator_gate_deferred = None
    return kind


def note_operator_launch(runner, hotkey: str | None = None, *,
                         landed_rate: float | None = None, pod_id: str = "") -> bool:
    """The approved leg's pod was rented (or its lane dispatched): mark the
    approval launched for THIS attempt and book the operator spend — only
    now, never for a rent that did not happen (review F3)."""
    loc = _leg_local(runner)
    a = getattr(loc, "operator_approval", None)
    if not a:
        return True
    loc.operator_approval = None
    if a.get("reservation"):
        with contextlib.suppress(Exception):
            OperatorSpend.for_runner(runner).release(a["reservation"])
    if hotkey is not None and a["hotkey"] != hotkey:
        return True                   # a stale approval of another leg on this thread
    from ..funding.operator_approvals import OperatorApprovals

    # Book the LANDED $/h × the expected wall when the rent price is known
    # (the projection was an upper bound; review R4).
    booked = (float(landed_rate) * float(a.get("hours") or 0.0) if landed_rate
              else float(a["cost"]))
    store = OperatorApprovals.for_work_root(runner.work_root)
    rec = store.get(a["queue"], a["hotkey"]) or {}
    ceiling = rec.get("max_cost_usd")
    if ceiling is not None and booked > float(ceiling):
        # money safety II: the LANDED cost broke the approval's ceiling — not
        # launched under it; the owner decides again at the real cost
        store.require_reapproval(a["queue"], a["hotkey"], cost_usd=booked, reason=(
            f"landed ${booked:.2f} > the approved ceiling ${float(ceiling):.2f} — re-approve"))
        alert(runner, "operator_leg_over_ceiling", a["hotkey"][:16],
              f"{a['hotkey'][:12]}: landed ≈${booked:.2f} > approved ceiling "
              f"${float(ceiling):.2f}; pod released, re-approval needed", severity="critical")
        return False
    store.mark_launched(
        a["queue"], a["hotkey"], ref=a["ref"], attempt_id=a["attempt_id"], cost_usd=booked)
    if not a.get("book"):
        return True
    # ``pod_id``: the launch already booked this allocation (every operator
    # launch is booked by the guarded provider) — relabel it to this leg at
    # its landed cost instead of booking it twice.
    OperatorSpend.for_runner(runner).charge(a["hotkey"], booked, kind=a["kind"],
                                            projected=float(a["cost"]), pod_id=pod_id,
                                            rate=landed_rate, relabel=bool(pod_id))
    return True


__all__ = ["AWAITING_APPROVAL_STATUS", "OperatorBreaker", "OperatorSpend", "alert",
           "BookedCpuProvider", "book_operator_allocation", "committed_spend",
           "operator_launch_check",
           "operator_runway", "release_reservation", "runway_tick", "settle_operator_pod",
           "would_rent_refusal",
           "clear_alert", "daily_cap_usd", "guard_operator_provider", "health_conditions",
           "is_balance_refusal", "operator_leg_gate", "operator_rent_preflight",
           "gate_or_adopt", "grandfathered", "note_operator_launch", "operator_leg_allocated",
           "take_deferred_gate", "payer_balance_refusal", "preapproved",
           "projected_operator_leg_cost"]
