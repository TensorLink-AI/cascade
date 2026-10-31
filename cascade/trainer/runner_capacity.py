"""Funded-rent capacity, SKU walls and latest-safe-start deadlines (TrainerRunner mixin)."""

from __future__ import annotations

import logging
import time

from ..funding.queue import OPERATOR_FAULT_RETRIES
from .lane_pool import _lium_provider

log = logging.getLogger("cascade.trainer")


class _FundedCapacityMixin:
    """Marketplace capacity waits, per-SKU walls and rent deadlines (DEC-CA-0036)."""

    def _sku_per_leg_active(self) -> bool:
        """``[round] funded_sku_per_leg``: every leg rents the cheapest listed
        GPU type that still fits instead of one locked type. Safe because
        rolling runs only past ``era_king_from_block``, where validators lift
        the gpu_mismatch gate on mixed types."""
        return bool(getattr(self.cfg.round, "funded_sku_per_leg", False))

    # Seconds between marketplace polls while a funded/king rent waits for a GPU.
    FUNDED_RENT_RETRY_SECONDS = 90.0
    # Wall-clock the manifest needs after the last leg ends (push + sign + publish).
    FUNDED_PUBLISH_MARGIN_SECONDS = 900.0
    # Consecutive rented pods that booted a stale worker image (pinned tag,
    # wrong code) before the leg gives up for the round as an infra fault.
    FUNDED_MAX_STALE_PODS = 3
    OPERATOR_FAULT_RETRIES = OPERATOR_FAULT_RETRIES

    def _funded_price_caps(self) -> dict:
        """The ``[round]`` price guards as ``apply_price_caps`` kwargs."""
        rnd = self.cfg.round
        try:
            default = max(int(c.max_train_seconds) for c in self.cfg.throne_contracts())
        except Exception:  # noqa: BLE001 — offline fakes
            default = 0
        contract = getattr(self.__dict__.get("_leg_local_obj"), "funded_contract", None)
        if contract is not None:
            default = int(getattr(contract, "max_train_seconds", default))
        return {
            "max_price_per_hour": float(getattr(rnd, "funded_max_price_per_hour", 0.0) or 0.0),
            "max_leg_cost_usd": float(getattr(rnd, "funded_max_leg_cost_usd", 0.0) or 0.0),
            "sku_wall_seconds": tuple(getattr(rnd, "funded_sku_wall_seconds", ()) or ()),
            "default_wall_seconds": float(default),
        }

    def _chosen_gpu_rent(self, choice, describe: str = "") -> tuple[tuple[str, ...], dict] | None:
        """DEC-CA-0060: ``((sku,), price_caps)`` for a miner-chosen GPU type,
        or None for the default path (no choice; a locked-type network; a
        type no longer in ``[round] funded_pod_skus`` — logged, the default
        path rents instead). The MINER's $/h cap replaces the operator caps
        (``funded_max_price_per_hour`` / ``funded_max_leg_cost_usd`` guard
        the default choice only); the wall table stays, for listing order."""
        gpu = str(getattr(choice, "gpu", "") or "")
        if not gpu:
            return None
        who = describe or "funded leg"
        if not self._sku_per_leg_active():
            log.warning("%s: chosen GPU %s ignored — [round] funded_sku_per_leg is off "
                        "(one locked type)", who, gpu)
            return None
        listed = tuple(getattr(self.cfg.round, "funded_pod_skus", ()) or ())
        sku = next((s for s in listed if s.lower() == gpu.lower()), None)
        if sku is None:
            log.warning("%s: chosen GPU %s is no longer in [round] funded_pod_skus — "
                        "renting the default (cheapest fitting) type", who, gpu)
            return None
        caps = dict(self._funded_price_caps())
        caps["max_price_per_hour"] = float(getattr(choice, "max_price_per_hour", 0.0) or 0.0)
        caps["max_leg_cost_usd"] = 0.0
        return (sku,), caps

    def _funded_skus_for_rent(self) -> tuple[str, ...]:
        """GPU types a rent may pick: the locked ``funded_pod_sku``, or — with
        ``[round] funded_sku_per_leg`` (the open market) — every listed type,
        cheapest fitting executor first."""
        rnd = self.cfg.round
        if self._sku_per_leg_active():
            skus = tuple(getattr(rnd, "funded_pod_skus", ()) or ())
            if skus:
                return skus
        locked = rnd.funded_pod_sku
        return (locked,) if locked else ()

    def _leg_wall_seconds(self, sku: str | None = None) -> float:
        """Wall one full-budget leg needs on ``sku``: the measured
        ``[round] funded_sku_wall_seconds`` entry, else the contract's
        ``max_train_seconds``. ``sku`` None/"" = the contract cap, except in
        per-leg mode where it is the FASTEST listed type (the last moment any
        rent could still fit; the rent itself picks only types that fit)."""
        from ..shared.config import funded_sku_wall_for

        rnd = self.cfg.round
        contract = getattr(self.__dict__.get("_leg_local_obj"), "funded_contract", None)
        cap = (int(contract.max_train_seconds) if contract is not None else
               max(int(c.max_train_seconds) for c in self.cfg.throne_contracts()))
        walls = tuple(getattr(rnd, "funded_sku_wall_seconds", ()) or ())
        if sku:
            return min(float(cap), funded_sku_wall_for(walls, sku, cap))
        if self._sku_per_leg_active() and walls:
            skus = self._funded_skus_for_rent()
            if skus:
                return min(float(cap), *(funded_sku_wall_for(walls, s, cap) for s in skus))
        return float(cap)

    def _funded_epoch_end_known(self) -> bool:
        """Whether the calling leg carries its target boundary (the
        thread-local ``end_wall`` every rolling leg sets): without it there is
        no latest safe start, so the per-SKU fit filter and the lane deadline
        stay off."""
        return getattr(getattr(self, "_leg_local", None), "end_wall", None) is not None

    def _skus_fitting_now(self, skus: tuple[str, ...]) -> tuple[str, ...]:
        """Of ``skus``, the types a leg started NOW would still finish inside
        the epoch (per-SKU latest safe start). Only per-leg mode filters —
        a locked round keeps its single type and the wait decides."""
        if not self._sku_per_leg_active() or not self._funded_epoch_end_known():
            return tuple(skus)
        now_fn = getattr(self, "_rent_wait_now", None) or time.time
        now = now_fn()
        return tuple(s for s in skus if now < self._funded_rent_wait_deadline_for(s))

    def _provider_capacity(self, provider, skus, exclude_ids=()) -> int:
        """Marketplace capacity summed over ``skus`` (adapters count per type)."""
        total = 0
        for s in skus:
            try:
                total += int(provider.capacity(s, exclude_ids=exclude_ids) or 0)
            except TypeError:
                total += int(provider.capacity(s) or 0)
        return total

    def _funded_rent_wait_deadline_for(self, sku: str | None, *,
                                       unknown_is_cap: bool = False) -> float:
        """The latest safe start for a leg on ``sku`` specifically: the
        round-wide deadline shifted by how much faster (or slower) that type
        is than the wall the round-wide figure assumes (``_leg_wall_seconds``
        with no SKU). No ``sku`` = the round-wide figure, or with
        ``unknown_is_cap`` the contract-cap figure (a lane of unknown type
        is assumed slowest). Unit fakes that stub the round-wide method keep
        working."""
        base = self._funded_rent_wait_deadline()
        if not sku and not unknown_is_cap:
            return base
        try:
            wall = (self._leg_wall_seconds(sku) if sku
                    else float(max(int(c.max_train_seconds)
                                   for c in self.cfg.throne_contracts())))
            return base + (self._leg_wall_seconds(None) - wall)
        except Exception:  # noqa: BLE001 — no contract context ⇒ the round-wide figure
            return base

    def _funded_rent_wait_deadline(self) -> float:
        """Latest wall-clock a rented leg may START and still land before its
        target boundary: the leg's ``end_wall`` − its wall (the contract's
        ``max_train_seconds``; in per-leg mode the FASTEST listed type's
        measured wall — see ``_leg_wall_seconds``) − publish margin. "Now"
        (no waiting) when the leg carries no target."""
        now = time.time()
        try:
            end_wall = getattr(self._leg_local, "end_wall", None)
            if end_wall is None:
                return now
            leg_s = int(self._leg_wall_seconds(None))
            return end_wall - leg_s - self.FUNDED_PUBLISH_MARGIN_SECONDS
        except Exception:  # noqa: BLE001 — a broken estimate must never hang a leg
            return now

    def _operator_lane_deadline(self, sku: str | None = None, *,
                                unknown_is_cap: bool = False) -> float | None:
        """Latest wall-clock an operator-lane FALLBACK leg may start — the same
        latest safe start as a marketplace rent — or None when the round's
        epoch end is unknown (then the lane wait is unbounded, as before)."""
        if not self._funded_epoch_end_known():
            return None
        return self._funded_rent_wait_deadline_for(sku, unknown_is_cap=unknown_is_cap)

    def _operator_lane_deadline_fn(self):
        """The lane pool's deadline argument: per-lane (``host.sku`` through
        the measured wall table) when ``funded_sku_wall_seconds`` is set,
        else the single round-wide number (or None = unbounded)."""
        if not self._funded_epoch_end_known():
            return None
        if tuple(getattr(self.cfg.round, "funded_sku_wall_seconds", ()) or ()):
            return lambda host: self._operator_lane_deadline(
                getattr(host, "sku", "") or None, unknown_is_cap=True)
        return self._operator_lane_deadline()

    def _operator_fallback_eligible(self, hotkey: str | None, now: float,
                                    deadline: float) -> bool:
        """Lium first, operator lanes for the spillover: may this waiting leg
        take an operator lane NOW? The JIT king (``hotkey`` None) always may.
        A funded entry REQUEUED after a failed leg (any recorded fault class)
        may at once; a FRESH entry
        keeps polling the marketplace (payer-billed) until the last
        ``funded_operator_fallback_fresh_window_seconds`` before the latest
        safe start. Window 0 ⇒ everyone at once."""
        if hotkey is None:
            return True
        window = int(getattr(self.cfg.round,
                             "funded_operator_fallback_fresh_window_seconds", 0) or 0)
        if window <= 0:
            return True
        carried = False
        try:
            queue = self._funded_queue()
            entry = queue.get(hotkey) if queue is not None else None
            carried = entry is not None and (
                int(getattr(entry, "attempts", 0) or 0) > 0
                or bool(getattr(entry, "last_error_class", "") or ""))
        except Exception as e:  # noqa: BLE001 — a torn queue read must not block the leg
            log.debug("fallback eligibility for %s: queue read failed (%s)", hotkey[:12], e)
        return carried or now >= deadline - window

    def _wait_for_funded_capacity(self, sku: str, *, describe: str,
                                  for_king: bool = False,
                                  hotkey: str | None = None,
                                  price_caps: dict | None = None,
                                  capacity_probe=None, allow_paid_fallback=False) -> bool:
        """Poll the marketplace for ``sku`` until it shows capacity or the leg's
        latest safe start passes. Legs start INDEPENDENTLY as GPUs appear; a
        leg still without a GPU at the deadline requeues. Returns True when
        capacity appeared, ``"operator"`` when the leg should take an operator
        lane instead, False otherwise. ``price_caps`` (a miner-chosen type,
        DEC-CA-0060) replaces the operator caps on the probe."""
        skus = (sku,) if isinstance(sku, str) else tuple(sku)
        sku = "/".join(skus)
        deadline = self._funded_rent_wait_deadline_for(skus[0] if len(skus) == 1 else None)
        sleep = getattr(self, "_rent_wait_sleep", None) or time.sleep
        now_fn = getattr(self, "_rent_wait_now", None) or time.time
        abort = getattr(self, "_funded_wait_abort", None)
        if for_king or getattr(self.__dict__.get("_leg_local_obj"), "funded_is_king", False):
            # The abort releases CHALLENGERS when the king cannot be rented; a
            # king leg (operator or payer-funded) never cancels on it — a stale
            # set from an earlier attempt would otherwise requeue the king.
            abort = None
        if for_king and self._operator_fallback_lanes():
            # The king is REQUIRED and never held back: with operator lanes
            # on file it takes one at once, latest safe start or not — a late
            # king costs a settlement, no king costs the era.
            log.info("%s: operator final lane(s) on file — the king takes one now "
                     "(operator-billed; never held back by the latest safe start)",
                     describe)
            return "operator"
        polled = 0
        while now_fn() < deadline:
            if abort is not None and abort.is_set():
                log.info("%s: capacity wait cancelled (the king cannot be rented "
                         "this attempt)", describe)
                return False
            # Hybrid fallback: operator lanes on file end the wait — the leg
            # runs there, operator-billed. Fresh legs keep polling the
            # marketplace until the fresh window (Lium first).
            if (self._operator_fallback_lanes()
                    and self._operator_fallback_eligible(hotkey, now_fn(), deadline)):
                log.info("%s: operator final lane(s) on file — leaving the %s wait "
                         "for an operator lane (operator-billed)", describe, sku)
                return "operator"
            if capacity_probe is not None:
                try:
                    n = capacity_probe(self._skus_fitting_now(skus) or skus)
                except Exception:
                    n = None
            else:
                n = self._probe_funded_capacity(self._skus_fitting_now(skus) or skus,
                                                self._claimed_executors(),
                                                **({"price_caps": price_caps}
                                                   if price_caps is not None else {}))
            if n and not for_king and self._king_pending():
                # Capacity is the king's first: keep polling until it has a pod.
                if not polled:
                    log.info("%s: %s capacity (%d) but the king still needs a pod — "
                             "yielding", describe, sku, n)
                polled += 1
                sleep(max(0.0, min(self.KING_YIELD_POLL_SECONDS, deadline - now_fn())))
                continue
            if n:
                if polled:
                    log.info("%s: %s capacity appeared (%d) after %d poll(s)",
                             describe, sku, n, polled)
                return True
            if n == 0 and allow_paid_fallback and hotkey:
                threshold = int(getattr(self.cfg.round, 'ops_paid_fallback_after_seconds', 0))
                queue = self._funded_queue()
                entry = queue.get(hotkey) if queue is not None else None
                if (threshold and entry is not None and now_fn() - entry.funded_at >= threshold
                        and self.cfg.round.ops_operator_leg_cap_usd > 0
                        and self.cfg.round.ops_operator_daily_cap_usd > 0):
                    return 'operator_rent'
            if not polled:
                log.info("%s: no %s capacity on lium — polling every %.0fs until %s "
                         "(latest start that still lands inside the epoch)",
                         describe, sku, self.FUNDED_RENT_RETRY_SECONDS,
                         time.strftime("%H:%M:%SZ", time.gmtime(deadline)))
            polled += 1
            sleep(max(0.0, min(self.FUNDED_RENT_RETRY_SECONDS, deadline - now_fn())))
        if self._operator_fallback_lanes():
            # Past the latest safe start with operator lanes on file: the leg
            # goes to the lane pool, which hands out a lane only if one is
            # FREE NOW (waiting would land it late); no free lane ⇒ the pool's
            # deadline requeues it unburned. Rented lanes never idle while
            # legs are queued.
            log.info("%s: past the latest safe start — taking an operator lane only "
                     "if one is free now (operator-billed)", describe)
            return "operator"
        return False

    def _claimed_executors(self) -> tuple[str, ...]:
        """Executors live rents hold (never re-picked); aged-out claims dropped."""
        from .runner_common import live_claimed_executors

        return live_claimed_executors(self)

    KING_YIELD_POLL_SECONDS = 5.0

    def _king_pending(self) -> bool:
        """True while the era's JIT king still needs a pod: the king is
        REQUIRED and must not lose the marketplace race to a challenger that
        spots the same freshly listed GPU. Off when the king does not rent
        (funded_king_rent off, non-rent mode) and once its rent reached any
        outcome."""
        if getattr(self.__dict__.get("_leg_local_obj"), "funded_is_king", False):
            return False
        rnd = self.cfg.round
        return bool(
            getattr(rnd, "funded_king_rent", False)
            and rnd.funded_pods == "rent"
            and self._funded_king_host is None
            and not self._king_rent_done
        )

    def _yield_to_king(self, describe: str) -> bool:
        """Hold a challenger's rent while the king still needs a pod, up to
        the round's latest safe start (or the king's abort — then the leg's
        own wait sees it and requeues). Returns as soon as the king is served."""
        if not self._king_pending():
            return True
        from .progress import report_progress

        report_progress(self, "waiting_gpu", "king_priority")
        deadline = self._funded_rent_wait_deadline()
        sleep = getattr(self, "_rent_wait_sleep", None) or time.sleep
        now_fn = getattr(self, "_rent_wait_now", None) or time.time
        abort = getattr(self, "_funded_wait_abort", None)
        log.info("%s: yielding the marketplace — the king rents first", describe)
        while self._king_pending() and now_fn() < deadline:
            if abort is not None and abort.is_set():
                return False
            sleep(max(0.0, min(self.KING_YIELD_POLL_SECONDS, deadline - now_fn())))
        return not self._king_pending()

    def _operator_fallback_lanes(self) -> list:
        """Operator FINAL lanes a waiting funded leg may fall back to: fresh
        from hosts.toml (lanes join mid-round), profile-only entries excluded,
        and only when ``[round] funded_operator_fallback`` is on."""
        if not bool(getattr(self.cfg.round, "funded_operator_fallback", False)):
            return []
        from .remote import RemoteDispatchError, is_profile_only, load_hosts

        hosts: list = []
        if getattr(self, "remote_hosts_path", None) is not None:
            try:
                hosts = list(load_hosts(self.remote_hosts_path))
            except RemoteDispatchError:
                hosts = []
        else:
            hosts = list(getattr(self, "remote_hosts", None) or [])
        return [h for h in hosts
                if getattr(h, "stage", "any") in ("any", "final") and not is_profile_only(h)]

    def _probe_funded_capacity(self, sku,
                               exclude_ids: tuple[str, ...] = (),
                               price_caps: dict | None = None) -> int | None:
        """``sku``'s (or, for a tuple of types, their summed) marketplace
        availability on the OPERATOR's key under the price caps, or None.
        ``exclude_ids`` = executors this round already claimed: a count that
        includes them is capacity no rent can use. ``price_caps`` overrides
        the operator caps (a miner-chosen type, DEC-CA-0060)."""
        skus = (sku,) if isinstance(sku, str) else tuple(sku)
        caps = price_caps if price_caps is not None else self._funded_price_caps()
        try:
            from ..provision.operator_pool import operator_provider

            prov = operator_provider(self, lambda: _lium_provider(self.cfg.round, price_caps=caps))
            return self._provider_capacity(prov, skus, exclude_ids)
        except Exception as e:  # noqa: BLE001 — a probe failure must not gate the round
            log.warning("funded capacity probe for %s failed: %s", "/".join(skus), e)
            return None
