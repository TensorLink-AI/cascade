"""Satellite validator (DEC-CA-0058): replay the main validator's signed
receipts and set the vector the replay derives — no model, no pool, no torch.

The owner's MAIN validator (``[netconfig] owner_hotkey``, the same key that
signs the DEC-CA-0057 overlays) runs the full path and publishes one signed
receipt per settlement under ``receipts/<main>/round-<id>.json``. A satellite
walks those receipts oldest first and, for each one:

a. verifies the main validator's signature (``main-signature``);
b. fetches the trainer manifest the receipt names (``manifests/round-<id>
   .json``), verifies the trainer's signature, and checks the published text's
   sha256 equals the receipt's ``manifest_sha256`` and the embedded manifest
   equals the published one (``manifest-signature`` / ``manifest-fetch`` /
   ``manifest-digest``);
c. checks continuity: the trainer's hash chain (``prev_round_id`` /
   ``prev_manifest_sha256``), the receipt's ``state_prior_sha256`` against
   the previous receipt's ``state_post`` (king, tenure, court, ladder, lane-B/C
   slots and pending), and the lane-C pending chain
   (:func:`cascade.audit.checks.check_lane_c_chain`) (``continuity`` /
   ``lane_c_chain``);
d. re-derives everything else with the code ``cascade-audit`` runs
   (:func:`cascade.audit.main.audit_receipt`, tier 0 — the verdict, the cohort
   correction, the bench gate replayed from the TRAINER-SIGNED bench report
   — or, from ``[v2] bench_source_from_block`` (DEC-CA-0063), from the MAIN
   validator's own signed report the receipt binds (``bench-source``; a
   satellite never sweeps),
   the king check and ladder, lane-B/C payouts, the weight vector) under the
   config the receipt was judged under (its DEC-CA-0045 blocks and its
   owner-signed DEC-CA-0057 overlay), then rebuilds the weight vector itself
   (:func:`cascade.audit.checks.derive_weights`) and requires it to equal the
   recorded one (``weights-derive``);
e. at every weight push, checks each paid UID against its OWN metagraph: the
   hotkey registered at that UID must be one the receipt pays
   (``reward-uid``).

All checks pass ⇒ the satellite sets the vector it derived through the chain
client's weight path (float32 → u16 quantisation, ``WEIGHT_RETRY_BLOCKS``
backoff). Any failure ⇒ it burns (``burn_uid``) while staying active and logs
the failing check at every push until a receipt replays cleanly. Main silent ⇒
it re-asserts the last verified vector until ``[satellite] stale_after_blocks``
past that settlement's boundary, then burns until a new receipt verifies.

A check that cannot run (storage read failure, the bench report or overlay not
readable, no sr25519 verifier) is NOT a failure: the receipt is retried next
poll and the satellite keeps its current vector (the stale clock still runs).

What a satellite cannot detect: fabricated score rows. The rows are replayed,
not re-scored — that is the public reproduce audit's job (checkpoints are
public; ``cascade-audit`` tier 2 re-scores).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..shared.activation import ActivationRecord, ActivationStore
from ..shared.config import ChainConfig
from ..shared.config.satellite import SatelliteConfig, satellite_config
from .activation_mixin import ActivationMixin
from .weights import WEIGHT_RETRY_BLOCKS, WEIGHTS_RATE_LIMIT_BLOCKS

log = logging.getLogger("cascade.satellite")

FOLLOWING, HOLDING, BURNING = "following", "holding", "burning"
# Audit rows whose WARN means "could not verify" (not "verified with a
# caveat"): the receipt is retried, never accepted, while one of these warns.
UNVERIFIABLE_WARN = frozenset({"receipt-signature", "manifest-signature", "netconfig",
                               "bench-gate", "king-check-bench", "bench-source",
                               "bench-coverage", "calendar-launch"})


class Transient(Exception):
    """A check could not run (storage, verifier). Retry; never a verdict."""


class _Gap(Exception):
    """The walk back from the newest receipt does not reach the anchor."""


@dataclass(frozen=True)
class Replay:
    """One receipt's replay. ``ok`` False ⇒ ``check`` names the first failing
    check. ``vector`` is the derived vector (None = the receipt records none,
    e.g. a gate rejection: the standing vector stays). ``advance`` False ⇒
    the receipt is not the main validator's (bad signature): the anchor does
    not move."""

    ok: bool
    check: str = ""
    detail: str = ""
    vector: list[float] | None = None
    burn_uid: int = 0
    attribution: dict = field(default_factory=dict)
    advance: bool = True


def _fail(check: str, detail: str, *, advance: bool = True) -> Replay:
    return Replay(ok=False, check=check, detail=detail, advance=advance)


def pin_main_validator(cfg: ChainConfig, owner: str) -> ChainConfig:
    """``cfg`` with ``[manifest] validator_hotkey`` = the main validator, in
    the parsed config AND its raw table (an overlay rebuilds from raw)."""
    raw = json.loads(json.dumps(getattr(cfg, "raw", None) or {}))
    raw.setdefault("manifest", {})["validator_hotkey"] = owner
    return replace(cfg, raw=raw, manifest=replace(cfg.manifest, validator_hotkey=owner))


def _is_not_found(exc: BaseException) -> bool:
    from ..shared.hippius import ObjectNotFound

    return isinstance(exc, ObjectNotFound)


def _anchor_of(receipt) -> dict:
    return {
        "round_id": str(receipt.round_id),
        "boundary": int(receipt.epoch_start_block),
        "manifest_sha256": str(getattr(receipt, "manifest_sha256", "") or ""),
        "state_post": getattr(receipt, "state_post", None),
        "lane_c": (None if getattr(receipt, "lane_c", None) is None
                   else [dict(r) for r in receipt.lane_c]),
    }


def _attribution(receipt) -> dict:
    """Who the receipt pays: every hotkey its posterior state pays (plus the
    verdict's king) and the ``uid -> hotkey`` pairs the lane payout records
    pin explicitly."""
    from ..shared.settlement_state import paid_hotkeys

    paid = paid_hotkeys(getattr(receipt, "state_post", None))
    v = receipt.verdict
    if v is not None and v.king_hotkey:
        paid.add(str(v.king_hotkey))
    pinned: dict[str, str] = {}
    for pay in (receipt.lane_b_pay, getattr(receipt, "lane_c_pay", None)):
        for w in (pay or {}).get("wins", ()) or ():
            if w.get("hotkey") is not None and w.get("uid") is not None:
                pinned[str(int(w["uid"]))] = str(w["hotkey"])
                paid.add(str(w["hotkey"]))
    return {"paid": sorted(paid), "pinned": pinned}


def check_reward_uids(vector: list[float], burn_uid: int, hotkeys: list[str],
                      attribution: dict) -> str | None:
    """DEC-CA-0058 §1e: ``None`` when every paid UID (weight > 0, not the
    burn UID) is registered on the satellite's own metagraph to a hotkey the
    receipt pays (and, for a lane win, to exactly the hotkey the payout
    record names); else why not."""
    paid = set(attribution.get("paid") or ())
    pinned = {int(u): str(h) for u, h in (attribution.get("pinned") or {}).items()}
    for u, w in enumerate(vector):
        if w <= 0 or u == burn_uid:
            continue
        if u >= len(hotkeys):
            return f"uid {u} carries weight but the metagraph has only {len(hotkeys)} uids"
        hk = str(hotkeys[u])
        if u in pinned and pinned[u] != hk:
            return f"uid {u} is registered to {hk[:12]}…, the payout record names {pinned[u][:12]}…"
        if hk not in paid:
            return f"uid {u} is registered to {hk[:12]}…, which the receipt does not pay"
    return None


def fit_vector(vector: list[float], n_uids: int) -> list[float] | None:
    """``vector`` resized to the metagraph's ``n_uids`` (zero-padded; a
    truncated tail must carry no weight, else None)."""
    v = [float(x) for x in vector]
    if len(v) <= n_uids:
        return v + [0.0] * (n_uids - len(v))
    if any(x > 0 for x in v[n_uids:]):
        return None
    return v[:n_uids]


# ── one receipt ──────────────────────────────────────────────────────────────


def replay_receipt(receipt, cfg: ChainConfig, *, reader: Any, prev: dict | None,
                   client=None) -> Replay:
    """Replay one main-validator receipt (DEC-CA-0058 §1a-d) against the
    previous replayed receipt's anchor ``prev`` (None = this satellite's
    first receipt: continuity starts here). Raises :class:`Transient` when a
    check cannot run."""
    from ..audit.checks import (
        FAIL,
        WARN,
        check_lane_c_chain,
        derive_weights,
        ladder_prior_mismatch,
    )
    from ..audit.main import apply_receipt_overlay, audit_receipt
    from ..shared.activation import apply_receipt_activation
    from ..shared.era import forfeited_hotkeys
    from ..shared.hippius import manifest_round_key
    from ..shared.manifest import load_manifest, verify_signature
    from ..shared.receipt import verify_receipt_signature
    from ..shared.settlement_state import state_digest

    owner = str(cfg.netconfig.owner_hotkey or "")
    rid = str(receipt.round_id)
    # a. the main validator's signature
    if not owner:
        return _fail("main-signature", "[netconfig] owner_hotkey is empty", advance=False)
    if str(receipt.validator_hotkey) != owner:
        return _fail("main-signature", f"signer {receipt.validator_hotkey[:12]}… is not the main "
                     f"validator {owner[:12]}…", advance=False)
    try:
        if not verify_receipt_signature(receipt, owner):
            return _fail("main-signature", "receipt signature does not verify against the main "
                         "validator", advance=False)
    except RuntimeError as e:          # SignatureVerifierUnavailable
        raise Transient(f"main-signature: cannot verify ({e})") from e
    # b. the trainer manifest it names
    trainer = str(cfg.manifest.trainer_hotkey or "")
    if not trainer:
        return _fail("manifest-signature", "[manifest] trainer_hotkey is unset")
    try:
        manifest = receipt.load_embedded_manifest()
    except (ValueError, KeyError) as e:
        return _fail("manifest-signature", f"embedded manifest unparseable: {e}")
    try:
        if not verify_signature(manifest, trainer):
            return _fail("manifest-signature", "embedded manifest is not signed by the trainer")
    except RuntimeError as e:
        raise Transient(f"manifest-signature: cannot verify ({e})") from e
    try:
        text = reader.get_text(manifest_round_key(rid))
    except Exception as e:  # noqa: BLE001
        if _is_not_found(e):
            return _fail("manifest-fetch", f"{manifest_round_key(rid)} is not published")
        raise Transient(f"manifest-fetch: {e}") from e
    want_sha = str(getattr(receipt, "manifest_sha256", "") or "")
    if not want_sha:
        return _fail("manifest-digest", "the receipt records no manifest_sha256")
    got_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if got_sha != want_sha:
        return _fail("manifest-digest", f"published manifest sha256 {got_sha[:12]}… != the "
                     f"receipt's {want_sha[:12]}…")
    try:
        published = load_manifest(text)
    except (ValueError, KeyError) as e:
        return _fail("manifest-digest", f"published manifest unparseable: {e}")
    if (published.canonical_body() != manifest.canonical_body()
            or published.signature != manifest.signature):
        return _fail("manifest-digest", "the embedded manifest differs from the published one")
    # the config the receipt was judged under (DEC-CA-0045 blocks, then the
    # owner-signed DEC-CA-0057 overlay it stamps)
    cfg_audit = pin_main_validator(cfg, owner)
    cfg_r = apply_receipt_activation(cfg_audit, receipt)
    cfg_r, nc_row = apply_receipt_overlay(receipt, cfg_r, store=reader)
    if nc_row is not None and nc_row.status == FAIL:
        return _fail("netconfig", nc_row.detail)
    if nc_row is not None and nc_row.status == WARN:
        raise Transient(f"netconfig: {nc_row.detail}")
    if cfg_r.v2.launch_at_utc and not cfg_r.v2.from_block:
        from ..shared.calendar_launch import receipt_calendar_config

        if client is None:
            raise Transient("calendar-launch: chain connection or resolved from_block required")
        try:
            cfg_r = receipt_calendar_config(cfg_r, receipt, client)
        except Exception as exc:
            raise Transient(f"calendar-launch: {exc}") from exc
        cfg_audit = cfg_r
    # c. continuity
    post = getattr(receipt, "state_post", None)
    if post is None:
        return _fail("continuity", "the receipt records no state_post")
    v = receipt.verdict
    if v is not None and post.get("king") != v.king_hotkey:
        return _fail("continuity", f"state_post king {post.get('king')!r} != the verdict's "
                     f"{v.king_hotkey!r}")
    if prev is not None:
        boundary, pb = int(receipt.epoch_start_block), int(prev["boundary"])
        if boundary <= pb:
            return _fail("continuity", f"boundary {boundary} is not after the previous "
                         f"receipt's {pb}")
        if manifest.prev_round_id and manifest.prev_round_id != prev["round_id"]:
            return _fail("continuity", f"manifest prev_round_id {manifest.prev_round_id} != the "
                         f"previous receipt's round {prev['round_id']}")
        if (manifest.prev_manifest_sha256 and prev.get("manifest_sha256")
                and manifest.prev_manifest_sha256 != prev["manifest_sha256"]):
            return _fail("continuity", "manifest prev_manifest_sha256 != the previous "
                         "receipt's manifest_sha256")
        want_prior = state_digest(prev.get("state_post"))
        got_prior = str(getattr(receipt, "state_prior_sha256", "") or "")
        if got_prior == want_prior:
            # the ladder record's prior key-slash state (DEC-CA-0055 escalating
            # slash) is the previous posterior's — the audit replays the step
            # from it, so it must be bound here
            why = ladder_prior_mismatch((getattr(receipt, "emission", None) or {}).get("ladder"),
                                        prev.get("state_post"))
            if why:
                return _fail("continuity", why)
        else:
            # DEC-CA-0048 forfeiture moves the throne on the BLOCK clock between
            # settlements: the one settlement that first meets the gate may
            # legitimately start from a different state.
            if not (forfeited_hotkeys(cfg_r.scoring, boundary)
                    and not forfeited_hotkeys(cfg_r.scoring, pb)):
                return _fail("continuity", f"prior state {got_prior[:12] or 'absent'} != the "
                             f"previous receipt's posterior {want_prior[:12]}…")
            log.warning("round=%s prior state differs from round %s's posterior across the "
                        "DEC-CA-0048 forfeiture gate; accepted", rid, prev["round_id"])
        prev_ns = SimpleNamespace(round_id=prev["round_id"], epoch_start_block=pb,
                                  lane_c=prev.get("lane_c"))
        row = check_lane_c_chain([prev_ns, receipt], cfg_r)
        if row.status == FAIL:
            return _fail("lane_c_chain", row.detail)
    else:
        log.warning("round=%s: first receipt this satellite replays — continuity is "
                    "anchored here", rid)
    # d. the audit's tier-0 re-derivation (verdict, cohort, bench gate from the
    # trainer-signed report, king check / ladder, lane payouts, weights)
    rows = audit_receipt(receipt, cfg_audit, tier=0, client=None,
                         netconfig_store=reader, bench_store=reader)
    for r in rows:
        if r.status == FAIL:
            return _fail(r.name, r.detail)
    for r in rows:
        if r.status == WARN and r.name in UNVERIFIABLE_WARN:
            raise Transient(f"{r.name}: {r.detail}")
    burn = int(cfg_r.scoring.burn_uid)
    if not receipt.weights:
        return Replay(ok=True, vector=None, burn_uid=burn, attribution=_attribution(receipt))
    vec, why = derive_weights(receipt, cfg_r)
    if vec is None:
        return _fail("weights-derive", why)
    got = [float(w) for w in receipt.weights]
    bad = [u for u, (a, b) in enumerate(zip(vec, got, strict=True))
           if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)]
    if bad:
        return _fail("weights-derive", f"uid {bad[0]}: derived {vec[bad[0]]} != recorded "
                     f"{got[bad[0]]}")
    return Replay(ok=True, vector=list(vec), burn_uid=burn, attribution=_attribution(receipt))


# ── storage ──────────────────────────────────────────────────────────────────


class StoreReader:
    """``get_text`` over the manifest-bucket store (``open_manifest_store``,
    so the R2 / HF fallbacks apply), opened lazily."""

    def __init__(self, cfg: ChainConfig, store: Any = None) -> None:
        self.cfg = cfg
        self._store = store

    def get_text(self, key: str) -> str:
        if self._store is None:
            from ..shared.hippius import open_manifest_store

            self._store = open_manifest_store(self.cfg.storage)
        return self._store.get_text(key)


# ── persisted state ──────────────────────────────────────────────────────────


@dataclass
class SatelliteState:
    """``anchor``: the last receipt replayed (verified or failed) — the next
    one must chain from it. ``vector`` / ``burn_uid`` / ``attribution``: the
    last VERIFIED vector and who it pays; ``verified_*``: its receipt.
    ``failing*``: the failing check of the last replayed receipt ("" = it
    replayed cleanly)."""

    anchor: dict | None = None
    vector: list[float] | None = None
    burn_uid: int = 0
    attribution: dict = field(default_factory=dict)
    verified_round: str = ""
    verified_boundary: int = 0
    failing: str = ""
    failing_detail: str = ""
    failing_round: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text: str) -> SatelliteState:
        obj = json.loads(text)
        return cls(**{k: obj[k] for k in cls.__dataclass_fields__ if k in obj})


def _load_state(path: Path) -> SatelliteState:
    from ..shared.atomic import read_text_with_backup

    st, source = read_text_with_backup(path, SatelliteState.from_json)
    if source == "absent":
        return SatelliteState()
    if st is None:
        raise RuntimeError(f"satellite state {path} (and .bak) does not parse")
    return st


# ── the runner ───────────────────────────────────────────────────────────────


@dataclass
class SatelliteRunner(ActivationMixin):
    """The satellite loop. Reuses :class:`ActivationMixin` as is, so a
    satellite posts its DEC-CA-0045 / DEC-CA-0057 readiness note exactly like a
    full validator (same tally, same note, its own persisted records)."""

    cfg: ChainConfig
    state_dir: Path
    reader: Any = None
    sat: SatelliteConfig = field(default_factory=SatelliteConfig)
    state: SatelliteState = field(default_factory=SatelliteState)
    hotkey: str = ""
    # ActivationMixin's attributes (no Cascade controller, no overlay view)
    cascade: Any = None
    netconfig: Any = None
    activation_store: ActivationStore | None = None
    _activation: ActivationRecord = field(default_factory=ActivationRecord, repr=False)
    _activation_tally: Any = field(default=None, repr=False)
    _signal_hotkey: str = field(default="", repr=False)
    _signal_sent: str = field(default="", repr=False)
    _forfeit: ActivationRecord = field(default_factory=ActivationRecord, repr=False)
    _forfeit_tally: Any = field(default=None, repr=False)
    _margin_v2: ActivationRecord = field(default_factory=ActivationRecord, repr=False)
    _margin_v2_tally: Any = field(default=None, repr=False)
    _prelaunch: ActivationRecord = field(default_factory=ActivationRecord, repr=False)
    _prelaunch_tally: Any = field(default=None, repr=False)
    _cfg_base: Any = field(default=None, repr=False)
    _settlement_block: int | None = field(default=None, repr=False)
    _netconfig_now: int | None = field(default=None, repr=False)
    # weight push bookkeeping
    _last_weight_block: int | None = field(default=None, repr=False)
    _last_pushed: list[float] | None = field(default=None, repr=False)
    _weight_retry_from: int | None = field(default=None, repr=False)
    _mode: str = field(default=BURNING, repr=False)
    _caught_up: bool = field(default=False, repr=False)
    _push_failing: str = field(default="", repr=False)

    @property
    def owner(self) -> str:
        return str(self.cfg.netconfig.owner_hotkey or "")

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir) / "satellite_state.json"

    def _persist(self) -> None:
        from ..shared.atomic import write_text_atomic

        try:
            write_text_atomic(self.state_path, self.state.to_json(), backup=True)
        except OSError as e:
            log.warning("satellite state not written (%s)", e)

    # ── receipts ──

    def _fetch_receipt(self, key: str):
        from ..shared.receipt import load_receipt

        return load_receipt(self.reader.get_text(key))

    def _prev_round_id(self, receipt) -> str:
        """The settlement before ``receipt``: the trainer's signed hash link,
        else the main validator's newest earlier entry in the receipt index."""
        prev = str((receipt.manifest or {}).get("prev_round_id") or "")
        if prev:
            return prev
        from ..shared.hippius import RECEIPT_INDEX_KEY

        try:
            doc = json.loads(self.reader.get_text(RECEIPT_INDEX_KEY))
        except Exception:  # noqa: BLE001 — no index: the gap is reported by the caller
            return ""
        rows = [r for r in (doc.get("rounds") or []) if isinstance(r, dict)
                and str(r.get("validator_hotkey") or "") == self.owner
                and int(r.get("epoch_start_block", 0) or 0) < int(receipt.epoch_start_block)]
        if not rows:
            return ""
        return str(max(rows, key=lambda r: int(r.get("epoch_start_block", 0)))["round_id"])

    def _pending(self, latest) -> list:
        """The receipts after the anchor up to ``latest``, oldest first."""
        from ..shared.hippius import receipt_round_key

        anchor = self.state.anchor
        if anchor is None:
            return [latest]
        out, cur = [latest], latest
        for _ in range(int(self.sat.max_walk)):
            if int(cur.epoch_start_block) <= int(anchor["boundary"]):
                raise _Gap(f"walked back to round {cur.round_id} (block "
                           f"{cur.epoch_start_block}) without meeting round "
                           f"{anchor['round_id']}")
            prev_id = self._prev_round_id(cur)
            if not prev_id:
                raise _Gap(f"round {cur.round_id} names no previous settlement")
            if prev_id == anchor["round_id"]:
                return list(reversed(out))
            try:
                cur = self._fetch_receipt(receipt_round_key(prev_id, self.owner))
            except Exception as e:  # noqa: BLE001
                if _is_not_found(e):
                    raise _Gap(f"the main validator published no receipt for round "
                               f"{prev_id}") from e
                raise Transient(f"receipt read: {e}") from e
            out.append(cur)
        raise _Gap(f"more than {self.sat.max_walk} receipts since round {anchor['round_id']}")

    def _record_failure(self, rid: str, check: str, detail: str) -> None:
        st = self.state
        if (st.failing, st.failing_round) != (check, rid):
            log.error("round=%s FAILED %s: %s — burning until a receipt replays cleanly",
                      rid, check, detail)
        st.failing, st.failing_detail, st.failing_round = check, detail, rid

    def sync(self, client=None) -> None:
        """Read the main validator's newest receipt and replay every receipt
        since the anchor, oldest first. Never raises."""
        from ..shared.hippius import receipt_latest_key

        self._caught_up = False
        try:
            latest = self._fetch_receipt(receipt_latest_key(self.owner))
        except Exception as e:  # noqa: BLE001 — silent or unreadable: hold
            log.info("main validator receipt unavailable (%s)", e)
            return
        anchor = self.state.anchor
        if anchor is not None and str(latest.round_id) == anchor["round_id"]:
            self._caught_up = not self.state.failing
            return
        try:
            chain = self._pending(latest)
        except Transient as e:
            log.warning("receipt walk deferred (%s); retrying next poll", e)
            return
        except _Gap as e:
            self._record_failure(str(latest.round_id), "continuity", str(e))
            self._persist()
            return
        for receipt in chain:
            rid = str(receipt.round_id)
            try:
                res = replay_receipt(receipt, self.cfg, reader=self.reader,
                                     prev=self.state.anchor, client=client)
            except Transient as e:
                log.warning("round=%s not verifiable yet (%s); retrying next poll", rid, e)
                break
            except Exception as e:  # noqa: BLE001 — a replay crash is a failure, not a pass
                res = _fail("replay", f"{type(e).__name__}: {e}")
            if not res.advance:
                self._record_failure(rid, res.check, res.detail)
                break
            self.state.anchor = _anchor_of(receipt)
            if not res.ok:
                self._record_failure(rid, res.check, res.detail)
                continue
            if self.state.failing:
                log.warning("round=%s replays cleanly — resuming after %s on round %s",
                            rid, self.state.failing, self.state.failing_round)
            st = self.state
            st.failing = st.failing_detail = st.failing_round = ""
            st.verified_round, st.verified_boundary = rid, int(receipt.epoch_start_block)
            if res.vector is not None:
                st.vector, st.burn_uid, st.attribution = res.vector, res.burn_uid, res.attribution
            log.info("round=%s verified (%s)", rid,
                     "vector derived" if res.vector is not None else "no vector recorded; "
                     "the standing vector stays")
        else:
            self._caught_up = not self.state.failing
        self._persist()

    # ── weights ──

    def desired(self, *, now_block: int, hotkeys: list[str]) -> tuple[list[float], str, str]:
        """``(vector, mode, why)`` for a push at ``now_block``."""
        from ..shared.chain import decayed_share_vector
        from ..shared.config import effective_epoch_blocks

        n = len(hotkeys)
        burn = int(self.cfg.scoring.burn_uid)
        burn_vec = decayed_share_vector([], n, decay=self.cfg.scoring.king_decay, burn_uid=burn)
        st = self.state
        self._push_failing = ""
        if st.failing:
            return burn_vec, BURNING, f"{st.failing} failed on round {st.failing_round}"
        if st.vector is None:
            return burn_vec, BURNING, "no verified receipt yet"
        age = int(now_block) - int(st.verified_boundary)
        if age > int(self.sat.stale_after_blocks):
            return burn_vec, BURNING, (f"main validator silent: {age} blocks since round "
                                       f"{st.verified_round} > stale_after_blocks="
                                       f"{self.sat.stale_after_blocks}")
        vec = fit_vector(st.vector, n)
        why = (check_reward_uids(st.vector, st.burn_uid, hotkeys, st.attribution)
               if vec is not None else f"vector has weight past the metagraph's {n} uids")
        if why is not None:
            self._push_failing = "reward-uid"
            return burn_vec, BURNING, f"reward-uid: {why}"
        grid = int(effective_epoch_blocks(self.cfg.round, int(now_block)))
        mode = FOLLOWING if (self._caught_up and age <= 2 * grid) else HOLDING
        return vec, mode, f"round {st.verified_round}"

    def push(self, client: Any, *, now_block: int) -> bool:
        """Set the desired vector when it changed or the re-assert interval
        elapsed (never inside the retry backoff or the chain's rate limit).
        True when a push was accepted."""
        try:
            hotkeys = [str(h) for h in client.registered_hotkeys()]
        except Exception as e:  # noqa: BLE001
            log.warning("metagraph read failed (%s); push skipped", e)
            return False
        vec, mode, why = self.desired(now_block=now_block, hotkeys=hotkeys)
        self._mode = mode
        last = self._last_weight_block
        if last is not None and now_block - last < WEIGHTS_RATE_LIMIT_BLOCKS:
            return False
        interval = max(int(self.cfg.validator.weight_set_interval_blocks),
                       WEIGHTS_RATE_LIMIT_BLOCKS)
        due = last is None or vec != self._last_pushed or now_block - last >= interval
        if not due:
            return False
        if (self._weight_retry_from is not None
                and now_block - self._weight_retry_from < WEIGHT_RETRY_BLOCKS):
            return False
        if mode == BURNING:
            log.warning("BURNING to uid %d at block %d: %s", self.cfg.scoring.burn_uid,
                        now_block, why)
        try:
            ok = client.set_weight_vector(list(vec))
        except Exception as e:  # noqa: BLE001 — retried after WEIGHT_RETRY_BLOCKS
            log.warning("weight set failed (%s); retrying in %d blocks", e, WEIGHT_RETRY_BLOCKS)
            ok = False
        if ok is False:
            self._weight_retry_from = int(now_block)
            return False
        self._weight_retry_from = None
        self._last_weight_block, self._last_pushed = int(now_block), list(vec)
        log.info("weights set (%s, %s): %s", mode, why,
                 {u: round(w, 4) for u, w in enumerate(vec) if w > 0})
        return True

    # ── status ──

    def status(self, now_block: int | None = None) -> dict:
        st = self.state
        return {
            "hotkey": self.hotkey, "main_validator": self.owner, "mode": self._mode,
            "block": now_block, "last_verified_round": st.verified_round,
            "last_verified_boundary": st.verified_boundary,
            "last_failing_check": st.failing or self._push_failing,
            "last_failing_detail": st.failing_detail, "last_failing_round": st.failing_round,
            "stale_after_blocks": int(self.sat.stale_after_blocks),
        }

    def write_status(self, now_block: int | None = None) -> dict:
        from ..shared.atomic import write_text_atomic

        doc = self.status(now_block)
        name = f"satellite-{self.hotkey or 'local'}.json"
        try:
            write_text_atomic(Path(self.state_dir) / "status" / name,
                              json.dumps(doc, sort_keys=True, indent=2))
        except OSError as e:
            log.debug("status not written (%s)", e)
        log.info("satellite status: mode=%s last_verified=%s failing=%s", doc["mode"],
                 doc["last_verified_round"] or "-", doc["last_failing_check"] or "-")
        return doc

    # ── loop ──

    def poll(self, client: Any) -> dict:
        """One pass: readiness note, receipts, weights, status."""
        now = int(client.current_block())
        with contextlib.suppress(Exception):   # activation never disturbs the replay
            self._activation_tick(client)
        self.sync(client=client)
        self.push(client, now_block=now)
        return self.write_status(now)

    def run_forever(self, client: Any) -> None:  # pragma: no cover — live loop
        with contextlib.suppress(Exception):
            self._activation_startup(client)
        while True:
            try:
                self.poll(client)
            except Exception:  # noqa: BLE001 — the loop must survive any one pass
                log.exception("satellite poll failed; retrying")
            time.sleep(int(self.sat.poll_seconds))


def build_satellite(cfg: ChainConfig, *, state_dir: Path, reader: Any = None,
                    hotkey: str = "") -> SatelliteRunner:
    """A runner with its persisted state, activation records and (when the
    owner key is set) its own read-only-applied, signalling overlay resolver."""
    state_dir = Path(state_dir)
    reader = reader if reader is not None else StoreReader(cfg)
    netconfig = None
    if cfg.netconfig.enabled:
        from ..shared.netconfig import NetConfigResolver

        netconfig = NetConfigResolver(cfg, state_path=state_dir / "netconfig_state.json",
                                      store=reader, signal=True)
    return SatelliteRunner(
        cfg=cfg, state_dir=state_dir, reader=reader, sat=satellite_config(cfg),
        state=_load_state(state_dir / "satellite_state.json"), hotkey=hotkey,
        netconfig=netconfig,
        activation_store=ActivationStore(state_dir / "activation_state.json"))


# ── CLI ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cascade-satellite",
                                description="cascade satellite validator (DEC-CA-0058).")
    p.add_argument("--chain-toml", type=Path, default=None)
    p.add_argument("--network", default="finney")
    p.add_argument("--wallet-name", default=None)
    p.add_argument("--wallet-hotkey", default=None)
    p.add_argument("--wallet-path", default=None)
    p.add_argument("--state-dir", type=Path, default=None,
                   help="Where the satellite keeps its state and status (default: the "
                        "directory of [validator] state_db_path).")
    p.add_argument("--once", action="store_true", help="Run one poll and exit.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def run_from_args(args: argparse.Namespace) -> int:
    from ..shared.chain import ChainClient
    from ..shared.config import load_chain_config

    cfg = load_chain_config(args.chain_toml)
    if not cfg.netconfig.owner_hotkey:
        print("[netconfig] owner_hotkey (the main validator) is empty", file=sys.stderr)
        return 2
    if not cfg.manifest.trainer_hotkey:
        print("[manifest] trainer_hotkey is empty", file=sys.stderr)
        return 2
    if args.wallet_name is None or args.wallet_hotkey is None:
        print("--wallet-name and --wallet-hotkey are required", file=sys.stderr)
        return 2
    state_dir = (getattr(args, "state_dir", None) or getattr(args, "cache_dir", None)
                 or Path(cfg.validator.state_db_path).parent)
    client = ChainClient.from_config(cfg, network=args.network, wallet_name=args.wallet_name,
                                     wallet_hotkey=args.wallet_hotkey,
                                     wallet_path=args.wallet_path)
    hotkey = str(client.hotkey_ss58() or "")
    runner = build_satellite(cfg, state_dir=Path(state_dir), hotkey=hotkey)
    from ..shared.logging_util import restore_cascade_logging

    restore_cascade_logging(args.log_level)
    log.info("satellite up: netuid=%s main validator %s, stale after %d blocks", cfg.netuid,
             runner.owner, runner.sat.stale_after_blocks)
    if getattr(args, "once", False):
        with contextlib.suppress(Exception):
            runner._activation_startup(client)
        print(json.dumps(runner.poll(client), indent=2, sort_keys=True))
        return 0
    runner.run_forever(client)
    return 0


def main(argv: list[str] | None = None) -> int:
    from ..shared.env import load_env_files

    load_env_files()
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
