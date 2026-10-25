"""Lane-A public benchmarks on the main validator's GPU, winners only (DEC-CA-0063).

From ``[v2] bench_source_from_block`` (``[v2] bench_source = "validator"``)
the lane-A bench rows the DEC-CA-0054 / DEC-CA-0061 bench gate and the
DEC-CA-0055 king check read come from the MAIN validator (``[netconfig]
owner_hotkey``), not from the trainer:

* the private pool is scored exactly as before; only a challenger that
  clears it (the existing winning-result path, ``_run_bench_gate``) is
  swept — together with the era king, whose rows are cached per checkpoint
  pointer (and invalidated when the challenger's sweep ran on a different
  pinned data revision) so it is swept once per king checkpoint. No winner
  ⇒ no sweep;
* when the king check is due, the king and the era init are swept (cached
  the same way);
* the sweeps run on this box's ``device`` or on the eval-offload pod
  (``eval_host_fn``, :func:`~cascade.validator.eval_offload.run_bench_via_host`),
  through the same sidecar every other benchmark path uses. Every
  checkpoint's sweep has its OWN allowance (``[v2] bench_sweep_hours``, its
  retries — :data:`VBENCH_SWEEP_ATTEMPTS` — included): a clearer's sweep
  never spends another clearer's time, and the king's sweep never spends a
  clearer's (a PAIRED pass — ``[eval] bench_paired``, challenger + an
  unswept king or king + init in one sidecar run — is charged in equal
  shares to its members' allowances and once in full to the cap,
  :meth:`ValidatorBenchMixin._vbench_paired_pass`; ``[eval] bench_workers``
  sets the sidecar's concurrency — neither changes a report byte, and only
  the main validator sweeps). ``[v2] bench_budget_hours`` caps the settlement's total; a
  clearer is started only while the cap still holds its whole allowance
  (plus the king's, when the king must be swept too) — one reached without
  it is published ``budget_deferred`` (never swept, never a loss) and the
  trainer carries its leg into the settlement that records this report
  (at most :data:`~cascade.shared.bench_report.VBENCH_MAX_CARRIES` times,
  inside the era), where it is judged again like any other leg;
* a sweep that fails is classified (:func:`~cascade.eval.benchmarks.
  classify_sweep_failure`): it is a ``checkpoint_fault`` only when EVERY
  attempt failed on positive checkpoint evidence (non-finite forecasts,
  every config failing at scoring) with no environmental marker — a
  challenger listed so FAILS the bench gate (a loss) on every validator;
  anything else (OOM / CUDA, timeouts, store / network, dataset fetch, a
  sidecar crash, an allowance spent before the retry) is ``infra_failed``
  (never a loss). Both lists are in the signed report;
* ONE report per such settlement — same :class:`~cascade.shared.bench_report.
  BenchReport` / rows format, signed by the main validator's hotkey — is
  published at ``bench_v/round-<id>.json`` (rows content-addressed under
  ``bench_v/rows/<id>/``) BEFORE the settlement is marked handled, even when
  nothing was swept. The FIRST published report of a round is final: it is
  never overwritten (a restarted main validator reads it like everyone else);
* THE ARBITER. Whether a settlement's report counts is decided by the
  trainer, not by when each validator happened to look: the signed manifest
  of a later settlement records ``validator_bench_reports[<id>]`` = the
  sha256 of the exact text the trainer first read, or ``"absent"`` after
  ``[v2] bench_report_timeout_settlements`` later settlements were built
  without it. A validator needs the report only when its own private-pool
  verdict sends a clearer to the bench gate, or the king check is due (both
  identical on every validator); otherwise it judges without it. When it
  needs it, EVERY validator — the main one included — HOLDS until that
  record exists, then judges from exactly the recorded bytes (or, on
  "absent", treats the clearers as infra). The main validator publishes its
  report first and holds with everyone else, so no validator ever judges
  from a different view of the store;
* infra (a recorded "absent", or an ``infra_failed`` entry) and a
  ``budget_deferred`` entry are never a loss: the clearer is skipped (streak
  untouched; the verdict note names it under ``bench_infra`` /
  ``bench_deferred``), the gate falls through to the next clearer, and with
  none left the king holds (inconclusive). A challenger's
  ``checkpoint_fault`` FAILS the gate (``bench_fault`` in the note; streak
  reset like any gate block); a king's / init's is infra. A report that fails its
  signature, an entry it omits, or rows failing their sha256 are
  *uncomputable* — the gate's existing semantics (``enforce`` ⇒
  inconclusive: the king holds), and the king check defers;
* the receipt binds the report (``bench_report`` = key + sha256 + signer,
  or ``record: "absent"``), so the audit and satellites replay from exactly
  those bytes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import time
from typing import Any

from ..shared.era import (
    BENCH_RULE_V2,
    bench_gate_rule_at,
    validator_bench_at,
    validator_bench_configured,
)
from ..shared.hippius import StorageError
from ..shared.manifest import BenchScores, TrainedEntry, TrainingManifest, load_manifest

log = logging.getLogger("cascade.validator")

# Bench suites of a validator sweep: the DEC-CA-0054 three, + FEV under the
# DEC-CA-0061 rule (the settlement's effective bench-gate rule).
SUITES_V1 = ("gift-eval", "boom", "time")
# Checkpoints whose sweep the main validator keeps in memory (the era king and
# the current init are the ones re-read across settlements).
VBENCH_CACHE_MAX = 32
# Attempts per sweep within one settlement (a failure is retried while the
# checkpoint's own allowance lasts; after the last one it is classified —
# checkpoint fault only when every attempt was the checkpoint's, else infra).
VBENCH_SWEEP_ATTEMPTS = 2
# The reason a clearer cut off by the settlement's sweep cap is published with.
BUDGET_DEFERRED_REASON = "bench budget exhausted (deferred to a later settlement)"
# The trainer's "absent" record of a validator bench report (DEC-CA-0063).
RECORD_ABSENT = "absent"


class ValidatorBenchHold(StorageError):
    """HOLD the settlement (never a verdict): the trainer has not yet recorded
    the validator bench report this settlement needs, or it could not be read.
    A :class:`StorageError`, so every lane-A read-hold path treats it alike."""


def _scores_dict(scores: BenchScores) -> dict:
    return dataclasses.asdict(scores)


def check_main_bench_signer(cfg: Any, signer: Any) -> None:
    """Refuse to run as the MAIN validator (wallet hotkey == ``[netconfig]
    owner_hotkey``) without a wallet that can sign: its bench report would be
    unsigned, uncomputable for every other validator. Raises
    :class:`RuntimeError` (the CLI exits at start). A no-op off the validator
    source or on any other validator."""
    if not validator_bench_configured(cfg):
        return
    owner = str(getattr(getattr(cfg, "netconfig", None), "owner_hotkey", "") or "")
    hk = getattr(signer, "hotkey", signer)
    if not owner or str(getattr(hk, "ss58_address", "") or "") != owner:
        return
    try:
        hk.sign(b"cascade-validator-bench-probe")
    except Exception as e:  # noqa: BLE001 — a public-only key cannot sign
        raise RuntimeError(
            f"this validator's hotkey is [netconfig] owner_hotkey ({owner}) — the MAIN "
            f"validator of DEC-CA-0063 — but its wallet cannot sign ({type(e).__name__}: "
            f"{e}); refusing to start rather than publish unsigned bench reports") from e


class ValidatorBenchMixin:
    """The main validator's winners-only lane-A sweep and its signed report."""

    # ── identity / activation ────────────────────────────────────────────────

    def _vbench_active(self, manifest: TrainingManifest) -> bool:
        return validator_bench_at(self.cfg, self._epoch_start_block(manifest))

    def _vbench_main_hotkey(self) -> str:
        return str(getattr(getattr(self.cfg, "netconfig", None), "owner_hotkey", "") or "")

    def _vbench_self_hotkey(self) -> str:
        signer = getattr(self, "bench_signer", None)
        hk = getattr(signer, "hotkey", signer)
        return str(getattr(hk, "ss58_address", "") or "")

    def _vbench_is_main(self) -> bool:
        main = self._vbench_main_hotkey()
        return bool(main) and self._vbench_self_hotkey() == main

    def _vbench_require_fev(self, manifest: TrainingManifest) -> bool:
        return bench_gate_rule_at(self.cfg, self._epoch_start_block(manifest)) == BENCH_RULE_V2

    # ── the trainer's record (the arbiter) ──────────────────────────────────

    def _vbench_record(self, round_id: str) -> str | None:
        """The trainer's signed record of ``round_id``'s validator bench
        report — the sha256 of its exact text, or :data:`RECORD_ABSENT` —
        from the EARLIEST later manifest that records it, else ``None`` (not
        recorded yet). The later manifests are the ones the settlement walk
        has pending after the one being judged (``_vbench_later_raw``)."""
        rid = str(round_id)
        for raw in getattr(self, "_vbench_later_raw", None) or ():
            m = raw if isinstance(raw, TrainingManifest) else self._vbench_parse_later(raw)
            if m is None:
                continue
            rec = (m.validator_bench_reports or {}).get(rid)
            if rec:
                return str(rec)
        return None

    def _vbench_parse_later(self, raw: str) -> TrainingManifest | None:
        cache = self.__dict__.setdefault("_vbench_parsed", {})
        if raw not in cache:
            try:
                cache[raw] = load_manifest(raw)
            except (ValueError, KeyError, TypeError):
                cache[raw] = None
            while len(cache) > 64:
                cache.pop(next(iter(cache)))
        return cache[raw]

    # ── the settlement's report ──────────────────────────────────────────────

    def _vbench_begin(self, manifest: TrainingManifest) -> None:
        """Open this settlement's validator bench state (a no-op off the
        validator source). Never reads the report itself: a validator needs
        it only when a clearer reaches the bench gate or the king check is
        due (:meth:`_vbench_resolve`, lazily). The main validator SWEEPS only
        while the round has neither a published report (the first one is
        final) nor a trainer record; otherwise it reads, exactly like every
        other validator. A store READ failure while checking for its own
        published report raises :class:`StorageError` (hold)."""
        self._vbench = None
        if not self._vbench_active(manifest):
            return
        from ..shared.bench_report import validator_bench_report_key

        rid = str(manifest.round_id)
        st: dict = {"round_id": rid, "main": self._vbench_is_main(),
                    "key": validator_bench_report_key(rid), "mode": "read",
                    "resolved": False, "absent": False, "report": None, "sha256": "",
                    "entries": {}, "rows": {}, "revisions": {}, "failed": {},
                    "deferred": {}, "faults": {}, "roles": {},
                    "requested": set(), "spent": 0.0,
                    "require_fev": self._vbench_require_fev(manifest)}
        if (st["main"] and self._vbench_record(rid) is None
                and self._vbench_own_published(rid) is None):
            st["mode"] = "sweep"
            prev = str(manifest.prev_round_id or "")
            if prev:
                self._vbench_seed_cache(prev)
        self._vbench = st

    def _vbench_own_published(self, round_id: str) -> str | None:
        """Main validator: the text already published at this round's report
        key, else ``None``. A read failure raises :class:`StorageError`."""
        from ..shared.bench_report import validator_bench_report_key
        from ..shared.hippius import ObjectNotFound

        try:
            return self._bench_store().get_text(validator_bench_report_key(round_id))
        except ObjectNotFound:
            return None
        except Exception as e:  # noqa: BLE001 — store down: whether a final report exists is unknown
            raise StorageError(f"validator_bench_report_unavailable: round={round_id}: "
                               f"{e}") from e

    def _vbench_resolve(self, st: dict) -> None:
        """Read mode: resolve this settlement's report through the trainer's
        record. No record yet ⇒ :class:`ValidatorBenchHold` (every validator
        holds alike); "absent" ⇒ every requested side is infra; a sha256 ⇒
        the published text, which must hash to it (else uncomputable), then
        its signature (else uncomputable)."""
        if st["resolved"] or st["mode"] != "read":
            return
        from ..shared.bench_report import load_bench_report, verify_bench_report_signature
        from ..shared.hippius import ObjectNotFound

        rid = st["round_id"]
        record = self._vbench_record(rid)
        if record is None:
            raise ValidatorBenchHold(
                f"validator_bench_unrecorded: round={rid} — the trainer has not yet recorded "
                "the main validator's bench report in a later settlement's manifest")
        st["resolved"] = True
        if record == RECORD_ABSENT:
            st["absent"] = True
            log.warning("validator-bench round=%s: the trainer recorded the report ABSENT — "
                        "its clearers are infra (skipped, never a loss)", rid)
            return
        try:
            text = self._bench_store().get_text(st["key"])
        except ObjectNotFound as e:
            st["resolved"] = False
            raise ValidatorBenchHold(f"validator_bench_report_missing: round={rid} (recorded "
                                     f"{record[:12]}… but not readable)") from e
        except Exception as e:  # noqa: BLE001 — store down / auth / network
            st["resolved"] = False
            raise ValidatorBenchHold(f"validator_bench_report_unavailable: round={rid}: "
                                     f"{e}") from e
        st["sha256"] = record
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != record:
            log.error("validator-bench round=%s: published bytes differ from the trainer's "
                      "record %s… — every entry uncomputable", rid, record[:12])
            return
        try:
            report = load_bench_report(text)
            if self.verify_signatures and not verify_bench_report_signature(
                    report, self._vbench_main_hotkey()):
                log.warning("validator-bench round=%s: report fails the main validator's "
                            "signature — every entry uncomputable", rid)
                report = None
        except (ValueError, KeyError, TypeError) as e:
            log.warning("validator-bench round=%s: report unparseable (%s) — every entry "
                        "uncomputable", rid, e)
            report = None
        st["report"] = report

    def _vbench_seed_cache(self, round_id: str) -> None:
        """Main validator: load its own published report of ``round_id`` into
        the sweep cache. Best-effort — anything missing is simply swept."""
        from ..shared.bench_report import (
            load_bench_report,
            load_bench_rows,
            validator_bench_report_key,
            verify_bench_report_signature,
        )

        try:
            text = self._bench_store().get_text(validator_bench_report_key(round_id))
            report = load_bench_report(text)
            if self.verify_signatures and not verify_bench_report_signature(
                    report, self._vbench_self_hotkey()):
                return
        except Exception:  # noqa: BLE001 — absent / unreadable: sweep instead
            return
        n = 0
        for e in report.entries:
            if e.trained_pointer in self._vbench_cache:
                continue
            try:
                rows = load_bench_rows(self._bench_store().get_text(e.rows_key), e.rows_sha256)
            except Exception:  # noqa: BLE001
                continue
            self._vbench_cache_put(e.trained_pointer, _scores_dict(e.scores), rows,
                                   e.data_revisions or {})
            n += 1
        if n:
            log.info("validator-bench: re-used %d sweep(s) from my report of round=%s",
                     n, round_id)

    def _vbench_cache_put(self, pointer: str, scores: dict, rows: dict,
                          revisions: dict | None = None) -> None:
        cache = self._vbench_cache
        cache.pop(pointer, None)
        cache[pointer] = {"scores": dict(scores), "rows": rows,
                          "revisions": dict(revisions or {})}
        while len(cache) > VBENCH_CACHE_MAX:
            cache.pop(next(iter(cache)))

    def _vbench_drop(self, pointer: str) -> None:
        """Main validator, sweep mode: forget ``pointer``'s rows (cache and
        this settlement's entries) — its data revision differs from the
        challenger's, so it is swept again."""
        self._vbench_cache.pop(pointer, None)
        st = self._vbench
        if st is not None:
            for k in ("entries", "rows", "revisions"):
                st[k].pop(pointer, None)
            st.get("from_cache", set()).discard(pointer)

    # ── one checkpoint's rows ────────────────────────────────────────────────

    def _vbench_side(self, entry: TrainedEntry, *, role: str, need_rows: bool = True,
                     reserve_for: TrainedEntry | None = None,
                     capped: bool = True,
                     pair_with: tuple[TrainedEntry, str] | None = None,
                     ) -> tuple[dict | None, dict | None]:
        """``(item rows, suite-level scores)`` for ``entry`` from this
        settlement's validator bench report — swept here on the main
        validator (cached per pointer) while the round has no report, read
        through the trainer's record elsewhere. ``(None, None)`` = no numbers
        (see :meth:`_vbench_outcome` for whether that is infra, a deferral, a
        checkpoint fault or uncomputable). Sweep mode: ``capped`` = the sweep
        may start only while the settlement cap holds its whole allowance
        (plus ``reserve_for``'s — the king a clearer's gate sweeps next — when
        that one is not swept yet); the king swept inside a clearer's gate
        passes ``capped=False`` (its slot was reserved). ``pair_with`` =
        ``(entry, role)`` of the checkpoint this decision reads next (the king
        of a clearer's gate; the init of the king check): when it needs a
        sweep too, both are swept in ONE paired sidecar pass
        (:meth:`_vbench_paired_pass`). Raises
        :class:`ValidatorBenchHold` while the record is pending,
        :class:`StorageError` on a rows READ failure."""
        st = self._vbench
        if st is None:
            return None, None
        ptr = entry.trained_pointer
        st["requested"].add(ptr)
        if st["mode"] == "read":
            self._vbench_resolve(st)
            return self._vbench_read_side(st, ptr, need_rows=need_rows)
        if ptr in st["entries"]:
            return st["rows"][ptr], _scores_dict(st["entries"][ptr].scores)
        if ptr in st["failed"] or ptr in st["deferred"] or ptr in st["faults"]:
            return None, None
        hit = self._vbench_cache_hit(ptr, st)
        if hit is None:
            got = self._vbench_sweep_bounded(entry, role=role, st=st, capped=capped,
                                             reserve_for=reserve_for, pair_with=pair_with)
            if got is None:
                return None, None
            scores, rows, revisions = got
            self._vbench_cache_put(ptr, scores, rows, revisions)
        else:
            scores, rows, revisions = hit["scores"], hit["rows"], hit.get("revisions") or {}
            st.setdefault("from_cache", set()).add(ptr)
            log.info("validator-bench round=%s: %s %s rows from the cache (no sweep)",
                     st["round_id"], role, entry.miner_hotkey[:12] or ptr[-16:])
        self._vbench_add(st, entry, role=role, scores=scores, rows=rows, revisions=revisions)
        return rows, dict(scores)

    def _vbench_cache_hit(self, ptr: str, st: dict) -> dict | None:
        hit = self._vbench_cache.get(ptr)
        if hit is not None and st["require_fev"] and hit["scores"].get("fev_crps") is None:
            hit = None                      # swept before FEV joined: sweep again
        # Custom sweep boundaries own their data discovery (including fixtures).
        # The real four-suite path must not reuse an old king/init forever
        # when BOTH sides are cached and the operator stages new data pins.
        if (hit is not None and st["require_fev"]
                and getattr(self, "bench_sweep_fn", None) is None
                and getattr(self, "bench_sweep_many_fn", None) is None):
            if "current_data_revisions" not in st:
                st["current_data_revisions"] = self._vbench_current_data_revisions()
            current = st["current_data_revisions"]
            if not current or hit.get("revisions") != current:
                hit = None
        return hit

    def _vbench_current_data_revisions(self):
        from .bench_identity import current_revisions

        data_dir = self.cfg.eval.gift_gate_data_dir or None
        host = self._eval_host()
        if host is None:
            return current_revisions(data_dir)
        import json
        import shlex

        from ..trainer.remote import build_ssh_argv, run_ssh

        args = [host.remote_python, "-m", "cascade.validator.bench_identity"]
        if data_dir:
            args += ["--data-dir", data_dir]
        try:
            command = (f"cd {shlex.quote(host.workdir)} && "
                       + " ".join(shlex.quote(a) for a in args))
            proc = run_ssh(build_ssh_argv(host, command), 30)
            identity = json.loads(proc.stdout) if proc.returncode == 0 else None
            return identity if isinstance(identity, dict) else None
        except Exception:
            return None  # unknown provenance forces a fresh sweep, never reuse

    def _vbench_outcome(self, entry: TrainedEntry) -> tuple[str, str] | None:
        """Why ``entry`` has no numbers, when the report says so:
        ``("infra", why)`` — a trainer-recorded "absent" report or an
        ``infra_failed`` entry; ``("deferred", why)`` — ``budget_deferred``
        (the settlement's sweep cap, item 9); ``("fault", why)`` —
        ``checkpoint_fault`` (item 10). ``None`` = present, or uncomputable.
        Read mode may raise :class:`ValidatorBenchHold`."""
        st = self._vbench
        if st is None:
            return None
        ptr = entry.trained_pointer
        if st["mode"] == "sweep":
            for kind, key in (("infra", "failed"), ("deferred", "deferred"),
                              ("fault", "faults")):
                if ptr in st[key]:
                    return kind, st[key][ptr]
            return None
        self._vbench_resolve(st)
        if st["absent"]:
            return "infra", "validator bench report recorded absent by the trainer"
        report = st["report"]
        if report is None:
            return None
        for kind, fn in (("infra", report.infra_reason), ("deferred", report.deferred_reason),
                         ("fault", report.fault_reason)):
            why = fn(ptr)
            if why is not None:
                return kind, why
        return None

    def _vbench_infra(self, entry: TrainedEntry) -> str | None:
        """Why ``entry`` has no numbers for a reason that is never ITS loss
        here — infra, a deferral, or (for a king / init, whose check this is)
        a checkpoint fault — else ``None`` (present, or uncomputable). Read
        mode may raise :class:`ValidatorBenchHold`."""
        out = self._vbench_outcome(entry)
        if out is None:
            return None
        kind, why = out
        return f"checkpoint fault: {why}" if kind == "fault" else why

    def _vbench_revisions(self, entry: TrainedEntry) -> dict:
        """The pinned data revisions ``entry``'s rows were swept on ({} =
        unknown: no check)."""
        st = self._vbench
        if st is None:
            return {}
        ptr = entry.trained_pointer
        if st["mode"] == "sweep":
            return dict(st["revisions"].get(ptr) or {})
        report = st["report"]
        e = report.entry_for_pointer(ptr) if report is not None else None
        return dict((e.data_revisions if e is not None else None) or {})

    def _vbench_cached(self, entry: TrainedEntry) -> bool:
        """Main validator, sweep mode: were ``entry``'s rows this settlement
        taken from the cache (not swept now)?"""
        st = self._vbench
        return bool(st is not None and st["mode"] == "sweep"
                    and entry.trained_pointer in st.get("from_cache", set()))

    def _vbench_add(self, st: dict, entry: TrainedEntry, *, role: str, scores: dict,
                    rows: dict, revisions: dict | None = None) -> None:
        from ..shared.bench_report import BenchEntry, dump_bench_rows, validator_bench_rows_key

        ptr = entry.trained_pointer
        text = dump_bench_rows(rows)
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        st["entries"][ptr] = BenchEntry(
            role=role, size=str(entry.size or ""), miner_hotkey=str(entry.miner_hotkey or ""),
            miner_uid=int(entry.miner_uid), trained_pointer=ptr, scores=BenchScores(**scores),
            rows_key=validator_bench_rows_key(st["round_id"], ptr, sha), rows_sha256=sha,
            data_revisions=(dict(revisions) if revisions else None))
        st["rows"][ptr] = rows
        st["revisions"][ptr] = dict(revisions or {})

    def _vbench_read_side(self, st: dict, ptr: str, *,
                          need_rows: bool) -> tuple[dict | None, dict | None]:
        from ..shared.bench_report import load_bench_rows
        from ..shared.hippius import ObjectNotFound

        report = st["report"]
        e = report.entry_for_pointer(ptr) if report is not None else None
        if e is None:
            return None, None
        if not need_rows:
            return {}, _scores_dict(e.scores)
        if not e.rows_key:
            return None, None
        try:
            text = self._bench_store().get_text(e.rows_key)
        except ObjectNotFound:
            return None, None
        except Exception as exc:  # noqa: BLE001 — store down / auth / network
            raise StorageError(f"validator_bench_rows_unavailable: round={st['round_id']}: "
                               f"{exc}") from exc
        try:
            rows = load_bench_rows(text, e.rows_sha256)
        except Exception as exc:  # noqa: BLE001 — forged/malformed ⇒ uncomputable
            log.warning("validator-bench round=%s: rows for %s unusable: %s",
                        st["round_id"], ptr[-16:], exc)
            return None, None
        return rows, _scores_dict(e.scores)

    # ── the sweep ────────────────────────────────────────────────────────────

    def _vbench_clock(self) -> float:
        fn = getattr(self, "vbench_clock", None)
        return float(fn()) if callable(fn) else time.monotonic()

    def _vbench_needs_sweep(self, entry: TrainedEntry | None, st: dict) -> bool:
        """Sweep mode: would ``entry`` be swept now (not in this settlement's
        report yet, not cached)?"""
        if entry is None:
            return False
        ptr = entry.trained_pointer
        if any(ptr in st[k] for k in ("entries", "failed", "deferred", "faults")):
            return False
        pre = st.get("paired", {}).get(ptr)
        if pre is not None and pre[0] is not None:
            return False                    # a paired pass already holds its rows
        return self._vbench_cache_hit(ptr, st) is None

    def _vbench_paired_on(self) -> bool:
        """``[eval] bench_paired``, and a sweep that can take several
        checkpoints (the sidecar, or an injected ``bench_sweep_many_fn``; the
        per-entry ``bench_sweep_fn`` test hook alone sweeps one at a time)."""
        if not bool(getattr(self.cfg.eval, "bench_paired", False)):
            return False
        return (self.bench_sweep_fn is None
                or getattr(self, "bench_sweep_many_fn", None) is not None)

    def _vbench_paired_pass(self, members: list[tuple[TrainedEntry, str]], st: dict,
                            allowance: float) -> None:
        """ONE sidecar pass over every member (each benchmark dataset built
        once, forecast by each checkpoint; each report byte-identical to its
        solo sweep). Charging (DEC-CA-0063 items 9-10): the pass may run for
        the members' combined allowance; its wall time counts ONCE, in full,
        toward the settlement cap and in EQUAL shares toward each member's own
        ``bench_sweep_hours``; for each member it is that member's first
        attempt. The outcome is parked in ``st["paired"]`` and consumed by the
        member's own :meth:`_vbench_sweep_bounded` (a failed member retries
        alone within what is left of its allowance)."""
        n = len(members)
        t0 = self._vbench_clock()
        got = self._vbench_sweep_many([e for e, _ in members], require_fev=st["require_fev"],
                                      timeout_s=max(1, int(allowance * n)))
        dt = max(0.0, self._vbench_clock() - t0)
        st["spent"] += dt
        for (e, _role), res in zip(members, got, strict=True):
            st.setdefault("paired", {})[e.trained_pointer] = (res[0], res[1], dt / n)
        log.info("validator-bench round=%s: paired pass over %s took %.2f h (%.2f h charged "
                 "to each)", st["round_id"], "+".join(r for _, r in members), dt / 3600.0,
                 dt / n / 3600.0)

    def _vbench_sweep_bounded(self, entry: TrainedEntry, *, role: str, st: dict,
                              capped: bool = True,
                              reserve_for: TrainedEntry | None = None,
                              pair_with: tuple[TrainedEntry, str] | None = None,
                              ) -> tuple[dict, dict, dict] | None:
        """Sweep ``entry`` with up to :data:`VBENCH_SWEEP_ATTEMPTS` attempts
        inside its OWN allowance (``[v2] bench_sweep_hours``), every second
        counted toward the settlement cap (``[v2] bench_budget_hours``).
        ``capped``: start only while the cap holds the whole allowance (+
        ``reserve_for``'s when it still needs a sweep) — otherwise a
        challenger is recorded ``budget_deferred`` and a king / init (the
        king check) ``infra_failed``. ``None`` ⇒ recorded in ``st``
        (deferred / failed / faults)."""
        v2 = self.cfg.v2
        cap = float(getattr(v2, "bench_budget_hours", 4.0)) * 3600.0
        allowance = float(getattr(v2, "bench_sweep_hours", 2.0)) * 3600.0
        ptr = entry.trained_pointer
        who = entry.miner_hotkey[:12] or ptr[-16:]
        st["roles"][ptr] = role
        pre = st.get("paired", {}).pop(ptr, None)   # swept by an earlier paired pass
        if pre is None and capped:
            need = allowance * (2 if self._vbench_needs_sweep(reserve_for, st) else 1)
            if cap - st["spent"] < need:
                if role == "challenger":
                    st["deferred"][ptr] = BUDGET_DEFERRED_REASON
                    log.warning("validator-bench round=%s: %s %s is BUDGET-DEFERRED (%.2f h "
                                "of the %.2f h cap used, %.2f h needed) — never swept, never "
                                "a loss; re-judged at a later settlement", st["round_id"],
                                role, who, st["spent"] / 3600.0, cap / 3600.0, need / 3600.0)
                else:
                    st["failed"][ptr] = "bench budget exhausted"
                    log.warning("validator-bench round=%s: %s %s — bench budget exhausted; "
                                "INFRA-FAILED", st["round_id"], role, who)
                return None
        if (pre is None and pair_with is not None and self._vbench_paired_on()
                and pair_with[0].trained_pointer != ptr
                and self._vbench_needs_sweep(pair_with[0], st)
                and pair_with[0].trained_pointer not in st.get("paired", {})
                and cap - st["spent"] >= 2 * allowance):
            self._vbench_paired_pass([(entry, role), pair_with], st, allowance)
            pre = st["paired"].pop(ptr)
        used = 0.0
        failures: list[tuple[str, str]] = []
        first = 1
        if pre is not None:
            got, failure, share = pre
            used = share                    # already counted toward the cap
            if got is not None:
                log.info("validator-bench round=%s: %s %s swept in a paired pass",
                         st["round_id"], role, who)
                return got
            failures.append(failure)
            first = 2
            log.warning("validator-bench round=%s: sweep of %s %s FAILED (paired pass, "
                        "attempt 1/%d, %s: %s)", st["round_id"], role, who,
                        VBENCH_SWEEP_ATTEMPTS, failure[0], failure[1])
        for attempt in range(first, VBENCH_SWEEP_ATTEMPTS + 1):
            left = allowance - used
            if left <= 0:
                break
            t0 = self._vbench_clock()
            got, failure = self._vbench_sweep(entry, require_fev=st["require_fev"],
                                              timeout_s=max(1, int(left)))
            dt = max(0.0, self._vbench_clock() - t0)
            used += dt
            st["spent"] += dt
            if got is not None:
                if attempt > 1:
                    log.info("validator-bench round=%s: %s %s swept on attempt %d",
                             st["round_id"], role, who, attempt)
                return got
            failures.append(failure)
            log.warning("validator-bench round=%s: sweep of %s %s FAILED (attempt %d/%d, "
                        "%s: %s)", st["round_id"], role, who, attempt, VBENCH_SWEEP_ATTEMPTS,
                        failure[0], failure[1])
        from ..eval.benchmarks import SWEEP_FAULT_CHECKPOINT

        if (len(failures) == VBENCH_SWEEP_ATTEMPTS
                and all(k == SWEEP_FAULT_CHECKPOINT for k, _ in failures)):
            # reproduced on every attempt, positive checkpoint evidence only
            st["faults"][ptr] = failures[-1][1]
            log.warning("validator-bench round=%s: %s %s is a CHECKPOINT FAULT (%s) — "
                        "published as such%s", st["round_id"], role, who, failures[-1][1],
                        "; the clearer fails the bench gate" if role == "challenger" else "")
            return None
        n = len(failures)
        reason = (f"sweep failed ({n} attempt{'s' if n != 1 else ''})"
                  + (f": {failures[-1][1]}" if failures else ""))
        if n < VBENCH_SWEEP_ATTEMPTS:
            reason += "; sweep allowance exhausted"
        st["failed"][ptr] = reason[:300]
        log.warning("validator-bench round=%s: %s %s is INFRA-FAILED (%s) — published as "
                    "such; never the miner's loss", st["round_id"], role, who, reason)
        return None

    def _vbench_sweep(self, entry: TrainedEntry, *, require_fev: bool,
                      timeout_s: int | None = None,
                      ) -> tuple[tuple[dict, dict, dict] | None, tuple[str, str]]:
        """Sweep one checkpoint on this validator's GPU (or the eval-offload
        pod) → ``((scores, item rows, data revisions), ("", ""))``, or
        ``(None, (kind, reason))`` on any failure (never raises) — ``kind``
        from :func:`~cascade.eval.benchmarks.classify_sweep_failure`; a raised
        exception (fetch, sidecar launch, offload) is always infra.
        ``bench_sweep_fn`` (tests) returns the raw sidecar report."""
        from ..eval.benchmarks import SWEEP_FAULT_INFRA

        try:
            if self.bench_sweep_fn is not None:
                report = self.bench_sweep_fn(entry)
            else:
                report = self._vbench_run_sidecar(entry, require_fev=require_fev,
                                                  timeout_s=timeout_s)
        except Exception as e:  # noqa: BLE001 — a raised sweep failure is infra
            log.warning("validator-bench: sweep of %s raised %s: %s",
                        entry.trained_pointer[-16:], type(e).__name__, e)
            return None, (SWEEP_FAULT_INFRA, f"sweep raised {type(e).__name__}")
        return self._vbench_parse_sweep(report, require_fev=require_fev)

    def _vbench_sweep_many(self, entries: list[TrainedEntry], *, require_fev: bool,
                           timeout_s: int | None = None,
                           ) -> list[tuple[tuple[dict, dict, dict] | None, tuple[str, str]]]:
        """:meth:`_vbench_sweep` for several checkpoints in one paired pass —
        one ``(got, failure)`` per entry; a raised exception is infra for all."""
        from ..eval.benchmarks import SWEEP_FAULT_INFRA

        try:
            fn = getattr(self, "bench_sweep_many_fn", None)
            if fn is not None:
                reports = list(fn(list(entries)))
            else:
                reports = self._vbench_run_sidecar_many(entries, require_fev=require_fev,
                                                        timeout_s=timeout_s)
            if len(reports) != len(entries):
                raise RuntimeError(f"{len(reports)} reports for {len(entries)} checkpoints")
        except Exception as e:  # noqa: BLE001 — a raised sweep failure is infra
            log.warning("validator-bench: paired sweep of %s raised %s: %s",
                        "+".join(e_.trained_pointer[-16:] for e_ in entries),
                        type(e).__name__, e)
            return [(None, (SWEEP_FAULT_INFRA, f"sweep raised {type(e).__name__}"))
                    for _ in entries]
        return [self._vbench_parse_sweep(r, require_fev=require_fev) for r in reports]

    @staticmethod
    def _vbench_parse_sweep(report: dict | None, *, require_fev: bool,
                            ) -> tuple[tuple[dict, dict, dict] | None, tuple[str, str]]:
        """One raw sidecar report → ``((scores, item rows, revisions), ("",
        ""))`` or ``(None, (kind, reason))``."""
        from ..eval.benchmarks import (
            classify_sweep_failure,
            extract_bench_scores,
            finite_bench_scores,
            item_rows_from_report,
        )

        scores = extract_bench_scores(report, require_fev=require_fev) if report else None
        rows = item_rows_from_report(report) if report else None
        if scores is None or rows is None or not finite_bench_scores(scores):
            return None, classify_sweep_failure(report, require_fev=require_fev)
        revs = report.get("data_revisions") if isinstance(report, dict) else None
        return (dict(scores), rows, ({str(k): str(v) for k, v in revs.items()}
                                     if isinstance(revs, dict) else {})), ("", "")

    def _vbench_run_sidecar(self, entry: TrainedEntry, *, require_fev: bool,
                            timeout_s: int | None = None) -> dict | None:  # pragma: no cover — GPU
        return self._vbench_run_sidecar_many([entry], require_fev=require_fev,
                                             timeout_s=timeout_s)[0]

    def _vbench_run_sidecar_many(self, entries: list[TrainedEntry], *, require_fev: bool,
                                 timeout_s: int | None = None,
                                 ) -> list[dict | None]:  # pragma: no cover — GPU
        """The sidecar on this box (or the eval-offload pod) over ``entries``
        in ONE pass, with ``[eval] bench_workers`` concurrency. Worker count
        and pairing are operator-local: the report is byte-identical either
        way, and only this (main) validator sweeps — every other one replays
        its signed rows (``_vbench_begin``: sweep mode needs ``st["main"]``)."""
        ec = self.cfg.eval
        suites = SUITES_V1 + (("fev",) if require_fev else ())
        cap = (int(ec.bench_hold_max_hours * 3600)
               if ec.cascade_bench_max_series == 0 else 1800) * max(1, len(entries))
        timeout_s = min(cap, int(timeout_s)) if timeout_s else cap
        num_samples = ec.benchmark_num_samples or ec.num_samples
        workers = getattr(ec, "bench_workers", "1")
        dests = [self._fetch_checkpoint_dir(e) for e in entries]
        host = self._eval_host()
        log.info("validator-bench: sweeping %s on %s (suites=%s, max_series=%d, workers=%s, "
                 "timeout=%ds)", "+".join(e.trained_pointer[-24:] for e in entries),
                 host.name if host is not None else self.device, ",".join(suites),
                 ec.cascade_bench_max_series, workers, timeout_s)
        if host is not None:
            from .eval_offload import run_bench_via_host_many

            return run_bench_via_host_many(
                host, dests, suites=",".join(suites), num_samples=num_samples,
                max_series=ec.cascade_bench_max_series,
                data_dir=(ec.gift_gate_data_dir or None), device="cuda", timeout_s=timeout_s,
                workers=workers)
        from ..eval.benchmarks import run_bench_reports

        return run_bench_reports(
            dests, project_dir=ec.benchmark_project_dir, suites=suites,
            num_samples=num_samples, max_series=ec.cascade_bench_max_series,
            device=self.device, data_dir=(ec.gift_gate_data_dir or None), timeout_s=timeout_s,
            workers=workers)

    # ── publish + receipt stamp ──────────────────────────────────────────────

    def _vbench_publish(self, manifest: TrainingManifest) -> None:
        """Main validator, sweep mode: sign and publish this settlement's
        report (rows first) — ALWAYS on a validator-sourced settlement, even
        with no entries, so the trainer can record it. Write-once: a report
        already at the key (a concurrent / earlier publish) is never
        overwritten — this validator then reads it like every other one. A
        publish or signing failure raises :class:`StorageError` (the caller
        rolls the settlement back and holds it). A no-op elsewhere."""
        from ..shared.bench_report import (
            BenchReport,
            dump_bench_report,
            dump_bench_rows,
            publish_validator_bench_report,
            sign_bench_report,
        )

        st = self._vbench
        if (st is None or not st["main"] or st["mode"] != "sweep"
                or st["round_id"] != str(manifest.round_id)):
            return
        roles = st["roles"]

        def _listed(d: dict) -> tuple:
            return tuple({"trained_pointer": p, "role": str(roles.get(p, "")),
                          "reason": str(r)} for p, r in sorted(d.items()))

        report = BenchReport(
            round_id=st["round_id"], created_block=int(manifest.created_block),
            entries=tuple(st["entries"].values()), infra_failed=_listed(st["failed"]),
            budget_deferred=_listed(st["deferred"]), checkpoint_fault=_listed(st["faults"]))
        signer = getattr(self, "bench_signer", None)
        if signer is None:
            raise StorageError(f"validator_bench_unsigned: round={st['round_id']} — the main "
                               "validator has no signing wallet (refusing to publish unsigned)")
        try:
            report = sign_bench_report(report, signer)
        except ValueError as e:
            raise StorageError(f"validator_bench_unsigned: round={st['round_id']}: {e}") from e
        store = self._bench_store()
        text = dump_bench_report(report)
        existing = self._vbench_own_published(st["round_id"])
        if existing is not None:
            log.error("validator-bench round=%s: a report is ALREADY published at %s — the "
                      "first one is final; reading it instead of overwriting", st["round_id"],
                      st["key"])
            st.update(mode="read", resolved=False, report=None, sha256="")
            return
        try:
            for e in report.entries:
                rows_text = dump_bench_rows(st["rows"][e.trained_pointer])
                try:
                    store.put_text(e.rows_key, rows_text, content_type="application/json",
                                   acl="public-read")
                except StorageError:
                    store.put_text(e.rows_key, rows_text, content_type="application/json")
            publish_validator_bench_report(store, text, st["round_id"])
        except Exception as e:  # noqa: BLE001 — the fleet waits on this object
            raise StorageError(f"validator_bench_publish_failed: round={st['round_id']}: "
                               f"{e}") from e
        st["published_sha"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        log.info("validator-bench round=%s: published %d entr%s [%s] + %d infra-failed + %d "
                 "budget-deferred + %d checkpoint-fault → %s", st["round_id"],
                 len(report.entries), "y" if len(report.entries) == 1 else "ies",
                 ", ".join(e.role for e in report.entries), len(report.infra_failed),
                 len(report.budget_deferred), len(report.checkpoint_fault), st["key"])

    def _vbench_awaiting_record(self, manifest: TrainingManifest) -> bool:
        """Must this settlement HOLD for the trainer's record? True on the
        main validator's sweep pass that needed the report (a clearer reached
        the gate, or the king check ran): it publishes, then holds with every
        other validator, and re-judges from the recorded bytes."""
        st = getattr(self, "_vbench", None)
        return bool(st is not None and st["round_id"] == str(manifest.round_id)
                    and st["mode"] == "sweep" and st["requested"])

    def _vbench_receipt_kw(self, manifest: TrainingManifest) -> dict:
        """``{"bench_report": {key, sha256, signer}}`` (or ``{key, signer,
        record: "absent"}``) for a settlement that judged on the validator
        report, else ``{}`` (receipt bytes unchanged — also on a
        validator-sourced settlement that never needed it)."""
        st = getattr(self, "_vbench", None)
        if st is None or st["round_id"] != str(manifest.round_id) or not st["resolved"]:
            return {}
        if st["absent"]:
            return {"bench_report": {"key": st["key"], "record": RECORD_ABSENT,
                                     "signer": self._vbench_main_hotkey()}}
        if not st["sha256"]:
            return {}
        return {"bench_report": {"key": st["key"], "sha256": st["sha256"],
                                 "signer": self._vbench_main_hotkey()}}

    # ── later readers (reign log, promotion members) ─────────────────────────

    def _fetch_validator_bench_report(self, round_id: str, *,
                                      raise_unavailable: bool = False) -> Any:
        """The main validator's signed report of ``round_id`` (DEC-CA-0063),
        or ``None`` (absent, malformed, bad signature). The current
        settlement's resolved report is answered from memory.
        ``raise_unavailable`` (the promotion member check) turns an ABSENT or
        unreadable report into a raised :class:`StorageError` — a hold, the
        trainer-report path's semantics: a member's check-settlement report is
        published before its receipt, so "not there" is never a verdict."""
        from ..shared.bench_report import (
            load_bench_report,
            validator_bench_report_key,
            verify_bench_report_signature,
        )
        from ..shared.hippius import ObjectNotFound

        rid = str(round_id)
        st = getattr(self, "_vbench", None)
        if st is not None and st["round_id"] == rid and st["report"] is not None:
            return st["report"]
        cached = self._vbench_reports.get(rid)
        if cached is not None:
            return cached
        try:
            text = self._bench_store().get_text(validator_bench_report_key(rid))
        except ObjectNotFound as e:
            if raise_unavailable:
                raise StorageError(f"validator_bench_report_unpublished: round={rid}") from e
            return None
        except Exception as e:  # noqa: BLE001 — store down / auth / network
            log.warning("validator-bench report round=%s could not be READ (%s: %s)",
                        rid, type(e).__name__, str(e)[-300:])
            if raise_unavailable:
                raise StorageError(f"validator_bench_report_unavailable: round={rid}: "
                                   f"{e}") from e
            return None
        try:
            report = load_bench_report(text)
        except (ValueError, KeyError, TypeError):
            return None
        if self.verify_signatures and not verify_bench_report_signature(
                report, self._vbench_main_hotkey()):
            return None
        self._vbench_reports[rid] = report
        while len(self._vbench_reports) > 16:
            self._vbench_reports.pop(next(iter(self._vbench_reports)))
        return report
