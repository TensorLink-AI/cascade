"""DEC-CA-0055 v2 king check, lane-A ladder bookkeeping and seed epochs.

At the settlement where the v2 check is due (:func:`cascade.shared.era.
v2_check_due` — a pass's warm start would land ``[v2]
reign_warm_start_blocks`` into the reign, the reign counted from no earlier
than the random-init window's end) every validator checks the reigning king's
checkpoint against the CURRENT init:

1. the private pool — the increment-unit LCB of king − init (the init in the
   king's slot and as the baseline: :func:`cascade.eval.koth.evaluate_round`,
   ``margin_mode = "increment"``) clears the dethrone margin in force;
2. the signed bench report (the trainer's; or, from ``[v2]
   bench_source_from_block``, the main validator's own sweep of THIS
   settlement — DEC-CA-0063) — under DEC-CA-0054 the king's mean CRPS and
   mean MASE on each of GIFT-Eval, BOOM and TIME strictly below the ``role =
   "init"`` entry's (six comparisons, all must hold); under DEC-CA-0061 rule
   v2 the gate's own rule (:func:`king_check_gate`: suite-equal average task
   ratio < 1, no suite's ratio LCB above 1 + tolerance at alpha/4, win rate
   >= min) with the king as challenger and the init as reference, on both
   entries' signed task rows (missing rows / too few tasks defer);
3. the king was not dethroned in the window (a dethrone restarts the clock,
   so a due check implies it).

The bench report read is the PREVIOUS settlement's (``prev_round_id``): the
check settlement's own report is published after its manifest, so it is never
there when the settlement is judged. No report, no king entry or no init
entry ⇒ ``"deferred"`` (re-checked next settlement; no share change, clock
not reset). Pass ⇒ ladder +10 % and the pass is the ONLY evidence a v2
promotion is accepted on (exactly the passing king pointer, k = 1); fail ⇒
ladder 10 %, no promotion. The clock restarts at the check settlement either
way. The record is stamped on the receipt (``king_check``); the trainer reads
it to decide whether to fire the promotion.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from ..shared.emission import (
    KeySlash,
    LadderState,
    crown_start,
    crowned_in_v2,
    key_slash_step,
    ladder_on,
    ladder_params,
    ladder_record,
    ladder_step,
)
from ..shared.era import (
    BENCH_RULE_V1,
    BENCH_RULE_V2,
    EraSpec,
    bench_gate_rule_at,
    min_effective_era,
    v2_check_due,
    v2_era_seed_block,
    v2_first_era,
    v2_first_warm_era,
    v2_settlement_active,
)
from ..shared.manifest import BenchScores, TrainedEntry, TrainingManifest, bench_scores_body

log = logging.getLogger("cascade.validator")

# The ONE notice verdict that is final for every node: the earliest
# chain-bound announcement of the record exists and already lies past the
# notice window of its effective era. The chain only grows forward, so no
# later settlement can move that announcement — the record is NEVER
# effective. Trainer (``LegOps.promotion_generations``) and validators
# (``EraGateMixin._stage_promotion_record``) both skip the generation on it.
NOTICE_LATE = "v2 promotion was announced too late for its effective era"

# The six suite means of the bench check, in report order.
BENCH_FIELDS = ("gifteval_crps", "gifteval_mase", "boom_crps", "boom_mase",
                "time_crps", "time_mase")
# DEC-CA-0061 in force for the settlement: the gate's four suites, so FEV's
# two means are recorded too (the bench half is then the gate's rule over the
# king's and the init's task rows, :func:`king_check_gate`).
BENCH_FIELDS_V2 = (*BENCH_FIELDS, "fev_crps", "fev_mase")
RULE_V1, RULE_V2 = BENCH_RULE_V1, BENCH_RULE_V2
# Deferral reason when the rule-v2 bench half is uncomputable (prefix; the
# gate's reason follows). The audit re-derives it.
KC_GATE_UNCOMPUTABLE = f"bench rule {RULE_V2} uncomputable: "


def bench_fields(rule: str = RULE_V1) -> tuple[str, ...]:
    return BENCH_FIELDS_V2 if rule == RULE_V2 else BENCH_FIELDS


def king_check_gate(king_rows, init_rows, king_scores, init_scores, params, *,
                    seed: int | str, rule: str = RULE_V2, king_revisions=None,
                    init_revisions=None):
    """CONSENSUS: the DEC-CA-0061 bench half of the king check — the gate's
    rule (average ratio, per-suite non-regression at ``alpha / 4``, win rate)
    with the KING as the challenger and the INIT as the reference, on their
    task rows. ``params`` = ``cfg.koth_params(block=boundary)``; the bootstrap
    seed is ``{seed}|king-check`` (``seed`` = the settlement's base seed, so
    the audit re-derives it). ``*_revisions`` = the DEC-CA-0063 pinned data
    revisions (a mismatch is uncomputable, as in the gate). Pure."""
    from ..eval.bench_gate import evaluate_bench_gate

    return evaluate_bench_gate(
        init_rows, king_rows, init_scores, king_scores, alpha=params.bootstrap_alpha,
        B=params.bootstrap_B, seed=f"{seed}|king-check", min_items=params.bench_gate_min_items,
        rule=rule, tolerance=params.bench_gate_tolerance,
        min_win_rate=params.bench_gate_min_win_rate, king_revisions=init_revisions,
        chal_revisions=king_revisions)


def king_check_gate_reason(reason: str) -> str:
    """The king check's failure reason for a failing rule-v2 bench half."""
    return f"bench rule {RULE_V2}: {reason}"


def bench_failures(king: BenchScores, init: BenchScores, rule: str = RULE_V1) -> list[str]:
    """CONSENSUS: the suite means where the king is NOT strictly below the
    init (empty = the bench half of the check passes) — the DEC-CA-0054
    king check (six GIFT-Eval/BOOM/TIME means). Under DEC-CA-0061 the bench
    half is :func:`king_check_gate` instead; ``rule`` RULE_V2 here only lists
    the eight means (a missing FEV mean counts as not below). Pure."""
    out = []
    for f in bench_fields(rule):
        kv, iv = getattr(king, f, None), getattr(init, f, None)
        if kv is None or iv is None:
            out.append(f"{f} missing (king {kv}, init {iv})")
            continue
        k, i = float(kv), float(iv)
        if not k < i:
            out.append(f"{f} king {k:.6g} >= init {i:.6g}")
    return out


# Reason strings the audit re-derives (cascade.audit.checks.check_king_check).
POOL_INCONCLUSIVE = "pool check inconclusive (too few windows/clusters)"


def pool_reason(lcb: float, margin: float) -> str:
    """The pool half's failure reason, as the receipt records it."""
    return f"pool increment LCB {lcb:.4f} < margin {margin:.4f}"


class KingCheckMixin:
    """The v2 king check, the ladder step per settlement, seed-epoch helpers."""

    # ── gates ────────────────────────────────────────────────────────────────

    def _v2_ladder_active(self, boundary: int) -> bool:
        """``[v2] king_ladder`` and the settlement at ``boundary`` judges a v2
        era."""
        return ladder_on(self.cfg) and v2_settlement_active(self.cfg, int(boundary))

    def _v2_epochs_active(self, era: EraSpec) -> bool:
        """Seed epochs are part of the DEC-CA-0055 economics bundle: gated on
        ``[v2] king_ladder`` so an already-live v2 (testnet) keeps per-era seeds."""
        return ladder_on(self.cfg) and bool(self.cfg.v2.active(int(era.start_block)))

    def _king_check_for(self, round_id: str) -> dict | None:
        rec = getattr(self, "_settlement_king_check", None)
        return rec[1] if rec is not None and rec[0] == str(round_id) else None

    def _king_check_scores_for(self, round_id: str) -> tuple:
        """The pool rows the king check scored itself (not already on the
        receipt's ``entry_scores``) — the receipt's ``king_check_scores``,
        ``()`` whenever no check is stamped (receipt bytes unchanged)."""
        if self._king_check_for(round_id) is None:
            return ()
        rec = getattr(self, "_settlement_king_check_scores", None)
        return tuple(rec[1]) if rec is not None and rec[0] == str(round_id) else ()

    def _synced_reign_start(self) -> int | None:
        """The reign clock ``v2_check_due`` reads — None when the check would
        not even be considered (no cascade, no king, or a cascade view that
        has not adopted the champion yet: :meth:`_maybe_king_check`)."""
        if self.cascade is None or self.state.king_hotkey is None:
            return None
        cs = self.cascade.state
        if cs.king_hotkey != self.state.king_hotkey or cs.reign_start_block is None:
            return None
        return int(cs.reign_start_block)

    # ── seed epochs ──────────────────────────────────────────────────────────

    def _v2_warm_eras(self) -> tuple[int, ...]:
        """Effective eras of the v2-lineage generations this validator knows
        (live and staged): the starts of the seed epochs after the first."""
        if self.cascade is None:
            return ()
        warm = v2_first_warm_era(self.cfg)
        if warm is None:
            return ()
        st = self.cascade.state
        eras = [int(st.generation_effective_era or 0)]
        if st.pending_generation and st.pending_generation > st.generation and st.pending_members:
            # (a staged generation WITHOUT members is the skip marker of a
            # never-effective record — no warm start, no seed epoch)
            eras.append(int(st.pending_effective_era or 0))
        return tuple(e for e in eras if e >= warm)

    def _era_seed_block(self, era: EraSpec) -> int:
        """The seed block ``era`` trains from: its seed epoch's first era's
        (DEC-CA-0055) for a v2 era under the economics bundle, else its own."""
        if not self._v2_epochs_active(era):
            return int(era.seed_block)
        return v2_era_seed_block(self.cfg, era, self._v2_warm_eras())

    def _epoch_start(self, era: EraSpec) -> int | None:
        from ..shared.era import v2_seed_epoch_start

        if not self._v2_epochs_active(era):
            return None
        return v2_seed_epoch_start(self.cfg, int(era.index), self._v2_warm_eras())

    def _v2_pending_kept(self) -> bool:
        """Does the staged generation survive a dethrone? Only a v2 king-only
        promotion under the economics bundle (DEC-CA-0055)."""
        if self.cascade is None or not ladder_on(self.cfg):
            return False
        st = self.cascade.state
        warm = v2_first_warm_era(self.cfg)
        return bool(st.pending_generation and warm is not None
                    and int(st.pending_effective_era) >= warm)

    # ── the settlement step ──────────────────────────────────────────────────

    def _v2_settlement_economics(self, manifest: TrainingManifest, windows, base_seed: int,
                                 outcome, pre_king: str | None) -> None:
        """After a scored settlement is judged (before it is persisted): run
        the king check when due, then fold the ladder step, the court and the
        key slash into the champion state. Stashes this settlement's ladder
        transition and king check for its receipt. A no-op outside v2 eras /
        without ``[v2] king_ladder``."""
        rid = str(manifest.round_id)
        boundary = self._epoch_start_block(manifest)
        if not self._v2_ladder_active(boundary):
            return
        st = self.state
        prev_bp = st.king_share_bp
        dethroned = bool(outcome is not None and outcome.transition.dethroned)
        king = manifest.entry_for_role("king")
        funded_by = str(getattr(king, "funded_by", "") or "") if king is not None else ""
        check = None
        prev_key: KeySlash | None = None
        key = KeySlash()
        crown = None
        if dethroned:
            new_king = outcome.transition.new_king_hotkey
            court = pre_king if pre_king and pre_king != new_king else None
            # margin-tiered start share (owner 2026-10-20): from the crowned
            # challenger's verdict LCB — the number the dethrone bar judged —
            # in the units it was taken in (tiers in increment units only)
            crown = crown_start(outcome.result.lcb, outcome.lcb_units, self.cfg)
            if crown.below_floor:
                log.warning("round=%s crowning LCB %s is below the lowest ladder_start_tiers "
                            "threshold %s (config mismatch with the dethrone bar?) — "
                            "starting at the lowest tier, %d bp", rid, crown.lcb, crown.tier,
                            crown.start_bp)
            # a crowning resets the escalating key slash (a fresh reign)
            st = replace(st, king_share_bp=crown.start_bp, king_slashed=False,
                         king_check_pass=None,
                         king_key_lapses=0, king_lapse_open=False, king_slash_until_block=0,
                         court_hotkey=court,
                         court_until_block=((boundary + ladder_params(self.cfg)["court_blocks"])
                                            if court else 0))
            event = "crowned"
        else:
            reign_start = self._synced_reign_start()
            check = self._maybe_king_check(manifest, windows, base_seed)
            crowned = crowned_in_v2(self.cfg, st.king_since_block)
            result = None if check is None else check["result"]
            bp = (ladder_step(prev_bp, crowned=False, check=result, cfg=self.cfg)
                  if crowned else None)
            # DEC-CA-0055 escalating key slash (owner 2026-10-20): replayed
            # from the reign's state and this manifest's funded_by alone
            prev_key = KeySlash(lapses=st.king_key_lapses, lapse_open=st.king_lapse_open,
                                slash_until_block=st.king_slash_until_block)
            key = key_slash_step(funded_by=funded_by, boundary=boundary,
                                 lapses=prev_key.lapses, lapse_open=prev_key.lapse_open,
                                 slash_until_block=prev_key.slash_until_block, cfg=self.cfg)
            st = replace(st, king_share_bp=bp, king_slashed=key.slashed,
                         king_key_lapses=key.lapses, king_lapse_open=key.lapse_open,
                         king_slash_until_block=key.slash_until_block)
            if result == "pass":
                st = replace(st, king_check_pass={
                    "round_id": rid, "block": int(boundary),
                    "king_pointer": check["king_pointer"], "king_hotkey": check["king_hotkey"]})
            elif result == "fail":
                st = replace(st, king_check_pass=None)
            event = (result or "none") if crowned else "carried"
        self.state = st
        ladder = LadderState(king_share_bp=st.king_share_bp, court_hotkey=st.court_hotkey,
                             court_until_block=st.court_until_block, slashed=st.king_slashed)
        lad = ladder_record(
            prev_bp=prev_bp, bp=(st.king_share_bp if event != "carried" else None),
            event=event, state=ladder, funded_by=funded_by, boundary=boundary,
            key=key, prev_key=prev_key, crown=crown)
        if not dethroned and reign_start is not None:
            # the reign clock the due-ness was judged from (BEFORE a check
            # restarts it), so the audit re-derives v2_check_due (DEC-CA-0058)
            lad["reign_start_block"] = int(reign_start)
        self._settlement_ladder = (rid, lad)
        self._settlement_king_check = (rid, check) if check is not None else None
        rows = getattr(self, "_king_check_rows", None)
        self._settlement_king_check_scores = (
            (rid, tuple(rows[1])) if check is not None and rows is not None
            and rows[0] == rid and rows[1] else None)
        log.info("round=%s v2 ladder: %s%s (share %s → %s bp, court %s until %s, slashed=%s"
                 "%s, key lapses %d)", rid, event,
                 (f" [{crown.units} LCB {crown.lcb}, tier {crown.tier}]"
                  if crown is not None else ""), prev_bp, st.king_share_bp,
                 (st.court_hotkey or "-")[:12], st.court_until_block, st.king_slashed,
                 f" [{key.reason} until {lad['slash_until']}]" if key.slashed else "",
                 st.king_key_lapses)

    def _maybe_king_check(self, manifest: TrainingManifest, windows,
                          base_seed: int) -> dict | None:
        """The king check record when it is due at this settlement, else None.
        On pass/fail the reign clock restarts here; a deferral leaves it."""
        if self.cascade is None or self.state.king_hotkey is None:
            return None
        cs = self.cascade.state
        if cs.king_hotkey != self.state.king_hotkey:
            return None                    # an adoption the Cascade step has not run yet
        boundary = self._epoch_start_block(manifest)
        if not v2_check_due(self.cfg, boundary=boundary, reign_start_block=cs.reign_start_block):
            return None
        rec = self._v2_king_check(manifest, windows, base_seed)
        if rec["result"] in ("pass", "fail"):
            self.cascade.restart_clock(block=boundary)
        log.info("round=%s v2 KING CHECK %s: %s", manifest.round_id, rec["result"].upper(),
                 "; ".join(rec["reasons"]) or "the bench half and the pool LCB improve")
        return rec

    def _v2_king_check(self, manifest: TrainingManifest, windows, base_seed: int) -> dict:
        """Judge the king check (see the module docstring). Never raises: an
        unreadable report / unscorable init defers instead (the settlement is
        already judged — a raise here would re-judge it)."""
        boundary = self._epoch_start_block(manifest)
        self._king_check_rows = None
        # DEC-CA-0061: (king rows, init rows) for the rule-v2 bench half,
        # read only when that rule is in force
        self._king_check_bench_rows = None
        self._king_check_need_rows = bench_gate_rule_at(self.cfg, boundary) == RULE_V2
        king = manifest.entry_for_role("king")
        pointer = str(self.state.king_pointer or (king.trained_pointer if king else ""))
        rec: dict = {"result": "deferred", "round_id": str(manifest.round_id),
                     "block": int(boundary), "king_hotkey": self.state.king_hotkey,
                     "king_pointer": pointer, "reasons": [],
                     "bench_round": str(manifest.prev_round_id or ""),
                     "warm_start_era": int(min_effective_era(self.cfg.round, boundary))}
        if king is None or not pointer:
            rec["reasons"].append("no king entry on the settlement")
            return rec
        vb = getattr(self, "_vbench", None)
        if vb is not None and vb.get("round_id") == str(manifest.round_id):
            # DEC-CA-0063: king + init swept on the main validator's GPU (or
            # read from its signed report) at THIS settlement
            got = self._v2_king_check_validator_bench(manifest, king, pointer, rec)
            if got is None:
                return rec
            kb_scores, ib_scores, ib_pointer = got
        else:
            got = self._v2_king_check_trainer_bench(manifest, pointer, rec)
            if got is None:
                return rec
            kb_scores, ib_scores, ib_pointer = got
        # DEC-CA-0061 in force ⇒ the bench half is the gate's rule with the
        # king as challenger and the init as reference, on their task rows;
        # DEC-CA-0054 ⇒ six strict suite means
        rule = bench_gate_rule_at(self.cfg, boundary)
        gate = None
        if rule == RULE_V2:
            kr, ir, krev, irev = (getattr(self, "_king_check_bench_rows", None)
                                  or (None, None, None, None))
            if kr is None or ir is None:
                rec["reasons"].append(KC_GATE_UNCOMPUTABLE + "signed bench rows missing for "
                                      + ("king" if kr is None else "init"))
                return rec
            gate = king_check_gate(kr, ir, bench_scores_body(kb_scores),
                                   bench_scores_body(ib_scores),
                                   self.cfg.koth_params(block=boundary), seed=base_seed,
                                   king_revisions=krev, init_revisions=irev)
            if not gate.computed:
                rec["reasons"].append(KC_GATE_UNCOMPUTABLE + gate.reason)
                rec["bench"] = {"rule": rule, "gate": gate.detail()}
                return rec
            bench_bad = [] if gate.passed else [king_check_gate_reason(gate.reason)]
        else:
            bench_bad = bench_failures(kb_scores, ib_scores, rule)
        # DEC-CA-0058: the suite means the bench half compared, so the
        # audit replays it from the record and checks it against the
        # signed report (and, under DEC-CA-0061, the gate's detail: the
        # rule, the three numbers and the per-suite ratios)
        rec["bench"] = {"king": bench_scores_body(kb_scores),
                        "init": bench_scores_body(ib_scores),
                        "init_pointer": str(ib_pointer),
                        **({"rule": rule, "gate": gate.detail()} if gate is not None else {})}
        return self._v2_king_check_pool(manifest, windows, base_seed, king, ib_pointer,
                                        bench_bad, rec)

    def _v2_king_check_trainer_bench(self, manifest: TrainingManifest, pointer: str,
                                     rec: dict) -> tuple | None:
        """The bench half's inputs from the PREVIOUS settlement's
        trainer-signed report: ``(king scores, init scores, init pointer)``,
        or None with the deferral reason appended."""
        report = None
        if manifest.prev_round_id:
            try:
                report = self._fetch_bench_report(str(manifest.prev_round_id), require=pointer)
            except Exception as e:  # noqa: BLE001 — defer, never raise past the verdict
                rec["reasons"].append(f"bench report unreadable: {e}")
                return None
        if report is None:
            rec["reasons"].append("no signed bench report for the previous settlement")
            return None
        kb, ib = report.entry_for_pointer(pointer), report.init_entry()
        if kb is None:
            rec["reasons"].append("bench report has no entry for the king pointer")
            return None
        if ib is None:
            rec["reasons"].append("bench report has no init entry")
            return None
        if manifest.warm_start_ckpt and ib.trained_pointer != manifest.warm_start_ckpt:
            rec["reasons"].append(f"bench init entry {ib.trained_pointer!r} is not the era init "
                                  f"{manifest.warm_start_ckpt!r}")
            return None
        if getattr(self, "_king_check_need_rows", False):
            try:
                self._king_check_bench_rows = (self._king_check_entry_rows(kb),
                                               self._king_check_entry_rows(ib), None, None)
            except Exception as e:  # noqa: BLE001 — defer, never raise past the verdict
                rec["reasons"].append(f"bench report unreadable: {e}")
                return None
        return kb.scores, ib.scores, ib.trained_pointer

    def _king_check_entry_rows(self, e) -> dict | None:
        """One trainer-report entry's sha256-pinned task rows (DEC-CA-0061
        king check), or None when absent / unusable. A store READ failure
        raises :class:`~cascade.shared.hippius.StorageError` like the gate's
        rows read; the caller defers on it (never past the verdict)."""
        from ..shared.bench_report import load_bench_rows
        from ..shared.hippius import ObjectNotFound

        if not e.rows_key:
            return None
        try:
            return load_bench_rows(self._bench_store().get_text(e.rows_key), e.rows_sha256)
        except ObjectNotFound:
            return None
        except ValueError:
            return None                     # forged / malformed rows: uncomputable

    def _v2_king_check_validator_bench(self, manifest: TrainingManifest, king: TrainedEntry,
                                       pointer: str, rec: dict) -> tuple | None:
        """DEC-CA-0063: the bench half's inputs from THIS settlement's
        validator bench report — the main validator sweeps the king and the
        current init now (cached per pointer); every other validator reads
        the signed report. The init is the era's warm start, else (a
        random-init seed epoch) the init pointer the trainer's report
        publishes. ``(king scores, init scores, init pointer)`` or None with
        the deferral reason appended."""
        from ..shared.hippius import StorageError
        from .validator_bench import ValidatorBenchHold

        rec["bench_source"] = "validator"
        rec["bench_round"] = str(manifest.round_id)
        init_ptr = str(manifest.warm_start_ckpt or "")
        st = getattr(self, "_vbench", None)
        if st is not None and st.get("mode") == "read":
            # Every reader — the main validator re-judging included — takes the
            # init from the report it judges on (the main validator chose it):
            # never its own read of the trainer's report, which can fail on
            # one validator while the main validator recorded a verdict.
            try:
                self._vbench_resolve(st)
            except StorageError as e:
                raise ValidatorBenchHold(str(e)) from e
            if st.get("absent"):
                rec["reasons"].append("validator bench infra: validator bench report recorded "
                                      "absent by the trainer")
                return None
            report = st.get("report")
            ie = report.init_entry() if report is not None else None
            if ie is not None:
                init_ptr = str(ie.trained_pointer)
        elif not init_ptr and manifest.prev_round_id:
            # the main validator's sweep pass: the random-init epoch's init
            # pointer is in the trainer's report — a READ failure HOLDS (a
            # deferral here would differ from the verdict peers later read)
            try:
                prev = self._fetch_bench_report(str(manifest.prev_round_id),
                                                raise_unavailable=True)
            except StorageError as e:
                raise ValidatorBenchHold(f"king check init pointer unreadable: {e}") from e
            ie = prev.init_entry() if prev is not None else None
            init_ptr = str(ie.trained_pointer) if ie is not None else ""
        if not init_ptr:
            rec["reasons"].append("no init pointer (random-init epoch with no trainer-published "
                                  "init checkpoint)")
            return None
        size = str(king.size or "")
        king_e = TrainedEntry(miner_hotkey=king.miner_hotkey, miner_uid=king.miner_uid,
                              role="king", gen_ref=king.gen_ref, trained_pointer=pointer,
                              corpus_digest="", train_block=0, size=size)
        init_e = TrainedEntry(miner_hotkey="", miner_uid=-1, role="king", gen_ref="",
                              trained_pointer=init_ptr, corpus_digest="", train_block=0,
                              size=size)
        need = bool(getattr(self, "_king_check_need_rows", False))
        try:
            kr, ks = self._vbench_side(king_e, role="king", need_rows=need,
                                       pair_with=(init_e, "init"))
            ir, is_ = self._vbench_side(init_e, role="init", need_rows=need)
            infra = self._vbench_infra(king_e) or self._vbench_infra(init_e)
        except StorageError as e:
            # a pending trainer record / unreadable report HOLDS the settlement
            # (the caller rolls the judgement back) — never a local deferral
            raise ValidatorBenchHold(str(e)) from e
        except Exception as e:  # noqa: BLE001 — defer, never raise past the verdict
            rec["reasons"].append(f"validator bench unreadable: {e}")
            return None
        if infra is not None:
            rec["reasons"].append(f"validator bench infra: {infra}")
            return None
        if ks is None:
            rec["reasons"].append("validator bench report has no entry for the king pointer")
            return None
        if is_ is None:
            rec["reasons"].append("validator bench report has no init entry")
            return None
        if need:
            self._king_check_bench_rows = (kr, ir, self._vbench_revisions(king_e),
                                           self._vbench_revisions(init_e))
        try:
            return BenchScores(**ks), BenchScores(**is_), init_ptr
        except (TypeError, ValueError) as e:
            rec["reasons"].append(f"validator bench scores unusable: {e}")
            return None

    def _v2_king_check_pool(self, manifest: TrainingManifest, windows, base_seed: int,
                            king: TrainedEntry, init_pointer: str, bench_bad: list,
                            rec: dict) -> dict:
        """The pool half (the init scored on this settlement's windows) and
        the final pass / fail."""
        from ..eval.koth import evaluate_round
        from ..shared.receipt import EntryScores

        boundary = self._epoch_start_block(manifest)
        rid = str(manifest.round_id)
        fresh: list = []           # rows scored here, not on the receipt's entry_scores
        self._king_check_rows = (rid, fresh)
        try:
            training = self._training_for(manifest)
            size = king.size or training.arch_preset
            stash = getattr(self, "_king_eval", None)
            if stash is not None and stash[0] == rid:
                king_scores, base_scores, tenure = stash[1], stash[2], stash[3]
            else:
                king_scores, base_scores, tenure = (
                    self._evaluate(king, windows, training), None, None)
                fresh.append(EntryScores(
                    role="king", size=size, hotkey=king.miner_hotkey, uid=king.miner_uid,
                    scores=self._window_records(king_scores, manifest)))
            if base_scores is None:
                base_scores = self._evaluate(TrainedEntry(
                    miner_hotkey=self.BASELINE_HOTKEY, miner_uid=-1, role="king",
                    gen_ref="", trained_pointer=init_pointer, corpus_digest="",
                    train_block=0, size=size), windows, training)
                fresh.append(EntryScores(
                    role="init", size=size, hotkey=self.BASELINE_HOTKEY, uid=-1,
                    scores=self._window_records(base_scores, manifest)))
            if tenure is None:
                from ..shared.era import tenure_rounds_at

                tenure = tenure_rounds_at(
                    self.cfg.round, self.cfg.scoring, block=boundary,
                    tenure_rounds=self.state.tenure_rounds,
                    king_since_block=self.state.king_since_block)
            rec["tenure_rounds"] = int(tenure)
            params = self.cfg.king_check_params(boundary)
            res = evaluate_round(base_scores, king_scores, params, seed=base_seed,
                                 king_tenure_rounds=tenure, baseline_scores=base_scores)
        except Exception as e:  # noqa: BLE001 — defer, never raise past the verdict
            rec["reasons"].append(f"pool check unscorable: {type(e).__name__}: {e}")
            return rec
        rec.update(lcb=None if res.inconclusive else float(res.lcb), margin=float(res.margin))
        if res.inconclusive:
            rec["reasons"].append(POOL_INCONCLUSIVE)
            return rec
        reasons = list(bench_bad)
        if not res.challenger_wins_round:
            reasons.append(pool_reason(res.lcb, res.margin))
        rec["reasons"] = reasons
        rec["result"] = "fail" if reasons else "pass"
        return rec

    # ── v2 promotions (k = 1, backed by a pass) ──────────────────────────────

    def _v2_record(self, record) -> bool:
        """Is ``record`` a v2-lineage promotion judged under the DEC-CA-0055
        rules (``[v2] king_ladder`` and it fired at/after ``[v2] from_block``)?
        A toto2 record fired before the v2 block keeps the DEC-CA-0013 rules
        (and never initialises a v2 era: :func:`v2_generation_usable`)."""
        first = v2_first_era(self.cfg)
        return (ladder_on(self.cfg) and first is not None
                and int(getattr(record, "fired_block", 0) or 0) >= int(self.cfg.v2.from_block))

    def _v2_promotion_reason(self, record, *, attesting: bool) -> str | None:
        """Why a v2 promotion record is rejected, else None: exactly ONE member;
        ``effective_era == min_effective_era(fired_block)`` and at/after
        :func:`v2_first_warm_era`; not a checkpoint already promoted; and —
        when this validator's clock can attest — the member is the king pointer
        of the last PASSING king check this validator judged, fired at or after
        that check's settlement."""
        members = record.member_ids()
        if len(members) != 1:
            return f"v2 promotion declares {len(members)} member(s); k = 1 (the king only)"
        eff = int(record.effective_era)
        want = int(min_effective_era(self.cfg.round, int(record.fired_block)))
        if eff != want:
            return (f"v2 promotion effective_era {eff} != min_effective_era {want} of its "
                    f"fire block {record.fired_block}")
        warm = v2_first_warm_era(self.cfg)
        if warm is None or eff < warm:
            return f"v2 promotion takes effect at era {eff}, inside the random-init window"
        st = self.cascade.state
        if members[0] in set(st.members) | set(st.pending_members):
            return f"v2 promotion re-promotes {members[0]!r} (already the init)"
        if not attesting:
            return self._v2_notice_reason(record)
        passed = self.state.king_check_pass
        if not passed:
            return "v2 promotion is not backed by a passing king check this validator judged"
        if members[0] != passed.get("king_pointer"):
            return (f"v2 promotion member {members[0]!r} is not the passing check's king "
                    f"pointer {passed.get('king_pointer')!r}")
        if int(record.fired_block) < int(passed.get("block", 0)):
            return (f"v2 promotion fired at block {record.fired_block}, before the passing "
                    f"check at {passed.get('block')}")
        return self._v2_notice_reason(record)

    def _v2_notice_reason(self, record) -> str | None:
        """A late record cannot rewrite the init of an already committed era.

        Judged from the settlement chain alone (``_promotion_check_manifest``
        back through predecessor hashes to the record's fire block), so every
        node reaches the same verdict from the same bytes. :data:`NOTICE_LATE`
        is returned exactly when the EARLIEST bound announcement exists and is
        late — final (the chain never gains an earlier announcement); every
        other string is a rejection that a later gate may lift (not announced
        YET, an unlinked or forked chain). A manifest of the chain that cannot
        be READ raises :class:`~cascade.shared.hippius.StorageError` (the
        poll loop holds the settlement — never a verdict), the same contract
        as :meth:`~cascade.validator.era_gate.EraGateMixin._verify_merged_init`."""
        import hashlib

        from ..shared.era import era_for_block
        from ..shared.hippius import StorageError
        from ..shared.manifest import dump_manifest, verify_signature
        from .lane_b import LaneBTransient

        manifest = getattr(self, "_promotion_check_manifest", None)
        if manifest is None:
            return "v2 promotion lacks settlement-chain context"
        wanted = hashlib.sha256(record.canonical_body()).hexdigest()
        seen = set()
        # The trainer repeats the notice until the effective era starts, so the
        # newest copy can fall inside the notice era. Notice is judged from the
        # EARLIEST announcement bound into the chain.
        earliest = None
        for _ in range(10000):
            if manifest.round_id in seen:
                return "v2 promotion notice chain contains a cycle"
            seen.add(manifest.round_id)
            if int(manifest.created_block) < int(record.fired_block):
                break  # the record cannot have been announced before it fired
            if (manifest.promotion_notices or {}).get(str(record.generation)) == wanted:
                earliest = manifest
            previous = manifest.prev_round_id
            if not previous:
                break
            if not manifest.prev_manifest_sha256:
                if earliest is not None:
                    break  # older manifests are not bound; judge what is
                return "v2 promotion notice is not bound by predecessor hashes"
            try:
                parent = self._lane_b_load_settlement(previous)
            except LaneBTransient as e:
                raise StorageError(f"promotion_notice_unverifiable: {e}") from e
            if (manifest.prev_manifest_sha256 and hashlib.sha256(
                    dump_manifest(parent).encode()).hexdigest() != manifest.prev_manifest_sha256):
                return "v2 promotion notice manifest chain hash differs"
            manifest = parent
        if earliest is not None:
            created = int(earliest.created_block)
            if (min_effective_era(self.cfg.round, created) > record.effective_era
                    or era_for_block(self.cfg.round, created).index >= record.effective_era):
                return NOTICE_LATE
            if self.verify_signatures and not verify_signature(earliest, self.cfg.manifest.trainer_hotkey):
                return "v2 promotion notice manifest signature is invalid"
            return None
        return "v2 promotion was not announced in the prior settlement chain"
