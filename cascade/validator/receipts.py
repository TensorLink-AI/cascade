"""Public round receipts, chain status telemetry and the re-entry receipt probe."""

from __future__ import annotations

import logging
from datetime import UTC
from typing import TYPE_CHECKING

from ..eval.window import EvalWindow
from ..shared.activation import summary as activation_summary
from ..shared.config import effective_epoch_blocks
from ..shared.era import (
    era_king_active,
    settlement_era,
)
from ..shared.manifest import TrainingManifest
from ..shared.receipt import (
    EvalContext,
    Participant,
    RoundReceipt,
    VerdictRecord,
    build_receipt,
)
from .manifest_chain import participants_from_commitments

if TYPE_CHECKING:
    from .loop import RoundOutcome

log = logging.getLogger("cascade.validator")


class ReceiptMixin:
    """Assemble, sign and publish receipts and status."""

    # ── public round receipts ────────────────────────────────────────────────

    def build_round_receipt(
        self,
        manifest: TrainingManifest,
        *,
        base_seed: int,
        epoch_start_block: int,
        epoch_block_hash: str,
        outcome: RoundOutcome | None = None,
        windows: list[EvalWindow] | None = None,
        reject_reason: str | None = None,
        participants: tuple[Participant, ...] = (),
        pool_provenance: tuple[str, str] = ("", ""),
        reward_uids: tuple[int, ...] = (),
        weights: tuple[float, ...] = (),
        validator_hotkey: str = "",
        era_base_seed: int = 0,
        lane_b: list[dict] | None = None,
        lane_b_pay: dict | None = None,
        emission: dict | None = None,
        king_check: dict | None = None,
        lane_c: list[dict] | None = None,
        lane_c_pay: dict | None = None,
        king_check_scores: tuple = (),
    ) -> RoundReceipt:
        """Assemble the round's public receipt (pure — no I/O, no signing).

        ``era_base_seed`` (DEC-CA-0043, era settlements only) is
        ``block_seed(era.seed_block)``: the era's training seeds derive from
        it, so the receipt's ``generation_seed`` / ``training_seed`` are the
        era's while ``base_seed`` stays the settlement boundary's.

        A gated-out manifest — or one with no (king, challenger) pair to score
        (a no-contest round: nothing failed, the king simply holds) — yields a
        ``rejected``-status receipt carrying the reason (the machine-readable
        status string is kept for compatibility); a scored round yields
        the full record: chain context, embedded manifest, participant set, the
        eval slice, every per-window score, the verdict, and the weight vector.
        """
        from ..trainer.contract import RoundSeeds

        era_start_block = 0
        if era_king_active(self.cfg.scoring, epoch_start_block):
            era_start_block = settlement_era(self.cfg.round, epoch_start_block).start_block
            seeds = RoundSeeds.derive(int(era_base_seed), self.cfg.training)
        else:
            era_base_seed = 0
            seeds = RoundSeeds.derive(base_seed, self.cfg.training)
        if reject_reason is not None:
            return build_receipt(
                round_id=manifest.round_id, status="rejected",
                epoch_start_block=epoch_start_block, epoch_block_hash=epoch_block_hash,
                base_seed=base_seed, seeds=seeds, manifest=manifest,
                participants=participants, reject_reason=reject_reason,
                reward_uids=reward_uids, weights=weights,
                validator_hotkey=validator_hotkey,
                era_start_block=era_start_block, era_base_seed=era_base_seed,
                activation_block=self.activation_block,
                margin_v2_block=self.margin_v2_block,
                lane_b=lane_b, lane_b_pay=lane_b_pay,
                v2_block=self._v2_receipt_block(epoch_start_block),
                prelaunch_burn_block=self.prelaunch_burn_block,
                emission=emission, king_check=king_check,
                king_check_scores=king_check_scores,
                lane_c=lane_c, lane_c_pay=lane_c_pay,
                **self._netconfig_receipt_kw(),
                **self._satellite_receipt_kw(manifest),
                **self._bench_report_receipt_kw(manifest),
            )
        if outcome is None or windows is None:
            raise ValueError("a scored receipt needs both outcome and windows")
        eval_context = EvalContext(
            pool_ref=pool_provenance[0],
            pool_digest=pool_provenance[1],
            window_ids=tuple(w.series_id for w in windows),
            n_windows=len(windows),
            num_samples=self.cfg.eval.num_samples,
        )
        verdict = VerdictRecord.from_round(
            outcome.result, outcome.transition,
            params=self.cfg.koth_params(block=epoch_start_block), bootstrap_seed=base_seed,
            king_tenure_rounds=outcome.king_tenure_rounds,
            cohort_k=outcome.cohort_k, cohort_lcbs=outcome.cohort_lcbs,
            cohort_geomeans=outcome.cohort_geomeans,
            cohort_per_horizon=outcome.cohort_per_horizon,
            cohort_stats=outcome.cohort_stats,
            cohort_per_domain=outcome.cohort_per_domain,
        )
        return build_receipt(
            round_id=manifest.round_id, status="scored",
            epoch_start_block=epoch_start_block, epoch_block_hash=epoch_block_hash,
            base_seed=base_seed, seeds=seeds, manifest=manifest,
            participants=participants, eval_context=eval_context,
            entry_scores=outcome.entry_scores, verdict=verdict,
            reward_uids=reward_uids, weights=weights,
            validator_hotkey=validator_hotkey,
            era_start_block=era_start_block, era_base_seed=era_base_seed,
            activation_block=self.activation_block,
            margin_v2_block=self.margin_v2_block,
            lane_b=lane_b, lane_b_pay=lane_b_pay,
            v2_block=self._v2_receipt_block(epoch_start_block),
            prelaunch_burn_block=self.prelaunch_burn_block,
            emission=emission, king_check=king_check,
            king_check_scores=king_check_scores,
            lane_c=lane_c, lane_c_pay=lane_c_pay,
            **self._netconfig_receipt_kw(),
            **self._satellite_receipt_kw(manifest),
            **self._bench_report_receipt_kw(manifest),
        )

    def _bench_report_receipt_kw(self, manifest: TrainingManifest) -> dict:
        """DEC-CA-0063: the validator bench report a settlement judged on
        (``{}`` — receipt bytes unchanged — everywhere else)."""
        fn = getattr(self, "_vbench_receipt_kw", None)
        return fn(manifest) if callable(fn) else {}

    def _satellite_state(self) -> dict:
        """DEC-CA-0058: this validator's settlement state as the receipt
        records it (:func:`cascade.shared.settlement_state.settlement_state`)."""
        from ..shared.settlement_state import settlement_state

        lane_b = self._lane_b() if self.cfg.lane_b.from_block else None
        lane_c = self._lane_c() if self.cfg.lane_c.from_block else None
        return settlement_state(self.state, lane_b=lane_b, lane_c=lane_c)

    def _capture_settlement_prior(self, round_id: str) -> None:
        """Remember the settlement state before ``round_id`` is applied
        (only while ``[netconfig] owner_hotkey`` is set; never raises)."""
        self._settlement_prior = None
        if not str(getattr(getattr(self.cfg, "netconfig", None), "owner_hotkey", "") or ""):
            return
        try:
            self._settlement_prior = (str(round_id), self._satellite_state())
        except Exception as e:  # noqa: BLE001 — the receipt then omits the prior digest
            log.warning("satellite prior state unavailable for round=%s: %s", round_id, e)

    def _satellite_receipt_kw(self, manifest: TrainingManifest) -> dict:
        """DEC-CA-0058 receipt stamps (``{}`` while ``[netconfig]
        owner_hotkey`` is empty — the receipt bytes are then unchanged): the
        judged manifest's published-text digest, the settlement state's prior
        digest (captured when the settlement started) and its posterior."""
        from ..shared.settlement_state import state_digest

        nc = getattr(self.cfg, "netconfig", None)
        if not str(getattr(nc, "owner_hotkey", "") or ""):
            return {}
        state = getattr(self, "state", None)
        kw: dict = {}
        if (state is not None and state.last_handled_round_id == str(manifest.round_id)
                and state.last_handled_manifest_sha):
            kw["manifest_sha256"] = str(state.last_handled_manifest_sha)
        prior = getattr(self, "_settlement_prior", None)
        if prior is not None and prior[0] == str(manifest.round_id):
            kw["state_prior_sha256"] = state_digest(prior[1])
        try:
            kw["state_post"] = self._satellite_state()
        except Exception as e:  # noqa: BLE001 — a receipt stamp never sinks a receipt
            log.warning("satellite state stamp unavailable for round=%s: %s",
                        manifest.round_id, e)
        return kw

    def _publish_round_receipt(
        self,
        client: object,
        manifest: TrainingManifest,
        base_seed: int,
        *,
        outcome: RoundOutcome | None = None,
        windows: list[EvalWindow] | None = None,
        reject_reason: str | None = None,
        window_source: object = None,
        reward_uids: tuple[int, ...] = (),
        weights: tuple[float, ...] = (),
        lane_b: list[dict] | None = None,
        lane_b_pay: dict | None = None,
        lane_c: list[dict] | None = None,
        lane_c_pay: dict | None = None,
    ) -> None:  # pragma: no cover — live-loop glue; assembly is unit-tested
        """Gather chain context, sign, and publish the round receipt.

        Best-effort by design: a receipt failure must never disturb weights or
        KOTH state (they are already committed), so every chain lookup degrades
        to an empty field and any publish error is logged and swallowed. The
        audit CLI treats missing context as WARN, not PASS.
        """
        from datetime import datetime

        from ..shared.calendar_launch import effective_commit_floor
        from ..shared.hippius import (
            open_manifest_store,
            publish_receipt,
            update_receipt_index,
        )
        from ..shared.receipt import dump_receipt, sign_receipt, summarize_receipt

        try:
            epoch_start = self._epoch_start_block(manifest)
            epoch_hash = ""
            era_base_seed = 0
            current_block: int | None = None
            participants: tuple[Participant, ...] = ()
            try:
                epoch_hash = client.block_hash(epoch_start)
                if era_king_active(self.cfg.scoring, epoch_start):
                    # DEC-CA-0055: a v2 era inside a seed epoch trains from
                    # the epoch's first era's seeds
                    era_base_seed = int(client.block_seed(self._era_seed_block(
                        settlement_era(self.cfg.round, epoch_start))))
                participants = participants_from_commitments(
                    client.poll_commitments(include_history=True),
                    cutoff_block=epoch_start,
                    floor_block=effective_commit_floor(
                        self.cfg, settlement_era(self.cfg.round, epoch_start).start_block),
                )
                # Anchor for the dashboard's next-round countdown (best-effort;
                # the client extrapolates block→wall-clock from this + as_of).
                current_block = int(client.current_block())
            except Exception as e:  # noqa: BLE001 — chain context is best-effort
                log.warning("receipt chain context unavailable for round=%s: %s",
                            manifest.round_id, e)
            provenance = ("", "")
            prov_fn = getattr(window_source, "provenance_for_round", None)
            if prov_fn is not None:
                try:
                    # Same epoch block that selected the round's windows, so the
                    # recorded provenance is the pool actually scored.
                    provenance = tuple(prov_fn(base_seed, block=epoch_start))
                except Exception as e:  # noqa: BLE001
                    log.warning("pool provenance unavailable for round=%s: %s",
                                manifest.round_id, e)
            wallet = getattr(client, "wallet", lambda: None)()
            hotkey_ss58 = str(getattr(getattr(wallet, "hotkey", None), "ss58_address", "") or "")

            receipt = self.build_round_receipt(
                manifest,
                base_seed=base_seed,
                epoch_start_block=epoch_start,
                epoch_block_hash=str(epoch_hash),
                outcome=outcome,
                windows=windows,
                reject_reason=reject_reason,
                participants=participants,
                pool_provenance=(provenance[0], provenance[1]),
                reward_uids=reward_uids,
                weights=weights,
                validator_hotkey=hotkey_ss58,
                era_base_seed=era_base_seed,
                lane_b=lane_b, lane_b_pay=lane_b_pay,
                lane_c=lane_c, lane_c_pay=lane_c_pay,
                emission=getattr(self, "_last_emission", None),
                king_check=self._king_check_for(str(manifest.round_id)),
                king_check_scores=self._king_check_scores_for(str(manifest.round_id)),
            )
            if wallet is not None:
                receipt = sign_receipt(receipt, wallet)
            else:
                log.warning("publishing an UNSIGNED receipt (no wallet) for round=%s",
                            manifest.round_id)
            store = open_manifest_store(self.cfg.storage)
            key = publish_receipt(store, dump_receipt(receipt), manifest.round_id,
                                  validator_hotkey=hotkey_ss58)
            # Operator-facing label only: a king-only round is a designed
            # no-contest, not a rejection — but the receipt's machine-readable
            # status stays "rejected" (audit checks, receipt bootstrap, and
            # dashboards branch on it, and archived signatures cover it).
            status_label = ("no-contest (king holds)"
                            if reject_reason == "no_king_challenger_pair"
                            else receipt.status)
            log.info("published %s receipt round=%s signed=%s → s3://%s/%s",
                     status_label, manifest.round_id, receipt.signature is not None,
                     self.cfg.storage.manifest_bucket, key)
            # Refresh the dashboard-facing rolling index (best-effort, and inside
            # the outer guard: a listing convenience must never disturb a round).
            try:
                now_iso = datetime.now(UTC).isoformat(timespec="seconds")
                # Schedule anchor for the "time until next round" countdown. The
                # next round begins at the next epoch boundary; the client turns
                # blocks into wall-clock via block_time_s, extrapolating current_block
                # from `as_of`. Bittensor blocks are ~12s regardless of epoch_blocks
                # (testnet shortens epochs, not block time), so it is a constant.
                chain = {
                    "as_of": now_iso,
                    "current_block": current_block,
                    "epoch_start_block": epoch_start,
                    # Length of the round STARTING at epoch_start, not the raw
                    # field: while a cadence change is pending the raw value is
                    # the post-switch length, and the client's next-boundary
                    # countdown (epoch_start + epoch_blocks) would point
                    # mid-round. A round never spans the activation block, so
                    # resolving at epoch_start is exact on both sides.
                    "epoch_blocks": int(effective_epoch_blocks(self.cfg.round, epoch_start)),
                    "block_time_s": 12.0,
                }
                update_receipt_index(
                    store, summarize_receipt(receipt),
                    updated_at=now_iso,
                    subnet={"netuid": self.cfg.subnet.netuid, "name": self.cfg.subnet.name},
                    chain=chain,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("receipt index update failed for round=%s: %s",
                            manifest.round_id, e)
            # OPTIONAL W&B settlement summary ([wandb] validator_summary, off
            # by default): after the weights and the receipt are out, so it
            # can never shape a verdict; never raises.
            if getattr(getattr(self.cfg, "wandb", None), "validator_summary", False):
                try:
                    from ..trainer.wandb_sink import log_validator_settlement

                    log_validator_settlement(self.cfg.wandb, summarize_receipt(receipt),
                                             validator_hotkey=hotkey_ss58)
                except Exception as e:  # noqa: BLE001
                    log.debug("validator wandb summary skipped: %s", e)
        except Exception as e:  # noqa: BLE001 — receipts must never disturb the round
            log.warning("receipt publication failed for round=%s: %s", manifest.round_id, e)

    def _publish_chain_status(self, client: object, store: object) -> None:  # pragma: no cover
        """Publish the dashboard's live ``status/chain.json`` (current block,
        epoch grid, stage windows, revealed submissions) on the poll cadence.

        Purely presentational and best-effort — it feeds the web dashboard's
        round-stage strip and live submissions panel between receipts, and any
        failure (chain flake, storage outage) is swallowed: status telemetry
        must never disturb a round.
        """
        from datetime import datetime

        from ..shared.chain_status import build_chain_status, publish_chain_status

        try:
            econ = getattr(client, "subnet_economics", lambda: None)()
            activation = (activation_summary(self.cfg, self._activation, self._activation_tally,
                                             self._forfeit, self._forfeit_tally,
                                             self._margin_v2, self._margin_v2_tally,
                                             self._prelaunch, self._prelaunch_tally)
                          if self.cfg.activation.enabled else None)
            status = build_chain_status(
                self.cfg,
                current_block=int(client.current_block()),  # type: ignore[attr-defined]
                commitments=client.poll_commitments(),  # type: ignore[attr-defined]
                network=str(getattr(client, "network", "")),
                as_of=datetime.now(UTC).isoformat(timespec="seconds"),
                economics=econ,
                activation=activation,
            )
            publish_chain_status(store, status)
        except Exception as e:  # noqa: BLE001 — telemetry only
            log.debug("chain status publish skipped: %s", e)

    def _probe_own_receipt_round(self, client: object, store: object) -> str | None:
        """Round id of this validator's own latest published receipt, or None.

        Boot fallback for state files that predate ``last_handled_round_id``:
        without it, the first restart after the upgrade would still re-judge
        the already-published round once. Only a receipt whose signature
        verifies against this validator's OWN hotkey seeds the marker — the
        receipt prefix is public-read, so an unverified object must never be
        able to make a validator skip a round. Any fault returns None (the
        stock re-judge behaviour); storage must never block startup.
        """
        from ..shared.hippius import receipt_latest_key
        from ..shared.receipt import load_receipt, verify_receipt_signature

        try:
            wallet = getattr(client, "wallet", lambda: None)()
            hotkey = str(getattr(getattr(wallet, "hotkey", None), "ss58_address", "") or "")
            if not hotkey:
                return None
            receipt = load_receipt(store.get_text(receipt_latest_key(hotkey)))
            if not verify_receipt_signature(receipt, hotkey):
                log.warning("re-entry probe: own latest receipt fails signature "
                            "verification; ignoring it")
                return None
            return str(receipt.round_id)
        except Exception as e:  # noqa: BLE001
            log.debug("re-entry probe: no usable own receipt (%s)", e)
            return None
