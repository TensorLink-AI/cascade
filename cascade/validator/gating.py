"""Manifest gating: signature, training contract, pool pin and GPU checks."""

from __future__ import annotations

import logging

from ..shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    locked_contract_terms,
    verify_signature,
)

log = logging.getLogger("cascade.validator")


# How long a gate's evidence may stay unREADable (auth/network/5xx — not
# absence) before the hold escalates from a warning to an error. Sized for
# observed Hippius blips (seconds-to-minutes) with a wide margin — ~15 polls at
# the default 120s manifest cadence. The lane-A gates (pool pin, warm-start
# provenance, era refs) NEVER turn a read failure into a reject: one
# validator's storage outage is not a verdict its peers, reading fine, would
# reach, so the round holds (no latch, no receipt) until the read succeeds —
# see :meth:`ManifestGateMixin._hold_read_failure`. Lane B still uses the grace
# as the point where lane A proceeds without it (its part stays unfolded).
POOL_PIN_READ_GRACE_SECONDS = 1800.0


class ManifestGateMixin:
    """The per-manifest gate and its read-grace bookkeeping."""

    # ── manifest gating ─────────────────────────────────────────────────────

    def check_manifest(self, manifest: TrainingManifest) -> str | None:
        """Return a rejection reason string, or None if the manifest is usable.

        Enforces (1) the trainer-hotkey signature, (2) the training-contract
        gate (see :meth:`_check_contract`), and (3) that the manifest targets
        our configured base architecture and eval dataset.
        """
        if self.verify_signatures and not verify_signature(manifest, self.cfg.manifest.trainer_hotkey):
            return "signature_invalid"
        contract_reason = self._check_contract(manifest)
        if contract_reason is not None:
            return contract_reason
        if manifest.base_arch_digest != self._training_for(manifest).base_arch_digest:
            return "base_arch_digest_mismatch"
        if manifest.eval_dataset != self.cfg.eval.eval_dataset:
            return "eval_dataset_mismatch"
        gpu_reason = self._check_gpu(manifest)
        if gpu_reason is not None:
            return gpu_reason
        ws_reason = self._check_warm_start(manifest)
        if ws_reason is not None:
            return ws_reason
        era_reason = self._check_era(manifest)
        if era_reason is not None:
            return era_reason
        return None

    def _check_contract(self, manifest: TrainingManifest) -> str | None:
        """Gate the round's training contract. Two regimes, block-selected.

        **Declared** (DEC-CA-0036; from ``[scoring] declared_contract_from_block``,
        and only for a manifest that actually publishes a ``contract_body``):
        the trainer ships the round's full contract in the signed manifest and
        this checks two things — that the body re-hashes to the
        ``contract_digest`` it declares, and that the body's LOCKED projection
        (:func:`locked_contract_terms`) matches ours. The locked terms are the
        only ones this validator consumes with its own local value — including
        ``train_seed_salt``, which the receipt assembler folds into the round's
        published ``training_seed`` via ``RoundSeeds.derive``; every other
        ``[training]`` term is recorded and audited but never gates, so the
        trainer ships runtime and recipe fixes without a fleet restart.

        **Strict** (legacy, and always for a body-less manifest): full digest
        equality against the local ``[training]``, with the block-gated
        ``prior_contract_digest`` transition for the round in flight across a
        contract-changing release.

        Note the asymmetry in where each side is read from: the digest
        self-consistency check reads the DECLARED body (it proves the trainer's
        own two statements agree), while the locked comparison comes from LOCAL
        config on our side — a pin sourced from the declaration would be the
        trainer gating itself.
        """
        declared_from = int(self.cfg.scoring.declared_contract_from_block)
        if (declared_from and manifest.contract_body is not None
                and self._epoch_start_block(manifest) >= declared_from):
            declared = contract_digest(manifest.contract_body)
            if declared != manifest.contract_digest:
                return (f"contract_body_mismatch: body hashes to {declared} "
                        f"but manifest declares {manifest.contract_digest}")
            want = locked_contract_terms(self._training_for(manifest))
            got = locked_contract_terms(manifest.contract_body)
            if got != want:
                return f"contract_locked_mismatch: {got} != {want}"
            return None
        want_contract = contract_digest(self._training_for(manifest))
        if manifest.contract_digest != want_contract:
            # Block-gated contract transition: a round whose epoch boundary
            # precedes contract_from_block was trained under the pinned prior
            # contract and is accepted against prior_contract_digest — so a
            # validator restarted onto a contract-changing release still
            # scores the round in flight (see ScoringConfig).
            prior = self.cfg.scoring.prior_contract_digest
            from_block = int(self.cfg.scoring.contract_from_block)
            if not (prior and from_block
                    and manifest.contract_digest == prior
                    and self._epoch_start_block(manifest) < from_block):
                return f"contract_digest_mismatch: {manifest.contract_digest} != {want_contract}"
        return None

    @staticmethod
    def check_pool_pin(
        manifest: TrainingManifest, window_source: object, *, block: int | None
    ) -> str | None:
        """Verify this round's eval pool against the trainer-signed pin.

        A pinned manifest carries the ``(key, sha256)`` of the snapshot the
        trainer screened on; the validator's own deterministic selection (same
        epoch block, same rule) must resolve to the identical pair. The pin is
        inside the signed body, so pool integrity descends from the trainer
        signature rather than the unsigned ``pool/index.json`` — a poisoned
        index or tampered tar surfaces here as a loud reject instead of scoring
        on attacker-chosen data. Unpinned manifests (older trainers) keep the
        legacy index-trust behaviour. Returns a reject reason or ``None``.

        Raises :class:`~cascade.shared.hippius.StorageError` when the pool
        index could not be READ (auth/network/5xx) — that is a transient, not a
        verdict, and the caller decides retry-vs-reject (the live loop retries
        within :data:`POOL_PIN_READ_GRACE_SECONDS`). Every other failure is a
        reject reason.
        """
        if not (manifest.eval_pool_key and manifest.eval_pool_sha256):
            return None
        prov_fn = getattr(window_source, "provenance_for_round", None)
        if prov_fn is None:
            return ("pool_pin_unverifiable: manifest pins the eval pool but this "
                    "validator's pool source reports no provenance")
        try:
            key, sha = prov_fn(int(manifest.round_id), block=block)
        except Exception as e:  # noqa: BLE001 — an unverifiable pin must reject, not crash
            from ..shared.hippius import StorageError

            if isinstance(e, StorageError):
                raise  # read failure, not a verdict — caller retries (see above)
            return f"pool_pin_unverifiable: provenance lookup failed: {e}"
        if not key or not sha:
            return ("pool_pin_unverifiable: manifest pins the eval pool but this "
                    "validator resolved no snapshot for the round")
        if (key, sha) != (manifest.eval_pool_key, manifest.eval_pool_sha256):
            return (f"pool_pin_mismatch: manifest signed {manifest.eval_pool_key}@"
                    f"{manifest.eval_pool_sha256[:12]}…, this validator resolved "
                    f"{key}@{sha[:12]}…")
        return None

    def _gate_read_failed(
        self, round_id: str, err: Exception, *, gate: str = "pool_pin_unverifiable",
        now: float | None = None,
    ) -> str | None:
        """Grace bookkeeping for a gate whose evidence could not be READ —
        the pool index at the pin gate, or the promotion record / bench report
        behind the warm-start gate (``gate`` names the reject reason).

        Returns ``None`` while ``round_id``'s read failures span less than
        :data:`POOL_PIN_READ_GRACE_SECONDS` — the caller then skips the cycle
        with no latch and no receipt, so the next manifest poll re-attempts the
        round from scratch. Once the grace expires, returns the past-grace
        reason (and forgets the round, so a later re-publish starts a fresh
        grace window). The live loop only uses this for lane B; the lane-A
        gates hold forever via :meth:`_hold_read_failure`.
        """
        import time

        now = time.monotonic() if now is None else now
        first = self._pin_read_first_failure.setdefault(str(round_id), now)
        waited = now - first
        if waited < POOL_PIN_READ_GRACE_SECONDS:
            log.warning(
                "%s: evidence unreadable for round=%s (%s); retrying "
                "next poll (%.0fs into %.0fs grace, no reject latched)",
                gate, round_id, err, waited, POOL_PIN_READ_GRACE_SECONDS,
            )
            return None
        self._pin_read_first_failure.pop(str(round_id), None)
        return (
            f"{gate}: provenance lookup failed persistently "
            f"({waited:.0f}s > {POOL_PIN_READ_GRACE_SECONDS:.0f}s grace): {err}"
        )

    def _hold_read_failure(
        self, key: str, err: Exception, *, gate: str, now: float | None = None,
    ) -> float:
        """A lane-A gate's evidence could not be READ: HOLD the round — no
        latch, no receipt — and retry next poll, however long it takes. A read
        failure is never a verdict (a reject latched here would never be
        re-judged, and peers reading fine would decide the round the other
        way). Logs a warning inside :data:`POOL_PIN_READ_GRACE_SECONDS`, an
        ERROR after it, with the attempt count and age for visibility. ``key``
        is per round AND per gate. Returns the age in seconds."""
        import time

        now = time.monotonic() if now is None else now
        first, count = self._read_holds.get(key, (now, 0))
        count += 1
        self._read_holds[key] = (first, count)
        waited = now - first
        if waited < POOL_PIN_READ_GRACE_SECONDS:
            log.warning("%s: evidence unreadable for %s (%s); holding, retrying next "
                        "poll (attempt %d, %.0fs; no reject latched)",
                        gate, key, err, count, waited)
        else:
            log.error("%s: evidence STILL unreadable for %s after %.0fs (attempt %d): %s "
                      "— the round HOLDS (no reject latched, later settlements wait); "
                      "fix storage/chain access", gate, key, waited, count, err)
        return waited

    def _clear_read_hold(self, key: str) -> None:
        """The gate's evidence was read: forget its hold clock."""
        held = self._read_holds.pop(key, None)
        if held is not None:
            log.info("%s: evidence readable again after %d held attempt(s)", key, held[1])

    def _check_gpu(self, manifest: TrainingManifest) -> str | None:
        """Matched-hardware gate for byte-exact re-derivation — PER SIZE.

        Each entry is judged against ITS size's pin: the per-size contract's
        ``expected_gpu`` (a ``SizeSpec.expected_gpu`` override, else the base
        ``[training]`` pin — DEC-CA-0027's size-conditional silicon). A size
        with no pin requires only that king and challenger ran the same GPU
        within that size — equal compute is already guaranteed by the token
        budget, but a byte-exact audit needs each comparison run on one SKU.
        Different SIZES may legitimately run different silicon (a 313M duel on
        H100 beside a 4M screen lineage on L40S); requiring one GPU across
        sizes would deadlock exactly the configuration the per-size pin
        exists for. Single-size manifests (every deployed round) are judged
        identically to the old whole-manifest rule.
        """
        training = self._training_for(manifest)
        registry = training.size_registry
        primary = training.arch_preset
        by_size: dict[str, list[TrainedEntry]] = {}
        for e in manifest.entries:
            by_size.setdefault(e.size, []).append(e)
        for size, entries in by_size.items():
            preset = size or primary
            # An unknown size tag falls back to the base pin. Defensive: both
            # contract regimes pin the legal preset set before we get here —
            # strict by full digest equality, declared by the locked projection
            # (which carries every extra size's preset) — so a tag outside the
            # registry means the trainer mislabelled its own entry, and the
            # base pin is the conservative reading.
            contract = registry.get(preset, training.primary_size)
            pinned = contract.expected_gpu
            gpus = {e.gpu_name for e in entries if e.gpu_name}
            if pinned:
                bad = sorted(g for g in gpus if g != pinned)
                if bad or any(not e.gpu_name for e in entries):
                    return (f"gpu_mismatch: size {preset!r} expected {pinned!r}, "
                            f"manifest has {sorted(gpus)!r}")
            elif len(gpus) > 1 and not self._era_active(manifest):
                # DEC-CA-0043 lifts the same-GPU fallback: legs pick their SKU
                # per entry, byte-exact audit across SKUs is given up (#294).
                return (f"gpu_mismatch: size {preset!r} king/challenger on "
                        f"different GPUs {sorted(gpus)!r}")
        return None
