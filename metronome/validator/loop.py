"""Validator loop — manifest → eval → KOTH decision → weights.

The validator never trains. Each round it:

1. Reads the current :class:`TrainingManifest` from the owner dataset repo and
   verifies its signature + that king and challenger share the contract digest
   (the controlled-experiment guarantee).
2. Pulls the king's and challenger's trained checkpoints and scores both on the
   *same* held-out eval windows.
3. Runs the paired-bootstrap KOTH verdict and folds it into the sticky
   champion state (``dethrone_cp`` consecutive wins to take the throne).
4. Sets winner-take-all weights on the reigning king's UID.

The pure orchestration in :meth:`ValidatorRunner.process_round` is testable by
injecting ``evaluate_fn`` and ``windows``; HF + torch + chain are isolated
behind the defaults.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..eval.koth import RoundResult, evaluate_round
from ..eval.scoring import WindowScore
from ..eval.window import EvalWindow
from ..shared.config import ChainConfig
from ..shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    parse_trained_pointer,
    verify_signature,
)
from . import state as state_mod
from .state import ChampionState, StateTransition

log = logging.getLogger("metronome.validator")

# Resolve a trained entry to its per-window scores on the eval set.
EvaluateFn = Callable[[TrainedEntry, list[EvalWindow]], list[WindowScore]]


@dataclass(frozen=True)
class RoundOutcome:
    result: RoundResult
    transition: StateTransition


@dataclass
class ValidatorRunner:
    cfg: ChainConfig
    state: ChampionState = field(default_factory=ChampionState)
    evaluate_fn: EvaluateFn | None = None     # injected in tests; defaults to registry+torch
    cache_dir: Path | None = None
    device: str = "cpu"
    verify_signatures: bool = True            # gate manifests on the trainer-hotkey signature

    # ── manifest gating ─────────────────────────────────────────────────────

    def check_manifest(self, manifest: TrainingManifest) -> str | None:
        """Return a rejection reason string, or None if the manifest is usable.

        Enforces (1) the trainer-hotkey signature, (2) the contract-digest match
        (king and challenger trained under the same terms), and (3) that the
        manifest targets our configured base architecture and eval dataset.
        """
        if self.verify_signatures and not verify_signature(manifest, self.cfg.manifest.trainer_hotkey):
            return "signature_invalid"
        want_contract = contract_digest(self.cfg.training)
        if manifest.contract_digest != want_contract:
            return f"contract_digest_mismatch: {manifest.contract_digest} != {want_contract}"
        if manifest.base_arch_digest != self.cfg.training.base_arch_digest:
            return "base_arch_digest_mismatch"
        if manifest.eval_dataset != self.cfg.eval.eval_dataset:
            return "eval_dataset_mismatch"
        return None

    # ── per-round decision ──────────────────────────────────────────────────

    def _evaluate(self, entry: TrainedEntry, windows: list[EvalWindow]) -> list[WindowScore]:
        if self.evaluate_fn is not None:
            return self.evaluate_fn(entry, windows)
        # Default path: fetch the checkpoint from the Hippius registry and score
        # it (registry + torch). The tar digest is re-verified on fetch.
        from ..shared.hippius import RegistryConfig, fetch_from_registry
        from .evaluator import evaluate_checkpoint

        cid = parse_trained_pointer(entry.trained_pointer)
        if cid is None:
            raise ValueError(f"malformed trained_pointer: {entry.trained_pointer!r}")
        reg = RegistryConfig.from_storage(self.cfg.storage)
        dest = Path(self.cache_dir or "./_eval_ckpts") / cid
        fetch_from_registry(
            cid, dest, reg, expected_tar_digest=entry.tar_digest or None
        )
        return evaluate_checkpoint(
            dest, windows, num_samples=self.cfg.eval.num_samples, device=self.device
        )

    def process_round(
        self,
        manifest: TrainingManifest,
        windows: list[EvalWindow],
        base_seed: int | str,
    ) -> RoundOutcome | None:
        """Evaluate one manifest against the eval windows and update state.

        Returns None (king holds, no state change) when the manifest carries no
        challenger or fails the contract gate. Otherwise returns the round
        outcome with the (already-applied) state transition.
        """
        reason = self.check_manifest(manifest)
        if reason is not None:
            log.warning("rejecting manifest round=%s: %s", manifest.round_id, reason)
            return None

        king_entry = manifest.entry_for_role("king")
        chal_entry = manifest.entry_for_role("challenger")
        if king_entry is None or chal_entry is None:
            log.info("manifest round=%s has no king/challenger pair; king holds", manifest.round_id)
            return None

        king_scores = self._evaluate(king_entry, windows)
        chal_scores = self._evaluate(chal_entry, windows)

        result = evaluate_round(
            king_scores,
            chal_scores,
            self.cfg.koth_params(),
            seed=base_seed,
            king_tenure_rounds=self.state.tenure_rounds,
        )
        transition = state_mod.apply_round(
            self.state,
            challenger_hotkey=chal_entry.miner_hotkey,
            challenger_uid=chal_entry.miner_uid,
            result=result,
            dethrone_cp=self.cfg.scoring.dethrone_cp,
        )
        self.state = transition.state
        log.info(
            "round=%s lcb=%.4f margin=%.4f win=%s %s king=%s tenure=%d",
            manifest.round_id, result.lcb, result.margin, result.challenger_wins_round,
            transition.note, self.state.king_hotkey, self.state.tenure_rounds,
        )
        return RoundOutcome(result=result, transition=transition)


    # ── live loop ────────────────────────────────────────────────────────────

    def run_forever(self, client: object, *, window_source: object) -> None:  # pragma: no cover
        """Poll the manifest bucket → evaluate → set weights, once per round.

        ``window_source`` is a :class:`metronome.validator.windows.WindowSource`
        (the loaded private pool). Each new manifest's ``round_id`` is the base
        seed; the same seed drives the rotating window slice so every validator
        scores the identical set.
        """
        import time

        from ..shared.hippius import S3Config, S3Store, read_latest_manifest
        from ..shared.manifest import load_manifest

        store = S3Store(S3Config.from_storage(self.cfg.storage, bucket=self.cfg.storage.manifest_bucket))
        poll = self.cfg.manifest.poll_seconds
        last_round: str | None = None
        while True:
            try:
                manifest = load_manifest(read_latest_manifest(store))
                if manifest.round_id != last_round:
                    base_seed = int(manifest.round_id)
                    windows = window_source.windows_for_round(base_seed, self.cfg.eval.n_windows)
                    outcome = self.process_round(manifest, windows, base_seed)
                    if outcome is not None and self.state.king_uid is not None:
                        client.set_winner_take_all_weights(self.state.king_uid, client.n_uids())
                    self._persist_state()
                    last_round = manifest.round_id
            except Exception as e:  # noqa: BLE001 — a service loop must not die on one round
                log.exception("round processing failed; retrying after poll: %s", e)
            time.sleep(poll)

    def _persist_state(self) -> None:  # pragma: no cover
        from . import state as state_mod

        try:
            Path(self.cfg.validator.state_db_path).write_text(
                state_mod.dumps(self.state), encoding="utf-8"
            )
        except Exception as e:  # noqa: BLE001
            log.warning("failed to persist validator state: %s", e)


def _load_state(path: str) -> ChampionState:
    """Load persisted champion state from ``state_db_path`` (JSON), or a fresh
    state if the file is absent/unreadable."""
    p = Path(path)
    if not p.is_file():
        return ChampionState()
    try:
        return state_mod.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("could not load validator state from %s (%s); starting fresh", path, e)
        return ChampionState()


def build_runner(
    *,
    chain_toml: Path | None = None,
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> ValidatorRunner:
    """Construct a runner from ``chain.toml``, restoring persisted champion
    state. Wallet/chain wiring for live weight-setting is attached by
    ``metronome-validator`` (see main.py)."""
    from ..shared.config import load_chain_config

    cfg = load_chain_config(chain_toml)
    return ValidatorRunner(
        cfg=cfg, state=_load_state(cfg.validator.state_db_path), cache_dir=cache_dir, device=device
    )
