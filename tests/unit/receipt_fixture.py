"""A fully self-consistent round-receipt fixture.

Shared by the receipt schema tests and the ``cascade-audit`` tests: every
derivable quantity in the fixture is *actually derived* (base seed from the
block hash, round seeds from the base seed, the verdict from the recorded
scores via ``evaluate_round``, weights via ``decayed_share_vector`` at the
config's own king_decay), so a Tier-0
audit of the untampered fixture passes every recomputation — and any single
mutation trips exactly the corresponding check.

``make_scored_receipt(cfg)`` pins the contract/base-arch digests to the given
chain config (audit recomputes them from chain.toml); ``cfg=None`` uses fixed
synthetic digests, giving a byte-stable receipt for the golden-schema test.
"""

from __future__ import annotations

import numpy as np

from cascade.eval.koth import KothParams, evaluate_round
from cascade.eval.scoring import WindowScore
from cascade.shared.chain import decayed_share_vector, seed_from_block_hash
from cascade.shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    format_trained_pointer,
)
from cascade.shared.receipt import (
    EntryScores,
    EvalContext,
    Participant,
    VerdictRecord,
    WindowScoreRecord,
    build_receipt,
)
from cascade.trainer.contract import RoundSeeds
from cascade.validator import state as state_mod

GEN_REF_KING = "alice/gen@sha256:" + "a" * 64
GEN_REF_CHAL = "bob/gen@sha256:" + "b" * 64
CKPT_KING = format_trained_pointer("cascade/ckpt-king@sha256:" + "c" * 64)
CKPT_CHAL = format_trained_pointer("cascade/ckpt-chal@sha256:" + "d" * 64)
POOL_REF = "cascade/eval-pool@sha256:" + "e" * 64

EPOCH_BLOCKS = 7200
EPOCH_START = 3 * EPOCH_BLOCKS
CREATED_BLOCK = EPOCH_START + 50
BLOCK_HASH = "0x" + "ab" * 32

# Fixed KOTH params for the cfg=None (golden) fixture; chain.toml values are
# used when a cfg is passed so the audit's recomputation matches its config.
_SYNTH_PARAMS = KothParams(
    win_margin_start=0.02, win_margin_end=0.02, margin_warmup_rounds=0,
    min_windows=200, bootstrap_B=500, bootstrap_alpha=0.05, dethrone_cp=1,
)


def make_scores(scale: float, seed: int, n: int = 256) -> list[WindowScore]:
    rng = np.random.default_rng(seed)
    return [
        WindowScore(
            series_id=f"w{i}",
            mase=float(rng.uniform(0.5, 1.5) * scale),
            qloss_per_q=(rng.uniform(0.1, 1.0, size=9) * scale).round(9),
            abs_target=float(round(rng.uniform(5.0, 10.0), 9)),
        )
        for i in range(n)
    ]


def make_manifest(cfg=None, *, base_seed: int, size: str = "",
                  trainer_wallet=None) -> TrainingManifest:
    if cfg is not None:
        cdigest = contract_digest(cfg.training)
        adigest = cfg.training.base_arch_digest
        dataset = cfg.eval.eval_dataset
        size = size or cfg.training.arch_preset
    else:
        cdigest, adigest, dataset = "1" * 64, "2" * 64, "cascade-private-v1"
        size = size or "toto2-4m"
    entries = [
        TrainedEntry("king_hk", 0, "king", GEN_REF_KING, CKPT_KING, "3" * 64,
                     CREATED_BLOCK, size=size),
        TrainedEntry("chal_hk", 1, "challenger", GEN_REF_CHAL, CKPT_CHAL, "4" * 64,
                     CREATED_BLOCK, size=size),
    ]
    manifest = TrainingManifest(
        round_id=str(base_seed),
        created_block=CREATED_BLOCK,
        contract_digest=cdigest,
        base_arch_digest=adigest,
        eval_dataset=dataset,
        entries=entries,
        signature="00ff" * 16,  # placeholder; audit signature checks use real signing
    )
    if trainer_wallet is not None:
        from cascade.shared.manifest import sign_manifest

        manifest = sign_manifest(manifest, trainer_wallet)
    return manifest


def make_scored_receipt(cfg=None, *, validator_hotkey: str = "", trainer_wallet=None):
    """A scored receipt whose every derived field genuinely derives.

    Returns ``(receipt, king_scores, chal_scores)`` so tests can re-feed the
    exact scores.
    """
    base_seed = seed_from_block_hash(BLOCK_HASH)
    if cfg is not None:
        seeds = RoundSeeds.derive(base_seed, cfg.training)
        params = cfg.koth_params()
        num_samples = cfg.eval.num_samples
        size = cfg.training.arch_preset
    else:
        # cfg=None keeps the fixture byte-stable: RoundSeeds.derive only reads
        # train_seed_salt, so a tiny stand-in contract suffices.
        class _Salt:
            train_seed_salt = 1337
        seeds = RoundSeeds.derive(base_seed, _Salt())
        params = _SYNTH_PARAMS
        num_samples = 100
        size = "toto2-4m"

    manifest = make_manifest(cfg, base_seed=base_seed, size=size,
                             trainer_wallet=trainer_wallet)
    king_scores = make_scores(1.0, 0)
    chal_scores = [
        WindowScore(s.series_id, s.mase * 0.6, s.qloss_per_q * 0.6, s.abs_target)
        for s in king_scores
    ]
    result = evaluate_round(king_scores, chal_scores, params,
                            seed=base_seed, king_tenure_rounds=0)
    transition = state_mod.apply_round(
        state_mod.genesis("king_hk", 0),
        challenger_hotkey="chal_hk", challenger_uid=1,
        result=result, dethrone_cp=params.dethrone_cp, keep_former_kings=1,
    )
    reward_uids = (1, 0) if transition.dethroned else (0,)
    # Use the SAME decay the validator reads from cfg — hardcoding a flat
    # split strands every audit fixture the moment [scoring] king_decay moves
    # (it did: 1.0 → 0.5, 2026-07-14).
    decay = cfg.scoring.king_decay if cfg is not None else 1.0
    weights = tuple(decayed_share_vector(list(reward_uids), 4,
                                         decay=decay, burn_uid=0))

    receipt = build_receipt(
        round_id=manifest.round_id,
        status="scored",
        epoch_start_block=EPOCH_START,
        epoch_block_hash=BLOCK_HASH,
        base_seed=base_seed,
        seeds=seeds,
        manifest=manifest,
        validator_hotkey=validator_hotkey,
        participants=(
            Participant("king_hk", 0, GEN_REF_KING, EPOCH_START - 100),
            Participant("chal_hk", 1, GEN_REF_CHAL, EPOCH_START - 10),
        ),
        eval_context=EvalContext(
            pool_ref=POOL_REF,
            pool_digest="sha256:" + "e" * 64,
            window_ids=tuple(s.series_id for s in king_scores),
            n_windows=len(king_scores),
            num_samples=num_samples,
        ),
        entry_scores=(
            EntryScores("king", size, "king_hk", 0,
                        tuple(WindowScoreRecord.from_score(s) for s in king_scores)),
            EntryScores("challenger", size, "chal_hk", 1,
                        tuple(WindowScoreRecord.from_score(s) for s in chal_scores)),
        ),
        verdict=VerdictRecord.from_round(result, transition, params=params,
                                         bootstrap_seed=base_seed, king_tenure_rounds=0),
        reward_uids=reward_uids,
        weights=weights,
    )
    return receipt, king_scores, chal_scores


def make_rejected_receipt(cfg=None, *, reason: str = "signature_invalid",
                          validator_hotkey: str = ""):
    base_seed = seed_from_block_hash(BLOCK_HASH)
    if cfg is not None:
        seeds = RoundSeeds.derive(base_seed, cfg.training)
    else:
        class _Salt:
            train_seed_salt = 1337
        seeds = RoundSeeds.derive(base_seed, _Salt())
    manifest = make_manifest(cfg, base_seed=base_seed)
    return build_receipt(
        round_id=manifest.round_id,
        status="rejected",
        epoch_start_block=EPOCH_START,
        epoch_block_hash=BLOCK_HASH,
        base_seed=base_seed,
        seeds=seeds,
        manifest=manifest,
        validator_hotkey=validator_hotkey,
        reject_reason=reason,
    )
