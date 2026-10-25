"""The audit checks — small pure functions from a receipt (+ config, + an
optional chain client) to :class:`CheckResult` s.

Design rules:

* Every check is individually callable and unit-testable; ``run_tier0`` /
  ``run_tier1`` just sequence them.
* A check that depends on chain history a lite node cannot serve (historical
  block hash, past commitment state) degrades to an explicit ``WARN`` — never
  a silent pass (the human table and the JSON both show it).
* FAIL means "the receipt contradicts a re-derivation"; WARN means "could not
  fully verify"; SKIP means "not applicable to this receipt".
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from types import SimpleNamespace

from ..shared.chain import decayed_share_vector, seed_from_block_hash
from ..shared.config import (
    ChainConfig,
    cohort_maxt_active,
    cohort_maxt_increment_active,
    effective_epoch_blocks,
)
from ..shared.manifest import (
    LOCKED_CONTRACT_FIELDS,
    TrainingManifest,
    contract_digest,
    contract_payload,
    locked_contract_terms,
)
from ..shared.receipt import RoundReceipt

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str            # PASS | FAIL | WARN | SKIP
    detail: str = ""


def _ok(name: str, detail: str = "") -> CheckResult:
    return CheckResult(name, PASS, detail)


def _fail(name: str, detail: str) -> CheckResult:
    return CheckResult(name, FAIL, detail)


def _warn(name: str, detail: str) -> CheckResult:
    return CheckResult(name, WARN, detail)


def _skip(name: str, detail: str) -> CheckResult:
    return CheckResult(name, SKIP, detail)


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def _bootstrap_seed(recorded: str) -> int | str:
    """The seed as ``evaluate_round`` saw it: the live loop passes an int, so a
    digit-string round-trips back to int; anything else stays a string (both
    forms are deterministic in the bootstrap)."""
    return int(recorded) if re.fullmatch(r"-?\d+", recorded) else recorded


# ── signatures ────────────────────────────────────────────────────────────────


def check_receipt_signature(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """The validator's signature over the canonical receipt body.

    Verified against ``[manifest] validator_hotkey`` when pinned; against the
    receipt's self-declared signer otherwise (valid-but-unpinned ⇒ WARN — the
    signature then only proves internal consistency, not who signed).
    """
    name = "receipt-signature"
    if not receipt.signature:
        return _fail(name, "receipt is unsigned")
    pinned = cfg.manifest.validator_hotkey
    signer = pinned or receipt.validator_hotkey
    if not signer:
        return _fail(name, "no signer: receipt carries no validator_hotkey and none is pinned")
    if pinned and receipt.validator_hotkey and receipt.validator_hotkey != pinned:
        return _fail(
            name,
            f"signer {receipt.validator_hotkey} != pinned [manifest] validator_hotkey {pinned}",
        )
    from ..shared.receipt import verify_receipt_signature

    try:
        valid = verify_receipt_signature(receipt, signer)
    except RuntimeError as e:  # bittensor unavailable
        return _warn(name, f"cannot verify: {e}")
    if not valid:
        return _fail(name, f"signature does not verify against {signer}")
    if not pinned:
        return _warn(name, f"valid, but signer {signer} is self-declared "
                           "(pin [manifest] validator_hotkey to trust it)")
    return _ok(name, f"signed by {signer}")


def check_manifest_signature(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """The trainer's signature on the embedded manifest, against
    ``[manifest] trainer_hotkey``."""
    name = "manifest-signature"
    from ..shared.manifest import verify_signature

    try:
        manifest = receipt.load_embedded_manifest()
    except (ValueError, KeyError) as e:
        return _fail(name, f"embedded manifest unparseable: {e}")
    if not cfg.manifest.trainer_hotkey:
        return _warn(name, "[manifest] trainer_hotkey unset in chain.toml; cannot verify")
    if not manifest.signature:
        return _fail(name, "embedded manifest is unsigned")
    try:
        valid = verify_signature(manifest, cfg.manifest.trainer_hotkey)
    except RuntimeError as e:  # bittensor unavailable
        return _warn(name, f"cannot verify: {e}")
    if not valid:
        return _fail(name, f"manifest signature does not verify against "
                           f"{cfg.manifest.trainer_hotkey}")
    return _ok(name, f"signed by {cfg.manifest.trainer_hotkey}")


# ── seeds + chain context ─────────────────────────────────────────────────────


def check_base_seed(receipt: RoundReceipt) -> CheckResult:
    """``base_seed`` (and the round id) re-derive from the recorded block hash."""
    name = "base-seed"
    derived = seed_from_block_hash(receipt.epoch_block_hash)
    if derived != receipt.base_seed:
        return _fail(name, f"seed_from_block_hash({receipt.epoch_block_hash[:18]}…) = "
                           f"{derived} != recorded base_seed {receipt.base_seed}")
    if receipt.round_id != str(receipt.base_seed):
        return _fail(name, f"round_id {receipt.round_id!r} != str(base_seed) "
                           f"{receipt.base_seed}")
    if receipt.manifest.get("round_id") != receipt.round_id:
        return _fail(name, f"embedded manifest round_id {receipt.manifest.get('round_id')!r} "
                           f"!= receipt round_id {receipt.round_id!r}")
    return _ok(name, f"base_seed {receipt.base_seed} derives from the recorded block hash")


def check_round_seeds(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """``RoundSeeds.derive(base_seed)`` reproduces the recorded seed pair."""
    name = "round-seeds"
    # DEC-CA-0043: an era settlement's training seeds derive from the era's
    # seed (the previous era's start block hash), recorded as era_base_seed;
    # the boundary's base_seed still names the round and draws the eval.
    from ..shared.era import era_king_active
    from ..trainer.contract import RoundSeeds

    if era_king_active(cfg.scoring, receipt.epoch_start_block):
        if not receipt.era_base_seed:
            return _fail(name, "era settlement records no era_base_seed")
        seeds = RoundSeeds.derive(receipt.era_base_seed, cfg.training)
    else:
        seeds = RoundSeeds.derive(receipt.base_seed, cfg.training)
    problems = []
    if seeds.generation_seed != receipt.generation_seed:
        problems.append(f"generation_seed {receipt.generation_seed} != derived "
                        f"{seeds.generation_seed}")
    if seeds.training_seed != receipt.training_seed:
        problems.append(f"training_seed {receipt.training_seed} != derived "
                        f"{seeds.training_seed}")
    if problems:
        return _fail(name, "; ".join(problems))
    if receipt.era_base_seed:
        return _ok(name, "generation + training seeds derive from era_base_seed")
    return _ok(name, "generation + training seeds derive from base_seed")


def check_activation(receipt: RoundReceipt, cfg: ChainConfig,
                     client: object | None = None) -> CheckResult:
    """DEC-CA-0045: the rollover block a receipt records was decided by the
    validators, not invented by one.

    A receipt without ``activation_block`` PASSES (pre-field, or the
    rollover is typed into chain.toml). A recorded block must be a boundary
    of the grid before it; with a chain, the validators' current notes
    must agree on the SAME block by the configured stake threshold (WARN
    without a chain, like every other chain-dependent half). ``cfg`` is the
    config the audit replays under — already carrying the recorded block
    (``apply_receipt_activation``), so the era checks that follow use it.
    """
    from ..shared.activation import (
        agreed_activation,
        apply_activation,
        configured_rollover,
        resolved_rollover,
    )

    name = "activation"
    block = int(receipt.activation_block or 0)
    if not block:
        return _ok(name, "no chain-decided rollover recorded (typed-in config applies)")
    typed = configured_rollover(cfg)
    if typed:
        # The owner pinned the decided block afterwards (the natural end
        # state): the receipt must have been judged under THAT block.
        if typed != block:
            return _fail(name, f"recorded activation_block {block} != the rollover typed "
                               f"into the audit config ({typed})")
        return _ok(name, f"rollover {block} matches the typed-in config")
    if not cfg.activation.enabled:
        return _fail(name, f"receipt records activation_block {block} but [activation] "
                           "is off in the loaded config")
    applied = resolved_rollover(cfg)
    if applied != block:
        # apply_receipt_activation could not arm this block on the loaded
        # config (or armed another) — replaying under it is impossible.
        try:
            apply_activation(cfg, block)
        except ValueError as e:
            return _fail(name, f"recorded activation_block {block} cannot be applied to "
                               f"the audit config: {e}")
        return _fail(name, f"recorded activation_block {block} != the rollover the audit "
                           f"config carries ({applied})")
    grid_before = int(cfg.round.epoch_blocks_prev or cfg.round.epoch_blocks)
    if block % max(1, grid_before):
        return _fail(name, f"activation_block {block} is not a boundary of the grid before "
                           f"it ({grid_before})")
    if client is None:
        return _warn(name, f"rollover {block} recorded; validator agreement not verified "
                           "(no chain)")
    try:
        validators = list(client.validator_stakes())  # type: ignore[attr-defined]
        signals = dict(client.read_plain_commitments())  # type: ignore[attr-defined]
        now_block = int(client.current_block())  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        return _warn(name, f"rollover {block} recorded; chain read failed ({e})")
    agreed = agreed_activation(cfg.activation.feature, validators, signals,
                               threshold=cfg.activation.threshold, block=now_block,
                               dormant_after_blocks=cfg.activation.dormant_after_blocks)
    # (No admissibility filter here: the audit must SEE a fleet note that
    # disagrees with the receipt, not silently drop it.)
    if agreed is None:
        return _warn(name, f"rollover {block} recorded; validators holding "
                           f"{cfg.activation.threshold:.0%} of stake do not (yet) name one "
                           "block in their notes")
    lock, act = agreed
    if act != block:
        return _fail(name, f"recorded activation_block {block} != the block validators "
                           f"agree on ({act}, locked in at {lock})")
    return _ok(name, f"rollover {block} agreed by validators holding "
                     f">= {cfg.activation.threshold:.0%} of stake (lock-in at {lock})")


def check_margin_v2(receipt: RoundReceipt, cfg: ChainConfig,
                    client: object | None = None) -> CheckResult:
    """DEC-CA-0049: the dethrone-bar-v2 block a receipt records was decided by
    the validators, not invented by one — the same shape as
    :func:`check_activation`. A receipt without ``margin_v2_block`` PASSES
    (pre-field, or the block is typed into chain.toml). ``cfg`` already
    carries the recorded block (``apply_receipt_activation``), so the
    ``koth-params`` check replays the verdict under the v2 bar."""
    from ..shared.activation import (
        agreed_activation,
        apply_margin_v2_activation,
        configured_margin_v2_block,
        margin_v2_feature,
        resolved_margin_v2_block,
    )

    name = "margin-v2"
    block = int(getattr(receipt, "margin_v2_block", 0) or 0)
    if not block:
        return _ok(name, "no chain-decided margin v2 recorded (typed-in config applies)")
    typed = configured_margin_v2_block(cfg)
    if typed:
        if typed != block:
            return _fail(name, f"recorded margin_v2_block {block} != the block typed into the "
                               f"audit config ({typed})")
        return _ok(name, f"margin v2 from {block} matches the typed-in config")
    spec = margin_v2_feature(cfg)
    if spec is None:
        return _fail(name, f"receipt records margin_v2_block {block} but the loaded config "
                           "defines no v2 bar (or [activation] is off)")
    applied = resolved_margin_v2_block(cfg)
    if applied != block:
        try:
            apply_margin_v2_activation(cfg, block)
        except ValueError as e:
            return _fail(name, f"recorded margin_v2_block {block} cannot be applied to the "
                               f"audit config: {e}")
        return _fail(name, f"recorded margin_v2_block {block} != the block the audit config "
                           f"carries ({applied})")
    if client is None:
        return _warn(name, f"margin v2 from {block} recorded; validator agreement not "
                           "verified (no chain)")
    try:
        validators = list(client.validator_stakes())  # type: ignore[attr-defined]
        signals = dict(client.read_plain_commitments())  # type: ignore[attr-defined]
        now_block = int(client.current_block())  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        return _warn(name, f"margin v2 from {block} recorded; chain read failed ({e})")
    agreed = agreed_activation(spec.name, validators, signals,
                               threshold=cfg.activation.threshold, block=now_block,
                               dormant_after_blocks=cfg.activation.dormant_after_blocks,
                               round_cfg=cfg.round, align=spec.align)
    if agreed is None:
        return _warn(name, f"margin v2 from {block} recorded; validators holding "
                           f"{cfg.activation.threshold:.0%} of stake do not (yet) name one "
                           "block in their notes")
    lock, act = agreed
    if act != block:
        return _fail(name, f"recorded margin_v2_block {block} != the block validators agree "
                           f"on ({act}, locked in at {lock})")
    return _ok(name, f"margin v2 from {block} agreed by validators holding "
                     f">= {cfg.activation.threshold:.0%} of stake (lock-in at {lock})")


def check_era(receipt: RoundReceipt, cfg: ChainConfig, client: object | None = None) -> CheckResult:
    """DEC-CA-0043 era envelope, replayed under the receipt's own block.

    Before ``era_king_from_block`` a receipt must carry NO era context (no
    ``era_start_block`` / ``era_base_seed``, no era stamp on the manifest).
    From it, the receipt's ``era_start_block`` and the manifest's era stamp
    must be the era every node derives from the boundary, every entry's
    ``train_block`` must sit inside the era's training window, and — with a
    chain — ``era_base_seed`` must be the seed of the era's ``seed_block``
    hash (WARN without a chain, like the boundary hash check).
    """
    from ..shared.era import EraSpec, era_king_active, era_length_blocks, settlement_era

    name = "era"
    stamp = receipt.manifest.get("era")
    if not era_king_active(cfg.scoring, receipt.epoch_start_block):
        if receipt.era_start_block or receipt.era_base_seed or stamp is not None:
            return _fail(name, f"era context recorded before era_king_from_block "
                               f"{cfg.scoring.era_king_from_block} (start "
                               f"{receipt.era_start_block}, seed {receipt.era_base_seed}, "
                               f"stamp={'yes' if stamp is not None else 'no'})")
        return _ok(name, "pre-era settlement; no era context (as required)")
    expected = settlement_era(cfg.round, receipt.epoch_start_block)
    if receipt.era_start_block != expected.start_block:
        return _fail(name, f"era_start_block {receipt.era_start_block} != derived "
                           f"{expected.start_block} for boundary {receipt.epoch_start_block}")
    epoch_note = ""
    if receipt.status == "scored" or stamp is not None:
        if stamp is None:
            return _fail(name, "scored era settlement whose manifest carries no era stamp")
        try:
            era = EraSpec.from_json(stamp)
        except ValueError as e:
            return _fail(name, f"era stamp malformed: {e}")
        if (era.index, era.start_block) == (expected.index, expected.start_block) \
                and era.seed_block != expected.seed_block:
            # DEC-CA-0055 seed epoch: a v2 era may train from its epoch's first
            # era's seed block (the first v2 era, or a v2 warm start's
            # effective era — the audit checks the shape; the warm start
            # itself is the promotion record's to show)
            start = _epoch_start_for_seed(cfg, era)
            if start is None:
                return _fail(name, f"manifest era {era.index} seed block {era.seed_block} is "
                                   f"neither its own ({expected.seed_block}) nor a legal "
                                   "v2 seed-epoch start's")
            expected = replace(expected, seed_block=int(era.seed_block))
            epoch_note = f" (seed epoch from era {start})"
        if (era.index, era.start_block, era.seed_block) != (
                expected.index, expected.start_block, expected.seed_block):
            return _fail(name, f"manifest era {era.index}@{era.start_block} (seed "
                               f"{era.seed_block}) != derived {expected.index}@"
                               f"{expected.start_block} (seed {expected.seed_block})")
        window_end = era.start_block + era_length_blocks(cfg.round, era.start_block)
        for e in receipt.manifest.get("entries", ()):
            tb = int(e.get("train_block", 0))
            if not (era.seed_block <= tb < window_end):
                return _fail(name, f"{e.get('role')} uid{e.get('miner_uid')} train_block "
                                   f"{tb} outside era window [{era.seed_block}, {window_end})")
    if not receipt.era_base_seed:
        return _fail(name, "era settlement records no era_base_seed")
    if client is None:
        return _warn(name, f"era {expected.index} start {expected.start_block} consistent; "
                           f"no chain connection to verify era_base_seed against block "
                           f"{expected.seed_block}")
    try:
        onchain = seed_from_block_hash(str(client.block_hash(expected.seed_block)))
    except Exception as e:  # noqa: BLE001 — lite node / pruned history
        return _warn(name, f"chain could not serve era seed block {expected.seed_block} "
                           f"({e}); era_base_seed not verified")
    if onchain != receipt.era_base_seed:
        return _fail(name, f"era_base_seed {receipt.era_base_seed} != seed of block "
                           f"{expected.seed_block} hash ({onchain})")
    return _ok(name, f"era {expected.index} start {expected.start_block}; era_base_seed "
                     f"matches block {expected.seed_block}{epoch_note}")


def _epoch_start_for_seed(cfg: ChainConfig, era) -> int | None:
    """DEC-CA-0055: the seed-epoch start era a v2 era's stamped seed block
    names, when it is a LEGAL one — ``[v2] king_ladder`` on, a v2 era, and the
    named start is the first v2 era or a warm-startable era
    (``>= v2_first_warm_era``) no later than the era itself; else None."""
    from ..shared.emission import ladder_on
    from ..shared.era import era_length_blocks, v2_first_era, v2_first_warm_era

    if not ladder_on(cfg) or not cfg.v2.active(int(era.start_block)):
        return None
    length = era_length_blocks(cfg.round, int(era.start_block))
    if int(era.seed_block) % length:
        return None
    start = int(era.seed_block) // length + 1
    first, warm = v2_first_era(cfg), v2_first_warm_era(cfg)
    if first is None or warm is None or not (first <= start < int(era.index)):
        return None
    return start if (start == first or start >= warm) else None


def check_epoch_alignment(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """The recorded boundary sits on an epoch multiple (the submission deadline
    is deterministic, not validator-chosen)."""
    name = "epoch-alignment"
    # Resolve the length in force AT the recorded boundary, not at the head:
    # a receipt written before a scheduled cadence change must keep validating
    # against the grid it was written on, forever.
    epoch_blocks = effective_epoch_blocks(cfg.round, receipt.epoch_start_block)
    if receipt.epoch_start_block % epoch_blocks != 0:
        return _fail(name, f"epoch_start_block {receipt.epoch_start_block} is not a "
                           f"multiple of [round] epoch_blocks {epoch_blocks}")
    created = int(receipt.manifest.get("created_block", 0))
    if created and not (receipt.epoch_start_block <= created
                        < receipt.epoch_start_block + epoch_blocks):
        return _fail(name, f"manifest created_block {created} falls outside the epoch "
                           f"[{receipt.epoch_start_block}, "
                           f"{receipt.epoch_start_block + epoch_blocks})")
    return _ok(name, f"boundary {receipt.epoch_start_block} on the epoch grid")


def check_block_hash_onchain(receipt: RoundReceipt, client: object | None) -> CheckResult:
    """The recorded epoch-boundary hash matches the chain (archive access needed
    for old rounds — a lite node degrades to WARN, never a silent pass)."""
    name = "block-hash-onchain"
    if client is None:
        return _warn(name, "no chain connection; recorded block hash not verified on-chain")
    try:
        onchain = str(client.block_hash(receipt.epoch_start_block))
    except Exception as e:  # noqa: BLE001 — lite node / pruned history
        return _warn(name, f"chain could not serve block {receipt.epoch_start_block} "
                           f"({e}); hash not verified")
    if onchain.lower() != receipt.epoch_block_hash.lower():
        return _fail(name, f"on-chain hash {onchain} != recorded {receipt.epoch_block_hash}")
    return _ok(name, f"block {receipt.epoch_start_block} hash matches the chain")


# ── contract digests ──────────────────────────────────────────────────────────


def receipt_training(receipt: RoundReceipt, cfg: ChainConfig):
    """The ``[training]`` contract the receipt's settlement trained under:
    resolved at the START of the era the settlement belongs to (DEC-CA-0050 —
    the v2 contract from ``[v2] from_block``), exactly as the validator's
    ``_training_for`` does. Without a v2 schedule, ``cfg.training`` itself."""
    from ..shared.era import settlement_contract_block

    t = cfg.training
    if not int(getattr(t, "v2_from_block", 0) or 0):
        return t
    return t.at_block(settlement_contract_block(cfg.round, int(receipt.epoch_start_block)))


def check_v2(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """DEC-CA-0050: the round was judged under ITS OWN block's v2 rules.

    * ``v2_block`` is stamped exactly when the settlement judges a v2 era (its
      era starts at or after ``[v2] from_block``), and equals that block;
    * a v2 settlement's entries were trained at the v2 contract's size and a
      declared warm start is a v2-lineage checkpoint of that size — a toto2
      checkpoint never initialises the MoE;
    * the first two v2 eras train from random init: no v2-lineage promotion
      can take effect before ``v2_first_era + 2`` (one full era of notice
      after the first v2 boundary), and older generations are toto2.

    The contract / base-arch checks already replay under
    :func:`receipt_training`."""
    from ..shared.era import settlement_era, v2_first_warm_era, v2_settlement_active

    name = "v2"
    if cfg.v2.launch_at_utc and not cfg.v2.from_block:
        return _warn(name, "dated launch is unresolved; use chain-backed audit or a resolved from_block")
    got = int(getattr(receipt, "v2_block", 0) or 0)
    want = (int(cfg.v2.from_block)
            if v2_settlement_active(cfg, int(receipt.epoch_start_block)) else 0)
    if got != want:
        return _fail(name, f"receipt v2_block {got} != {want} derived for boundary "
                           f"{receipt.epoch_start_block} ([v2] from_block "
                           f"{int(getattr(cfg.v2, 'from_block', 0) or 0)})")
    if not want:
        return _ok(name, "pre-v2 settlement (no v2 rules)")
    if receipt.status == "rejected":
        # the gate's own reason is replayed by the contract / era checks
        return _ok(name, f"rejected v2 settlement; v2_block {want} stamped")
    contract = receipt_training(receipt, cfg)
    preset = contract.arch_preset
    for e in receipt.manifest.get("entries", ()):
        size = str(e.get("size", "") or preset)
        if size != preset:
            return _fail(name, f"{e.get('role')} uid{e.get('miner_uid')} trained at "
                               f"{size!r}, the v2 contract is {preset!r}")
    ws = str(receipt.manifest.get("warm_start_ckpt", "") or "")
    if ws:
        ws_size = str(receipt.manifest.get("warm_start_size", "") or "")
        if ws_size != preset:
            return _fail(name, f"v2 era initialised from a {ws_size!r} checkpoint; the v2 "
                               f"lineage is {preset!r}")
        era = settlement_era(cfg.round, int(receipt.epoch_start_block))
        warm = v2_first_warm_era(cfg)
        if warm is not None and era.index < warm:
            return _fail(name, f"era {era.index} is warm-started, but no v2-lineage "
                               f"generation can take effect before era {warm}")
    return _ok(name, f"v2 settlement (from block {want}) judged under the {preset} "
                     "contract")


def check_contract_digest(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """The round trained under a contract this auditor accepts.

    Replays each round under its OWN rule, matching the validator
    (:meth:`~cascade.validator.loop.ValidatorRunner._check_contract`):

    * **Declared** (DEC-CA-0036) — from ``declared_contract_from_block``, for a
      receipt carrying a published ``contract_body``: the body must re-hash to
      the declared digest, and its locked projection must equal the local one.
      The non-locked terms are reported by :func:`check_contract_declaration`,
      not gated here.
    * **Strict** — full equality against the local ``chain.toml``, with the
      block-gated ``prior_contract_digest`` transition.

    The declared path is what makes a historical replay honest: a round's terms
    come from the round, so re-pinning ``chain.toml`` does not silently
    change what an audit of an older round compares against.
    """
    name = "contract-digest"
    training = receipt_training(receipt, cfg)
    want = contract_digest(training)
    got = str(receipt.manifest.get("contract_digest", ""))
    body = receipt.manifest.get("contract_body")
    declared_from = int(cfg.scoring.declared_contract_from_block)
    if (declared_from and isinstance(body, dict)
            and int(receipt.epoch_start_block) >= declared_from):
        rehashed = contract_digest(body)
        if rehashed != got:
            return _fail(name, f"published contract_body hashes to {rehashed[:16]}… "
                               f"but the manifest declares {got[:16]}…")
        want_locked = locked_contract_terms(training)
        got_locked = locked_contract_terms(body)
        if got_locked != want_locked:
            return _fail(name, f"locked contract terms differ from chain.toml: "
                               f"{got_locked} != {want_locked}")
        return _ok(name, f"contract_body hashes to the declared {got[:16]}… and its "
                         "locked terms match chain.toml")
    if got != want:
        prior = cfg.scoring.prior_contract_digest
        from_block = int(cfg.scoring.contract_from_block)
        if (prior and from_block and got == prior
                and int(receipt.epoch_start_block) < from_block):
            return _ok(name, f"contract_digest {got[:16]}… matches the pinned prior "
                             f"contract (epoch {receipt.epoch_start_block} < "
                             f"contract_from_block {from_block})")
        return _fail(name, f"manifest contract_digest {got[:16]}… != local chain.toml "
                           f"{want[:16]}… (different training contract)")
    return _ok(name, f"contract_digest {want[:16]}… matches chain.toml")


def check_contract_declaration(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """Report how the round's DECLARED contract differs from the local one.

    Under DEC-CA-0036 the non-locked ``[training]`` terms — recipe, budget,
    runtime image, submission surface — are trainer-declared: they are signed
    and recorded but never reject a round, so a trainer-side fix ships without
    a validator deploy. That trade only holds if the drift is *visible*, which
    is this check's whole job. It never FAILs (the terms are not consensus);
    it WARNs when the round trained under terms this chain.toml does not
    describe, and names every field so a reader can see exactly what moved.
    """
    name = "contract-declaration"
    body = receipt.manifest.get("contract_body")
    if not isinstance(body, dict):
        return _skip(name, "manifest publishes no contract_body (pre-DEC-CA-0036 trainer)")
    local = contract_payload(receipt_training(receipt, cfg))
    locked = set(LOCKED_CONTRACT_FIELDS) | {"extra_sizes"}
    drift = sorted(
        k for k in set(body) | set(local)
        if k not in locked and _jsonable(body.get(k)) != _jsonable(local.get(k))
    )
    if not drift:
        return _ok(name, "declared contract matches chain.toml on every term")
    detail = ", ".join(
        f"{k}: declared {body.get(k)!r} vs local {local.get(k)!r}" for k in drift
    )
    return _warn(name, f"round trained under {len(drift)} declared term(s) that differ "
                       f"from this chain.toml — recorded, not gated ({detail})")


def _jsonable(value: object) -> object:
    """Normalise for cross-source comparison: a contract read back from JSON has
    lists where the dataclass had tuples, so compare both as lists."""
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def check_base_arch_digest(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """The frozen-architecture digests: the manifest matches the chain.toml pin,
    and the pin matches a recomputation from the local model source."""
    name = "base-arch-digest"
    from ..trainer.contract import compute_base_arch_digest

    got = str(receipt.manifest.get("base_arch_digest", ""))
    training = receipt_training(receipt, cfg)
    if got != training.base_arch_digest:
        return _fail(name, f"manifest base_arch_digest {got[:16]}… != chain.toml pin "
                           f"{training.base_arch_digest[:16]}…")
    for size in training.all_sizes():
        computed = compute_base_arch_digest(size)
        if computed != size.base_arch_digest:
            return _fail(name, f"[{size.arch_preset}] pinned digest "
                               f"{size.base_arch_digest[:16]}… does not recompute from the "
                               f"local model source ({computed[:16]}…) — local checkout "
                               "differs from the published architecture")
    return _ok(name, "manifest matches the pin; pin recomputes from local source")


# ── participants / cutoff ─────────────────────────────────────────────────────


def check_commit_cutoff(
    receipt: RoundReceipt, client: object | None = None
) -> CheckResult:
    """Every participant committed strictly before the epoch boundary, and every
    trained entry's generator is a participant's committed pointer.

    With a reachable chain, each participant's ``(gen_ref, commit_block)`` is
    also checked against the currently visible reveals; a lite node that cannot
    serve the historical state degrades to WARN.
    """
    name = "commit-cutoff"
    if not receipt.participants:
        return _warn(name, "receipt carries no participant set (chain was unreachable "
                           "at receipt time); cutoff not verifiable")
    late = [p for p in receipt.participants if p.commit_block >= receipt.epoch_start_block]
    if late:
        return _fail(name, "participants committed at/after the boundary "
                           f"{receipt.epoch_start_block}: "
                           + ", ".join(f"{p.hotkey}@{p.commit_block}" for p in late))
    by_hotkey = {p.hotkey: p for p in receipt.participants}
    try:
        manifest = receipt.load_embedded_manifest()
    except (ValueError, KeyError) as e:
        return _fail(name, f"embedded manifest unparseable: {e}")
    for e in manifest.entries:
        p = by_hotkey.get(e.miner_hotkey)
        if p is None:
            return _fail(name, f"trained entry {e.miner_hotkey} ({e.role}) is not in the "
                               "participant set")
        if p.gen_ref != e.gen_ref:
            return _fail(name, f"entry {e.miner_hotkey} trained gen_ref {e.gen_ref[:32]}… "
                               f"but the participant committed {p.gen_ref[:32]}…")
    chain_note = ""
    if client is not None:
        chain_note = _cutoff_chain_note(receipt, client)
        if chain_note.startswith("FAIL:"):
            return _fail(name, chain_note[5:])
    else:
        chain_note = "; chain payloads not cross-checked (no connection)"
        return _warn(name, f"all {len(receipt.participants)} participant(s) pre-cutoff and "
                           f"entries match the recorded set{chain_note}")
    return _ok(name, f"all {len(receipt.participants)} participant(s) pre-cutoff; "
                     f"entries match{chain_note}")


def _cutoff_chain_note(receipt: RoundReceipt, client: object) -> str:
    """Cross-check recorded participants against currently visible reveals.

    Reads the FULL retained reveal history per hotkey, so a participant stays
    verifiable even after re-committing for a later round. Only a definite
    contradiction (same hotkey, same block, different payload) is a FAIL; a
    reveal the chain no longer retains (pruned history, deregistered hotkey,
    lite node) is noted as unverifiable.
    """
    from ..interface.validation import parse_commit

    try:
        try:
            commitments = client.poll_commitments(include_history=True)
        except TypeError:  # pre-history client: latest reveal only
            commitments = client.poll_commitments()
    except Exception as e:  # noqa: BLE001
        return f"; chain payload cross-check unavailable ({e})"
    by_hotkey: dict[str, dict[int, str]] = {}
    for c in commitments:
        by_hotkey.setdefault(c.hotkey, {})[int(c.commit_block)] = c.payload
    unverifiable = 0
    for p in receipt.participants:
        payload = by_hotkey.get(p.hotkey, {}).get(int(p.commit_block))
        if payload is None:
            unverifiable += 1  # reveal no longer on chain / deregistered / lite node
            continue
        parsed = parse_commit(payload)
        if parsed is None or parsed.ref != p.gen_ref:
            return (f"FAIL:participant {p.hotkey} recorded gen_ref {p.gen_ref[:32]}… "
                    f"but chain shows {getattr(parsed, 'ref', payload)[:32]}… "
                    f"at the same block {p.commit_block}")
    if unverifiable:
        return f"; {unverifiable} participant(s) unverifiable (reveal no longer on chain)"
    return "; chain payloads match"


# ── verdict + weights ─────────────────────────────────────────────────────────


def _duelled_hotkeys(receipt: RoundReceipt) -> list[str]:
    """Challenger hotkeys the validator actually evaluated, in record order.

    Read off ``entry_scores`` rather than the manifest: the manifest lists what
    the trainer *advanced*, but a challenger can be dropped before the duel (a
    size the king does not cover), and an inconclusive round stops the cohort
    early. ``entry_scores`` is the record of what was scored, and it is signed.
    """
    out: list[str] = []
    for r in receipt.entry_scores:
        if r.role == "challenger" and r.hotkey not in out:
            out.append(r.hotkey)
    return out


def _pooled_scores(
    receipt: RoundReceipt, manifest: TrainingManifest, *, hotkey: str | None = None
):
    """Rebuild the pooled king/challenger score lists exactly as the validator
    pooled them (paired sizes in manifest order, per-size scores concatenated).

    ``hotkey`` selects which challenger of a multi-finalist cohort to pool
    (DEC-CA-0012); ``None`` takes the last one recorded, which is the challenger
    the receipt's verdict belongs to under the cohort rule — the crowned clearer,
    or the challenger that stopped an inconclusive cohort. Keying by ``(role,
    size)`` alone would silently collapse the cohort to one arbitrary entrant,
    which is the bug this parameter exists to prevent.
    """
    king_by_size = {e.size: e for e in manifest.entries_for_role("king")}
    if hotkey is None:
        duelled = _duelled_hotkeys(receipt)
        hotkey = duelled[-1] if duelled else None
    chal_by_size = {
        e.size: e for e in manifest.entries_for_role("challenger")
        if hotkey is None or e.miner_hotkey == hotkey
    }
    paired = [s for s in manifest.sizes() if s in king_by_size and s in chal_by_size]
    recs = {(r.role, r.size, r.hotkey): r for r in receipt.entry_scores}
    king, chal = [], []
    for size in paired:
        k = recs.get(("king", size, king_by_size[size].miner_hotkey))
        c = recs.get(("challenger", size, chal_by_size[size].miner_hotkey))
        if k is None or c is None:
            raise KeyError(f"entry_scores missing for size {size!r}")
        king += [w.to_score() for w in k.scores]
        chal += [w.to_score() for w in c.scores]
    return king, chal, paired


def _baseline_pooled(receipt: RoundReceipt, paired: list[str]):
    """The shared warm-start init's scores from the receipt, pooled over
    ``paired`` — or ``None`` when no ``role == "baseline"`` rows were
    recorded.

    Baseline rows are appended whenever the validator scored the init — for
    the increment margin AND/OR the init-baseline floor ([scoring]
    init_gate_mode), so their presence alone does not name the margin
    denomination. The judged rule is the conjunction the caller applies:
    "increment" ⟺ the receipt's params say ``margin_mode = "increment"``
    (the unmodified config params) AND baseline rows exist (the validator's
    in-round fallback to "level" — random-init round, size mismatch —
    records none). A level-mode round with the floor on has rows but keeps
    the level rule; the floor itself replays through the recorded
    ``init_gate_mode`` param. Like ``wql_mode``, the judged mode cannot ride
    the signed ``VerdictRecord`` without invalidating archived receipts —
    the replay is self-verifying instead.
    """
    rows = [r for r in receipt.entry_scores if r.role == "baseline"]
    if not rows:
        return None
    by_size = {r.size: r for r in rows}
    out = []
    for size in paired:
        r = by_size.get(size)
        if r is None:
            raise KeyError(f"baseline entry_scores missing for size {size!r}")
        out += [w.to_score() for w in r.scores]
    if not out:
        # Baseline rows exist but carry no scores: a malformed receipt, not a
        # level-judged round — raising here surfaces it as a check FAILURE
        # (callers catch ValueError) instead of a truthiness slip that would
        # replay under the wrong rule or crash evaluate_round.
        raise ValueError("baseline entry_scores present but empty")
    return out


def _cohort_maxt_lcbs(receipt: RoundReceipt, manifest: TrainingManifest, params,
                      baseline=None):
    """``{hotkey: max-T family-wise LCB}`` for a cohort round replayed under
    the shared-resample correction (DEC-CA-0038) — the same call the validator
    made (:func:`cascade.eval.koth.cohort_maxt_lcb_map`), off the receipt's own
    scores. ``params`` is the UNMODIFIED recorded set (max-T needs no
    ``alpha/k``). ``baseline`` (the pooled init rows) puts the bound in
    INCREMENT units — pass it exactly when the validator did: increment judged
    AND ``cohort_maxt_increment_from_block`` reached for this receipt's block.
    Raises on unpaired/​unrebuildable scores."""
    from ..eval.koth import cohort_maxt_lcb_map

    duelled = _duelled_hotkeys(receipt)
    king_scores = None
    cohort: list[tuple[str, list]] = []
    for hk in duelled:
        king_scores, chal, paired = _pooled_scores(receipt, manifest, hotkey=hk)
        if not paired:
            raise ValueError(f"challenger {hk} has no paired size")
        cohort.append((hk, chal))
    return cohort_maxt_lcb_map(
        king_scores, cohort, params,
        seed=_bootstrap_seed(receipt.verdict.bootstrap_seed), wql_mode="geomean",
        baseline_scores=baseline)


def _cohort_params(params, manifest: TrainingManifest):
    """``params`` with the duel's family-wise alpha applied (DEC-CA-0012).

    ``bootstrap_alpha / k`` where ``k = len(manifest.duel_cohort())`` — the same
    derivation the validator ran, off the same signed manifest. ``k <= 1`` returns
    ``params`` unchanged, so a single-challenger round (every round before this
    shipped) replays bit-identically.
    """
    from dataclasses import replace

    k = len(manifest.duel_cohort()[0])
    if k <= 1:
        return params
    return replace(params, bootstrap_alpha=params.bootstrap_alpha / k)


def check_duel_cohort(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """Verify the cohort duel's SELECTION, not just its verdict (DEC-CA-0012).

    With one challenger this is a no-op skip. With a cohort it replays every
    recorded challenger against the king under the corrected alpha and checks the
    two things the verdict alone cannot show:

    1. Every duelled challenger reproduces a verdict at ``bootstrap_alpha / k``
       — so no challenger was judged under a looser bound than the cohort earns.
    2. The challenger the verdict belongs to is the best OBSERVED relative
       improvement among those that cleared the margin (lowest ``chal_geomean``,
       the king's being shared). This is what catches a validator crowning a
       clearer that was not the best, which is exactly the artifact the rejected
       sequential rule would have produced.
    3. The receipt's PUBLISHED ``cohort_k`` and ``cohort_lcbs`` match what the
       signed manifest implies and what the scores replay to. Those two fields are
       the round's math in public form — with ``margin`` they are the complete
       input to the selection rule — so anyone can check "who cleared and was the
       crowned one the best" without re-running the bootstrap; this check
       confirms the published numbers are the ones that actually decided.

    Note the correction is applied to the bootstrap QUANTILE, not to the margin:
    the LCB becomes the ``alpha/k`` percentile of the paired distribution while
    ``margin`` stays flat. Replaying with a raised margin instead will not
    reproduce these numbers, and is not the rule.

    A gift-gate block legitimately moves the crown down the clearer list, so when
    the recorded verdict shows the gate ran, requirement 2 is relaxed to "the
    crowned challenger cleared" — the sidecar numbers are not recomputable at
    Tier 0, the same limitation :func:`check_verdict` already documents.
    """
    name = "duel-cohort"
    if receipt.verdict is None:
        return _skip(name, "no verdict on this receipt")
    try:
        manifest = receipt.load_embedded_manifest()
    except (ValueError, KeyError) as e:
        return _fail(name, f"cannot load embedded manifest: {e}")
    duelled = _duelled_hotkeys(receipt)
    if len(duelled) <= 1:
        return _skip(name, "single-challenger round; no cohort to verify")

    from ..eval.koth import KothParams, evaluate_round

    v = receipt.verdict
    try:
        params = KothParams(**v.params)
    except TypeError as e:
        return _fail(name, f"recorded params do not form a KothParams: {e}")
    k = len(manifest.duel_cohort()[0])
    if k < len(duelled):
        return _fail(name, f"{len(duelled)} challengers were scored but the signed "
                           f"manifest's cohort is {k} — the alpha correction the "
                           f"round was judged under cannot be re-derived")
    # The receipt PUBLISHES the k it used; it must equal the one the signed
    # manifest implies, or the alpha the round was judged under is not the alpha
    # it claims. (0 = a pre-cohort receipt, which never reaches here.)
    if v.cohort_k and v.cohort_k != k:
        return _fail(name, f"receipt records cohort_k={v.cohort_k} but the signed "
                           f"manifest's cohort is {k}")
    duel_params = _cohort_params(params, manifest)

    replayed: list[tuple[str, object]] = []
    for hk in duelled:
        try:
            king, chal, paired = _pooled_scores(receipt, manifest, hotkey=hk)
        except (ValueError, KeyError) as e:
            return _fail(name, f"cannot rebuild pooled scores for {hk}: {e}")
        if not paired:
            return _fail(name, f"challenger {hk} has no paired size")
        # Margin denomination: "increment" ⟺ the recorded params say so AND
        # baseline rows exist (the in-round fallback records none). Baseline
        # rows on a level round belong to the init floor (see
        # _baseline_pooled) — level rule, rows passed through for the floor.
        try:
            baseline = _baseline_pooled(receipt, paired)
        except (ValueError, KeyError) as e:
            return _fail(name, f"cannot rebuild baseline scores: {e}")
        from dataclasses import replace as _dc_replace

        judged_increment = (
            duel_params.margin_mode == "increment" and baseline is not None
        )
        gate_on = str(getattr(duel_params, "init_gate_mode", "off") or "off") != "off"
        mode_params = (
            _dc_replace(duel_params, margin_mode="increment") if judged_increment
            else cfg.judged_level_params(duel_params, receipt.epoch_start_block)
        )
        res = evaluate_round(
            king, chal, mode_params,
            seed=_bootstrap_seed(v.bootstrap_seed),
            king_tenure_rounds=v.king_tenure_rounds,
            baseline_scores=(
                baseline if (judged_increment or (gate_on and baseline is not None))
                else None
            ),
        )
        replayed.append((hk, res))

    # Cohort family-wise correction (DEC-CA-0038): from cohort_maxt_from_block
    # the round was judged under the shared-resample max-T, not Bonferroni
    # alpha/k. Resolve the rule from THIS receipt's block (exactly as the
    # validator did) and, when armed, replace every replayed lcb with its
    # joint bound and re-decide the win — so the cohort_lcbs check below
    # verifies the numbers that actually decided the round. Bonferroni rounds
    # (and every pre-gate round) skip this untouched.
    use_maxt = cohort_maxt_active(cfg.scoring, receipt.epoch_start_block)
    if use_maxt:
        from ..eval.koth import with_cohort_lcb

        # Increment units for the joint bound (DEC-CA-0039 stacked): exactly
        # when the round was increment-judged AND its block reached the gate.
        maxt_baseline = (
            baseline if judged_increment and cohort_maxt_increment_active(
                cfg.scoring, receipt.epoch_start_block) else None)
        # The UNMODIFIED recorded params (no alpha/k) with the judged margin
        # mode — the same pair the validator handed cohort_maxt_lcb_map.
        maxt_params = (
            _dc_replace(params, margin_mode="increment") if judged_increment
            else cfg.judged_level_params(params, receipt.epoch_start_block))
        try:
            lcbs = _cohort_maxt_lcbs(receipt, manifest, maxt_params, maxt_baseline)
        except (ValueError, KeyError) as e:
            return _fail(name, f"cannot replay cohort max-T bound: {e}")
        replayed = [(hk, with_cohort_lcb(res, lcbs[hk], params))
                    for hk, res in replayed]

    # The verdict belongs to the LAST challenger recorded (the validator appends
    # in evaluation order and the crowned/​stopping challenger is scored last).
    crowned = duelled[-1]
    clearers = [(hk, r) for hk, r in replayed if r.challenger_wins_round]
    problems = []
    # Every published per-challenger LCB must be the one that replays. This is
    # what lets a third party check the selection arithmetically — the recorded
    # bounds and the recorded margin are the whole input to "who cleared" — so a
    # published bound that does not reproduce is a hard failure, not a warning.
    for hk, res in replayed:
        recorded = (v.cohort_lcbs or {}).get(hk)
        if recorded is None:
            problems.append(f"no published LCB for duelled challenger {hk}")
        elif not _close(recorded, None if math.isnan(res.lcb) else res.lcb):
            problems.append(f"published LCB for {hk} is {recorded} but replays as "
                            f"{res.lcb}")
    for hk in (v.cohort_lcbs or {}):
        if hk not in {h for h, _ in replayed}:
            problems.append(f"published an LCB for {hk}, which was never scored")
    problems += _cohort_stats_problems(v.cohort_stats, replayed)
    if v.inconclusive:
        if clearers:
            problems.append(
                f"round recorded inconclusive but {len(clearers)} challenger(s) "
                f"replay as conclusive wins")
    elif clearers:
        gate_ran = v.gift_gate_passed is not None or v.bench_gate_passed is not None
        best = min(clearers, key=lambda t: t[1].chal_geomean)[0]
        if crowned not in {hk for hk, _ in clearers}:
            problems.append(f"crowned {crowned} did not clear the margin on replay")
        elif not gate_ran and crowned != best:
            problems.append(
                f"crowned {crowned} is not the best clearer; {best} has a better "
                f"observed geomean and no public-benchmark gate is recorded")
    if problems:
        return _fail(name, "; ".join(problems))
    rule = (
        f"cohort max-T family-wise correction (DEC-CA-0038, alpha="
        f"{params.bootstrap_alpha:.4f})" if use_maxt else
        f"Bonferroni alpha={duel_params.bootstrap_alpha:.5f} "
        f"(={params.bootstrap_alpha:.4f}/{k})"
    )
    return _ok(name, f"{len(duelled)} challengers replayed under {rule}; "
                     f"{len(clearers)} cleared; crowned {crowned[:12]}")


def _cohort_stats_problems(published: dict | None, replayed: list) -> list[str]:
    """Every published per-challenger diagnostic must be the one that replays.

    ``cohort_stats`` is optional on the receipt (receipts before it shipped carry
    none), so its absence is not a problem; but once published it is signed, so
    it must cover every duelled challenger, name no one else, and every number
    must reproduce from the recorded scores — the same standard as the LCBs.
    ``wilcoxon_p`` is compared only when the replay could compute one (scipy
    is optional for the auditor).
    """
    if not published:
        return []
    from ..shared.receipt import cohort_stats_of

    problems: list[str] = []
    scored = {hk for hk, _ in replayed}
    for hk in published:
        if hk not in scored:
            problems.append(f"published stats for {hk}, which was never scored")
    for hk, res in replayed:
        rec = published.get(hk)
        if rec is None:
            problems.append(f"no published stats for duelled challenger {hk}")
            continue
        want = cohort_stats_of(res)
        for key in ("geomean", "win_rate", "boot_p50", "boot_p95"):
            if not _close(rec.get(key), want.get(key)):
                problems.append(f"published {key} for {hk} is {rec.get(key)} but "
                                f"replays as {want.get(key)}")
        if want.get("wilcoxon_p") is not None and not _close(
                rec.get("wilcoxon_p"), want["wilcoxon_p"]):
            problems.append(f"published wilcoxon_p for {hk} is {rec.get('wilcoxon_p')} "
                            f"but replays as {want['wilcoxon_p']}")
        # per_domain_win_rate is a non-gating diagnostic, and unlike the
        # bootstrap inputs the window's DOMAIN label is not carried on the
        # receipt (``WindowScoreRecord`` keeps only the cluster ``source``). So
        # a Tier-0 replay collapses every window into the ``"unknown"`` bucket
        # and cannot reconstruct the published split — the same Tier-0 limit the
        # gift-gate sidecar has. Compare it only when the replay actually
        # resolved domains (a future receipt that carries them), exactly as
        # ``wilcoxon_p`` above is compared only when the replay could compute one.
        got_pd, want_pd = rec.get("per_domain_win_rate") or {}, want.get("per_domain_win_rate") or {}
        replay_resolved_domains = bool(want_pd) and set(want_pd) != {"unknown"}
        if replay_resolved_domains and (set(got_pd) != set(want_pd) or any(
                not _close(got_pd[d][0], want_pd[d][0]) or int(got_pd[d][1]) != int(want_pd[d][1])
                for d in want_pd)):
            problems.append(f"published per_domain_win_rate for {hk} does not replay")
    return problems


def check_koth_params(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """The recorded decision parameters equal the published ``[scoring]``."""
    name = "koth-params"
    if receipt.verdict is None:
        return _skip(name, "no verdict on this receipt")
    from dataclasses import asdict

    from ..eval.koth import KothParams

    # The values in force for THIS receipt's epoch: a scheduled margin change
    # is replayed per round, exactly as the validator resolved it.
    want = dict(asdict(cfg.koth_params(block=receipt.epoch_start_block)))
    try:
        # Normalise through the dataclass: a receipt written before a params
        # field existed compares at that field's default instead of failing on
        # a missing key (the gift_gate / margin_mode precedent — fields only
        # ever ship with behaviour-preserving defaults).
        got = dict(asdict(KothParams(**receipt.verdict.params)))
    except TypeError as e:
        return _fail(name, f"recorded params do not form a KothParams: {e}")
    if got != want:
        diff = {k for k in set(want) | set(got) if want.get(k) != got.get(k)}
        return _fail(name, f"recorded KothParams differ from chain.toml [scoring] on: "
                           f"{sorted(diff)}")
    return _ok(name, "recorded params match chain.toml [scoring]")


def check_verdict(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """Recompute the KOTH verdict from the receipt's own scores.

    Feeds the recorded per-window scores back into ``evaluate_round`` with the
    recorded params/seed/tenure and compares lcb, margin, window count, and the
    win/inconclusive bits. A gift-gate override is applied from the recorded
    gate outcome (the sidecar run itself is not recomputable at Tier 0).
    """
    name = "verdict"
    if receipt.verdict is None:
        return _skip(name, "no verdict on this receipt")
    from ..eval.koth import KothParams, evaluate_round

    v = receipt.verdict
    try:
        params = KothParams(**v.params)
    except TypeError as e:
        return _fail(name, f"recorded params do not form a KothParams: {e}")
    try:
        manifest = receipt.load_embedded_manifest()
        king, chal, paired = _pooled_scores(receipt, manifest)
    except (ValueError, KeyError) as e:
        return _fail(name, f"cannot rebuild pooled scores: {e}")
    if not paired:
        return _fail(name, "verdict recorded but no paired (king, challenger) size")

    # Replay under the rule that DECIDED the round, not necessarily today's.
    # The receipt cannot record which rule it used (a new VerdictRecord field
    # would change the signed canonical body and invalidate every archived
    # receipt — see the note in cascade/shared/receipt.py), so we replay under
    # each known aggregation and take the one that reproduces the recorded LCB.
    # This is self-verifying: reproducing an LCB to 1e-9 under the wrong rule is
    # not something a mis-scored or tampered receipt does by accident.
    # Family-wise alpha (DEC-CA-0012): the round was judged at
    # ``bootstrap_alpha / k``, k = the advanced cohort size. The receipt records
    # the UNMODIFIED config params on purpose — a mutated alpha there would fail
    # ``check_koth_params`` against the published ``[scoring]`` — so re-derive the
    # correction from the signed manifest, exactly as the validator did. k = 1
    # leaves params untouched, so every pre-cohort receipt replays unchanged.
    duel_params = _cohort_params(params, manifest)

    # Margin denomination: "increment" ⟺ the recorded params say so AND
    # baseline rows exist (the validator's in-round fallback to "level"
    # records none). Rows on a level round belong to the init floor
    # ([scoring] init_gate_mode, replayed via the recorded param) — level
    # rule, rows passed through so the floor replays identically.
    try:
        baseline = _baseline_pooled(receipt, paired)
    except (ValueError, KeyError) as e:
        return _fail(name, f"cannot rebuild baseline scores: {e}")
    from dataclasses import replace as _dc_replace

    judged_increment = (
        duel_params.margin_mode == "increment" and baseline is not None
    )
    gate_on = str(getattr(duel_params, "init_gate_mode", "off") or "off") != "off"
    duel_params = (
        _dc_replace(duel_params, margin_mode="increment") if judged_increment
        else cfg.judged_level_params(duel_params, receipt.epoch_start_block)
    )
    replay_baseline = (
        baseline if (judged_increment or (gate_on and baseline is not None))
        else None
    )

    def _replay(mode: str):
        return evaluate_round(
            king, chal, duel_params,
            seed=_bootstrap_seed(v.bootstrap_seed),
            king_tenure_rounds=v.king_tenure_rounds,
            wql_mode=mode,
            baseline_scores=replay_baseline,
        )

    result = _replay("geomean")
    scoring_note = ""
    # Cohort family-wise correction (DEC-CA-0038): from cohort_maxt_from_block
    # the crowned challenger's lcb is its JOINT max-T bound (it depends on the
    # whole cohort), so a solo replay cannot reproduce it. Resolve the rule
    # from this receipt's block and, when armed, swap in the crowned bound +
    # re-decide the win before comparing. k <= 1 and pre-gate rounds never
    # enter this branch (the crowned solo replay is the whole story there).
    duelled = _duelled_hotkeys(receipt)
    if len(duelled) > 1 and cohort_maxt_active(cfg.scoring, receipt.epoch_start_block):
        from ..eval.koth import with_cohort_lcb

        maxt_baseline = (
            baseline if judged_increment and cohort_maxt_increment_active(
                cfg.scoring, receipt.epoch_start_block) else None)
        maxt_params = (
            _dc_replace(params, margin_mode="increment") if judged_increment
            else cfg.judged_level_params(params, receipt.epoch_start_block))
        try:
            lcbs = _cohort_maxt_lcbs(receipt, manifest, maxt_params, maxt_baseline)
        except (ValueError, KeyError) as e:
            return _fail(name, f"cannot replay cohort max-T bound: {e}")
        result = with_cohort_lcb(result, lcbs[duelled[-1]], params)
        scoring_note = " (cohort max-T family-wise correction, DEC-CA-0038)"
    elif not _close(None if math.isnan(result.lcb) else result.lcb, v.lcb):
        legacy = _replay("pooled")
        if _close(None if math.isnan(legacy.lcb) else legacy.lcb, v.lcb):
            result = legacy
            scoring_note = (
                " (judged under the pre-2026-07-28 pooled-MWSQL rule; "
                "reproduced under that rule, not the current one)"
            )
    problems = []
    if result.n_windows != v.n_windows:
        problems.append(f"n_windows {v.n_windows} != recomputed {result.n_windows}")
    if not _close(result.margin, v.margin):
        problems.append(f"margin {v.margin} != recomputed {result.margin}")
    recomputed_lcb = None if math.isnan(result.lcb) else result.lcb
    if not _close(recomputed_lcb, v.lcb):
        problems.append(f"lcb {v.lcb} != recomputed {recomputed_lcb}")
    gate_note = ""
    if v.gift_gate_passed is not None or (
        params.gift_gate_mode == "enforce"
        and result.challenger_wins_round
        and v.inconclusive
    ):
        # The public-benchmark gate ran (or was uncomputable under enforce);
        # its sidecar numbers are not recomputable here — apply the recorded
        # outcome to the private-pool result and note it.
        from ..eval.gift_gate import GiftGateResult
        from ..eval.koth import apply_gift_gate

        gate = GiftGateResult(
            computed=v.gift_gate_passed is not None,
            passed=bool(v.gift_gate_passed),
            lcb=(v.gift_lcb if v.gift_lcb is not None else float("nan")),
            tolerance=params.gift_gate_tolerance,
            n_configs=0, king_agg=float("nan"), chal_agg=float("nan"),
            reason="recorded outcome (not recomputable at Tier 0)",
        )
        result = apply_gift_gate(result, gate, mode=params.gift_gate_mode)
        gate_note = "; gift-gate outcome taken as recorded"
    bench_mode = str(getattr(params, "bench_gate_mode", "off") or "off")
    if result.challenger_wins_round and (
        v.bench_gate_passed is not None or (bench_mode == "enforce" and v.inconclusive)
    ):
        # The three-suite bench gate (DEC-CA-0054) ran on the trainer-signed
        # rows (or was uncomputable under enforce); the audit does not replay
        # it — apply the recorded outcome, the same as the gift gate.
        from ..eval.bench_gate import uncomputable_bench_gate
        from ..eval.koth import apply_bench_gate

        bgate = uncomputable_bench_gate("recorded outcome (not replayed)")
        if v.bench_gate_passed is not None:
            bgate = _dc_replace(bgate, computed=True, passed=bool(v.bench_gate_passed))
        result = apply_bench_gate(result, bgate, mode=bench_mode)
        gate_note += "; bench-gate outcome taken as recorded"
    if result.challenger_wins_round != v.challenger_wins_round:
        problems.append(f"challenger_wins_round {v.challenger_wins_round} != recomputed "
                        f"{result.challenger_wins_round}")
    if result.inconclusive != v.inconclusive:
        problems.append(f"inconclusive {v.inconclusive} != recomputed {result.inconclusive}")
    if problems:
        return _fail(name, "; ".join(problems))
    lcb_txt = "n/a" if v.lcb is None else f"{v.lcb:.5f}"
    return _ok(name, f"lcb={lcb_txt} margin={v.margin:.5f} win={v.challenger_wins_round} "
                     f"reproduced over {len(paired)} size(s){gate_note}{scoring_note}")


def _bench_gate_pair(receipt: RoundReceipt, cfg: ChainConfig, manifest: TrainingManifest):
    """``(king, challenger)`` entries the bench gate judged: the decided
    challenger (recorded last) and the king at the settlement's primary size,
    else the first size both cover — the validator's ``gate_size`` rule."""
    from ..shared.era import settlement_contract_block

    t = cfg.training
    if int(getattr(t, "v2_from_block", 0) or 0):
        t = t.at_block(settlement_contract_block(cfg.round, int(receipt.epoch_start_block)))
    primary = t.arch_preset
    duelled = _duelled_hotkeys(receipt)
    if not duelled:
        return None, None
    kings = {(e.size or primary): e for e in manifest.entries_for_role("king")}
    chals = {(e.size or primary): e for e in manifest.entries_for_role("challenger")
             if e.miner_hotkey == duelled[-1]}
    common = [sz for sz in kings if sz in chals]
    if not common:
        return None, None
    size = primary if primary in common else common[0]
    return kings[size], chals[size]


def _same_detail(a, b) -> bool:
    """Recursive equality with float tolerance (``_close``)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_detail(a[k], b[k]) for k in a)
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _close(float(a), float(b))
    return a == b


def _detail_verdict(detail: dict) -> bool | None:
    """The DEC-CA-0061 outcome the recorded detail itself implies
    (:func:`cascade.eval.bench_gate.detail_verdict`: the three numbers
    re-derived from the recorded per-suite ratios, then the thresholds).
    ``None`` when the detail says uncomputable."""
    from ..eval.bench_gate import detail_verdict

    return detail_verdict(detail)


def _bench_thresholds(params: dict) -> dict:
    """DEC-CA-0061 rule v2's thresholds from a verdict's recorded params
    (dropped at their defaults, so the defaults refill them)."""
    from ..eval.bench_gate import DEFAULT_MIN_WIN_RATE, DEFAULT_TOLERANCE

    return {"tolerance": float(params.get("bench_gate_tolerance", DEFAULT_TOLERANCE)),
            "min_win_rate": float(params.get("bench_gate_min_win_rate", DEFAULT_MIN_WIN_RATE))}


def check_bench_gate(receipt: RoundReceipt, cfg: ChainConfig,
                     bench: dict | None = None) -> CheckResult:
    """Replay the lane-A bench gate (DEC-CA-0054 / DEC-CA-0061 rule v2
    "avg_noninf_winrate") from the trainer-signed bench report (or, on a
    DEC-CA-0063 settlement, the main validator's report the receipt binds).

    ``bench`` maps a trained pointer to its ``(item rows, suite-level scores)``
    as the signed report pins them (``(None, None)`` = absent); ``None`` =
    the report was not fetched. The recorded rule must be the one in force
    at the receipt's block (``bench_gate_rule_at``); the outcome — and, under
    DEC-CA-0061, the recorded detail (the three numbers, the thresholds and
    every suite's ratio / ratio LCB / win rate) — must match the replay.
    Without the report, a DEC-CA-0061 receipt is still checked for internal
    consistency (its recorded numbers must imply its recorded outcome)."""
    from ..eval.bench_gate import (
        RULE_V1,
        RULE_V2,
        checkpoint_fault_bench_gate,
        evaluate_bench_gate,
        uncomputable_bench_gate,
    )
    from ..shared.era import bench_gate_rule_at

    name = "bench-gate"
    v = receipt.verdict
    if v is None:
        return _skip(name, "no verdict on this receipt")
    params = dict(v.params)
    rule = str(params.get("bench_gate_rule", RULE_V1))
    want = bench_gate_rule_at(cfg, receipt.epoch_start_block)
    detail = v.bench_gate_detail
    if v.bench_gate_passed is None and detail is None:
        return _skip(name, "bench gate did not run (or uncomputable under DEC-CA-0054; "
                           "see verdict)")
    if rule != want:
        return _fail(name, f"recorded rule {rule} != {want} in force at block "
                           f"{receipt.epoch_start_block}")
    if rule == RULE_V2 and detail is None:
        return _fail(name, f"rule {RULE_V2} gate outcome recorded without its per-suite detail")
    if rule == RULE_V1 and detail is not None:
        return _fail(name, "per-suite detail recorded under the DEC-CA-0054 rule")
    if detail is not None:
        if detail.get("rule") != rule:
            return _fail(name, f"detail rule {detail.get('rule')!r} != params rule {rule}")
        implied = _detail_verdict(detail)
        if implied != v.bench_gate_passed or detail.get("passed") != v.bench_gate_passed:
            return _fail(name, f"recorded bench_gate_passed={v.bench_gate_passed} but the "
                               f"recorded comparisons imply {implied}")
    if bench is None:
        return _warn(name, f"rule {rule}: signed bench report not fetched; outcome "
                           f"{v.bench_gate_passed} not replayed"
                           + (" (detail internally consistent)" if detail is not None else ""))
    try:
        manifest = receipt.load_embedded_manifest()
    except (ValueError, KeyError) as e:
        return _fail(name, f"embedded manifest unparseable: {e}")
    king, chal = _bench_gate_pair(receipt, cfg, manifest)
    if king is None or chal is None:
        return _fail(name, "gate outcome recorded but no king/challenger pair to replay")
    king_rows, king_scores = bench.get(king.trained_pointer, (None, None))
    chal_rows, chal_scores = bench.get(chal.trained_pointer, (None, None))
    fault = (bench.get("_faults") or {}).get(chal.trained_pointer)
    thr = _bench_thresholds(params)
    if fault is not None:
        # DEC-CA-0063 item 10: the signed report's checkpoint fault fails it
        gate = checkpoint_fault_bench_gate(fault, rule=rule, **thr)
    elif king_rows is None or chal_rows is None:
        gate = uncomputable_bench_gate("signed bench rows missing", rule=rule, **thr)
    else:
        gate = evaluate_bench_gate(
            king_rows, chal_rows, king_scores, chal_scores,
            alpha=float(params.get("bootstrap_alpha", 0.05)),
            B=int(params.get("bootstrap_B", 1000)),
            seed=_bootstrap_seed(v.bootstrap_seed),
            min_items=int(params.get("bench_gate_min_items", 10)), rule=rule, **thr,
            # DEC-CA-0063 validator reports: the pinned data revisions
            king_revisions=(bench.get("_revisions") or {}).get(king.trained_pointer),
            chal_revisions=(bench.get("_revisions") or {}).get(chal.trained_pointer))
    replayed = gate.passed if gate.computed else None
    if replayed != v.bench_gate_passed:
        return _fail(name, f"rule {rule}: recorded bench_gate_passed={v.bench_gate_passed} "
                           f"!= replayed {replayed} ({gate.reason or 'pass'})")
    if rule == RULE_V2 and not _same_detail(gate.detail(), detail):
        return _fail(name, f"rule {RULE_V2}: recorded detail (avg ratio / worst suite LCB / "
                           "win rate / per-suite ratios) differs from the replay over the "
                           "signed rows")
    nums = (f" avg_ratio={gate.avg_ratio:.5f} worst_lcb={gate.worst_lcb:.5f} "
            f"win_rate={gate.win_rate:.4f}" if rule == RULE_V2 and gate.computed
            and gate.suites else "")
    return _ok(name, f"rule {rule}: replayed {replayed} from the signed rows{nums}"
                     + (f" ({gate.reason})" if gate.reason else ""))


def check_bench_source(receipt: RoundReceipt, cfg: ChainConfig,
                       vbench: dict | None = None) -> CheckResult:
    """DEC-CA-0063: who produced the lane-A bench rows this settlement read.

    On a settlement whose era takes its rows from the main validator
    (:func:`~cascade.shared.era.validator_bench_at`) a judged receipt must
    bind the main validator's report (``bench_report`` = key, sha256 of the
    published text, signer = ``[netconfig] owner_hotkey``) and a king check
    must say ``bench_source = "validator"``; elsewhere neither may appear.
    ``vbench`` = :func:`cascade.audit.main.fetch_validator_bench` (``None`` =
    not fetched ⇒ WARN): the published bytes must hash to the bound sha256."""
    from ..shared.bench_report import validator_bench_report_key
    from ..shared.era import validator_bench_at

    name = "bench-source"
    active = validator_bench_at(cfg, int(receipt.epoch_start_block))
    br = receipt.bench_report
    kc = receipt.king_check if isinstance(receipt.king_check, dict) else None
    kc_src = (kc or {}).get("bench_source")
    if not active:
        if br is not None:
            return _fail(name, "bench_report bound on a settlement whose lane-A bench rows "
                               "are the trainer's")
        if kc_src:
            return _fail(name, f"king check records bench_source={kc_src!r} on a "
                               "trainer-sourced settlement")
        return _skip(name, "trainer-sourced settlement")
    v = receipt.verdict
    # DEC-CA-0063: a validator needs the report only when a clearer reached
    # the bench gate or the king check read it — otherwise it judges without
    gate_ran = v is not None and (v.bench_gate_passed is not None
                                  or v.bench_gate_detail is not None
                                  or "bench_infra " in str(v.note or "")
                                  or "bench_deferred " in str(v.note or ""))
    need = gate_ran or (kc is not None and kc_src == "validator"
                        and isinstance(kc.get("bench"), dict))
    if br is None:
        if need:
            return _fail(name, "validator-sourced settlement read the bench gate / king check "
                               "without binding the main validator's bench report")
        return _ok(name, "validator-sourced settlement judged without needing the report "
                         "(no clearer reached the bench gate)")
    problems = []
    owner = str(cfg.netconfig.owner_hotkey or "")
    if owner and str(br.get("signer", "")) != owner:
        problems.append(f"signer {br.get('signer')!r} != [netconfig] owner_hotkey {owner!r}")
    if str(br.get("key", "")) != validator_bench_report_key(str(receipt.round_id)):
        problems.append(f"key {br.get('key')!r} is not this settlement's report key")
    if kc is not None and kc_src != "validator":
        problems.append(f"king check bench_source {kc_src!r} != 'validator'")
    if problems:
        return _fail(name, "; ".join(problems))
    if br.get("record") == "absent":
        return _ok(name, "the trainer recorded the main validator's report ABSENT (timeout): "
                         "the settlement's clearers were infra (skipped)")
    if vbench is None:
        return _warn(name, "main validator's bench report not fetched; binding not verified")
    if vbench.get("text_sha") is None:
        return _fail(name, f"the bound report {br.get('key')!r} is not published")
    if vbench["text_sha"] != str(br.get("sha256", "")):
        return _fail(name, "the published report's sha256 differs from the bytes the "
                           "receipt binds")
    if not vbench.get("verified"):
        return _ok(name, "bound report verified by digest; it fails the main validator's "
                         "signature, so every entry replays uncomputable")
    return _ok(name, f"bound to the main validator's signed report "
                     f"({len(vbench['report'].entries)} entr"
                     f"{'y' if len(vbench['report'].entries) == 1 else 'ies'})")


def _bench_infra_hotkeys(note: str, segment: str = "bench_infra") -> list[str]:
    """The hotkeys a verdict note's ``bench_infra`` (or ``bench_deferred`` /
    ``bench_fault``) segment names (DEC-CA-0063: ``"; bench_infra <hk>:
    <why>; <hk>: <why>"``)."""
    tag = f"; {segment} "
    i = str(note or "").find(tag)
    if i < 0:
        return []
    out = []
    for part in note[i + len(tag):].split("; "):
        hk, sep, _ = part.partition(": ")
        if not sep or " " in hk:
            break
        out.append(hk)
    return out


def check_bench_coverage(receipt: RoundReceipt, cfg: ChainConfig,
                         vbench: dict | None = None) -> CheckResult:
    """DEC-CA-0063 (report-only): the main validator's report must cover what
    the private-pool verdict sent to the bench gate. The decided challenger
    of a verdict that cleared the pool under an enforcing / shadow gate
    reached the gate, so its pointer and the king's must each be in the
    report — as an entry or as ``infra_failed``; an OMITTED one is a
    violation (the main validator skipped a clearer it reached). Every
    hotkey the verdict note records as ``bench_infra`` must be infra-failed
    in the report (or the trainer recorded the whole report absent); every
    ``bench_deferred`` one budget-deferred, every ``bench_fault`` one a
    challenger ``checkpoint_fault`` (items 9-10) — and a report that lists a
    clearer deferred or faulted must be matched by the note (a validator
    that judged it otherwise did not apply the signed outcome)."""
    from ..shared.era import bench_gate_mode_at, validator_bench_at

    name = "bench-coverage"
    if not validator_bench_at(cfg, int(receipt.epoch_start_block)):
        return _skip(name, "trainer-sourced settlement")
    v = receipt.verdict
    if v is None:
        return _skip(name, "no verdict on this receipt")
    br = receipt.bench_report or {}
    if br.get("record") == "absent":
        return _ok(name, "report recorded absent by the trainer: every clearer infra")
    infra_hks = _bench_infra_hotkeys(v.note)
    deferred_hks = _bench_infra_hotkeys(v.note, "bench_deferred")
    fault_hks = _bench_infra_hotkeys(v.note, "bench_fault")
    cleared = (v.lcb is not None and v.lcb >= v.margin and v.gift_gate_passed is not False
               and bench_gate_mode_at(cfg, int(receipt.epoch_start_block)) != "off")
    if not cleared and not (infra_hks or deferred_hks or fault_hks):
        return _ok(name, "no clearer reached the bench gate")
    if vbench is None:
        return _warn(name, "a clearer reached the bench gate but the main validator's report "
                           "was not fetched; coverage not checked")
    if vbench.get("report") is None:
        return _ok(name, "the bound report is unsigned / unparseable: every entry replays "
                         "uncomputable, coverage moot")
    report = vbench["report"]
    try:
        manifest = receipt.load_embedded_manifest()
    except (ValueError, KeyError) as e:
        return _fail(name, f"embedded manifest unparseable: {e}")
    king, chal = _bench_gate_pair(receipt, cfg, manifest)
    problems = []
    by_hk = {e.miner_hotkey: e for e in manifest.entries_for_role("challenger")}
    for hk in infra_hks:
        e = by_hk.get(hk)
        if e is None:
            problems.append(f"bench_infra names {hk[:12]}… which is not a challenger")
        elif report.infra_reason(e.trained_pointer) is None and (
                king is None or report.infra_reason(king.trained_pointer) is None):
            problems.append(f"bench_infra {hk[:12]}… but the report records no infra-failed "
                            "sweep for it or the king")
    for seg, hks, fn in (("bench_deferred", deferred_hks, report.deferred_reason),
                         ("bench_fault", fault_hks, report.fault_reason)):
        for hk in hks:
            e = by_hk.get(hk)
            if e is None:
                problems.append(f"{seg} names {hk[:12]}… which is not a challenger")
            elif fn(e.trained_pointer) is None:
                problems.append(f"{seg} {hk[:12]}… but the report does not list it so")
    for seg, listed, hks in (("budget_deferred", report.budget_deferred, deferred_hks),
                             ("checkpoint_fault", report.checkpoint_fault, fault_hks)):
        if seg == "budget_deferred" and bench_gate_mode_at(
                cfg, int(receipt.epoch_start_block)) != "enforce":
            continue                   # a shadow gate never skips a clearer
        for f in listed:
            if str(f.get("role", "")) != "challenger":
                continue
            hk = next((h for h, e in by_hk.items()
                       if e.trained_pointer == str(f.get("trained_pointer", ""))), None)
            if hk is not None and hk not in hks:
                problems.append(f"the report lists {hk[:12]}… {seg} but the verdict note "
                                "does not apply it")
    if (cleared and chal is not None and chal.miner_hotkey not in infra_hks
            and chal.miner_hotkey not in deferred_hks):
        for role, ent in (("challenger", chal), ("king", king)):
            if ent is None:
                continue
            p = ent.trained_pointer
            if (report.entry_for_pointer(p) is None and report.infra_reason(p) is None
                    and report.fault_reason(p) is None):
                if role == "king" and (report.infra_reason(chal.trained_pointer) is not None
                                       or report.fault_reason(chal.trained_pointer)
                                       is not None):
                    continue           # the challenger was infra / faulted: no king needed
                problems.append(f"OMITTED: the {role} {p[-24:]} reached the bench gate but the "
                                "report has neither an entry nor an infra-failed record for it")
    if problems:
        return _fail(name, "; ".join(problems))
    return _ok(name, "every clearer that reached the bench gate is covered by the report")


def check_transition(receipt: RoundReceipt) -> CheckResult:
    """Internal consistency of the recorded state transition.

    Fully checkable for ``dethrone_cp = 1``; a multi-round streak depends on
    prior-round validator state, which needs the receipt chain (WARN).
    """
    name = "transition"
    v = receipt.verdict
    if v is None:
        return _skip(name, "no verdict on this receipt")
    try:
        manifest = receipt.load_embedded_manifest()
    except (ValueError, KeyError) as e:
        return _fail(name, f"embedded manifest unparseable: {e}")
    # The challenger the verdict belongs to — the last one scored, which under the
    # cohort rule is the crowned clearer (DEC-CA-0012). Taking the manifest's
    # FIRST challenger would misattribute the throne on any multi-finalist round.
    duelled = _duelled_hotkeys(receipt)
    chal = next(
        (e for e in manifest.entries_for_role("challenger")
         if duelled and e.miner_hotkey == duelled[-1]),
        manifest.entry_for_role("challenger"),
    )
    king = manifest.entry_for_role("king")
    if not v.challenger_wins_round and v.dethroned:
        return _fail(name, "dethroned recorded without a round win")
    cp = int(v.params.get("dethrone_cp", 1))
    if cp == 1 and v.challenger_wins_round:
        if not v.dethroned:
            return _fail(name, "dethrone_cp=1 and a round win, but dethroned=False")
        if chal is not None and v.king_hotkey != chal.miner_hotkey:
            return _fail(name, f"dethrone recorded but resulting king {v.king_hotkey!r} "
                               f"is not the challenger {chal.miner_hotkey!r}")
    if not v.dethroned and king is not None and v.king_hotkey not in (
        king.miner_hotkey, None
    ):
        # Legitimate steady state, not a contradiction: the trainer reads its
        # king from on-chain incentive while the validator's state tracks its
        # own throne — they can diverge (e.g. after a
        # service outage or an interim-king promotion). The vote still goes to
        # the manifest king, which check_weights verifies. Surface it, don't
        # fail it — a receipt-chain audit is what would confirm the lineage.
        return _warn(name, f"validator-state king {v.king_hotkey!r} differs from the "
                           f"manifest king {king.miner_hotkey!r} (trainer reads incentive; "
                           "state tracks its own throne)")
    if cp > 1:
        return _warn(name, f"dethrone_cp={cp}: streak state spans rounds; verify the "
                           "receipt chain for full confirmation")
    return _ok(name, f"transition '{v.note}' consistent with the verdict")


# ── the DEC-CA-0055 king check (re-derived, DEC-CA-0058) ─────────────────────

# The bench half could not be verified against the trainer-signed report: a
# satellite treats this row as "could not verify" (retry), never as a pass.
KING_CHECK_BENCH = "king-check-bench"

# Deferral reasons that name a transient (a store read, an unscorable eval):
# not derivable from published data — WARN, never FAIL.
_KC_TRANSIENT = ("bench report unreadable", "pool check unscorable",
                 "validator bench unreadable", "validator bench scores unusable")


def _king_check_live(cfg: ChainConfig, boundary: int) -> bool:
    """Does the DEC-CA-0055 king check run at the settlement at ``boundary``
    (a v2 settlement under ``[v2] king_ladder`` — the validator's
    ``_v2_ladder_active``)?"""
    from ..shared.emission import ladder_on
    from ..shared.era import v2_settlement_active

    return bool(v2_settlement_active(cfg, int(boundary)) and ladder_on(cfg))


def _king_check_pool_rows(receipt: RoundReceipt, kc_role: str, es_role: str):
    """The rows the king check fed ``evaluate_round`` for one side, and where
    they came from: its own ``king_check_scores`` rows (``kc_role``) when it
    scored them itself, else the verdict's ``entry_scores`` rows (``es_role``,
    pooled in record order — the duel's paired-size order) it reused. ``(None,
    "")`` when neither is recorded."""
    rows = [r for r in receipt.king_check_scores if r.role == kc_role]
    src = "king_check_scores"
    if not rows:
        rows = [r for r in receipt.entry_scores if r.role == es_role]
        src = "entry_scores"
    if not rows:
        return None, ""
    return [w.to_score() for r in rows for w in r.scores], src


def _king_check_tenure(receipt: RoundReceipt, cfg: ChainConfig, kc: dict):
    """``(tenure, problem)``: the tenure the check judged at — the verdict's
    (the check reuses the duel's ``tenure_at_decision``) on a scored receipt,
    else :func:`tenure_rounds_at` from the emission record's crowning block
    under block tenure, else the recorded value."""
    from ..shared.era import tenure_blocks_active, tenure_rounds_at

    got = kc.get("tenure_rounds")
    boundary = int(receipt.epoch_start_block)
    want = None
    if receipt.verdict is not None:
        want = int(receipt.verdict.king_tenure_rounds)
    else:
        ksb = (receipt.emission or {}).get("king_since_block")
        if ksb is not None and tenure_blocks_active(cfg.scoring, boundary):
            want = tenure_rounds_at(cfg.round, cfg.scoring, block=boundary,
                                    tenure_rounds=0, king_since_block=int(ksb))
    if want is not None:
        if got is not None and int(got) != want:
            return None, f"recorded tenure {got} != {want} the settlement derives"
        return want, None
    if got is None:
        return None, "no tenure recorded and none derivable"
    return int(got), None


def _king_check_replay(receipt: RoundReceipt, cfg: ChainConfig, kc: dict):
    """Re-run the check's pool half on the recorded rows:
    ``evaluate_round(init, king, cfg.king_check_params(boundary), seed=base_seed,
    king_tenure_rounds=tenure, baseline_scores=init)`` — the validator's call.
    Returns ``(result, init_source, problem)``."""
    from ..eval.koth import evaluate_round

    king, _ = _king_check_pool_rows(receipt, "king", "king")
    init, init_src = _king_check_pool_rows(receipt, "init", "baseline")
    if king is None or init is None:
        missing = [s for s, r in (("king", king), ("init", init)) if r is None]
        return None, init_src, f"no recorded {'/'.join(missing)} rows to replay the pool half"
    tenure, why = _king_check_tenure(receipt, cfg, kc)
    if why is not None:
        return None, init_src, why
    params = cfg.king_check_params(int(receipt.epoch_start_block))
    try:
        res = evaluate_round(init, king, params, seed=int(receipt.base_seed),
                             king_tenure_rounds=tenure, baseline_scores=init)
    except ValueError as e:
        return None, init_src, f"pool rows do not replay: {e}"
    return res, init_src, None


def _kc_bench_scores(rec: dict | None):
    from ..shared.manifest import bench_scores_from_json

    try:
        return bench_scores_from_json(dict(rec or {}))
    except (KeyError, TypeError, ValueError):
        return None


def _kc_deferred(receipt: RoundReceipt, cfg: ChainConfig, kc: dict, manifest,
                 bench: dict | None) -> CheckResult:
    """A recorded deferral: its stated reason against what is derivable."""
    from ..validator.king_check import KC_GATE_UNCOMPUTABLE, POOL_INCONCLUSIVE

    name = "king-check"
    reasons = [str(r) for r in (kc.get("reasons") or ())]
    if not reasons:
        return _fail(name, "deferred with no reason")
    why = reasons[0]
    report = None if bench is None else bench.get("report")
    king = manifest.entry_for_role("king")
    ptr = str(kc.get("king_pointer") or "")
    if why == "no king entry on the settlement":
        if king is not None and ptr:
            return _fail(name, "deferred for 'no king entry' but the manifest has a king "
                               f"entry and the record names pointer {ptr[:40]!r}")
        return _ok(name, "deferred: no king entry (manifest agrees)")
    if why.startswith(_KC_TRANSIENT):
        return _warn(name, f"deferred for a transient ({why[:120]}); not derivable")
    if why.startswith("validator bench infra: "):
        # DEC-CA-0063: the bound report (or the trainer's "absent" record)
        # must say so — an infra deferral is derivable from published bytes
        br = receipt.bench_report or {}
        if br.get("record") == "absent":
            return _ok(name, "deferred: the trainer recorded the validator bench report absent")
        if report is None:
            return _warn(name, f"deferred for {why[:80]!r}; validator report not re-read")
        if not report.infra_failed:
            return _fail(name, f"deferred for {why[:80]!r} but the bound validator report "
                               "records no infra-failed sweep")
        return _ok(name, f"deferred: {why[:80]} (bound validator report agrees)")
    if why.startswith(KC_GATE_UNCOMPUTABLE):
        # DEC-CA-0061: the rule-v2 bench half (king vs init on task rows)
        # was uncomputable — missing rows (FEV included), too few tasks
        from ..shared.era import BENCH_RULE_V2, bench_gate_rule_at

        if bench_gate_rule_at(cfg, int(receipt.epoch_start_block)) != BENCH_RULE_V2:
            return _fail(name, f"deferred for an uncomputable {BENCH_RULE_V2} bench half "
                               "outside a settlement under that rule")
        gd = (kc.get("bench") or {}).get("gate")
        if gd is not None and (gd.get("computed") or gd.get("rule") != BENCH_RULE_V2):
            return _fail(name, "deferred as uncomputable but the recorded gate detail says "
                               "computed (or names another rule)")
        return _ok(name, f"deferred: {why[:120]}")
    if why == "no signed bench report for the previous settlement":
        if not manifest.prev_round_id:
            return _ok(name, "deferred: the settlement has no previous round to read")
        if report is not None:
            return _warn(name, f"deferred for no bench report, but a signed report for "
                               f"round {kc.get('bench_round')} is published now")
        return _ok(name, "deferred: no signed bench report"
                         + ("" if bench is not None else " (report not re-read)"))
    if why.startswith("no init pointer"):
        # DEC-CA-0063: a random-init epoch whose init checkpoint pointer the
        # trainer's report did not carry
        if manifest.warm_start_ckpt:
            return _fail(name, f"deferred for {why[:60]!r} but the settlement warm-starts "
                               f"from {manifest.warm_start_ckpt[:40]!r}")
        return _ok(name, "deferred: random-init epoch with no published init pointer")
    if why in ("validator bench report has no entry for the king pointer",
               "validator bench report has no init entry"):
        # DEC-CA-0063: the receipt binds the exact report bytes, so an entry
        # present in it contradicts the deferral (no re-publish excuse)
        if report is None:
            return _ok(name, f"deferred: {why[:80]} (validator report uncomputable or not "
                             "re-read)")
        present = (report.entry_for_pointer(ptr) is not None if why.endswith("king pointer")
                   else report.init_entry() is not None)
        if present:
            return _fail(name, f"deferred for {why[:80]!r} but the bound validator report "
                               "has it")
        return _ok(name, f"deferred: {why[:80]} (bound validator report agrees)")
    if why in ("bench report has no entry for the king pointer",
               "bench report has no init entry") or why.startswith("bench init entry "):
        if report is None:
            return _ok(name, f"deferred: {why[:80]} (report not available to re-read)")
        kb, ib = report.entry_for_pointer(ptr), report.init_entry()
        stale = ((why.endswith("king pointer") and kb is not None)
                 or (why.endswith("init entry") and ib is not None)
                 or (why.startswith("bench init entry ") and ib is not None
                     and ib.trained_pointer == manifest.warm_start_ckpt))
        if stale:
            return _warn(name, f"deferred for {why[:80]!r}, but the signed report now has "
                               "it (re-published after the settlement?)")
        return _ok(name, f"deferred: {why[:80]} (signed report agrees)")
    if why == POOL_INCONCLUSIVE:
        res, _, problem = _king_check_replay(receipt, cfg, kc)
        if problem is not None:
            return _fail(name, f"deferred as inconclusive but {problem}")
        if not res.inconclusive:
            return _fail(name, f"deferred as inconclusive but the recorded rows replay "
                               f"conclusive (lcb {res.lcb:.5f}, {res.n_windows} windows)")
        if kc.get("lcb") is not None or not _close(float(kc.get("margin", float("nan"))),
                                                   float(res.margin)):
            return _fail(name, f"inconclusive record lcb={kc.get('lcb')} margin="
                               f"{kc.get('margin')} != replay (None, {res.margin})")
        return _ok(name, f"deferred: pool half replays inconclusive ({res.n_windows} windows)")
    return _warn(name, f"deferred for an unrecognised reason {why[:120]!r}")


def _kc_gate_replay(receipt: RoundReceipt, cfg: ChainConfig, kc: dict, manifest,
                    bench: dict | None, res, problems: list) -> CheckResult:
    """A pass / fail king check judged under DEC-CA-0061 rule v2: the bench
    half is the gate's rule, king as challenger, init as reference
    (:func:`cascade.validator.king_check.king_check_gate`). The recorded
    gate detail must imply its outcome; with the signed rows of both
    checkpoints (``bench["rows"]``) the gate is replayed from them under
    ``cfg.koth_params(block)`` and seed ``{base_seed}|king-check``."""
    from ..shared.era import bench_gate_rule_at
    from ..validator.king_check import RULE_V2 as KC_RULE_V2
    from ..validator.king_check import king_check_gate, king_check_gate_reason, pool_reason

    name = "king-check"
    result = str(kc.get("result"))
    boundary = int(receipt.epoch_start_block)
    rb = kc.get("bench") or {}
    want_rule = bench_gate_rule_at(cfg, boundary)
    if want_rule != KC_RULE_V2:
        problems.append(f"bench half rule {KC_RULE_V2} != {want_rule} in force at {boundary}")
    gd = rb.get("gate")
    if not isinstance(gd, dict) or gd.get("rule") != KC_RULE_V2:
        return _fail(name, f"{result} recorded under {KC_RULE_V2} without its gate detail")
    implied = _detail_verdict(gd)
    if implied is None:
        return _fail(name, f"{result} recorded but the bench gate detail says uncomputable")
    if gd.get("passed") is not implied:
        problems.append(f"gate detail passed={gd.get('passed')} but its numbers imply {implied}")
    init_ptr = str(rb.get("init_pointer", ""))
    if manifest.warm_start_ckpt and init_ptr != manifest.warm_start_ckpt:
        problems.append(f"bench init {init_ptr!r} is not the era init "
                        f"{manifest.warm_start_ckpt!r}")
    want_reasons = [] if implied else [king_check_gate_reason(str(gd.get("reason", "")))]
    if not res.challenger_wins_round:
        want_reasons.append(pool_reason(res.lcb, res.margin))
    want = "fail" if want_reasons else "pass"
    if result != want:
        problems.append(f"recorded {result} but the recorded gate detail and pool rows give "
                        f"{want}" + (f" ({'; '.join(want_reasons)})" if want_reasons else ""))
    elif [str(r) for r in (kc.get("reasons") or ())] != want_reasons:
        problems.append(f"reasons {kc.get('reasons')} != re-derived {want_reasons}")
    if problems:
        return _fail(name, "; ".join(problems))
    summary = (f"{result} reproduced: pool lcb {res.lcb:.5f} vs margin {res.margin:.5f}; "
               f"bench {KC_RULE_V2} avg_ratio {gd.get('avg_ratio')} worst_lcb "
               f"{gd.get('worst_lcb')} win_rate {gd.get('win_rate')}")
    rows = (bench or {}).get("rows") or {}
    ptr = str(kc.get("king_pointer") or "")
    king_side, init_side = rows.get(ptr, (None, None)), rows.get(init_ptr, (None, None))
    if king_side[0] is None or init_side[0] is None:
        return _warn(KING_CHECK_BENCH, f"{summary}; the signed task rows of "
                                       f"{'king' if king_side[0] is None else 'init'} were "
                                       "not fetched — the bench half is internally "
                                       "consistent but NOT replayed")
    gate = king_check_gate(king_side[0], init_side[0], king_side[1], init_side[1],
                           cfg.koth_params(block=boundary), seed=int(receipt.base_seed),
                           king_revisions=(rows.get("_revisions") or {}).get(ptr),
                           init_revisions=(rows.get("_revisions") or {}).get(init_ptr))
    if not _same_detail(gate.detail(), gd):
        return _fail(name, f"{result} recorded but the gate replayed over the signed rows "
                           f"differs from the recorded detail ({gate.reason or 'pass'})")
    return _ok(name, f"{summary}; bench half replayed from the signed task rows")


def check_king_check(receipt: RoundReceipt, cfg: ChainConfig,
                     bench: dict | None = None) -> CheckResult:
    """Re-derive the DEC-CA-0055 v2 king check from published data instead of
    trusting the recorded pass/fail (it sets the lane-A ladder share and is
    the only evidence a k = 1 promotion is accepted on).

    * Due-ness: from the ladder record's ``reign_start_block`` (the clock
      :func:`v2_check_due` read, stamped before a check restarts it). A check
      recorded where it is not due FAILS; a missing check where it is due
      FAILS; a missing check on a receipt with no recorded clock WARNS (the
      validator legitimately skips while its cascade view has not adopted the
      champion — not derivable from the receipt).
    * Pool half: ``evaluate_round`` replayed on the recorded rows (the king's
      ``entry_scores`` / the init's ``baseline`` rows, or the check's own
      ``king_check_scores``) under ``cfg.king_check_params(boundary)``; the
      result, lcb and margin must match.
    * Bench half: :func:`bench_failures` on the recorded suite means, and
      those means against the PREVIOUS settlement's trainer-signed report
      (``bench`` = ``{"report": BenchReport | None}``, None = not fetched —
      then the row is :data:`KING_CHECK_BENCH` WARN after the rest verifies).
    * Deferrals: the stated reason against what is derivable; transient
      reasons WARN."""
    from ..shared.emission import CHECK_RESULTS, crowned_in_v2
    from ..shared.era import min_effective_era, v2_check_due
    from ..validator.king_check import RULE_V2 as KC_RULE_V2
    from ..validator.king_check import bench_failures, bench_fields, pool_reason

    name = "king-check"
    boundary = int(receipt.epoch_start_block)
    kc = receipt.king_check
    live = _king_check_live(cfg, boundary)
    emission = receipt.emission or {}
    lad = emission.get("ladder")
    reign = None if lad is None else lad.get("reign_start_block")
    if kc is None:
        if receipt.king_check_scores:
            return _fail(name, "king_check_scores recorded without a king check")
        if not live:
            return _skip(name, "not a v2 settlement under [v2] king_ladder")
        if lad is None:
            return _skip(name, "no ladder step at this settlement (not judged)")
        if lad.get("event") == "crowned":
            return _ok(name, "dethroning settlement: no king check (the clock restarts)")
        if reign is None:
            return _warn(name, "no king check and no reign clock recorded (the validator's "
                               "cascade view had not adopted the king, so the check was not "
                               "considered); due-ness not derivable")
        if v2_check_due(cfg, boundary=boundary, reign_start_block=int(reign)):
            return _fail(name, f"the king check was due at {boundary} (reign from {reign}) "
                               "but none is recorded")
        return _ok(name, f"not due at {boundary} (reign from {reign})")

    if not live:
        return _fail(name, "king check recorded outside a v2 settlement under [v2] king_ladder")
    result = str(kc.get("result"))
    if result not in CHECK_RESULTS:
        return _fail(name, f"result {result!r} is not one of {CHECK_RESULTS}")
    try:
        manifest = receipt.load_embedded_manifest()
    except (ValueError, KeyError) as e:
        return _fail(name, f"embedded manifest unparseable: {e}")
    problems = []
    if str(kc.get("round_id")) != str(receipt.round_id):
        problems.append(f"round_id {kc.get('round_id')} != the receipt's {receipt.round_id}")
    if int(kc.get("block", -1)) != boundary:
        problems.append(f"block {kc.get('block')} != the settlement boundary {boundary}")
    if kc.get("bench_source") == "validator":
        # DEC-CA-0063: the main validator swept king + init at THIS settlement
        if str(kc.get("bench_round", "")) != str(receipt.round_id):
            problems.append(f"bench_round {kc.get('bench_round')!r} != this settlement "
                            f"{receipt.round_id!r} (validator bench source)")
    elif str(kc.get("bench_round", "")) != str(manifest.prev_round_id or ""):
        problems.append(f"bench_round {kc.get('bench_round')!r} != the manifest's previous "
                        f"round {manifest.prev_round_id!r}")
    want_era = int(min_effective_era(cfg.round, boundary))
    if int(kc.get("warm_start_era", -1)) != want_era:
        problems.append(f"warm_start_era {kc.get('warm_start_era')} != {want_era}")
    if (receipt.verdict is not None and receipt.verdict.dethroned) or (
            lad is not None and lad.get("event") == "crowned"):
        problems.append("a king check recorded at a dethroning settlement")
    # due-ness, from the clock the validator read
    if reign is None:
        problems.append("no reign clock recorded on the ladder (due-ness not derivable)")
    else:
        if int(reign) > boundary:
            problems.append(f"reign start {reign} is after the settlement {boundary}")
        ksb = emission.get("king_since_block")
        if crowned_in_v2(cfg, ksb) and int(reign) < int(ksb):
            problems.append(f"reign start {reign} precedes the crowning at {ksb}")
        if not v2_check_due(cfg, boundary=boundary, reign_start_block=int(reign)):
            problems.append(f"recorded at {boundary} but not due (reign from {reign}, "
                            f"[v2] reign_warm_start_blocks {cfg.v2.reign_warm_start_blocks})")
    if problems:
        return _fail(name, "; ".join(problems))
    if result == "deferred":
        return _kc_deferred(receipt, cfg, kc, manifest, bench)

    # pass / fail: replay both halves
    res, init_src, problem = _king_check_replay(receipt, cfg, kc)
    if problem is not None:
        return _fail(name, f"{result} recorded but {problem}")
    if res.inconclusive:
        return _fail(name, f"{result} recorded but the pool half replays inconclusive")
    if not _close(kc.get("lcb"), float(res.lcb)):
        problems.append(f"lcb {kc.get('lcb')} != replayed {res.lcb}")
    if not _close(kc.get("margin"), float(res.margin)):
        problems.append(f"margin {kc.get('margin')} != replayed {res.margin} "
                        "(cfg.king_check_params at this block)")
    rb = kc.get("bench")
    if rb is not None and rb.get("rule") == KC_RULE_V2:
        return _kc_gate_replay(receipt, cfg, kc, manifest, bench, res, problems)
    kb_rec = _kc_bench_scores(None if rb is None else rb.get("king"))
    ib_rec = _kc_bench_scores(None if rb is None else rb.get("init"))
    if kb_rec is None or ib_rec is None:
        return _fail(name, f"{result} recorded without the bench suite means it compared")
    init_ptr = str(rb.get("init_pointer", ""))
    if manifest.warm_start_ckpt and init_ptr != manifest.warm_start_ckpt:
        problems.append(f"bench init {init_ptr!r} is not the era init "
                        f"{manifest.warm_start_ckpt!r}")
    if init_src == "entry_scores" and init_ptr != manifest.warm_start_ckpt:
        problems.append("the init rows reused are the verdict baseline (the era init), not "
                        f"the bench init {init_ptr!r}")
    # DEC-CA-0061 in force ⇒ the bench half used the gate's four suites
    from ..shared.era import bench_gate_rule_at

    kc_rule = str(rb.get("rule", "dec-ca-0054"))
    want_rule = bench_gate_rule_at(cfg, boundary)
    if kc_rule != want_rule:
        problems.append(f"bench half rule {kc_rule} != {want_rule} in force at {boundary}")
    want_reasons = bench_failures(kb_rec, ib_rec, kc_rule)
    if not res.challenger_wins_round:
        want_reasons.append(pool_reason(res.lcb, res.margin))
    want = "fail" if want_reasons else "pass"
    if result != want:
        problems.append(f"recorded {result} but the recorded rows and suite means give {want}"
                        + (f" ({'; '.join(want_reasons)})" if want_reasons else ""))
    elif [str(r) for r in (kc.get("reasons") or ())] != want_reasons:
        problems.append(f"reasons {kc.get('reasons')} != re-derived {want_reasons}")
    if problems:
        return _fail(name, "; ".join(problems))
    summary = (f"{result} reproduced: pool lcb {res.lcb:.5f} vs margin {res.margin:.5f} over "
               f"{res.n_windows} windows; bench "
               f"{len(bench_fields(kc_rule)) - len(bench_failures(kb_rec, ib_rec, kc_rule))}"
               f"/{len(bench_fields(kc_rule))}")
    src = ("main validator's" if kc.get("bench_source") == "validator"
           else "trainer-signed")
    if bench is None or bench.get("report") is None:
        why = "not fetched" if bench is None else "absent or unsigned"
        return _warn(KING_CHECK_BENCH, f"{summary}; the {src} bench report for round "
                                       f"{kc.get('bench_round')} is {why} — the bench half is "
                                       "internally consistent but NOT verified")
    report = bench["report"]
    ptr = str(kc.get("king_pointer") or "")
    kb, ib = report.entry_for_pointer(ptr), report.init_entry()
    if kb is None or ib is None:
        return _fail(name, f"{result} recorded but the signed report for round "
                           f"{kc.get('bench_round')} has no "
                           f"{'king entry' if kb is None else 'init entry'}")
    if ib.trained_pointer != init_ptr:
        problems.append(f"recorded bench init {init_ptr!r} != the signed report's "
                        f"{ib.trained_pointer!r}")
    for side, got, signed in (("king", kb_rec, kb.scores), ("init", ib_rec, ib.scores)):
        bad = [f for f in bench_fields(kc_rule)
               if getattr(got, f, None) is None or getattr(signed, f, None) is None
               or not _close(float(getattr(got, f)), float(getattr(signed, f)))]
        if bad:
            problems.append(f"recorded {side} suite means differ from the signed report "
                            f"on {bad}")
    if problems:
        return _fail(name, "; ".join(problems))
    return _ok(name, f"{summary}; bench half replayed from the {src} report")


def _paid_full_pay_why(w: dict, tiers, *, at_win: bool, ttl: int,
                       state_entry: list | None) -> str | None:
    """Why a payout record's win carries a forged / inconsistent gain-tiered
    full-pay period (owner 2026-10-20), else None. Every paid win carries
    ``gain_lcb`` / ``tier`` / ``full_eras`` / ``below_floor``; ``full_eras``
    is an integer in ``[0, win_ttl_eras]``; a win made AT this settlement
    (``at_win``) is re-derived from its recorded gain LCB under the tiers in
    force here (:func:`cascade.eval.expert_lane.full_pay_mismatch`) — an
    older win's tier is fixed at its own settlement and is bound across
    receipts by the settlement state instead: it must equal the receipt's
    ``state_post`` slot (``state_entry``, when the receipt carries one)."""
    import math

    from ..eval.expert_lane import full_pay_mismatch
    from ..shared.settlement_state import slot_full_pay

    if not all(k in w for k in ("gain_lcb", "tier", "full_eras")):
        return "paid without a gain-tiered full-pay record (gain_lcb / tier / full_eras)"
    n, lcb = w.get("full_eras"), w.get("gain_lcb")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0 or (ttl and n > ttl):
        return f"full_eras {n!r} is not an integer in [0, win_ttl_eras={ttl}]"
    if lcb is None or not math.isfinite(float(lcb)):
        return f"gain_lcb {lcb!r} is not finite"
    if at_win:
        why = full_pay_mismatch(w, tiers)
        if why:
            return f"full-pay tier forged: {why}"
    elif bool(w.get("below_floor")) != (float(lcb) < float(w.get("tier") or 0.0)):
        return (f"below_floor {w.get('below_floor')!r} contradicts gain_lcb {lcb} vs its "
                f"tier {w.get('tier')}")
    st = slot_full_pay(state_entry)
    if state_entry is not None and st != {k: w.get(k) for k in ("gain_lcb", "tier",
                                                                 "full_eras")}:
        return (f"full-pay record {[w.get(k) for k in ('gain_lcb', 'tier', 'full_eras')]} "
                f"!= the settlement state's {st}")
    return None


def _lane_b_paid_wins(receipt: RoundReceipt, cfg: ChainConfig,
                      n_uids: int) -> tuple[list, str | None]:
    """The receipt's ``lane_b_pay`` wins re-derived from ``[lane_b]``:
    ``(wins, None)``, or ``([], why)`` on the first inconsistency (decay clock,
    era length, unknown/duplicate domain, out-of-range uid, age, expiry,
    multiplier)."""
    import math

    from ..eval.expert_lane import LaneBWin, reward_multiplier
    from ..shared.era import era_length_blocks
    from ..shared.lane_b import lane_b_slots, win_expired

    lb = cfg.lane_b
    # DEC-CA-0059: one slot per lane-B target (domains + horizon targets)
    slots = lane_b_slots(lb)
    pay = receipt.lane_b_pay or {}
    post = getattr(receipt, "state_post", None)
    post = post if isinstance(post, dict) and "lane_b" in post else None
    block = int(pay.get("block", -1))
    if block != int(receipt.epoch_start_block):
        return [], (f"lane-B decay clock at block {block}, but the settlement "
                    f"is {receipt.epoch_start_block}")
    eb = int(era_length_blocks(cfg.round, block))
    if int(pay.get("era_blocks", -1)) != eb:
        return [], f"lane-B era length {pay.get('era_blocks')} != {eb} in force"
    if "share" in pay or cfg.lane_c.active(block):
        # DEC-CA-0056: once lane C is live the record carries lane B's
        # (reduced) pool and slot count
        from ..shared.lane_c import lane_c_shares

        want_share = lane_c_shares(lb, cfg.lane_c, block)[0]
        if not math.isclose(float(pay.get("share", -1.0)), want_share, rel_tol=0,
                            abs_tol=1e-12) or int(pay.get("slots", -1)) != len(slots):
            return [], (f"lane-B pay record share/slots {pay.get('share')}/{pay.get('slots')} "
                        f"!= {want_share}/{len(slots)} with lane C")
    wins, seen = [], set()
    for w in pay.get("wins", ()):
        d, uid = str(w.get("domain")), int(w.get("uid", -1))
        if d not in slots or d in seen:
            return [], f"lane-B payout for unknown/duplicate domain {d!r}"
        seen.add(d)
        if not 0 <= uid < n_uids:
            return [], f"lane-B payout to out-of-range uid {uid}"
        age = max(0, (block - int(w.get("merge_block", 0))) // eb)
        if int(w.get("age", -1)) != age:
            return [], (f"domain {d}: recorded age {w.get('age')} != {age} "
                        f"eras since block {w.get('merge_block')}")
        if win_expired(age, lb):
            return [], (f"domain {d}: win paid at age {age} eras, past "
                        f"win_ttl_eras={lb.win_ttl_eras}")
        why = _paid_full_pay_why(
            w, lb.full_pay_tiers, at_win=int(w.get("merge_block", -1)) == block,
            ttl=int(lb.win_ttl_eras),
            state_entry=None if post is None else (post.get("lane_b") or {}).get(d))
        if why:
            return [], f"domain {d}: {why}"
        mult = reward_multiplier(age, full_eras=int(w["full_eras"]),
                                 half_life_rounds=lb.half_life_rounds, floor=lb.floor)
        if not math.isclose(float(w.get("multiplier", -1.0)), mult, rel_tol=1e-12,
                            abs_tol=1e-15):
            return [], f"domain {d}: multiplier {w.get('multiplier')} != {mult} at age {age}"
        wins.append(LaneBWin(uid=uid, domain=d, multiplier=mult))
    return wins, None


def _lane_c_paid_wins(receipt: RoundReceipt, cfg: ChainConfig,
                      n_uids: int) -> tuple[list, str | None]:
    """The receipt's ``lane_c_pay`` wins re-derived from ``[lane_c]``
    (DEC-CA-0056): ``(wins, None)`` or ``([], why)`` on the first
    inconsistency (lane C not active, payout clock, era length, share / slot
    count, unknown / duplicate target, uid range, age, expiry, multiplier)."""
    from ..shared.era import era_length_blocks
    from ..shared.lane_c import LaneCWin, full_pay_tiers_for, target_multiplier, win_expired

    lc = cfg.lane_c
    pay = receipt.lane_c_pay or {}
    post = getattr(receipt, "state_post", None)
    post = (post.get("lane_c") or {}).get("slots") if (
        isinstance(post, dict) and "lane_c" in post) else None
    block = int(pay.get("block", -1))
    if not lc.active(int(receipt.epoch_start_block)):
        return [], (f"lane-C payout record at block {receipt.epoch_start_block}, but lane C "
                    f"is not active there ([lane_c] from_block={lc.from_block})")
    if block != int(receipt.epoch_start_block):
        return [], (f"lane-C payout clock at block {block}, but the settlement is "
                    f"{receipt.epoch_start_block}")
    eb = int(era_length_blocks(cfg.round, block))
    if int(pay.get("era_blocks", -1)) != eb:
        return [], f"lane-C era length {pay.get('era_blocks')} != {eb} in force"
    from ..shared.lane_c import lane_c_split, wind_down_pays

    # concurrent competitions (owner 2026-10-20): the per-competition split
    # and states, and the per-target slot fractions, re-derived from config
    comps, slot = lane_c_split(lc, block, eb, getattr(cfg, "marketcast", None))
    live = tuple(t for t in lc.targets if t in slot)
    winding = {t for c in comps if c["state"] == "winding_down" for t in c["targets"]}
    if not math.isclose(float(pay.get("share", -1.0)), float(lc.share), rel_tol=0,
                        abs_tol=1e-15) or int(pay.get("slots", -1)) != len(live):
        return [], (f"lane-C pay record share/slots {pay.get('share')}/{pay.get('slots')} != "
                    f"[lane_c] {lc.share}/{len(live)} (slot-bearing targets at {block})")
    if json.loads(json.dumps(pay.get("competitions"))) != json.loads(json.dumps(comps)):
        return [], (f"lane-C per-competition shares/states {pay.get('competitions')} != "
                    f"{comps} re-derived at {block}")
    got_frac = pay.get("slot_frac")
    if not isinstance(got_frac, dict) or set(got_frac) != set(slot) or any(
            not math.isclose(float(got_frac[t]), slot[t], rel_tol=0, abs_tol=1e-15)
            for t in slot):
        return [], f"lane-C slot fractions {got_frac} != {slot} re-derived at {block}"
    wins, seen = [], set()
    for w in pay.get("wins", ()):
        t, uid = str(w.get("target")), int(w.get("uid", -1))
        if t not in live or t in seen:
            return [], f"lane-C payout for unknown/duplicate target {t!r}"
        seen.add(t)
        if not 0 <= uid < n_uids:
            return [], f"lane-C payout to out-of-range uid {uid}"
        if int(w.get("win_block", 0)) > block:
            return [], f"target {t}: win block lies after the payout boundary"
        age = max(0, (block - int(w.get("win_block", 0))) // eb)
        if int(w.get("age", -1)) != age:
            return [], (f"target {t}: recorded age {w.get('age')} != {age} eras since block "
                        f"{w.get('win_block')}")
        if win_expired(age, lc):
            return [], f"target {t}: win paid at age {age} eras, past win_ttl_eras={lc.win_ttl_eras}"
        why = _paid_full_pay_why(
            w, full_pay_tiers_for(t, lc, getattr(cfg, "marketcast", None)),
            at_win=int(w.get("win_block", -1)) == block, ttl=int(lc.win_ttl_eras),
            state_entry=None if post is None else (post or {}).get(t))
        if why:
            return [], f"target {t}: {why}"
        if bool(w.get("wind_down", False)) != (t in winding):
            return [], (f"target {t}: wind_down flag {w.get('wind_down')} but its competition "
                        f"is {'' if t in winding else 'not '}winding down at {block}")
        if t in winding:
            # a winding-down competition pays full while the win is inside its
            # own full-pay period (no decay tail), else nothing
            held = SimpleNamespace(win_block=int(w.get("win_block", 0)),
                                   full_eras=int(w["full_eras"]))
            if not wind_down_pays(lc.competition_of(t), held, block, eb):
                return [], (f"target {t}: paid in wind-down at age {age}, past its full-pay "
                            f"period ({w.get('full_eras')} eras) or the wind-down window")
            mult = 1.0
        else:
            mult = target_multiplier(age, lc, int(w["full_eras"]))
        if not math.isclose(float(w.get("multiplier", -1.0)), mult, rel_tol=1e-12,
                            abs_tol=1e-15):
            return [], f"target {t}: multiplier {w.get('multiplier')} != {mult} at age {age}"
        wins.append(LaneCWin(uid=uid, target=t, multiplier=mult))
    return wins, None


def key_slash_mismatch(lad: dict, cfg: ChainConfig, *, boundary: int, funded_by: str,
                       dethroned: bool) -> str:
    """DEC-CA-0055 escalating key slash: why the ladder record's key-slash
    fields do NOT follow from its recorded prior state and the manifest's
    ``funded_by`` (:func:`cascade.shared.emission.key_slash_step`), "" when
    they do. A dethrone resets the state (no prior recorded, no slash). The
    prior itself is bound across receipts by :func:`ladder_prior_mismatch`
    (satellite continuity)."""
    from ..shared.emission import KeySlash, key_slash_step

    got = KeySlash(lapses=int(lad.get("key_lapses", -1)),
                   lapse_open=bool(lad.get("lapse_open", False)),
                   slash_until_block=int(lad.get("slash_until_block", -1)),
                   reason=str(lad.get("slash_reason", "") or ""))
    if dethroned:
        if got != KeySlash() or "prev_key_lapses" in lad:
            return "a dethrone must reset the key slash (no lapses, no window, no prior)"
        return ""
    if "prev_key_lapses" not in lad:
        return "the ladder record carries no prior key-slash state (prev_key_lapses)"
    want = key_slash_step(funded_by=funded_by, boundary=boundary,
                          lapses=int(lad.get("prev_key_lapses", 0)),
                          lapse_open=bool(lad.get("prev_lapse_open", False)),
                          slash_until_block=int(lad.get("prev_slash_until_block", 0)), cfg=cfg)
    if got != want:
        return (f"key slash {got} does not follow from the prior "
                f"({lad.get('prev_key_lapses')}, {lad.get('prev_lapse_open')}, "
                f"{lad.get('prev_slash_until_block')}) and funded_by {funded_by or 'n/a'!r} "
                f"(expected {want})")
    if bool(lad.get("slashed", False)) != want.slashed:
        return (f"slashed={lad.get('slashed')} but the key slash says "
                f"{want.reason or 'not slashed'!r}")
    return ""


def receipt_crown_units(receipt: RoundReceipt, cfg: ChainConfig) -> str:
    """The units of the receipt's verdict LCB
    (:func:`cascade.shared.emission.crown_units`), re-derived exactly as
    :func:`check_verdict` replays it: increment ⟺ the recorded params say
    ``margin_mode = "increment"`` AND baseline rows exist, and — when the LCB
    is the cohort's joint max-T bound (more than one duelled challenger, max-T
    in force) — ``cohort_maxt_increment_from_block`` reached. Raises
    ``ValueError`` / ``KeyError`` / ``TypeError`` on a malformed receipt."""
    from ..eval.koth import KothParams
    from ..shared.emission import crown_units

    v = receipt.verdict
    if v is None:
        raise ValueError("no verdict on this receipt")
    params = KothParams(**v.params)
    _, _, paired = _pooled_scores(receipt, receipt.load_embedded_manifest())
    b = int(receipt.epoch_start_block)
    return crown_units(
        margin_mode=params.margin_mode,
        has_baseline=_baseline_pooled(receipt, paired) is not None,
        maxt=len(_duelled_hotkeys(receipt)) > 1 and cohort_maxt_active(cfg.scoring, b),
        maxt_increment=cohort_maxt_increment_active(cfg.scoring, b))


def crown_start_mismatch(lad: dict, cfg: ChainConfig, *, verdict_lcb: float | None,
                         units: str) -> str:
    """DEC-CA-0055 margin-tiered start share (owner 2026-10-20): why a
    crowning's ladder record does NOT follow from the verdict it came from, ""
    when it does. The recorded ``crown_lcb`` must BE the verdict's LCB (the
    signed number :func:`check_verdict` replays), ``crown_units`` the units
    re-derived from the receipt (:func:`receipt_crown_units`); the tier, the
    start share and the ladder ``bp`` are then re-derived
    (:func:`cascade.shared.emission.crown_start`) — never trusted."""
    from ..shared.emission import crown_record, crown_start

    if "crown_lcb" not in lad or "crown_units" not in lad:
        return "a crowning's ladder record carries no crown_lcb / crown_units"
    got_lcb = lad.get("crown_lcb")
    if got_lcb != verdict_lcb and not (got_lcb is None and verdict_lcb is None):
        return f"crown_lcb {got_lcb!r} != the verdict's lcb {verdict_lcb!r}"
    if lad.get("crown_units") != units:
        return f"crown_units {lad.get('crown_units')!r} != the receipt's {units!r}"
    want = crown_record(crown_start(verdict_lcb, units, cfg))
    for k, w in want.items():
        if lad.get(k) != w:
            return f"{k} {lad.get(k)!r} != {w!r} re-derived from the {units} LCB {verdict_lcb!r}"
    if lad.get("bp") != want["crown_start_bp"]:
        return (f"a crowning must start the ladder at crown_start_bp "
                f"{want['crown_start_bp']}; got bp {lad.get('bp')!r}")
    return ""


def ladder_prior_mismatch(lad: dict | None, prior_state: dict | None) -> str:
    """Satellite continuity (DEC-CA-0058): the ladder record's prior
    key-slash state must equal the previous receipt's ``state_post`` ladder
    (:func:`cascade.shared.settlement_state.settlement_state`). "" when it
    does, or when there is nothing to bind (no ladder / a crowning)."""
    if not lad or "prev_key_lapses" not in lad:
        return ""
    prior = (prior_state or {}).get("ladder") or {}
    want = (int(prior.get("key_lapses", 0) or 0), bool(prior.get("lapse_open", False)),
            int(prior.get("slash_until_block", 0) or 0))
    got = (int(lad.get("prev_key_lapses", 0)), bool(lad.get("prev_lapse_open", False)),
           int(lad.get("prev_slash_until_block", 0)))
    if got != want:
        return (f"ladder prior key slash {got} != the previous receipt's posterior {want}")
    return ""


def _check_emission_weights(name: str, receipt: RoundReceipt,
                            cfg: ChainConfig) -> CheckResult | None:
    """DEC-CA-0055: rebuild a vector shaped by the pre-launch burn, the
    carried-king cap, the v2 lane-A ladder or the key slash from the
    receipt's ``emission`` record, and check the record itself against the
    config and the receipt's own verdict / king check / manifest.

    ``None`` = no DEC-CA-0055 rule applies to this receipt (the legacy checks
    run). A receipt the config says MUST carry a record (a pre-launch burn in
    force at its boundary, or a v2 settlement under ``carryover_king_share``
    / ``king_ladder``) but carries none FAILS — the record is what binds the
    validator to the rule."""
    from ..shared.emission import (
        CHECK_RESULTS,
        EmissionPolicy,
        LadderState,
        crowned_in_v2,
        emission_policy,
        ladder_on,
        ladder_params,
        ladder_step,
    )
    court_blocks = ladder_params(cfg)["court_blocks"]   # the overlay-settable court window
    from ..shared.era import v2_settlement_active

    boundary = int(receipt.epoch_start_block)
    rec = getattr(receipt, "emission", None)
    in_v2 = v2_settlement_active(cfg, boundary)
    c_pay = getattr(receipt, "lane_c_pay", None)
    if rec is None and c_pay is not None:
        # DEC-CA-0056: a lane-C vector with no DEC-CA-0055 rule shaping lane A
        # — rebuilt with an inert policy; a rule the config says applies but
        # the receipt omits still fails below
        if not in_v2 and not emission_policy(cfg, boundary=boundary,
                                             king_since_block=None).inert:
            return _fail(name, "the pre-launch burn is in force but the receipt carries no "
                               "emission record")
        if in_v2 and (float(cfg.v2.carryover_king_share or 0.0) > 0.0 or ladder_on(cfg)):
            return _fail(name, "v2 settlement under a DEC-CA-0055 rule carries no emission "
                               "record")
        return _rebuild_lane_vector(name, receipt, cfg, EmissionPolicy(), None)
    if rec is None:
        if not in_v2 and not emission_policy(cfg, boundary=boundary,
                                             king_since_block=None).inert:
            return _fail(name, f"the pre-launch burn ({cfg.v2.prelaunch_burn}) is in force "
                               f"from block {cfg.v2.prelaunch_burn_from_block}, but the "
                               "receipt carries no emission record")
        if in_v2 and float(cfg.v2.carryover_king_share or 0.0) > 0.0:
            return _fail(name, "v2 settlement under [v2] carryover_king_share carries no "
                               "emission record (the cap or the ladder always applies)")
        if getattr(receipt, "king_check", None) is not None:
            return _fail(name, "king_check stamped without an emission record")
        return None
    if int(rec.get("boundary", -1)) != boundary:
        return _fail(name, f"emission record decided at block {rec.get('boundary')}, the "
                           f"settlement is {boundary}")
    ksb = rec.get("king_since_block")
    lad = rec.get("ladder")
    state = None
    if lad is not None:
        state = LadderState(king_share_bp=lad.get("bp"),
                            court_hotkey=lad.get("court_hotkey") or None,
                            court_until_block=int(lad.get("court_until_block", 0) or 0),
                            slashed=bool(lad.get("slashed", False)))
    want_pol = emission_policy(cfg, boundary=boundary, king_since_block=ksb, ladder=state)
    got_pol = EmissionPolicy.from_record(rec.get("policy") or {})
    if want_pol != got_pol:
        return _fail(name, f"emission policy {got_pol.as_record()} != the config's "
                           f"{want_pol.as_record()} at block {boundary}")
    if lad is not None and not (in_v2 and ladder_on(cfg)):
        return _fail(name, "a ladder record outside a v2 settlement under [v2] king_ladder")
    if lad is None and in_v2 and ladder_on(cfg) and receipt.status == "scored":
        return _fail(name, "scored v2 settlement under [v2] king_ladder carries no ladder "
                           "record")
    v = receipt.verdict
    if lad is not None:
        event = str(lad.get("event"))
        dethroned = bool(v is not None and v.dethroned)
        try:
            manifest = receipt.load_embedded_manifest()
            king = manifest.entry_for_role("king")
        except (ValueError, KeyError):
            king = None
        funded = str(getattr(king, "funded_by", "") or "") if king is not None else ""
        if str(lad.get("funded_by", "")) != funded:
            return _fail(name, f"ladder funded_by {lad.get('funded_by')!r} != the manifest "
                               f"king entry's {funded!r}")
        kc = receipt.king_check
        check = None if kc is None else str(kc.get("result"))
        if check is not None and check not in CHECK_RESULTS:
            return _fail(name, f"king_check result {check!r} is not one of {CHECK_RESULTS}")
        why = key_slash_mismatch(lad, cfg, boundary=boundary, funded_by=funded,
                                 dethroned=dethroned)
        if why:
            return _fail(name, why)
        if not dethroned and any(str(k).startswith("crown_") for k in lad):
            return _fail(name, "crown_* fields on a ladder record without a dethrone")
        if dethroned:
            if event != "crowned" or state.slashed:
                return _fail(name, "a dethrone must record event 'crowned' and no slash")
            try:
                units = receipt_crown_units(receipt, cfg)
            except (ValueError, KeyError, TypeError) as e:
                return _fail(name, f"cannot re-derive the crowning LCB's units: {e}")
            why = crown_start_mismatch(lad, cfg, verdict_lcb=v.lcb, units=units)
            if why:
                return _fail(name, why)
            if ksb != boundary:
                return _fail(name, f"dethroned at {boundary} but king_since_block is {ksb}")
            if state.court_hotkey and state.court_until_block != boundary + court_blocks:
                return _fail(name, f"court until {state.court_until_block}, expected "
                                   f"{boundary + court_blocks} (dethrone + {court_blocks})")
            if check is not None:
                return _fail(name, "a king check recorded at a dethroning settlement")
        elif receipt.status == "scored":
            if crowned_in_v2(cfg, ksb):
                want_bp = ladder_step(lad.get("prev_bp"), crowned=False, check=check, cfg=cfg)
                if lad.get("bp") != want_bp or event != (check or "none"):
                    return _fail(name, f"ladder {lad.get('prev_bp')} → {lad.get('bp')} "
                                       f"({event}) does not follow from check {check!r} "
                                       f"(expected {want_bp})")
            elif event != "carried" or lad.get("bp") is not None:
                return _fail(name, "a carried (pre-v2-crowned) king runs no ladder")
        if state.court_hotkey and state.court_until_block > boundary + court_blocks:
            return _fail(name, f"court until {state.court_until_block} is past "
                               f"{boundary} + {court_blocks}")
    return _rebuild_lane_vector(name, receipt, cfg, got_pol, lad)


def _lane_vector_want(receipt: RoundReceipt, cfg: ChainConfig, got_pol,
                      n_uids: int) -> tuple[list[float] | None, int, str | None]:
    """The lane-kept-apart vector (lane A's court, lane B's ``lane_b_pay``,
    lane C's ``lane_c_pay``) under ``got_pol``: ``(want, n paid lane-C
    slots, None)``, or ``(None, 0, why)`` on the first inconsistency."""
    from ..shared.emission import emission_vector, lane_parts

    boundary = int(receipt.epoch_start_block)
    burn = cfg.scoring.burn_uid
    wins, pay = [], receipt.lane_b_pay
    if pay is not None:
        wins, why = _lane_b_paid_wins(receipt, cfg, n_uids)
        if why is not None:
            return None, 0, why
    elif cfg.lane_b.active(boundary):
        return None, 0, ("lane B is live at this settlement but the receipt carries no "
                         "payout record to rebuild it")
    c_wins, c_pay = [], getattr(receipt, "lane_c_pay", None)
    if c_pay is not None:
        c_wins, why = _lane_c_paid_wins(receipt, cfg, n_uids)
        if why is not None:
            return None, 0, why
    elif cfg.lane_c.active(boundary):
        return None, 0, ("lane C is live at this settlement but the receipt carries no "
                         "lane-C payout record to rebuild it")
    if c_pay is not None and pay is None:
        return None, 0, "a lane-C payout record without lane B's: lane C without lane B is refused"
    a, b = lane_parts(list(receipt.reward_uids), n_uids, king_decay=cfg.scoring.king_decay,
                      burn_uid=burn, lane_b_cfg=(cfg.lane_b if pay is not None else None),
                      wins=wins, lane_c_cfg=(cfg.lane_c if c_pay is not None else None),
                      c_wins=c_wins,
                      c_slot_frac=(dict(c_pay["slot_frac"]) if c_pay is not None else None))
    want = emission_vector(a, b if (pay is not None or c_pay is not None) else None, got_pol,
                           burn_uid=burn, lane_a_uids=list(receipt.reward_uids))
    return want, len(c_wins), None


def _rebuild_lane_vector(name: str, receipt: RoundReceipt, cfg: ChainConfig, got_pol,
                         lad: dict | None) -> CheckResult:
    """Rebuild the vector from the lanes kept apart (lane A's court, lane B's
    ``lane_b_pay``, lane C's ``lane_c_pay`` — DEC-CA-0056, taken from lane
    B's share) under ``got_pol`` and compare it uid for uid."""
    got = [float(w) for w in receipt.weights]
    pay, c_pay = receipt.lane_b_pay, getattr(receipt, "lane_c_pay", None)
    want, n_c_wins, why = _lane_vector_want(receipt, cfg, got_pol, len(got))
    if why is not None:
        return _fail(name, why)
    bad = [u for u, (g, x) in enumerate(zip(got, want, strict=True))
           if not math.isclose(g, x, rel_tol=1e-9, abs_tol=1e-12)]
    lanes = "A+B+C" if c_pay is not None else ("A+B" if pay is not None else "A")
    if bad:
        u = bad[0]
        return _fail(name, f"uid {u} holds {got[u]}, the lane {lanes} rebuild gives {want[u]} "
                           f"(policy {got_pol.as_record()})")
    if got_pol.inert:
        return _ok(name, f"lane {lanes} vector rebuilds exactly ({n_c_wins} paid lane-C "
                         f"slot(s))")
    return _ok(name, f"DEC-CA-0055 vector rebuilds exactly (policy {got_pol.as_record()}"
                     f"{'' if lad is None else ', ladder ' + str(lad.get('event'))}"
                     f"{', lane C' if c_pay is not None else ''})")


def _check_lane_b_weights(name: str, receipt: RoundReceipt, cfg: ChainConfig,
                          lane_a: list[float]) -> CheckResult:
    """DEC-CA-0050 weight check, selected by ``cfg.lane_b.active(block)``.

    With the receipt's ``lane_b_pay`` record (every vector a lane-B validator
    sets) the vector is REBUILT exactly: the decay clock is this settlement's
    block, each paid win's age (in ERAS of the grid in force) and multiplier
    re-derive from ``[lane_b]``, no win at or past ``win_ttl_eras`` is paid,
    each domain slot pays ``share / n_domains × multiplier``
    (never more), and the unpaid lane-B share goes to ``burn_uid`` (or back to
    lane A under ``remainder = "lane_a"``). Without the record (a validator
    whose push fell back to the legacy vector, e.g. ``force_burn``) only the
    decomposition is checked: every lane-A recipient holds at least
    ``(1 − share) ×`` its court share and no non-burn uid draws more lane-B
    weight than every domain slot together."""
    import math

    from ..eval.expert_lane import split_emissions

    lb = cfg.lane_b
    got = [float(w) for w in receipt.weights]
    if abs(sum(got) - 1.0) > 1e-6:
        return _fail(name, f"lane-B vector sums to {sum(got)}, not 1")
    from ..shared.lane_b import lane_b_slots

    n_slots = max(1, len(lane_b_slots(lb)))     # DEC-CA-0059: domains + horizons
    passing = sum(1 for r in (receipt.lane_b or ()) if r.get("passed"))
    pay = getattr(receipt, "lane_b_pay", None)
    if pay is not None:
        wins, why = _lane_b_paid_wins(receipt, cfg, len(got))
        if why is not None:
            return _fail(name, why)
        want_map = split_emissions({u: a for u, a in enumerate(lane_a) if a > 0}, wins,
                                   share_b=lb.share, n_domain_slots=n_slots,
                                   burn_uid=cfg.scoring.burn_uid, remainder=lb.remainder)
        want = [0.0] * len(got)
        for u, w in want_map.items():
            want[u] += w
        bad = [u for u, (g, x) in enumerate(zip(got, want, strict=True))
               if not math.isclose(g, x, rel_tol=1e-9, abs_tol=1e-12)]
        if bad:
            u = bad[0]
            return _fail(name, f"uid {u} holds {got[u]}, the lane-A + lane-B rebuild gives "
                               f"{want[u]} ({len(wins)} paid domain win(s))")
        return _ok(name, f"lane-B vector rebuilds exactly (share {lb.share}, {len(wins)} paid "
                         f"domain win(s), {passing} passing legs this round)")
    excess = []
    for u, (g, a) in enumerate(zip(got, lane_a, strict=True)):
        base = (1.0 - lb.share) * a
        if g + 1e-9 < base:
            return _fail(name, f"uid {u} holds {g} < its lane-A share {base}")
        excess.append(g - base)
    if abs(sum(excess) - lb.share) > 1e-6:
        return _fail(name, f"lane-B remainder sums to {sum(excess)}, expected {lb.share}")
    slot = lb.share / n_slots
    for u, e in enumerate(excess):
        if (u != cfg.scoring.burn_uid and lb.remainder == "burn"
                and e > slot * n_slots + 1e-9):
            return _fail(name, f"uid {u} draws {e} lane-B weight > every domain slot")
    return _warn(name, "lane-B weight decomposition consistent, but the receipt carries no "
                       f"payout record to rebuild it ({passing} passing legs this round)")


LANE_B_STAGES = ("stale", "selection", "merge", "verdict", "targets")


def _lane_b_row_verdict(r: dict, lb, slots: tuple[str, ...]) -> str | None:
    """Why ``r`` (one lane-B verdict row) contradicts ``[lane_b]``, else None.
    A multi-target row (DEC-CA-0059, it names ``targets``) is re-derived in
    full; a single-target row carries no point gain, so only ``passed ⇒``
    every bar is checked (its bytes are the pre-0059 row)."""
    import math

    d = str(r.get("domain"))
    multi = "targets" in r
    if d not in slots:
        return f"row for unconfigured target {d!r}"
    if r.get("stage") not in LANE_B_STAGES:
        return f"{d}: unknown stage {r.get('stage')!r}"
    if r.get("stage") != "verdict":
        return f"{d}: a {r.get('stage')}-stage row passed" if r.get("passed") else None
    lcb = r.get("target_lcb")
    lcb_ok = lcb is not None and not math.isnan(float(lcb)) and float(lcb) >= lb.gain_margin
    guards = r.get("guard_lcbs") or {}
    pubs = r.get("public_lcbs") or {}
    guards_ok = (all(float(v) >= -lb.tau_domain for v in guards.values())
                 and all(float(v) >= -lb.tau_public for v in pubs.values()))
    declared = list(r.get("targets") or [d])
    if any(t in guards for t in declared):
        return f"{d}: a declared target {sorted(set(declared) & set(guards))} is also a guard"
    if not multi:
        if r.get("passed") and not (lcb_ok and guards_ok):
            return f"{d}: passed with LCB {lcb} / guards {guards} {pubs} below the bars"
        return None
    k = len(declared)
    if not math.isclose(float(r.get("alpha") or -1.0), lb.alpha / k, rel_tol=1e-12):
        return f"{d}: gain level {r.get('alpha')} != alpha/k = {lb.alpha}/{k}"
    inc = str(r.get("inconclusive") or "")
    gain = r.get("target_gain")
    if inc:
        want = False
        if r.get("passed") or guards or pubs:
            return f"{d}: inconclusive ({inc}) row passed or carries guards"
    else:
        n, cl = int(r.get("target_n") or 0), int(r.get("target_clusters") or 0)
        if n < lb.min_target_windows or cl < lb.min_target_clusters:
            return f"{d}: {n} windows / {cl} clusters is inconclusive, row says conclusive"
        want = lcb_ok and gain is not None and float(gain) > 0.0 and guards_ok
        if bool(r.get("guard_reason")) == guards_ok:
            return f"{d}: guard_reason {r.get('guard_reason')!r} contradicts the guard LCBs"
    if bool(r.get("passed")) != want:
        return (f"{d}: passed={r.get('passed')} but the re-derived per-target verdict is "
                f"{want} (LCB {lcb} vs {lb.gain_margin}, gain {gain}, guards ok {guards_ok})")
    return None


def check_lane_b(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """DEC-CA-0050 / DEC-CA-0059 lane-B verdict consistency, selected by
    ``cfg.lane_b.active(block)`` (the vector itself: :func:`check_weights`):

    * every verdict row names a configured slot (domains + horizon targets)
      and a known stage; only a ``verdict``-stage row may pass;
    * a passing row clears ``gain_margin`` and every guard (``−tau_domain`` /
      ``−tau_public``); no declared target is also a guard;
    * a multi-target leg (rows naming ``targets``): 1 .. ``max_domains_per_leg``
      configured targets, one row each; every row's gain level is ``alpha /
      k``; each per-target verdict is RE-DERIVED from its row (floors, LCB,
      point gain, guards); the guards were run ONCE — every conclusive row of
      the leg carries the same guard / public LCBs and guard reason — and a
      guard failure leaves no target of the leg passing;
    * slots: a pay-record win merged at THIS settlement is the best passing
      row of its target (highest target LCB), and a target with a passing row
      is never still paid to an older win;
    * merge-once: the embedded manifest's era-init merges are distinct;
    * merge guard (``[lane_b] merge_guard_from_block``): they are also
      row-disjoint — no (layer, row) is written by two merges.

    Re-scoring the candidates needs the private pool and the checkpoints —
    Tier-1 replay, not Tier-0."""
    import json

    name = "lane_b"
    lb = cfg.lane_b
    boundary = int(receipt.epoch_start_block)
    if not lb.active(boundary):
        return _skip(name, "lane B not active at this settlement")
    from ..shared.lane_b import lane_b_slots, merge_identities

    slots = lane_b_slots(lb)
    rows = [r for r in (getattr(receipt, "lane_b", None) or ()) if "error" not in r]
    legs: dict[tuple, list[dict]] = {}
    for r in rows:
        why = _lane_b_row_verdict(r, lb, slots)
        if why is not None:
            return _fail(name, why)
        if "targets" in r:
            legs.setdefault((r.get("hotkey"), r.get("leg")), []).append(r)
    for (hk, _leg), rs in legs.items():
        declared = list(rs[0].get("targets") or [])
        if (not 1 <= len(declared) <= lb.max_domains_per_leg
                or any(t not in slots for t in declared) or len(set(declared)) != len(declared)):
            return _fail(name, f"{hk}: declared targets {declared} are not 1.."
                               f"{lb.max_domains_per_leg} configured slots")
        if sorted(str(r.get("domain")) for r in rs) != sorted(declared) or any(
                list(r.get("targets") or []) != declared for r in rs):
            return _fail(name, f"{hk}: rows {[r.get('domain') for r in rs]} do not cover the "
                               f"declared targets {declared} once each")
        conc = [r for r in rs if r.get("stage") == "verdict" and not r.get("inconclusive")]
        sig = {(json.dumps(r.get("guard_lcbs") or {}, sort_keys=True),
                json.dumps(r.get("public_lcbs") or {}, sort_keys=True),
                str(r.get("guard_reason") or "")) for r in conc}
        if len(sig) > 1:
            return _fail(name, f"{hk}: the leg's guards differ between its targets "
                               "(guards must run once per leg)")
        if any(r.get("guard_reason") for r in conc) and any(r.get("passed") for r in rs):
            return _fail(name, f"{hk}: a guard failed but a target of the leg passed")
    best: dict[str, float] = {}
    for r in rows:
        if r.get("passed"):
            d = str(r.get("domain"))
            best[d] = max(best.get(d, float("-inf")), float(r.get("target_lcb")))
    pay = getattr(receipt, "lane_b_pay", None) or {}
    acceptance = lb.acceptance_at_settlement(boundary)
    if acceptance:
        why = _lane_b_acceptance_pay(receipt, cfg)
        if why:
            return _fail(name, why)
    for w in (() if acceptance else (pay.get("wins", ()) or ())):
        d = str(w.get("domain"))
        if int(w.get("merge_block", -1)) == boundary:
            mine = [r for r in rows if r.get("passed") and str(r.get("domain")) == d
                    and r.get("hotkey") == w.get("hotkey")]
            if not mine:
                return _fail(name, f"slot {d} won at this settlement by {w.get('hotkey')!r} "
                                   "without a passing row")
            if max(float(r.get("target_lcb")) for r in mine) < best[d]:
                return _fail(name, f"slot {d} went to {w.get('hotkey')!r}, not the best "
                                   f"passing target LCB {best[d]}")
            # gain-tiered full-pay (owner 2026-10-20): the tier is derived from
            # THIS row's target LCB (per target of a multi-target leg) — the
            # pay record's gain_lcb must be it, and its tier re-derives from it
            if w.get("gain_lcb") is None or float(w.get("gain_lcb")) != best[d]:
                return _fail(name, f"slot {d}: pay record gain_lcb {w.get('gain_lcb')!r} != "
                                   f"the winning row's target LCB {best[d]}")
            from ..eval.expert_lane import full_pay_mismatch

            why = full_pay_mismatch(w, lb.full_pay_tiers)
            if why:
                return _fail(name, f"slot {d}: full-pay tier forged: {why}")
        elif d in best:
            return _fail(name, f"slot {d} has a passing row at this settlement but is still "
                               f"paid to the win from block {w.get('merge_block')}")
    merges = list((receipt.manifest or {}).get("warm_start_merges") or ())
    if len(merge_identities(merges)) != len(merges):
        return _fail(name, "an era-init merge is declared more than once (merge-once)")
    if lb.merge_guard_at_settlement(boundary):
        # DEC-CA-0050 merge guards: the era-init merges are row-disjoint
        from ..trainer.expert_lane import merge_row_overlap

        hit = merge_row_overlap(merges)
        if hit is not None:
            return _fail(name, f"era-init merges overlap on MoE layer {hit[0]} expert row "
                               f"{hit[1]} (merge guard)")
    n_multi = len(legs)
    return _ok(name, f"{len(rows)} verdict row(s) consistent ({n_multi} multi-target leg(s), "
                     f"{len(best)} target(s) won, {len(slots)} slots)")


def _lane_b_acceptance_pay(receipt, cfg):
    """Tier-0 binding: newly paid slots need signed merge-acceptance evidence.
    Tier-1 validators re-score the checkpoints and verify each marginal tier.
    """
    from ..shared.era import era_length_blocks, era_start_block

    manifest = receipt.manifest or {}
    era = manifest.get("era") or {}
    if not era:
        return "acceptance payouts require an era stamp"
    start = int(era["start_block"])
    block = min(start, int(era["seed_block"]) + era_length_blocks(cfg.round, start))
    if block < cfg.lane_b.acceptance_from_block:
        return None  # an already-running seed epoch keeps its legacy init
    before = str(manifest.get("warm_start_base", ""))
    approved, previous = {}, None
    for merge in manifest.get("warm_start_merges", ()):
        proof = merge.get("acceptance") or {}
        if proof.get("block") != block or proof.get("before") != before:
            return "merge acceptance block/checkpoint chain differs"
        qualified = merge.get("qualified_wins") or []
        if not qualified:
            return "merge has no qualification evidence"
        order = (int(qualified[0].get("commit_block", 0)), qualified[0]["hotkey"],
                 merge["rebased_pointer"])
        if previous is not None and order <= previous:
            return "accepted merges are not in commit order"
        previous = order
        wins = proof.get("wins") or []
        if not wins or proof.get("targets") != [w["domain"] for w in wins]:
            return "accepted merge has no matching target rewards"
        for win in wins:
            domain = win["domain"]
            if domain in approved or not any(
                    q["domain"] == domain and q["hotkey"] == win["hotkey"] for q in qualified):
                return "accepted target is duplicated or was not qualified"
            if (not float(win.get("target_lcb", -1)) >= cfg.lane_b.gain_margin
                    or int(win.get("qualified_block", block + 1)) > block):
                return "accepted target has insufficient gain or a future qualification"
            approved[domain] = win
        before = str(proof.get("candidate", ""))
    pay = receipt.lane_b_pay or {}
    for win in pay.get("wins", ()) or ():
        clock = int(win["merge_block"])
        if clock >= cfg.lane_b.acceptance_from_block and era_start_block(cfg.round, clock) != clock:
            return "qualified leg paid before a warm-start acceptance"
        if clock > int(receipt.epoch_start_block):
            return "lane-B reward starts in the future"
        if clock == block and clock >= cfg.lane_b.acceptance_from_block:
            source = approved.get(win["domain"])
            if source is None or source["hotkey"] != win["hotkey"]:
                return "paid qualification has no accepted merge"
            if any(win.get(k) != source.get(sk) for k, sk in (
                    ("gain_lcb", "target_lcb"), ("tier", "tier"),
                    ("full_eras", "full_eras"), ("below_floor", "below_floor"))):
                return "payout differs from the accepted marginal gain/tier"
    return None


# "duplicate" (review L6 / DEC-CA-0064 B5): a later commit of a tree / weights
# an earlier commit holds
# "ended": an entry of a competition past its until_block (concurrent
# competitions, owner 2026-10-20) — refused, never judged
LANE_C_STAGES = ("target", "stale", "binding", "fault", "verdict", "duplicate", "ended")
# DEC-CA-0064 review fixes: stages only an ``mkt:`` row may carry
MKT_STAGES = ("infra",)
LANE_C_OUTCOMES = ("", "pending", "deferred", "dropped", "confirmed", "below holder LCB", "not_selected")


def _lane_c_best_key(x: dict) -> tuple:
    return (-float(x["gain_lcb"]), int(x.get("train_block", 0) or 0), str(x.get("hotkey")))


def _lane_c_holder_kept(row: dict, boundary: int, round_id: str) -> bool:
    """:func:`cascade.shared.lane_c.win_beats` re-derived from the RECORDED
    values: does the row's ``holder`` keep its slot against this confirmed
    row (a win at ``boundary``)?"""
    from ..shared.lane_c import TargetWin, win_beats

    h = row.get("holder")
    if not h:
        return False
    t = str(row.get("target"))
    new = TargetWin(target=t, hotkey=str(row.get("hotkey")), uid=0, round_id=str(round_id),
                    win_block=int(boundary), base_pointer=str(row.get("base_pointer", "")),
                    gain_lcb=float(row.get("gain_lcb") or 0.0))
    cur = TargetWin(target=t, hotkey=str(h.get("hotkey")), uid=0,
                    round_id=str(h.get("round_id", "")), win_block=int(h.get("win_block", 0)),
                    base_pointer=str(h.get("base_pointer", "")),
                    gain_lcb=float(h.get("gain_lcb") or 0.0))
    return not win_beats(new, cur)


def check_lane_c(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """DEC-CA-0056 verdict consistency, selected by ``cfg.lane_c.active``:

    * before lane C is active the receipt carries neither ``lane_c`` nor
      ``lane_c_pay`` (pre-lane-C receipts are byte-identical);
    * every verdict row names a configured target, a known stage and outcome;
      a PASSING row is a ``verdict``-stage row whose gain LCB clears
      ``gain_margin`` with a positive finite point estimate on at least
      ``min_target_windows`` windows over ``min_target_clusters`` clusters,
      whose public-guard LCBs clear ``−tau_public``, and whose level is
      ``alpha / k`` — ``k`` = the target's JUDGE-phase rows (Bonferroni) — or
      ``alpha`` for a CONFIRM-phase row;
    * per target, the judge-phase best passer (highest gain LCB; ties:
      earliest train_block, then hotkey) and only it is ``pending``; a
      ``deferred`` row is a first-time cluster-inconclusive verdict;
    * a confirm row is ``confirmed`` / ``below holder LCB`` iff it passed,
      the split re-derived from the recorded holder (strictly higher LCB on
      the same king, or a newer king, takes the slot), else ``dropped``;
    * the slot the payout record shows won at THIS block is exactly the
      ``confirmed`` row's (when its holder was paid).

    MarketCast-only lane C (owner 2026-10-20): every row is an ``mkt:`` row
    (MarketCast terms); the non-mkt branches here (cluster deferral, public
    guards) cannot fire, since no other target loads. The verdict itself is
    MarketCast's replay (``cascade.audit.marketcast_reproduce``) — Tier-1,
    not Tier-0. The pending chain across settlements:
    :func:`check_lane_c_chain`."""
    from ..shared.lane_c import competition_past_end

    name = "lane_c"
    lc = cfg.lane_c
    boundary = int(receipt.epoch_start_block)
    rows = getattr(receipt, "lane_c", None)
    pay = getattr(receipt, "lane_c_pay", None)
    if not lc.active(boundary):
        if rows is not None or pay is not None:
            return _fail(name, f"lane-C record on a settlement at {boundary}, before lane C "
                               f"([lane_c] from_block={lc.from_block})")
        return _skip(name, "lane C not active at this settlement")
    if receipt.status == "rejected" and rows is None:
        return _skip(name, "rejected round; lane C folds through the next settlement's walk")
    rid = str(getattr(receipt, "round_id", "") or "")
    vrows = [r for r in rows or () if "error" not in r]
    ks: dict[tuple, int] = {}
    for r in vrows:
        if str(r.get("phase", "judge")) == "judge":
            g = _lane_c_group(r)
            ks[g] = ks.get(g, 0) + 1
    mkt_fams = _mkt_owner_families(vrows)
    best: dict[tuple, dict] = {}
    confirmed: dict[str, dict] = {}
    for r in vrows:
        t, stage = str(r.get("target")), str(r.get("stage"))
        phase, outcome = str(r.get("phase", "judge")), str(r.get("outcome", "") or "")
        who = f"row {r.get('hotkey')} ({phase})"
        terms = _lane_c_row_terms(t, lc, cfg.marketcast)
        if stage not in LANE_C_STAGES + (MKT_STAGES if terms["mkt"] else ()):
            return _fail(name, f"{who}: unknown stage {stage!r}")
        if phase not in ("judge", "confirm") or outcome not in LANE_C_OUTCOMES:
            return _fail(name, f"{who}: unknown phase/outcome {phase!r}/{outcome!r}")
        if stage != "target" and t not in lc.targets:
            return _fail(name, f"{who}: target {t!r} is not configured")
        comp = lc.competition_of(t)
        if comp is not None and competition_past_end(comp, boundary) and (
                r.get("passed") or stage != "ended" or outcome not in ("", "dropped")):
            # concurrent competitions: after its until_block a competition
            # judges nothing — every row is an uncounted "ended" refusal
            return _fail(name, f"{who}: competition {comp.kind!r} ended at block "
                               f"{comp.until_block}, but the row is {stage!r}/{outcome!r}")
        if stage == "ended" and (comp is None or not competition_past_end(comp, boundary)):
            return _fail(name, f"{who}: an 'ended' row for a competition that has not ended")
        mk = dict(r.get("mkt") or {}) if terms["mkt"] else {}
        registered = mk.get("at") == "register"
        infra_waits = bool(terms["mkt"] and stage == "infra" and not mk.get("error_class"))
        waits = bool(terms["mkt"] and not r.get("passed") and stage == "verdict"
                     and mk.get("inconclusive") and mk.get("within_wait")) or infra_waits
        if terms["mkt"] and stage == "infra" and mk.get("error_class") and outcome != "dropped":
            return _fail(name, f"{who}: an infra drop past max_infra_deferrals is {outcome!r}")
        k = 1 if (phase != "judge" or registered) else (
            _mkt_owner_k(r, mkt_fams) if terms["mkt"] else ks.get(_lane_c_group(r), 1))
        want_alpha = float(terms["alpha"]) / k
        if r.get("alpha") is None or not math.isclose(float(r["alpha"]), want_alpha,
                                                      rel_tol=1e-12):
            return _fail(name, f"{who}: level {r.get('alpha')} != {want_alpha} (alpha / k, "
                               f"k = {k})")
        if phase == "confirm":
            if outcome not in (("confirmed", "below holder LCB") if r.get("passed")
                               else (("pending",) if waits else ("dropped",))):
                return _fail(name, f"{who}: outcome {outcome!r} inconsistent with passed="
                                   f"{bool(r.get('passed'))}")
            if r.get("passed"):
                kept = _lane_c_holder_kept(r, boundary, rid)
                if kept != (outcome == "below holder LCB"):
                    return _fail(name, f"{who}: outcome {outcome!r} but the recorded LCBs "
                                       f"say the slot {'stays' if kept else 'moves'}")
                if outcome == "confirmed":
                    # Only the final slot holder is a confirmed winner, even
                    # when several backlog rounds resolve at this settlement.
                    if t in confirmed:
                        return _fail(name, f"target {t}: two confirmed rows")
                    confirmed[t] = r
        else:
            if outcome in ("confirmed", "below holder LCB"):
                return _fail(name, f"{who}: confirm-only outcome {outcome!r}")
            if outcome == "deferred" and not waits and (
                    terms["mkt"] or r.get("passed") or r.get("origin_round")
                    or int(r.get("clusters", 0) or 0) >= int(lc.min_target_clusters)
                    or stage != "verdict"):
                return _fail(name, f"{who}: deferred without a first-time cluster-"
                                   f"inconclusive verdict")
        if not r.get("passed"):
            if outcome == "pending" and not (phase == "confirm" and waits):
                return _fail(name, f"{who}: pending without a pass")
            continue
        lcb, gain, n = r.get("gain_lcb"), r.get("gain"), int(r.get("n", 0) or 0)
        if stage != "verdict" or lcb is None or gain is None or not math.isfinite(float(gain)):
            return _fail(name, f"{who} passed without a verdict / finite gain")
        if not float(gain) > 0.0:
            return _fail(name, f"{who} passed with a non-positive point gain {gain}")
        if not float(lcb) >= float(terms["margin"]):
            return _fail(name, f"{who} passed with gain LCB {lcb} < margin {terms['margin']}")
        if n < int(terms["min_windows"]):
            return _fail(name, f"{who} passed on {n} < {terms['min_windows']} windows")
        if int(r.get("clusters", 0) or 0) < int(terms["min_clusters"]):
            return _fail(name, f"{who} passed on {r.get('clusters')} < "
                               f"{terms['min_clusters']} clusters")
        if terms["mkt"] and mk.get("coverage_passed") is False:
            return _fail(name, f"{who} passed with a failed coverage gate")
        for g, v in (r.get("public_lcbs") or {}).items():
            if not float(v) >= -float(lc.tau_public):
                return _fail(name, f"{who} passed with public guard {g} LCB "
                                   f"{v} < -{lc.tau_public}")
        missing = [s_ for s_ in lc.guard_suites(t)
                   if not any(str(k).startswith(f"{s_}_") for k in (r.get("public_lcbs") or {}))]
        if missing:
            return _fail(name, f"{who} passed without the public guard "
                               f"{missing} configured for {t}")
        if phase == "confirm":
            # gain-tiered full-pay (owner 2026-10-20): the tier a confirmed
            # win records, re-derived from this row's gain LCB
            from ..eval.expert_lane import full_pay_mismatch
            from ..shared.lane_c import full_pay_tiers_for

            fp = dict(r.get("full_pay") or {})
            fp["gain_lcb"] = lcb
            why = full_pay_mismatch(fp, full_pay_tiers_for(t, lc, cfg.marketcast))
            if why:
                return _fail(name, f"{who}: full-pay tier forged: {why}")
        if phase == "judge":
            g = _lane_c_group(r)
            cur = best.get(g)
            if cur is None or _lane_c_best_key(r) < _lane_c_best_key(cur):
                best[g] = r
    for r in vrows:
        if str(r.get("phase", "judge")) != "judge":
            continue
        t, is_best = str(r.get("target")), best.get(_lane_c_group(r)) is r
        if is_best != (str(r.get("outcome", "") or "") == "pending"):
            return _fail(name, f"target {t}: the pending entry is not the best passer "
                               f"(highest gain LCB) — row {r.get('hotkey')}")
    paid_now = {str(w.get("target")): w for w in ((pay or {}).get("wins") or ())
                if int(w.get("win_block", -1)) == boundary}
    for t, w in paid_now.items():
        if t not in confirmed:
            return _fail(name, f"target {t} shows a win at this settlement with no confirmed "
                               f"row")
        if str(w.get("hotkey")) != str(confirmed[t].get("hotkey")):
            return _fail(name, f"target {t}: slot went to {w.get('hotkey')}, the confirmed "
                               f"fine-tune is {confirmed[t].get('hotkey')}")
        # gain-tiered full-pay (owner 2026-10-20): the slot's tier is the
        # confirmed row's, derived from that row's gain LCB
        c_lcb = confirmed[t].get("gain_lcb")
        if w.get("gain_lcb") is None or c_lcb is None or float(w["gain_lcb"]) != float(c_lcb):
            return _fail(name, f"target {t}: pay record gain_lcb {w.get('gain_lcb')!r} != the "
                               f"confirmed row's {c_lcb!r}")
        if {k: w.get(k) for k in ("tier", "full_eras", "below_floor")} != {
                k: (confirmed[t].get("full_pay") or {}).get(k)
                for k in ("tier", "full_eras", "below_floor")}:
            return _fail(name, f"target {t}: pay record tier {w.get('tier')!r} / full_eras "
                               f"{w.get('full_eras')!r} != the confirmed row's full_pay "
                               f"{confirmed[t].get('full_pay')!r}")
    unpaid = sorted(set(confirmed) - set(paid_now))
    note = f" ({len(unpaid)} winner(s) unpaid: deregistered)" if unpaid and pay else ""
    return _ok(name, f"{len(rows or ())} lane-C row(s), {len(best)} pending, "
                     f"{len(confirmed)} confirmed win(s) consistent with the payout "
                     f"record{note}")


def _mkt_owner(r: dict) -> str:
    return str(dict(r.get("mkt") or {}).get("owner") or f"hk:{r.get('hotkey')}")


def _mkt_owner_families(rows) -> dict[tuple, dict[str, int]]:
    """Red-team F6 (DEC-CA-0064): per MarketCast family group
    (:func:`_lane_c_group`) the judge-phase entries per OWNER (``mkt.owner``:
    the coldkey the trainer stamped, else the hotkey)."""
    out: dict[tuple, dict[str, int]] = {}
    for r in rows:
        if str(r.get("target", "")).startswith("mkt:") and str(r.get("phase", "judge")) == "judge":
            o = out.setdefault(_lane_c_group(r), {})
            o[_mkt_owner(r)] = o.get(_mkt_owner(r), 0) + 1
    return out


def _mkt_owner_k(r: dict, fams: dict) -> int:
    """``k = C × m`` of an ``mkt:`` judge row: the group's distinct owners ×
    this owner's entries (a weighted Bonferroni whose levels sum to alpha)."""
    o = fams.get(_lane_c_group(r), {})
    return max(1, len(o) * o.get(_mkt_owner(r), 1))


def _lane_c_group(r: dict) -> tuple:
    """The Bonferroni family / best-passer group of a verdict row: its
    target — and, for a DEC-CA-0064 ``mkt:`` row, its MarketCast round and
    whether it was a registration row (a settlement may apply several
    MarketCast rounds)."""
    t = str(r.get("target"))
    if not t.startswith("mkt:"):
        return (t,)
    mk = dict(r.get("mkt") or {})
    return (t, str(mk.get("round", "")), str(mk.get("at", "")))


def _lane_c_row_terms(target: str, lc, mc) -> dict:
    """The verdict terms a lane-C row is checked against: ``[lane_c]``'s,
    or ``[marketcast]``'s for a DEC-CA-0064 ``mkt:`` row."""
    if str(target).startswith("mkt:") and mc is not None:
        return {"mkt": True, "alpha": float(mc.alpha), "margin": float(mc.gain_margin),
                "min_windows": int(mc.min_windows), "min_clusters": int(mc.min_clusters)}
    return {"mkt": False, "alpha": float(lc.alpha), "margin": float(lc.gain_margin),
            "min_windows": int(lc.min_target_windows),
            "min_clusters": int(lc.min_target_clusters)}


def _lane_c_carry_key(r: dict) -> tuple:
    return (str(r.get("target")), str(r.get("hotkey")), str(r.get("trained_pointer")))


def _mkt_chain_step(open_: dict, rows: list[dict], rid: str) -> str:
    """One receipt of the DEC-CA-0064 MarketCast carry check (``open_``
    mutated): a MarketCast entry left ``pending`` / ``deferred`` is next
    seen — in whichever later receipt applies its round — as a confirm /
    carried row naming the same origin, exactly once. Carries opened before
    the receipts given are unknown and pass (a MarketCast round may apply
    several settlements later). "" = consistent."""
    for r in rows:
        key = _lane_c_carry_key(r)
        phase = str(r.get("phase", "judge"))
        if phase == "confirm" or r.get("origin_round"):
            kind = "confirm" if phase == "confirm" else "deferred"
            want = open_.pop(key, None)
            if want is not None and want != (kind, str(r.get("origin_round", ""))):
                return (f"round {rid}: MarketCast {kind} row {key} names origin "
                        f"{r.get('origin_round')!r}; it was left {want}")
    for r in rows:
        out = str(r.get("outcome", "") or "")
        mk = dict(r.get("mkt") or {})
        if out == "pending":
            open_[_lane_c_carry_key(r)] = ("confirm", str(mk.get("round", "")))
        elif out == "deferred":
            open_[_lane_c_carry_key(r)] = ("deferred", str(r.get("origin_round")
                                                           or mk.get("round", "")))
    return ""


def check_lane_c_chain(receipts, cfg: ChainConfig) -> CheckResult:
    """DEC-CA-0056 pending continuity across CONSECUTIVE lane-C receipts
    (``receipts`` in chain order, no gaps): every entry a settlement left
    ``pending`` is the next settlement's confirm row (originating there),
    every ``deferred`` one its carried judge row, and nothing is confirmed or
    carried that was never pending / deferred. A settlement without a king
    (``no_king`` error row) carries the pending set over untouched."""
    name = "lane_c_chain"
    carry: dict[tuple, tuple[str, str]] = {}      # key -> (kind, origin round)
    mkt_open: dict[tuple, tuple[str, str]] = {}   # DEC-CA-0064 MarketCast carries
    n = 0
    for rc in receipts:
        if not cfg.lane_c.active(int(rc.epoch_start_block)):
            continue
        n += 1
        rows = list(getattr(rc, "lane_c", None) or ())
        if any(r.get("error") == "no_king" for r in rows):
            # the pending set waits for a king; MarketCast never does
            bad = _mkt_chain_step(mkt_open, [r for r in rows if "error" not in r and str(
                r.get("target", "")).startswith("mkt:")], str(getattr(rc, "round_id", "")))
            if bad:
                return _fail(name, bad)
            continue
        rid = str(getattr(rc, "round_id", "") or "")
        vrows = [r for r in rows if "error" not in r]
        bad = _mkt_chain_step(mkt_open, [r for r in vrows
                                         if str(r.get("target", "")).startswith("mkt:")], rid)
        if bad:
            return _fail(name, bad)
        vrows = [r for r in vrows if not str(r.get("target", "")).startswith("mkt:")]
        got_confirm = {_lane_c_carry_key(r): str(r.get("origin_round", ""))
                       for r in vrows if r.get("phase") == "confirm"}
        got_carried = {_lane_c_carry_key(r): str(r.get("origin_round", ""))
                       for r in vrows if r.get("phase", "judge") == "judge"
                       and r.get("origin_round")}
        want_confirm = {k: o for k, (kind, o) in carry.items() if kind == "confirm"}
        want_carried = {k: o for k, (kind, o) in carry.items() if kind == "deferred"}
        if got_confirm != want_confirm:
            return _fail(name, f"round {rid}: confirm rows {sorted(got_confirm)} != the "
                               f"previous settlement's pending {sorted(want_confirm)}")
        if got_carried != want_carried:
            return _fail(name, f"round {rid}: carried rows {sorted(got_carried)} != the "
                               f"previous settlement's deferred {sorted(want_carried)}")
        carry = {}
        for r in vrows:
            out = str(r.get("outcome", "") or "")
            if out == "pending":
                carry[_lane_c_carry_key(r)] = ("confirm", rid)
            elif out == "deferred":
                carry[_lane_c_carry_key(r)] = ("deferred", str(r.get("origin_round") or rid))
    return _ok(name, f"pending chain consistent over {n} lane-C settlement(s)")


def check_weights(
    receipt: RoundReceipt, cfg: ChainConfig, client: object | None = None
) -> CheckResult:
    """The recorded weight vector is exactly the geometrically-decayed share of
    the recorded reward UIDs (``[scoring] king_decay``); compared against
    on-chain weights when reachable."""
    name = "weights"
    if receipt.status == "rejected":
        return _skip(name, "rejected round; no weights expected")
    if not receipt.weights:
        return _warn(name, "no weight vector recorded (the weight extrinsic failed that "
                           "round and is retried the next)")
    shaped = _check_emission_weights(name, receipt, cfg)
    if shaped is not None:
        return shaped
    if cfg.lane_c.active(receipt.epoch_start_block):
        # DEC-CA-0056: every lane-C vector carries ``lane_c_pay`` (rebuilt
        # above); only a pure burn (force_burn) may omit it
        burn = cfg.scoring.burn_uid
        if any(float(w) > 0 for u, w in enumerate(receipt.weights) if u != burn):
            return _fail(name, "lane C is live at this settlement but the receipt carries no "
                               "lane-C payout record to rebuild the vector")
        return _warn(name, "lane C live but the vector is a pure burn (force_burn?) with no "
                           "lane-C payout record")
    want = decayed_share_vector(
        list(receipt.reward_uids), len(receipt.weights),
        decay=cfg.scoring.king_decay, burn_uid=cfg.scoring.burn_uid,
    )
    # Selected by the CONFIG at the receipt's block, never by row presence: a
    # lane-B validator's resync receipt once carried no rows, and a strict
    # lane-A comparison then false-FAILed its (correct) lane-B vector.
    if cfg.lane_b.active(receipt.epoch_start_block):
        return _check_lane_b_weights(name, receipt, cfg, want)
    if [float(w) for w in receipt.weights] != want:
        return _fail(name, f"recorded weights {list(receipt.weights)} != "
                           f"decayed_share_vector({list(receipt.reward_uids)}, "
                           f"decay={cfg.scoring.king_decay}) = {want}")
    # The round's vote mirrors the validator's _king_uid_to_vote: the MANIFEST
    # king holds the throne vote unless this round dethroned, in which case the
    # new (state) king takes it. The state king can legitimately differ from
    # the manifest king on a no-dethrone round (see check_transition).
    v = receipt.verdict
    if v is not None and not receipt.reward_uids:
        # A scored round that deliberately burned: either the king was
        # unregistered at vote time, or the operator ran [validator]
        # force_burn. The vector already recomputed above (empty uids ⇒ the
        # burn share), so this is consistent — surface it, don't fail it.
        return _warn(name, "scored round recorded a deliberate burn (empty "
                           "reward_uids): king unregistered at vote time, or "
                           "the validator ran with [validator] force_burn")
    if v is not None:
        if v.dethroned:
            voted = v.king_uid
        else:
            try:
                king = receipt.load_embedded_manifest().entry_for_role("king")
            except (ValueError, KeyError):
                king = None
            voted = king.miner_uid if king is not None else v.king_uid
        if voted is not None and voted not in receipt.reward_uids:
            return _fail(name, f"the round's king vote (uid {voted}) is not among "
                               f"reward_uids {list(receipt.reward_uids)}")
    if client is None:
        return _warn(name, "vector recomputes; on-chain weights not compared "
                           "(no chain connection)")
    note = _weights_chain_note(receipt, client)
    if note.startswith("WARN:"):
        return _warn(name, f"vector recomputes, but {note[5:]}")
    return _ok(name, f"equal-share vector over uids {list(receipt.reward_uids)}{note}")


def _weights_chain_note(receipt: RoundReceipt, client: object) -> str:
    """Best-effort comparison against the validator's current on-chain row.

    Weights are u16-normalised on chain and overwritten by later rounds, so
    only the *support* (which UIDs carry weight) is compared, and only when the
    signer's row is addressable. A mismatch is a WARN, never a FAIL: the chain
    is stale in both directions — a freshly set row lags inclusion (the
    extrinsic is async; commit-reveal delays it further) and a newer round
    overwrites older receipts' rows. The falsifying weight checks are the pure
    recomputations above.
    """
    try:
        row = client.weights_for_hotkey(receipt.validator_hotkey)  # may not exist
    except AttributeError:
        return "; on-chain weight comparison not supported by this client"
    except Exception as e:  # noqa: BLE001
        return f"; on-chain weights unavailable ({e})"
    if row is None:
        return "; validator has no on-chain weight row (not registered here?)"
    got_support = {i for i, w in enumerate(row) if w > 0}
    want_support = {i for i, w in enumerate(receipt.weights) if w > 0}
    if got_support != want_support:
        return (f"WARN:on-chain weight support {sorted(got_support)} != receipt "
                f"{sorted(want_support)} — inclusion lag or a later round moved "
                "weights; compare the newest receipt")
    return "; on-chain weight support matches"


def derive_weights(receipt: RoundReceipt, cfg: ChainConfig) -> tuple[list[float] | None, str]:
    """DEC-CA-0058: the weight vector rebuilt from the receipt's records
    (reward UIDs, the DEC-CA-0055 ``emission`` policy, ``lane_b_pay``,
    ``lane_c_pay``) with the same builders :func:`check_weights` compares
    against, over ``len(receipt.weights)`` UIDs. ``(vector, "")``, or
    ``(None, why)`` when the receipt records no vector or one that cannot be
    rebuilt. ``cfg`` must be the config the receipt was judged under (its
    DEC-CA-0045 blocks and DEC-CA-0057 overlay applied). A satellite sets
    this vector only after :func:`check_weights` passed on the same receipt
    and the two agree uid for uid."""
    from ..shared.emission import EmissionPolicy, lane_parts

    if not receipt.weights:
        return None, "no weight vector recorded"
    n = len(receipt.weights)
    burn = cfg.scoring.burn_uid
    boundary = int(receipt.epoch_start_block)
    rec = getattr(receipt, "emission", None)
    if rec is not None or getattr(receipt, "lane_c_pay", None) is not None:
        pol = (EmissionPolicy.from_record(rec.get("policy") or {}) if rec is not None
               else EmissionPolicy())
        want, _, why = _lane_vector_want(receipt, cfg, pol, n)
        return (want, "") if why is None else (None, why)
    burn_vec = decayed_share_vector([], n, decay=cfg.scoring.king_decay, burn_uid=burn)
    if cfg.lane_c.active(boundary):
        # only a pure burn (force_burn) may omit the lane-C record
        if [float(w) for w in receipt.weights] == burn_vec:
            return burn_vec, ""
        return None, "lane C live but no lane-C payout record"
    if cfg.lane_b.active(boundary):
        if receipt.lane_b_pay is None:
            return None, "lane-B vector without a payout record cannot be rebuilt"
        wins, why = _lane_b_paid_wins(receipt, cfg, n)
        if why is not None:
            return None, why
        a, b = lane_parts(list(receipt.reward_uids), n, king_decay=cfg.scoring.king_decay,
                          burn_uid=burn, lane_b_cfg=cfg.lane_b, wins=wins)
        return [x + y for x, y in zip(a, b, strict=True)], ""
    return decayed_share_vector(list(receipt.reward_uids), n,
                                decay=cfg.scoring.king_decay, burn_uid=burn), ""


def check_status(receipt: RoundReceipt) -> CheckResult:
    """Surface the receipt's own claim (a rejected round is a valid, auditable
    public record — the reason is signed)."""
    name = "status"
    if receipt.status == "rejected":
        return _ok(name, f"rejected round; signed reason: {receipt.reject_reason!r}")
    return _ok(name, "scored round")


# ── tier runners ──────────────────────────────────────────────────────────────


# On a status=rejected receipt, the check that re-detects the recorded
# rejection reason CONFIRMS the validator's gate rather than contradicting the
# receipt — its FAIL is converted to a PASS noting the confirmation.
_REJECTION_CHECK_FOR_REASON = (
    ("signature_invalid", "manifest-signature"),
    ("era_", "era"),
    ("manifest_chain_broken", "era"),
    ("contract_digest_mismatch", "contract-digest"),
    ("base_arch_digest_mismatch", "base-arch-digest"),
)


def _confirm_rejection(receipt: RoundReceipt, results: list[CheckResult]) -> list[CheckResult]:
    if receipt.status != "rejected" or not receipt.reject_reason:
        return results
    confirmed = {check for reason, check in _REJECTION_CHECK_FOR_REASON
                 if receipt.reject_reason.startswith(reason)}
    return [
        CheckResult(r.name, PASS, f"re-detects the recorded rejection "
                                  f"({receipt.reject_reason!r}); the gate was right")
        if r.status == FAIL and r.name in confirmed else r
        for r in results
    ]


def check_funded_roster(receipt: RoundReceipt,
                        roster: dict | None) -> CheckResult:
    """Seniority evidence for a funded round, against the trainer's UNSIGNED
    ``funded/round-<id>.json`` — so this WARNs, never FAILs.

    Boundary-synchronous roster (``seated`` / ``waiting``): every manifest
    challenger seated; ``seated`` in reveal-block order; no waiting entry
    outranks a seated one. Rolling roster (DEC-CA-0043, ``mode = "rolling"``
    with ``rents``): seniority is "earliest reveal gets first pick of FITTING
    executors", not strict serialization — an earlier entry waiting for an
    H100 under the price cap is legitimately overtaken by a later one that
    fits a 4090 now. The claim is therefore per rent: no more-senior QUEUED
    entry that fit the same executors was passed over (``passed_over``
    empty), and every settled challenger was seated. Reveal order among the
    settled legs is not a claim (legs land when their walls end)."""
    name = "funded-roster"
    if not roster:
        return _skip(name, "no funded roster published for this round")
    # receipt.manifest is the embedded RAW manifest dict; entries are dicts.
    challengers = [str(e.get("miner_hotkey", "?"))
                   for e in (receipt.manifest.get("entries") or ())
                   if e.get("role") == "challenger"]
    seated = roster.get("seated") or []
    seated_hk = {str(s.get("hotkey")) for s in seated if isinstance(s, dict)}
    unseated = [hk for hk in challengers if hk not in seated_hk]
    if unseated:
        return _warn(name, f"manifest challenger(s) not on the seated roster: {unseated}")
    if str(roster.get("mode", "")) == "rolling":
        jumped = [(r.get("hotkey"), r.get("passed_over")) for r in (roster.get("rents") or [])
                  if isinstance(r, dict) and r.get("passed_over")]
        if jumped:
            return _warn(name, f"rent(s) passed over a more-senior fitting entry: "
                               f"{jumped[:3]}{'…' if len(jumped) > 3 else ''}")
        return _ok(name, f"{len(seated)} settled leg(s) seated; every rent took the "
                         f"most senior fitting entry")
    order = [(int(s["reveal_block"]), str(s["hotkey"])) for s in seated
             if isinstance(s, dict) and s.get("reveal_block") is not None]
    if order != sorted(order):
        return _warn(name, f"seated order is not reveal-block order: {order}")
    waiting = [int(w["reveal_block"]) for w in (roster.get("waiting") or [])
               if isinstance(w, dict) and w.get("reveal_block") is not None]
    if order and waiting and min(waiting) < max(b for b, _ in order):
        return _warn(name, f"a waiting entry (reveal {min(waiting)}) had seniority over "
                           f"a seated one (reveal {max(b for b, _ in order)}) — the queue "
                           f"was jumped")
    return _ok(name, f"{len(seated)} seated in reveal order; {len(waiting)} waiting behind")

def _mkt_train_time(r: dict, m: dict, cfg: ChainConfig, manifests: dict | None,
                    cache: dict) -> tuple[float | None, str]:
    """``(train_time, source)`` of a MarketCast row: from the TRAINER-signed
    lane-C manifest of the round that carried the entry (``m["entry_round"]``,
    signature verified) when the auditor fetched it, else the row's own copy
    (``source`` = "row": the validator's word, reported as unverified)."""
    from ..shared.lane_c_manifest import load_lane_c_manifest, verify_lane_c_manifest

    er = str(m.get("entry_round", "") or "")
    text = (manifests or {}).get(er) if er else None
    if text is not None:
        cm = cache.get(er)
        if cm is None:
            try:
                cm = load_lane_c_manifest(text)
                ok = verify_lane_c_manifest(cm, str(cfg.manifest.trainer_hotkey or ""))
            except Exception:  # noqa: BLE001 — unparseable / no SDK: fall back to the row
                ok = False
            cm = cache[er] = cm if ok else False
        if cm:
            for e in cm.entries:
                if (e.miner_hotkey, e.trained_pointer) == (str(r.get("hotkey")),
                                                           str(r.get("trained_pointer"))):
                    from ..validator.marketcast import mkt_owner

                    # red-team F6: the owner the trainer stamped (coldkey)
                    cache[("owner", e.miner_hotkey, e.trained_pointer)] = mkt_owner(e)
                    t = dict(e.marketcast or {}).get("train_time")
                    return (float(t), "manifest") if t is not None else (None, "manifest")
    t = m.get("train_time")
    return (None if t is None else float(t)), "row"


def check_marketcast(receipt: RoundReceipt, cfg: ChainConfig,
                     reports: dict | None = None, *, lane_c_manifests: dict | None = None,
                     quantiles: dict | None = None, snapshots: dict | None = None
                     ) -> CheckResult:
    """DEC-CA-0064 MarketCast replay (joins :func:`run_tier0` ONLY at a
    settlement where ``[marketcast]`` is active, so earlier audit output is
    byte-identical). Nothing here trusts the main validator's arithmetic.

    Every ``mkt:`` lane-C row of an applied MarketCast round is checked
    against the main validator's signed report it binds (``reports``:
    ``{key: text}``): the text's sha256 is the bound one; the report verifies
    against ``[netconfig] owner_hotkey`` and names the row's round. Then:

    * ``after`` is the TRAINER's dispatch time + ``start_margin_seconds``
      (``train_time`` from the trainer-signed lane-C manifest that carried
      the entry, ``lane_c_manifests``: ``{round: text}``; and on a confirm,
      ≥ the recorded ``fresh_after``) — never the snapshot anchor;
    * with the published snapshot (``snapshots``: ``{digest: MktSnapshot}``)
      the report's window set must EQUAL the forward-only set
      :func:`~cascade.pool.marketcast.snapshot_windows` re-cuts (nothing
      omitted, nothing added); without it, every window must be forward-only
      and no (asset, horizon) cell hold more than ``windows_per_cell``;
    * with the quantile sidecar (``quantiles``: ``{key: bytes}``, sha256
      bound in the signed report) AND the snapshot, every per-window loss
      and coverage tally is recomputed on CPU from the snapshot targets and
      must match exactly (:func:`~cascade.shared.marketcast_report.
      score_windows`);
    * the verdict re-derived from the rows (``verdict_from_rows`` at the
      recorded ``k``: paired geomean LCB, coverage gate on ``mkt:returns``,
      min windows / clusters) reproduces the row's pass, gain LCB, windows
      and clusters exactly;
    * an ``infra`` row is consistent only when no usable rows existed (no /
      unverifiable report, the entry missing or ``infra_deferred``, or rows
      cut after another instant);
    * red-team fixes (2026-10-20): the tail tallies are recomputed with the
      losses; the row's ``masked_cells`` must be the re-derived count (F3);
      a judge-phase row's ``k`` must be ``C × m`` over the round's owners
      (F6); a signed ``near_dup`` record must be the row's, its distance is
      recomputed from the two sidecars (F4), and the stage must follow
      ``near_dup_tol`` (a near-duplicate never reaches a verdict; a
      ``duplicate`` row on ``near_dup`` alone must be below it).

    PASS needs every input; a missing one WARNs with what was not verified.
    While inactive, a receipt may carry no ``mkt:`` row."""
    name = "marketcast"
    mc = cfg.marketcast
    boundary = int(receipt.epoch_start_block)
    rows = getattr(receipt, "lane_c", None) or ()
    mkt = [r for r in rows if str(r.get("target", "")).startswith("mkt:")]
    if not mc.active(boundary):
        if getattr(receipt, "marketcast", None) is not None or mkt:
            return _fail(name, "MarketCast record on a settlement while [marketcast] is inert")
        return _skip(name, "MarketCast not active at this settlement")
    if not mkt:
        return _ok(name, "no MarketCast rows at this settlement")
    from ..eval.marketcast import COVERAGE_GATE_SUITES
    from ..pool.marketcast import snapshot_windows
    from ..shared.marketcast_report import (
        behaviour_distance,
        decode_quantiles,
        load_mkt_report,
        mkt_text_digest,
        score_windows,
        verdict_from_rows,
        verify_mkt_report,
    )
    from ..validator.marketcast import anchor_time

    replayed, recomputed = 0, 0
    missing: dict[str, int] = {}
    notes: list[str] = []
    parsed: dict[str, object] = {}
    mcache: dict = {}

    def _miss(what: str) -> None:
        missing[what] = missing.get(what, 0) + 1

    # red-team F6: per MarketCast round + target the judge-phase family by owner
    fam = _mkt_owner_families([r for r in mkt if "error" not in r])
    for r in mkt:
        m = dict(r.get("mkt") or {})
        who = f"row {r.get('hotkey')} {r.get('target')} ({r.get('phase')})"
        stage = str(r.get("stage"))
        if m.get("at") == "register" or stage not in ("verdict", "infra", "duplicate"):
            continue
        if stage == "duplicate" and not m.get("near_dup"):
            continue                      # exact-identity duplicates: not a report verdict
        if str(r.get("phase")) == "judge" and m.get("round"):
            want_k = _mkt_owner_k(r, fam)
            if int(r.get("k", 1) or 1) != want_k:
                return _fail(name, f"{who}: k={r.get('k')} but the round's family by owner "
                                   f"gives {want_k} (red-team F6)")
        key = str(m.get("report_key", "") or "")
        text = (reports or {}).get(key) if key else None
        rep, ent = None, None
        if text is not None and m.get("report_sha256"):
            if mkt_text_digest(text) != str(m.get("report_sha256", "")):
                return _fail(name, f"{who}: report {key} sha256 is not the bound one")
            rep = parsed.get(key)
            if rep is None:
                try:
                    rep = load_mkt_report(text)
                except (ValueError, KeyError, TypeError) as e:
                    if stage == "verdict":
                        return _fail(name, f"{who}: report {key} unparseable: {e}")
                    rep = False
                if rep:
                    try:
                        ok = verify_mkt_report(rep, str(cfg.netconfig.owner_hotkey or ""))
                    except Exception:  # noqa: BLE001 — no SDK: cannot verify
                        return _warn(name, "the bittensor SDK is unavailable: report "
                                           "signatures not verified")
                    if not ok or str(rep.round_id) != str(m.get("round", rep.round_id)):
                        rep = False
                parsed[key] = rep
            if rep:
                ent = rep.entry_for(str(r.get("hotkey")), str(r.get("trained_pointer")),
                                    str(r.get("phase")))
        elif text is None and key and m.get("report_sha256"):
            _miss("report")
            continue
        elif stage in ("verdict", "duplicate"):
            return _fail(name, f"{who}: a {stage} without a bound report")
        tt, src = _mkt_train_time(r, m, cfg, lane_c_manifests, mcache)
        if src == "row":
            _miss("lane-C manifest (train_time)")
        own = mcache.get(("owner", str(r.get("hotkey")), str(r.get("trained_pointer"))))
        if own is not None and str(m.get("owner") or "") != own:
            return _fail(name, f"{who}: owner {m.get('owner')!r} but the trainer-signed lane-C "
                               f"manifest gives {own!r} (red-team F6)")
        want_after = None if tt is None else max(
            tt + float(mc.start_margin_seconds), float(m.get("fresh_after", 0.0) or 0.0))
        usable = bool(rep) and ent is not None and not ent.deferred and want_after is not None \
            and math.isclose(float(ent.after), want_after, rel_tol=0, abs_tol=1e-6)
        if stage == "infra":
            if usable:
                return _fail(name, f"{who}: deferred as infra although the signed report "
                                   f"carries usable rows for it")
            replayed += 1
            continue
        if not rep:
            if r.get("passed"):
                return _fail(name, f"{who} passed on a report that fails the main "
                                   f"validator's signature / round")
            continue
        if ent is None:
            if r.get("passed"):
                return _fail(name, f"{who} passed without rows in the report")
            continue
        if ent.fault:
            return _fail(name, f"{who}: a faulted entry was judged at the verdict stage")
        if want_after is None or not usable:
            return _fail(name, f"{who}: windows start after {ent.after}, the trainer's dispatch "
                               f"time gives {want_after}")
        # red-team F4: the near-duplicate record and the stage it implies
        from ..validator.marketcast import near_duplicate_of

        if dict(m.get("near_dup") or {}) != dict(ent.near_dup or {}):
            return _fail(name, f"{who}: near_dup {m.get('near_dup')} is not the signed "
                               f"report's {ent.near_dup}")
        is_dup = near_duplicate_of(ent, mc) is not None
        if is_dup != (stage == "duplicate"):
            return _fail(name, f"{who}: stage {stage} but the signed near_dup "
                               f"{ent.near_dup.get('mean_rel')} vs near_dup_tol "
                               f"{mc.near_dup_tol} makes it {'a' if is_dup else 'no'} duplicate")
        snap_rec = rep.snapshot
        if "anchor_time" in snap_rec and abs(anchor_time(int(r.get("train_block", 0)),
                                                         snap_rec) - float(tt)) > 3600:
            notes.append(f"{who}: the snapshot anchor puts train_block "
                         f"{abs(anchor_time(int(r.get('train_block', 0)), snap_rec) - tt):.0f}s "
                         f"from the trainer's train_time")
        nonfinite = [w.wid for w in ent.windows if not math.isfinite(float(w.ft))]
        if nonfinite:
            # review M11: a non-finite fine-tune forecast is a fault, never masked
            return _fail(name, f"{who}: fine-tune loss non-finite on {nonfinite[:2]} — that "
                               f"entry is a fault, not a verdict")
        cells: dict[tuple, int] = {}
        for w in ent.windows:
            if not (w.start > float(ent.after) and w.end <= int(snap_rec.get("end", 0))):
                return _fail(name, f"{who}: window {w.wid} is not forward-only "
                                   f"(after {ent.after}, snapshot end {snap_rec.get('end')})")
            c = (w.cluster, w.horizon)
            cells[c] = cells.get(c, 0) + 1
        over = [c for c, n in cells.items() if n > int(mc.windows_per_cell)]
        if over:
            return _fail(name, f"{who}: {over[:2]} hold more than windows_per_cell="
                               f"{mc.windows_per_cell} windows")
        suite = str(r["target"]).split(":", 1)[1]
        snap = (snapshots or {}).get(str(snap_rec.get("digest", "")))
        if snap is None:
            _miss("snapshot")
        else:
            want = snapshot_windows(snap, mc, suite, after=float(ent.after))
            got_ids = [w.wid for w in ent.windows]
            want_ids = [w.wid for w in want]
            if got_ids != want_ids:
                gone = sorted(set(want_ids) - set(got_ids))
                extra = sorted(set(got_ids) - set(want_ids))
                return _fail(name, f"{who}: the report's windows are not the snapshot's "
                                   f"forward-only set (omitted {gone[:2]}{'…' if len(gone) > 2 else ''}"
                                   f" {len(gone)}, added {extra[:2]} {len(extra)})")
            data = (quantiles or {}).get(ent.quantiles_key) if ent.quantiles_key else None
            if data is None:
                _miss("quantile sidecar")
            else:
                import hashlib

                if hashlib.sha256(bytes(data)).hexdigest() != ent.quantiles_sha256:
                    return _fail(name, f"{who}: quantile sidecar {ent.quantiles_key} is not "
                                       f"the bound one")
                try:
                    qs = decode_quantiles(data, [w.steps for w in want],
                                          len(mc.quantile_knots))
                except ValueError as e:
                    return _fail(name, f"{who}: quantile sidecar unusable: {e}")
                again = score_windows([w.target for w in want], [q[0] for q in qs],
                                      [q[1] for q in qs], knots=tuple(mc.quantile_knots),
                                      coverage_levels=tuple(rep.coverage_levels
                                                            or mc.coverage_levels),
                                      gated=suite in COVERAGE_GATE_SUITES,
                                      tail_levels=tuple(rep.tail_levels or mc.tail_levels))
                for w, (b, f, ci, cn, tl) in zip(ent.windows, again, strict=True):
                    same = all((math.isnan(x) and math.isnan(y)) or x == y
                               for x, y in ((float(w.base), b), (float(w.ft), f)))
                    if (not same or tuple(w.cov_in) != tuple(ci) or int(w.cov_n) != int(cn)
                            or tuple(w.tail) != tuple(tl)):
                        return _fail(name, f"{who}: window {w.wid} recomputes to base={b} "
                                           f"ft={f} cov={ci}/{cn} tail={tl}; the report says "
                                           f"base={w.base} ft={w.ft} cov={w.cov_in}/{w.cov_n} "
                                           f"tail={w.tail}")
                recomputed += 1
                nd = dict(ent.near_dup or {})
                if nd:
                    rdata = (quantiles or {}).get(str(nd.get("quantiles_key", "")))
                    if rdata is None:
                        _miss("near-dup reference sidecar")
                    else:
                        import numpy as np

                        if hashlib.sha256(bytes(rdata)).hexdigest() != nd.get(
                                "quantiles_sha256"):
                            return _fail(name, f"{who}: near-dup reference sidecar is not the "
                                               f"bound one")
                        try:
                            rq = decode_quantiles(rdata, [w.steps for w in want],
                                                  len(mc.quantile_knots))
                        except ValueError as e:
                            return _fail(name, f"{who}: near-dup reference sidecar unusable: {e}")
                        if any(not np.array_equal(a[0], b_[0]) for a, b_ in zip(rq, qs,
                                                                                 strict=True)):
                            return _fail(name, f"{who}: near-dup reference sidecar's base plane "
                                               f"is not the entry's")
                        dm, dx = behaviour_distance([q[0] for q in qs], [q[1] for q in qs],
                                                    [q[1] for q in rq])
                        if (dm, dx, len(want)) != (float(nd.get("mean_rel")),
                                                   float(nd.get("max_rel")),
                                                   int(nd.get("n_windows", -1))):
                            return _fail(name, f"{who}: near_dup recomputes to mean={dm} "
                                               f"max={dx} over {len(want)} windows; the report "
                                               f"says {nd}")
        if stage == "duplicate":
            replayed += 1
            continue
        v = verdict_from_rows(ent, mc, suite=suite, k=int(r.get("k", 1) or 1),
                              seed=f"{m.get('round', '')}|{r.get('hotkey')}|{r.get('target')}",
                              coverage_levels=rep.coverage_levels or None,
                              tail_levels=rep.tail_levels or None)
        if int(m.get("masked_cells", 0) or 0) != int(v.diagnostics.get("masked_cells", 0)):
            return _fail(name, f"{who}: masked_cells={m.get('masked_cells')} but the rows "
                               f"re-derive {v.diagnostics.get('masked_cells')} (red-team F3)")
        lcb = r.get("gain_lcb")
        same_lcb = (lcb is None and not math.isfinite(v.gain_lcb)) or (
            lcb is not None and math.isfinite(v.gain_lcb)
            and math.isclose(float(lcb), float(v.gain_lcb), rel_tol=1e-12, abs_tol=1e-15))
        if (bool(v.passed) != bool(r.get("passed")) or not same_lcb
                or int(v.n) != int(r.get("n", 0) or 0)
                or int(v.n_clusters) != int(r.get("clusters", 0) or 0)):
            return _fail(name, f"{who}: the rows re-derive passed={v.passed} LCB={v.gain_lcb} "
                               f"n={v.n} clusters={v.n_clusters} ({v.reason}); the row says "
                               f"passed={r.get('passed')} LCB={lcb} n={r.get('n')} "
                               f"clusters={r.get('clusters')}")
        replayed += 1
    detail = (f"{replayed} MarketCast row(s) replayed, {recomputed} with every loss recomputed "
              f"from the snapshot + quantile sidecar")
    if missing or notes:
        what = ", ".join(f"{n} {k}(s) not fetched" for k, n in sorted(missing.items()))
        return _warn(name, "; ".join(x for x in (detail, what, *notes[:2]) if x))
    return _ok(name, detail)


def run_tier0(
    receipt: RoundReceipt, cfg: ChainConfig, client: object | None = None,
    *, funded_roster: dict | None = None, bench: dict | None = None,
    king_bench: dict | None = None, validator_bench: dict | None = None,
    marketcast_reports: dict | None = None, marketcast_inputs: dict | None = None,
) -> list[CheckResult]:
    """All Tier-0 checks, in a stable order. CPU-only, seconds; ``client`` is an
    optional chain connection (None ⇒ the chain-dependent halves WARN).
    ``check_marketcast`` joins only at a settlement where MarketCast is active
    (DEC-CA-0064), so every earlier receipt's output is unchanged."""
    results = [
        check_status(receipt),
        check_receipt_signature(receipt, cfg),
        check_manifest_signature(receipt, cfg),
        check_base_seed(receipt),
        check_round_seeds(receipt, cfg),
        check_epoch_alignment(receipt, cfg),
        check_activation(receipt, cfg, client),
        check_margin_v2(receipt, cfg, client),
        check_v2(receipt, cfg),
        check_era(receipt, cfg, client),
        check_block_hash_onchain(receipt, client),
        check_contract_digest(receipt, cfg),
        check_contract_declaration(receipt, cfg),
        check_base_arch_digest(receipt, cfg),
        check_commit_cutoff(receipt, client),
        check_koth_params(receipt, cfg),
        check_verdict(receipt, cfg),
        check_duel_cohort(receipt, cfg),
        check_bench_gate(receipt, cfg, bench),
        check_bench_source(receipt, cfg, validator_bench),
        check_bench_coverage(receipt, cfg, validator_bench),
        check_funded_roster(receipt, funded_roster),
        check_transition(receipt),
        check_lane_b(receipt, cfg),
        check_lane_c(receipt, cfg),
        *([check_marketcast(receipt, cfg, marketcast_reports, **(marketcast_inputs or {}))]
          if cfg.marketcast.active(int(receipt.epoch_start_block)) else []),
        check_king_check(receipt, cfg, king_bench),
        check_weights(receipt, cfg, client),
    ]
    return _confirm_rejection(receipt, results)
