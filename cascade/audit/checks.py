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

import math
import re
from dataclasses import dataclass

from ..shared.chain import equal_share_vector, seed_from_block_hash
from ..shared.config import ChainConfig
from ..shared.manifest import TrainingManifest, contract_digest
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
    from ..trainer.contract import RoundSeeds

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
    return _ok(name, "generation + training seeds derive from base_seed")


def check_epoch_alignment(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """The recorded boundary sits on an epoch multiple (the submission deadline
    is deterministic, not validator-chosen)."""
    name = "epoch-alignment"
    epoch_blocks = max(1, cfg.round.epoch_blocks)
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


def check_contract_digest(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """The manifest's contract digest equals the one recomputed from the local
    ``chain.toml`` — the round trained under the published contract."""
    name = "contract-digest"
    want = contract_digest(cfg.training)
    got = str(receipt.manifest.get("contract_digest", ""))
    if got != want:
        return _fail(name, f"manifest contract_digest {got[:16]}… != local chain.toml "
                           f"{want[:16]}… (different training contract)")
    return _ok(name, f"contract_digest {want[:16]}… matches chain.toml")


def check_base_arch_digest(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """The frozen-architecture digests: the manifest matches the chain.toml pin,
    and the pin matches a recomputation from the local model source."""
    name = "base-arch-digest"
    from ..trainer.contract import compute_base_arch_digest

    got = str(receipt.manifest.get("base_arch_digest", ""))
    if got != cfg.training.base_arch_digest:
        return _fail(name, f"manifest base_arch_digest {got[:16]}… != chain.toml pin "
                           f"{cfg.training.base_arch_digest[:16]}…")
    for size in cfg.training.all_sizes():
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

    Only a definite contradiction (same hotkey, same block, different payload)
    is a FAIL; a hotkey that re-committed since the round is unverifiable on a
    lite node and noted as such.
    """
    from ..interface.validation import parse_commit

    try:
        commitments = client.poll_commitments()
    except Exception as e:  # noqa: BLE001
        return f"; chain payload cross-check unavailable ({e})"
    latest = {c.hotkey: c for c in commitments}
    unverifiable = 0
    for p in receipt.participants:
        c = latest.get(p.hotkey)
        if c is None:
            unverifiable += 1  # deregistered / lite node without the reveal
            continue
        parsed = parse_commit(c.payload)
        if c.commit_block == p.commit_block:
            if parsed is None or parsed.ref != p.gen_ref:
                return (f"FAIL:participant {p.hotkey} recorded gen_ref {p.gen_ref[:32]}… "
                        f"but chain shows {getattr(parsed, 'ref', c.payload)[:32]}… "
                        f"at the same block {p.commit_block}")
        else:
            unverifiable += 1  # re-committed since; history not addressable
    if unverifiable:
        return f"; {unverifiable} participant(s) re-committed since (history unverifiable)"
    return "; chain payloads match"


# ── verdict + weights ─────────────────────────────────────────────────────────


def _pooled_scores(receipt: RoundReceipt, manifest: TrainingManifest):
    """Rebuild the pooled king/challenger score lists exactly as the validator
    pooled them (paired sizes in manifest order, per-size scores concatenated)."""
    king_by_size = {e.size: e for e in manifest.entries_for_role("king")}
    chal_by_size = {e.size: e for e in manifest.entries_for_role("challenger")}
    paired = [s for s in manifest.sizes() if s in king_by_size and s in chal_by_size]
    recs = {(r.role, r.size): r for r in receipt.entry_scores}
    king, chal = [], []
    for size in paired:
        k = recs.get(("king", size))
        c = recs.get(("challenger", size))
        if k is None or c is None:
            raise KeyError(f"entry_scores missing for size {size!r}")
        king += [w.to_score() for w in k.scores]
        chal += [w.to_score() for w in c.scores]
    return king, chal, paired


def check_koth_params(receipt: RoundReceipt, cfg: ChainConfig) -> CheckResult:
    """The recorded decision parameters equal the published ``[scoring]``."""
    name = "koth-params"
    if receipt.verdict is None:
        return _skip(name, "no verdict on this receipt")
    from dataclasses import asdict

    want = dict(asdict(cfg.koth_params()))
    got = dict(receipt.verdict.params)
    if got != want:
        diff = {k for k in set(want) | set(got) if want.get(k) != got.get(k)}
        return _fail(name, f"recorded KothParams differ from chain.toml [scoring] on: "
                           f"{sorted(diff)}")
    return _ok(name, "recorded params match chain.toml [scoring]")


def check_verdict(receipt: RoundReceipt) -> CheckResult:
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

    result = evaluate_round(
        king, chal, params,
        seed=_bootstrap_seed(v.bootstrap_seed),
        king_tenure_rounds=v.king_tenure_rounds,
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
    if result.challenger_wins_round != v.challenger_wins_round:
        problems.append(f"challenger_wins_round {v.challenger_wins_round} != recomputed "
                        f"{result.challenger_wins_round}")
    if result.inconclusive != v.inconclusive:
        problems.append(f"inconclusive {v.inconclusive} != recomputed {result.inconclusive}")
    if problems:
        return _fail(name, "; ".join(problems))
    lcb_txt = "n/a" if v.lcb is None else f"{v.lcb:.5f}"
    return _ok(name, f"lcb={lcb_txt} margin={v.margin:.5f} win={v.challenger_wins_round} "
                     f"reproduced over {len(paired)} size(s){gate_note}")


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
    chal = manifest.entry_for_role("challenger")
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
        return _fail(name, f"no dethrone, but resulting king {v.king_hotkey!r} is not the "
                           f"manifest king {king.miner_hotkey!r}")
    if cp > 1:
        return _warn(name, f"dethrone_cp={cp}: streak state spans rounds; verify the "
                           "receipt chain for full confirmation")
    return _ok(name, f"transition '{v.note}' consistent with the verdict")


def check_weights(
    receipt: RoundReceipt, cfg: ChainConfig, client: object | None = None
) -> CheckResult:
    """The recorded weight vector is exactly ``equal_share_vector`` of the
    recorded reward UIDs; compared against on-chain weights when reachable."""
    name = "weights"
    if receipt.status == "rejected":
        return _skip(name, "rejected round; no weights expected")
    if not receipt.weights:
        return _warn(name, "no weight vector recorded (the weight extrinsic failed that "
                           "round and is retried the next)")
    want = equal_share_vector(
        list(receipt.reward_uids), len(receipt.weights), burn_uid=cfg.scoring.burn_uid
    )
    if [float(w) for w in receipt.weights] != want:
        return _fail(name, f"recorded weights {list(receipt.weights)} != "
                           f"equal_share_vector({list(receipt.reward_uids)}) = {want}")
    v = receipt.verdict
    if v is not None and v.king_uid is not None and v.king_uid not in receipt.reward_uids:
        return _fail(name, f"resulting king uid {v.king_uid} is not among reward_uids "
                           f"{list(receipt.reward_uids)}")
    if client is None:
        return _warn(name, "vector recomputes; on-chain weights not compared "
                           "(no chain connection)")
    note = _weights_chain_note(receipt, client)
    if note.startswith("FAIL:"):
        return _fail(name, note[5:])
    return _ok(name, f"equal-share vector over uids {list(receipt.reward_uids)}{note}")


def _weights_chain_note(receipt: RoundReceipt, client: object) -> str:
    """Best-effort comparison against the validator's current on-chain row.

    Weights are u16-normalised on chain and overwritten by later rounds, so
    only the *support* (which UIDs carry weight) is compared, and only when the
    signer's row is addressable; anything else is noted, not failed.
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
        return (f"FAIL:on-chain weight support {sorted(got_support)} != receipt "
                f"{sorted(want_support)} (a later round may have moved weights — "
                "check the newest receipt first)")
    return "; on-chain weight support matches"


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


def run_tier0(
    receipt: RoundReceipt, cfg: ChainConfig, client: object | None = None
) -> list[CheckResult]:
    """All Tier-0 checks, in a stable order. CPU-only, seconds; ``client`` is an
    optional chain connection (None ⇒ the chain-dependent halves WARN)."""
    results = [
        check_status(receipt),
        check_receipt_signature(receipt, cfg),
        check_manifest_signature(receipt, cfg),
        check_base_seed(receipt),
        check_round_seeds(receipt, cfg),
        check_epoch_alignment(receipt, cfg),
        check_block_hash_onchain(receipt, client),
        check_contract_digest(receipt, cfg),
        check_base_arch_digest(receipt, cfg),
        check_commit_cutoff(receipt, client),
        check_koth_params(receipt, cfg),
        check_verdict(receipt),
        check_transition(receipt),
        check_weights(receipt, cfg, client),
    ]
    return _confirm_rejection(receipt, results)
