"""Verify the immutable v2 baseline before allowing its first paid leg."""
from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from ..shared.atomic import write_text_atomic
from ..shared.manifest import contract_digest


def check_worker_release(cfg, *, launched: bool = False):
    """Check release pins before the dated launch, without renting a pod.

    ``launched`` (a verified ``v2-launch.json`` root exists): the whole-tree
    comparison of THIS process's source against ``[round]
    worker_code_fingerprint`` is a launch-day check only. After launch the
    trainer ships its own releases while the pods keep the pinned image, so
    the two legitimately differ; the pin names the image and every pod
    verifies its own code against it (provision health, leg pins). Comparing
    the trainer's tree blocked every king leg after the first trainer release
    past the image (2026-11-12: era 2637 king leg refused for 7 h). The image
    pins must still name one release."""
    if not cfg.v2.launch_at_utc and not cfg.v2.from_block:
        return
    from ..provision.codeprint import canonical_fingerprint
    from ..provision.core import image_digest_of

    root = Path(__file__).resolve().parents[2]
    current = canonical_fingerprint((str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
                                    for p in sorted((root / "cascade").rglob("*.py")))
    contract = cfg.training.v2_contract
    # DEC-CA-0067 automatic repin: this release may be the AFTER pins (the box
    # already runs the code the switch moves to) — either release qualifies
    v2 = getattr(cfg, "v2", None)
    if (getattr(v2, "worker_code_fingerprint_after", "") == current
            and cfg.round.worker_code_fingerprint != current):
        image = image_digest_of(v2.funded_pod_image_after)
        from ..shared.config import image_digest

        if contract is None or not image or image_digest(v2.train_image_digest_after) != image:
            raise ValueError("v2 funded_pod_image_after and train_image_digest_after must name "
                             "the same release")
        from ..shared.image_qualification import qualification_status

        st = qualification_status(cfg)
        if not st.usable:
            raise ValueError(f"v2 after release is not qualified (owner rule): {st.reason}")
        return
    if cfg.round.worker_code_fingerprint != current and not launched:
        raise ValueError("v2 worker image is not qualified for this release: rebuild from this source "
                         f"and repin worker_code_fingerprint (expected {current}) and image digests")
    image = image_digest_of(cfg.round.funded_pod_image)
    if contract is None or not image or contract.train_image_digest != image:
        raise ValueError("v2 funded_pod_image and training train_image_digest must name the same release")


def identity(cfg, ref):
    contract = cfg.training.v2_contract
    if contract is None:
        raise ValueError("v2 training contract is missing")
    from ..shared.hippius import HubRef

    HubRef.parse(ref)  # An immutable digest, never a mutable repository name.
    # The baseline is the king's ref: qualified under the king-leg policy.
    policy = cfg.for_king(ref).generator_for_contract(contract)
    source = Path(__file__).resolve().parents[1]
    verifier = hashlib.sha256()
    for file in ("interface/generator.py", "interface/static_guard.py", "miner/verify.py",
                 "trainer/sandbox.py", "generation/tier.py"):
        verifier.update((source / file).read_bytes())
    return {"ref": ref, "contract_digest": contract_digest(contract),
            "verifier_sha256": verifier.hexdigest(),
            "generator_policy": hashlib.sha256(json.dumps(asdict(policy), sort_keys=True).encode()).hexdigest()}


def proof_path(work_root, ref):
    return Path(work_root) / "v2-baselines" / (hashlib.sha256(ref.encode()).hexdigest() + ".json")


def qualify(cfg, ref, work_root):
    from ..miner.training_context import with_contract
    from ..miner.verify import verify_repo
    from ..shared.hippius import HubConfig, fetch_from_hub

    check_worker_release(cfg, launched=(Path(work_root) / "v2-launch.json").exists())
    expected = identity(cfg, ref)
    effective = with_contract(cfg.for_king(ref), cfg.training.v2_contract)
    with tempfile.TemporaryDirectory(prefix="cascade-v2-baseline-") as directory:
        fetch_from_hub(ref, directory, HubConfig.from_storage(cfg.storage))
        report = verify_repo(directory, effective)
    if not report.ok or report.runtime_skipped or not report.corpus_digest:
        raise ValueError("v2 baseline verification failed: " + report.render())
    proof = {**expected, "verification": report.to_dict()}
    write_text_atomic(proof_path(work_root, ref), json.dumps(proof, sort_keys=True))
    return proof


_ROOT_KEYS = ("contract_digest", "generator_policy", "verifier_sha256")


def _proof_problem(proof, expected, keys) -> str | None:
    """Why ``proof`` does not authorize ``expected`` (None: it does)."""
    if not isinstance(proof, dict):
        return "proof_malformed"
    stale = [k for k in keys if proof.get(k) != expected[k]]
    if stale:
        return "proof_stale:" + ",".join(stale)
    report = proof.get("verification") or {}
    if not (report.get("ok") and report.get("scope") == "full"
            and not report.get("runtime_skipped") and report.get("corpus_digest")):
        return "proof_not_full"
    return None


def ready(cfg, ref, work_root):
    root = Path(work_root) / "v2-launch.json"
    check_worker_release(cfg, launched=root.exists())
    expected = identity(cfg, ref)
    anchor = cfg.manifest.validator_hotkey
    if not anchor:
        raise ValueError("v2 launch needs a pinned validator receipt anchor")
    for path in (root, proof_path(work_root, ref)):
        try:
            proof = json.loads(path.read_text())
        except FileNotFoundError:
            continue
        # A verified launch root remains valid when the first v2 winner replaces
        # the inherited king. Policy changes require a new qualification.
        keys = _ROOT_KEYS if path == root else tuple(expected)
        if _proof_problem(proof, expected, keys) is None:
            if path != root:
                write_text_atomic(root, json.dumps(proof, sort_keys=True))
            return True
    return False


def preflight_report(cfg, ref, work_root) -> dict:
    """Read-only launch check for ``ref`` (``--preflight-v2-baseline-check``):
    what :func:`ready` would decide, with EVERY blocker named at once, so a
    fresh (testnet) deployment learns all of them before its first king leg.
    Fetches nothing, verifies nothing, rents nothing and writes nothing (not
    even the launch root :func:`ready` writes on success).

    ``{"ref", "ready", "blockers": [code, ...], "proof", "launch_root",
    "accepted_by", "next"}``. Codes: ``worker_release: <reason>``,
    ``identity: <reason>`` (bad ref, no v2 contract), ``no_validator_anchor``,
    ``no_proof``, ``proof_stale:<keys>``, ``proof_not_full``,
    ``proof_malformed``, ``proof_unreadable``."""
    work_root = Path(work_root)
    root = work_root / "v2-launch.json"
    blockers: list[str] = []
    try:
        check_worker_release(cfg, launched=root.exists())
    except ValueError as exc:
        blockers.append(f"worker_release: {exc}")
    try:
        expected = identity(cfg, ref)
    except Exception as exc:  # noqa: BLE001 — a bad ref is StorageError; report, never raise
        expected = None
        blockers.append(f"identity: {exc}")
    if not getattr(cfg.manifest, "validator_hotkey", ""):
        blockers.append("no_validator_anchor")
    proof_file = proof_path(work_root, ref)
    accepted_by = None
    if expected is not None:
        problems = []
        for path in (root, proof_file):
            try:
                proof = json.loads(path.read_text())
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as exc:
                problems.append(f"proof_unreadable: {path.name} ({type(exc).__name__})")
                continue
            problem = _proof_problem(proof, expected, _ROOT_KEYS if path == root else tuple(expected))
            if problem is None:
                accepted_by = str(path)
                break
            problems.append(problem)
        if accepted_by is None:
            blockers.extend(problems or ["no_proof"])
    ok = not blockers
    nxt = ("none: the first v2 king leg may rent" if ok else
           f"cascade-trainer --chain-toml CHAIN_TOML --work-root {work_root} "
           f"--preflight-v2-baseline {ref}")
    return {"ref": ref, "ready": ok, "blockers": blockers, "proof": str(proof_file),
            "launch_root": str(root), "accepted_by": accepted_by, "next": nxt}
