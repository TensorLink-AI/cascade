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
        keys = ("contract_digest", "generator_policy", "verifier_sha256") if path == root else tuple(expected)
        report = proof.get("verification") or {}
        if (all(proof.get(k) == expected[k] for k in keys) and report.get("ok")
                and report.get("scope") == "full" and not report.get("runtime_skipped")
                and report.get("corpus_digest")):
            if path != root:
                write_text_atomic(root, json.dumps(proof, sort_keys=True))
            return True
    return False
