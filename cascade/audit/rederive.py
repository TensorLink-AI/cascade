"""Tier 1 + Tier 2 re-derivation — the expensive halves of the audit.

Tier 1 (CPU, minutes): fetch each pinned generator and re-run it in the same
sandbox the trainer used, at the receipt's ``generation_seed``, and byte-compare
the corpus digest per (entry, size). ``cache_reuse`` re-derives exactly.
``stream_cpu`` digests cover the *consumed training prefix* — re-deriving one
means re-streaming the full per-size token budget, which costs about as much as
the round's own generation; that runs only under ``--full-stream``, else WARN.
``stream_gpu`` is tolerance/same-hardware by design and never byte-compares on
CPU (WARN).

Tier 2 (GPU, ``[train]`` extra, **experimental**): fetch the published
checkpoint, re-train from the contract under the receipt's seeds and token
budget, and compare — byte-exact when this runtime matches the recorded
``gpu_name`` *and* the pinned ``train_image_digest``, else eval-score
comparison within a documented tolerance.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from ..shared.config import ChainConfig
from ..shared.receipt import RoundReceipt
from .checks import FAIL, PASS, SKIP, WARN, CheckResult

log = logging.getLogger("cascade.audit")

# Relative tolerance on geomean(CRPS, MASE) for Tier 2's score comparison on
# mismatched hardware. Deterministic kernels on a different GPU SKU reorder
# float reductions; observed drift is well under 1% — 5% catches a swapped
# corpus/checkpoint while never tripping on numerics. Documented in AUDIT.md.
TIER2_SCORE_RTOL = 0.05


def _fetch_generator(gen_ref: str, dest: Path, cfg: ChainConfig) -> Path:
    """Fetch a pinned generator, preferring anonymous access.

    Audit runs credential-free where possible: try the authenticated client
    only if the anonymous pull fails (public Hub repos need no token; see
    OPEN_QUESTIONS.md on public-read).
    """
    from ..shared.hippius import HubAuthError, HubConfig, fetch_from_hub

    try:
        return fetch_from_hub(gen_ref, dest, HubConfig.from_storage(cfg.storage))
    except HubAuthError:
        # No credentials in the environment — try an anonymous snapshot pull.
        from hippius_hub import snapshot_download  # type: ignore

        from ..shared.hippius import ALLOW_PATTERNS, HubRef

        ref = HubRef.parse(gen_ref)
        return Path(snapshot_download(
            repo_id=ref.repo, revision=ref.digest, local_dir=str(dest),
            allow_patterns=ALLOW_PATTERNS, token=None,
        ))


def _rederive_digest(
    repo_dir: Path,
    generation_seed: int,
    cfg: ChainConfig,
    *,
    mode: str,
    token_budget: int,
    max_wall_seconds: int | None = None,
) -> str:
    """Re-derive one corpus digest exactly as the trainer derived it."""
    if mode == "cache_reuse":
        from ..trainer.corpus import build_round_corpus

        return build_round_corpus(
            repo_dir, generation_seed, cfg.generator, "cache_reuse",
            use_sandbox=True, blocked=cfg.static_guard.blocked,
        ).digest
    # stream_cpu: drain the same budget-capped stream and read the rolling digest.
    from ..trainer.stream import open_round_stream

    with open_round_stream(
        mode, repo_dir, generation_seed, cfg.generator,
        token_budget=token_budget, use_sandbox=True,
        blocked=cfg.static_guard.blocked,
        max_wall_seconds=max_wall_seconds,
    ) as rs:
        for _ in rs.series():
            pass
        return rs.digest


def run_tier1(
    receipt: RoundReceipt,
    cfg: ChainConfig,
    *,
    workdir: Path,
    full_stream: bool = False,
) -> list[CheckResult]:
    """Re-derive every trained entry's corpus digest from its pinned generator."""
    if receipt.status == "rejected":
        return [CheckResult("corpus", SKIP, "rejected round; nothing was trained")]
    try:
        manifest = receipt.load_embedded_manifest()
    except (ValueError, KeyError) as e:
        return [CheckResult("corpus", FAIL, f"embedded manifest unparseable: {e}")]

    registry = cfg.training.size_registry
    results: list[CheckResult] = []
    digest_cache: dict[tuple, str] = {}
    for entry in manifest.entries:
        name = f"corpus:{entry.role}:{entry.size or cfg.training.arch_preset}"
        contract = registry.get(entry.size) if entry.size else cfg.training.primary_size
        if contract is None:
            results.append(CheckResult(
                name, FAIL, f"entry size {entry.size!r} is not a configured size"))
            continue
        mode = contract.corpus_mode
        if mode == "stream_gpu":
            results.append(CheckResult(
                name, WARN, "stream_gpu corpus is tolerance/same-hardware by design; "
                            "not byte-re-derivable on CPU (Tier 2 audits it)"))
            continue
        if mode == "stream_cpu" and not full_stream:
            results.append(CheckResult(
                name, WARN, "stream_cpu digest covers the full consumed training budget "
                            f"(~{contract.train_tokens:,} points); pass --full-stream to "
                            "re-stream and byte-compare"))
            continue
        cache_key = (entry.gen_ref, mode, contract.train_tokens if mode != "cache_reuse" else 0)
        try:
            if cache_key in digest_cache:
                digest = digest_cache[cache_key]
            else:
                gen_dir = workdir / "generators" / hashlib.sha256(
                    entry.gen_ref.encode()).hexdigest()[:16]
                _fetch_generator(entry.gen_ref, gen_dir, cfg)
                digest = _rederive_digest(
                    gen_dir, receipt.generation_seed, cfg,
                    mode=mode, token_budget=contract.train_tokens,
                    max_wall_seconds=contract.max_train_seconds,
                )
                digest_cache[cache_key] = digest
        except Exception as e:  # noqa: BLE001 — fetch/build failure is a WARN, not proof
            results.append(CheckResult(
                name, WARN, f"could not re-derive ({type(e).__name__}: {e})"))
            continue
        if digest != entry.corpus_digest:
            results.append(CheckResult(
                name, FAIL, f"re-derived corpus digest {digest[:16]}… != recorded "
                            f"{entry.corpus_digest[:16]}… (gen {entry.gen_ref[:32]}…, "
                            f"seed {receipt.generation_seed})"))
        else:
            results.append(CheckResult(
                name, PASS, f"corpus digest {digest[:16]}… re-derives byte-identically"))
    if not results:
        results.append(CheckResult("corpus", WARN, "manifest carries no trained entries"))
    return results


# ── Tier 2 (experimental) ─────────────────────────────────────────────────────


def _dir_file_digests(d: Path) -> dict[str, str]:
    return {
        p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(d.rglob("*")) if p.is_file()
    }


def run_tier2(
    receipt: RoundReceipt,
    cfg: ChainConfig,
    *,
    workdir: Path,
    device: str = "cuda",
    trainer_spec: str = "cascade.trainer.toto2_trainer:Toto2Trainer",
) -> list[CheckResult]:
    """EXPERIMENTAL: re-train each entry from the contract and compare.

    Needs the ``[train]`` extra and a GPU (~the round's own compute per entry).
    Byte-exact checkpoint comparison only when this runtime matches the entry's
    recorded ``gpu_name`` AND the contract's ``train_image_digest`` pin;
    otherwise the retrained checkpoint is scored against the published one on
    the receipt's eval slice within :data:`TIER2_SCORE_RTOL`.
    """
    if receipt.status == "rejected":
        return [CheckResult("retrain", SKIP, "rejected round; nothing was trained")]
    try:
        import torch  # noqa: F401
    except ImportError:
        return [CheckResult("retrain", WARN,
                            "torch unavailable; install the [train] extra for Tier 2")]
    try:
        manifest = receipt.load_embedded_manifest()
    except (ValueError, KeyError) as e:
        return [CheckResult("retrain", FAIL, f"embedded manifest unparseable: {e}")]

    import os

    import torch

    from ..shared.manifest import parse_trained_pointer
    from ..trainer.contract import TRAIN_IMAGE_DIGEST_ENV, _image_sha256
    from ..trainer.main import _load_trainer
    from ..trainer.stream import open_round_stream

    local_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    pinned_img = _image_sha256(cfg.training.train_image_digest)
    runtime_img = _image_sha256(os.environ.get(TRAIN_IMAGE_DIGEST_ENV, ""))
    image_matched = pinned_img is not None and runtime_img == pinned_img

    registry = cfg.training.size_registry
    base_trainer = _load_trainer(trainer_spec)
    results: list[CheckResult] = []
    # Warm-start pin (DEC-CA-0003): when the signed manifest records a promoted
    # init, re-derivation must start from THAT checkpoint, not random init. The
    # pointer is content-addressed, so the fetch is byte-pinned. A manifest that
    # pins an init we cannot fetch fails every matching entry loudly.
    warm_dir: Path | None = None
    if manifest.warm_start_ckpt:
        ws_ref = parse_trained_pointer(manifest.warm_start_ckpt)
        if ws_ref is None:
            return [CheckResult("retrain", FAIL,
                                f"malformed warm_start_ckpt {manifest.warm_start_ckpt!r}")]
        warm_dir = workdir / "warm_start" / hashlib.sha256(ws_ref.encode()).hexdigest()[:16]
        try:
            _fetch_generator(ws_ref, warm_dir, cfg)
        except Exception as e:  # noqa: BLE001 — a pinned-but-unfetchable init fails the audit
            return [CheckResult("retrain", FAIL,
                                f"pinned warm-start init unfetchable ({type(e).__name__}: {e})")]
    for entry in manifest.entries:
        name = f"retrain:{entry.role}:{entry.size or cfg.training.arch_preset}"
        contract = registry.get(entry.size) if entry.size else cfg.training.primary_size
        if contract is None:
            results.append(CheckResult(name, FAIL, f"unknown size {entry.size!r}"))
            continue
        try:
            gen_dir = workdir / "generators" / hashlib.sha256(
                entry.gen_ref.encode()).hexdigest()[:16]
            _fetch_generator(entry.gen_ref, gen_dir, cfg)
            ref = parse_trained_pointer(entry.trained_pointer)
            if ref is None:
                results.append(CheckResult(
                    name, FAIL, f"malformed trained_pointer {entry.trained_pointer!r}"))
                continue
            published_dir = workdir / "published" / hashlib.sha256(ref.encode()).hexdigest()[:16]
            _fetch_generator(ref, published_dir, cfg)  # same tolerant fetch path

            out_dir = workdir / "retrained" / f"{entry.role}-{entry.size or 'primary'}"
            out_dir.mkdir(parents=True, exist_ok=True)
            # The pinned init applies to the size it was trained at; other sizes
            # trained (and re-derive) from random init.
            entry_warm = (warm_dir if warm_dir is not None
                          and (entry.size or cfg.training.arch_preset) ==
                          (manifest.warm_start_size or cfg.training.arch_preset) else None)
            with open_round_stream(
                contract.corpus_mode, gen_dir, receipt.generation_seed, cfg.generator,
                token_budget=contract.train_tokens, use_sandbox=True,
                blocked=cfg.static_guard.blocked,
                max_wall_seconds=contract.max_train_seconds,
            ) as rs:
                base_trainer.train(
                    rs.series(), contract,
                    training_seed=receipt.training_seed,
                    token_budget=contract.train_tokens,
                    out_dir=out_dir,
                    **({"warm_start_dir": entry_warm} if entry_warm is not None else {}),
                )
        except Exception as e:  # noqa: BLE001
            results.append(CheckResult(name, WARN, f"re-train failed "
                                                   f"({type(e).__name__}: {e})"))
            continue

        exact_ok = image_matched and entry.gpu_name and entry.gpu_name == local_gpu
        if exact_ok:
            ours, theirs = _dir_file_digests(out_dir), _dir_file_digests(published_dir)
            weights_ours = {k: v for k, v in ours.items() if k.endswith(".safetensors")}
            weights_theirs = {k: v for k, v in theirs.items() if k.endswith(".safetensors")}
            if weights_ours == weights_theirs and weights_ours:
                results.append(CheckResult(name, PASS,
                                           "checkpoint re-trains byte-identically "
                                           f"(gpu={local_gpu}, image={pinned_img})"))
            else:
                results.append(CheckResult(name, FAIL,
                                           "matched GPU + image but the retrained "
                                           "checkpoint bytes differ"))
            continue
        results.append(_tier2_score_compare(
            name, receipt, cfg, out_dir, published_dir, device=device,
            why=("gpu_name/image not matched: recorded gpu="
                 f"{entry.gpu_name!r} vs local {local_gpu!r}, image pin matched="
                 f"{image_matched}"),
        ))
    return results


def _tier2_score_compare(
    name: str, receipt: RoundReceipt, cfg: ChainConfig,
    retrained: Path, published: Path, *, device: str, why: str,
) -> CheckResult:
    """Score both checkpoints on the receipt's eval slice; compare geomeans."""
    try:
        from ..eval.scoring import global_geomean
        from ..validator.evaluator import evaluate_checkpoint
        from ..validator.pool import load_pool

        source = load_pool(cfg)
        n = receipt.eval_context.n_windows if receipt.eval_context else cfg.eval.n_windows
        windows = source.windows_for_round(receipt.base_seed, n)
        ours = global_geomean(evaluate_checkpoint(
            retrained, windows, num_samples=cfg.eval.num_samples, device=device))
        theirs = global_geomean(evaluate_checkpoint(
            published, windows, num_samples=cfg.eval.num_samples, device=device))
    except Exception as e:  # noqa: BLE001
        return CheckResult(name, WARN, f"score comparison unavailable "
                                       f"({type(e).__name__}: {e}); {why}")
    rel = abs(ours - theirs) / max(abs(theirs), 1e-12)
    if rel > TIER2_SCORE_RTOL:
        return CheckResult(name, FAIL,
                           f"retrained geomean {ours:.5f} vs published {theirs:.5f} "
                           f"(rel diff {rel:.3%} > {TIER2_SCORE_RTOL:.0%}); {why}")
    return CheckResult(name, PASS,
                       f"retrained geomean {ours:.5f} ≈ published {theirs:.5f} "
                       f"(rel diff {rel:.3%} ≤ {TIER2_SCORE_RTOL:.0%}); {why}")
