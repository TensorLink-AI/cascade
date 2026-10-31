"""``cascade-generate``: materialise one leg's corpus on this machine
(DEC-CA-0053). The orchestrator runs it on the rented CPU VM over ssh.

Exit codes: 0 ok, 2 refused (tier disabled / bad args), 3 generator fault,
4 infra fault.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

EXIT_OK, EXIT_REFUSED, EXIT_GENERATOR, EXIT_INFRA = 0, 2, 3, 4


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cascade-generate", description=__doc__)
    p.add_argument("repo_dir", type=Path, help="the fetched generator repo")
    p.add_argument("out_dir", type=Path, help="where shards + corpus.json go")
    p.add_argument("--seed", type=int, required=True, help="the leg's generation seed")
    p.add_argument("--chain-toml", type=Path, default=None)
    p.add_argument("--at-block", type=int, default=None,
                   help="Era-start block selecting the effective generation contract.")
    p.add_argument("--generator-ref", default="", help="repo@digest, recorded in the index")
    p.add_argument("--window-seconds", type=float, default=None,
                   help="override [generation] window_seconds")
    p.add_argument("--vcpus", type=int, default=None,
                   help="the rented tier's vCPUs (one sandbox child each); pair it with "
                        "--window-seconds (the orchestrator sends both)")
    p.add_argument("--no-sandbox", action="store_true",
                   help="run in-process (local testing only)")
    p.add_argument("--allow-disabled", action="store_true",
                   help="run even though [generation] enabled = false (local testing)")
    p.add_argument("--upload-urls", type=Path, default=None,
                   help="JSON {name: pre-signed PUT link}: upload every shard, then "
                        "corpus.json, deleting each local file once it is up")
    p.add_argument("--progress-id", default="", help="Operator attempt identity for progress.")
    p.add_argument("--progress-file", type=Path, default=None, help="Atomic local progress JSON file.")
    p.add_argument("--json", action="store_true", help="print only the JSON summary")
    p.add_argument("--min-channels", type=int, default=None,
                   help="width floor for this leg (a king leg under [generator] "
                        "king_min_channels); default: the contract's policy")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.json else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Pre-import what the unsandboxed parent runs after the children: it never
    # imports a module once miner code has had the machine.
    import concurrent.futures  # noqa: F401

    from ..shared.config import load_chain_config
    from . import shards as _shards  # noqa: F401
    from . import transfer as _transfer  # noqa: F401
    from .tier import generate_corpus

    cfg = load_chain_config(args.chain_toml) if args.chain_toml else load_chain_config()
    if args.vcpus is not None:
        # The orchestrator picked the tier and sends its window too; the VM's
        # own chain.toml copy need not list the fallback tiers.
        if args.vcpus <= 0 or cfg.generation.children > 0:
            print(json.dumps({"ok": False, "error": "--vcpus needs a positive tier and children = 0"}))
            return EXIT_REFUSED
        cfg = replace(cfg, generation=replace(cfg.generation, vcpus=int(args.vcpus), fallback_vcpus=()))
    if not (cfg.generation.enabled or args.allow_disabled):
        print(json.dumps({"ok": False, "error": "[generation] enabled = false"}))
        return EXIT_REFUSED
    if not args.repo_dir.is_dir():
        print(json.dumps({"ok": False, "error": f"no generator repo at {args.repo_dir}"}))
        return EXIT_REFUSED
    # Arrow's allocator must be initialized on the parent thread before any
    # writer thread is started (pyarrow 25's default mimalloc is unsafe here).
    _shards._pa()
    # The CPU VM runs miner code: a missing network namespace refuses the run
    # instead of degrading to the Python-level socket guard.
    contract = cfg.throne_contracts_at(args.at_block)[0]
    if (args.at_block is None and contract.corpus_mode != "shards"
            and cfg.training.v2_contract is not None):
        # This CLI prepares a shard leg, never a legacy streaming leg.
        contract = cfg.training.v2_contract
    gen_cfg = cfg.generator_for_contract(contract)
    if args.min_channels is not None:
        if not 1 <= args.min_channels <= gen_cfg.min_channels:
            print(json.dumps({"ok": False, "error": "--min-channels may only lower the floor"}))
            return EXIT_REFUSED
        gen_cfg = replace(gen_cfg, min_channels=args.min_channels)
    if not args.no_sandbox:
        gen_cfg = replace(gen_cfg, sandbox_strict=True)
    from .progress import PROGRESS_NAME, CpuProgress

    urls = json.loads(args.upload_urls.read_text()) if args.upload_urls else {}
    progress = CpuProgress(args.progress_file or args.out_dir / PROGRESS_NAME,
                           attempt_id=args.progress_id, url=urls.get(PROGRESS_NAME))
    try:
        idx = generate_corpus(
            args.repo_dir, args.seed, gen_cfg, cfg.generation, args.out_dir,
            deadline_s=args.window_seconds, use_sandbox=not args.no_sandbox,
            generator_ref=args.generator_ref, progress=progress)
        summary = {k: idx.get(k) for k in (
            "corpus_digest", "n_series", "points", "stop_reason", "faults",
            "cpu_seconds", "wall_seconds", "cpu_model", "n_children", "stored_bytes", "raw_bytes", "shard_encoding")}
        summary["ok"] = not idx["faults"]
        if idx["faults"]:
            progress({"stage": "generator_fault" if idx["stop_reason"] == "generator_fault" else "infra_fault"})
            print(json.dumps(summary, sort_keys=True))
            return EXIT_GENERATOR if idx["stop_reason"] == "generator_fault" else EXIT_INFRA
        if args.upload_urls is not None:
            try:
                progress({"stage": "uploading"})
                summary["uploaded"] = upload_corpus_dir(args.out_dir, idx, args.upload_urls, progress=progress)
            except Exception as exc:  # noqa: BLE001 — upload is infrastructure, not a miner failure
                progress({"stage": "infra_fault"})
                summary.update(ok=False, error=f"upload: {type(exc).__name__}")
                print(json.dumps(summary, sort_keys=True))
                return EXIT_INFRA
        progress({"stage": "uploaded"})  # The orchestrator still verifies/registers the corpus.
        print(json.dumps(summary, sort_keys=True))
        return EXIT_OK
    finally:
        progress.close()


def upload_corpus_dir(out_dir: Path, index: dict, urls_path: Path, *, workers: int = 8, progress=None) -> int:
    """Upload every shard ``index`` lists through its pre-signed link (in
    parallel), then ``corpus.json`` last, so a reader never sees an index
    whose shards are not all in place. Each local shard is deleted once up.
    Returns the number of objects uploaded."""
    import concurrent.futures as cf

    from .shards import INDEX_NAME
    from .transfer import put_file

    urls = json.loads(urls_path.read_text(encoding="utf-8"))
    names = [sh["name"] for c in index["children"] for sh in c["shards"]]
    missing = [n for n in [*names, INDEX_NAME] if n not in urls]
    if missing:
        raise RuntimeError(f"no upload link for {missing[:3]} (+{max(0, len(missing) - 3)})")

    total_bytes = sum((out_dir / n).stat().st_size for n in names)
    uploaded_bytes = 0

    def one(name: str) -> int:
        size = (out_dir / name).stat().st_size
        put_file(urls[name], out_dir / name)
        (out_dir / name).unlink(missing_ok=True)
        return size

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for count, future in enumerate(cf.as_completed([ex.submit(one, n) for n in names]), 1):
            uploaded_bytes += future.result()
            if progress:
                try:
                    progress(dict(stage="uploading", uploaded_shards=count, upload_total_shards=len(names),
                                  uploaded_bytes=uploaded_bytes, upload_total_bytes=total_bytes))
                except Exception:  # noqa: BLE001 — logging cannot fail an otherwise successful upload
                    logging.getLogger(__name__).warning("upload progress observer unavailable")
    put_file(urls[INDEX_NAME], out_dir / INDEX_NAME)
    return len(names) + 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
