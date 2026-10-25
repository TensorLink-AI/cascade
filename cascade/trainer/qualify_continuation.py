"""Local crash/resume qualification on an already available CPU or GPU.

No provisioning, network, credentials or submissions. Each stage runs in a
fresh process; the interrupted worker exits without cleanup after a durable
snapshot. This checks A/B numerics on the selected device, not cloud lifecycle
behaviour or MarketCast's external model/data integration.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..shared.atomic import write_text_atomic
from ..shared.config import load_chain_config
from ..shared.manifest import contract_digest
from .continuation import PROTOCOL, Continuation, SnapshotStore, digest_file
from .toto2_trainer import Toto2Trainer


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain-toml", default="chain.toml")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument(
        "--tiny",
        action="store_true",
        help="Small architecture smoke check; not production-size qualification.",
    )
    p.add_argument("--cases", default="a,a-warm,b,anneal")
    p.add_argument("--phase", choices=("baseline", "interrupt", "resume"), help=argparse.SUPPRESS)
    p.add_argument("--case", help=argparse.SUPPRESS)
    return p


def _contract(args):
    cfg = load_chain_config(args.chain_toml)
    c = cfg.training.v2_contract or cfg.training
    c = replace(c, continuation_protocol=PROTOCOL)
    if args.tiny:
        c = replace(
            c,
            d_model=32,
            num_layers=2,
            num_heads=2,
            head_dim=16,
            patch_size=4,
            d_ff=40,
            mlp_expansion=2,
            num_quantiles=9,
            cpm_c_max=2,
            context_length=32,
            horizon=8,
            batch_size=4,
            max_train_seconds=600,
            umup_base_d_model=32,
            micro_batches=2,
        )
    if c.base_arch != "cascade_moe":
        raise ValueError("qualification requires a Cascade MoE contract")
    if args.case == "anneal":
        c = replace(c, anneal_fraction=0.2, power_ema_eval=-1, lr_schedule="wsd")
    return c


def _stream(contract):
    rng = np.random.default_rng(101)
    for _ in range(contract.batch_size * 64):
        yield np.cumsum(rng.normal(size=(2, contract.context_length)), axis=-1)


def _stage(args):
    import torch

    torch.set_num_threads(1 if args.tiny else 8)
    c = _contract(args)
    root = args.work_dir / args.case
    root.mkdir(parents=True, exist_ok=True)
    budget = c.batch_size * c.context_length * 8
    trainer = Toto2Trainer(device=args.device, dtype=args.dtype)
    warm = root / "base" if args.case in {"b", "a-warm"} else None
    if warm is not None and args.phase == "baseline":
        trainer.train(_stream(c), c, training_seed=7, token_budget=budget, out_dir=warm)
    selection = {0: [0, 1]} if args.case == "b" else None
    binding = {
        "contract": contract_digest(c),
        "init": {p.name: digest_file(p) for p in warm.glob("*.safetensors")} if warm else {},
        "seed": 7,
        "experts": selection,
        "dtype": args.dtype,
    }
    context = None
    if args.phase != "baseline":
        context = Continuation(
            SnapshotStore(root / "partial", binding, max_bytes=64 * 1024**3), interval=0
        )
        if args.phase == "interrupt":
            save = context.save

            def crash(state):
                save(state)
                if state["step"] == 3:
                    os._exit(86)

            context.save = crash
    result = trainer.train(
        _stream(c),
        c,
        training_seed=7,
        token_budget=budget,
        out_dir=root / args.phase,
        warm_start_dir=warm,
        expert_selection=selection,
        **({"continuation": context} if context else {}),
    )
    write_text_atomic(root / (args.phase + ".json"), json.dumps(result.metrics, sort_keys=True))
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    if args.phase:
        return _stage(args)
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise ValueError("qualification needs an empty work directory")
    cases = args.cases.split(",")
    if not cases or any(c not in {"a", "a-warm", "b", "anneal"} for c in cases):
        raise ValueError("unknown qualification case")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "cascade-continuation-qualification/1",
        "device": args.device,
        "dtype": args.dtype,
        "tiny": args.tiny,
        "cases": {},
        "scope": "local optimizer-boundary crash/resume; no provider or MarketCast qualification",
    }
    for case in cases:
        cmd = [
            sys.executable,
            "-m",
            "cascade.trainer.qualify_continuation",
            "--chain-toml",
            args.chain_toml,
            "--work-dir",
            str(args.work_dir),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--case",
            case,
            *(["--tiny"] if args.tiny else []),
        ]
        for phase, expected in (("baseline", 0), ("interrupt", 86), ("resume", 0)):
            # No credentials are passed or loaded; local training code has no
            # provider/storage client. Child logs remain visible on stderr.
            result = subprocess.run([*cmd, "--phase", phase], check=False)
            if result.returncode != expected:
                raise RuntimeError(
                    f"{case}/{phase} returned {result.returncode}, expected {expected}"
                )
        root = args.work_dir / case
        expected = {p.name: digest_file(p) for p in (root / "baseline").glob("*.safetensors")}
        actual = {p.name: digest_file(p) for p in (root / "resume").glob("*.safetensors")}
        before = json.loads((root / "baseline.json").read_text())
        after = json.loads((root / "resume.json").read_text())
        if (
            not expected
            or expected != actual
            or any(before[k] != after[k] for k in ("steps", "tokens_seen", "final_loss"))
        ):
            raise RuntimeError(f"{case} recovery differs from uninterrupted training")
        report["cases"][case] = {
            "equal": True,
            "tensor_files": expected,
            "steps": before["steps"],
            "tokens": before["tokens_seen"],
        }
    import torch

    report.update(
        torch_version=torch.__version__,
        gpu_name=torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
    )
    write_text_atomic(args.work_dir / "qualification.json", json.dumps(report, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
