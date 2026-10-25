"""Publish the reproducible random initialization used by every founder."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save

from ..shared.direct_moe import canonical, digest
from ..trainer.direct_model import build_direct_model
from .storage import open_objects, publish


def initialize(architecture, seed, objects):
    # Initialization happens on CPU on every hardware profile; this prevents a
    # CUDA RNG implementation from silently changing the founding checkpoint.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = build_direct_model(architecture, device="cpu")
        weights = publish(objects, save({k: v.contiguous() for k, v in model.state_dict().items()}))
    manifest = {
        "kind": "random-initialization",
        "seed": seed,
        "architecture": digest(architecture),
        "ranks": {"0": {"weights": weights}},
    }
    publish(objects, canonical(manifest).encode())
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    manifest = initialize(spec["architecture"], spec["seed"], open_objects(spec["storage"]))
    args.output.write_text(canonical({"anchor": digest(manifest), "manifest": manifest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
