#!/usr/bin/env python3
"""Write a complete offline miner configuration for a supplied ladder size."""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def size_aliases():
    from cascade.shared.model_names import parameter_count_label

    rows = json.loads((ROOT / "deploy/presets/pyramid-ladder.json").read_text())["sizes"]
    return {parameter_count_label(row["total_params"]).lower(): alias for alias, row in rows.items()}


def resolve_size(size):
    return size_aliases().get(size.lower(), size.lower())


def training_table(size):
    size = resolve_size(size)
    row = json.loads((ROOT / "deploy/presets/pyramid-ladder.json").read_text())["sizes"][size]
    table = tomllib.loads((ROOT / "deploy/presets/cascade-moe-28m.toml").read_text())["training"]
    table.update(base_arch="cascade_pyramid", arch_preset=f"cascade-moe-v2-{size}",
        d_model=row["width"], d_ff=row["dense_ffn"], num_heads=row["heads"], num_quantiles=21,
        target_train_hours=3.0, ref_throughput_tokens_per_s=0, training_steps=27500,
        batch_size=512, batch_denomination="sequences", budget_denomination="points",
        base_lr=0.002125, warmup_fraction=80/27500, micro_batches=1,
        power_ema_sigma_rels=[], power_ema_eval=-1, copula_samples=4,
        copula_patches=127, router_lr_scale=1.0, route_bias_rate_scale=1.0,
        max_train_seconds=18000)
    from cascade.shared.config.training import parse_training_contract
    from cascade.trainer.contract import compute_base_arch_digest

    table["base_arch_digest"] = compute_base_arch_digest(parse_training_contract(table))
    return table


def _value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return json.dumps(value, allow_nan=False)
    if isinstance(value, list) and not any(isinstance(v, dict) for v in value):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    raise TypeError(f"unsupported TOML value {type(value).__name__}")


def dump_toml(raw):
    lines = []

    def key(name):
        return name if re.fullmatch(r"[A-Za-z0-9_-]+", name) else json.dumps(name)

    def table(obj, path=(), array=False):
        if path:
            header = ".".join(map(key, path))
            lines.extend(["", f"[[{header}]]" if array else f"[{header}]"])
        for name, value in obj.items():
            if not isinstance(value, dict) and not (isinstance(value, list) and value and isinstance(value[0], dict)):
                lines.append(f"{key(name)} = {_value(value)}")
        for name, value in obj.items():
            if isinstance(value, dict):
                table(value, (*path, name))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                for item in value:
                    table(item, (*path, name), True)

    table(raw)
    text = "\n".join(lines) + "\n"
    if tomllib.loads(text) != raw:
        raise ValueError("TOML serialization changed the configuration")
    return text


def local_config(size, template=None):
    from cascade.shared.config import parse_chain_config

    size = resolve_size(size)
    raw = tomllib.loads(Path(template or ROOT / "chain.toml").read_text())
    raw["training"] = training_table(size)
    raw["v2"].pop("training", None)
    raw["v2"].update(from_block=0, launch_at_utc="", commit_open_at_utc="", commit_from_block=0)
    raw["lane_b"]["from_block"] = 0
    raw["lane_c"]["from_block"] = 0
    raw["netconfig"]["owner_hotkey"] = ""
    raw["round"].update(screen_size=raw["training"]["arch_preset"],
                        throne_sizes=[raw["training"]["arch_preset"]])
    raw["eval"].update(context_length=4096, horizon=64)
    parse_chain_config(raw)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", required=True, type=str.lower,
        choices=[*size_aliases(), "2m", "7m", "28m", "120m"],
        help="Total parameter size; earlier active-size aliases are also accepted")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--template", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; choose a new local file")
    text = dump_toml(local_config(args.size, args.template))
    with args.out.open("x") as file:
        file.write("# Offline model experiment; hardware throughput is unmeasured.\n" + text)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
