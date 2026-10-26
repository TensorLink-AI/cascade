#!/usr/bin/env python
"""Build a reviewable benchmark/model-size catalog; no inference or publication.

GIFT-Eval results use the pinned leaderboard and our existing aggregation.
Parameter counts come from revision-pinned official Hub checkpoint metadata.
Other suites stay absent until measured with the Cascade benchmark pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cascade/website/data/benchmark-models.json"


def model_metadata(repo, revision=None):
    endpoint=f"https://huggingface.co/api/models/{repo}"+(f"/revision/{revision}" if revision else "")
    with urllib.request.urlopen(endpoint, timeout=30) as response:
        metadata = json.load(response)
    requested_revision = revision
    revision = metadata.get("sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"Unpinned model metadata: {repo}")
    if requested_revision and revision != requested_revision:
        raise ValueError(f"Model metadata revision changed: {repo}")
    count = (metadata.get("safetensors") or {}).get("total")
    return {"parameters": count if type(count) is int and count > 0 else None,
            "size_source": f"https://huggingface.co/api/models/{repo}/revision/{revision}",
            "model_revision": revision}


def add_measured_reference(catalog, doc):
    """Only native same-pipeline loss pairs; never convert skill/rank scores."""
    source = str(doc.get("source", ""))
    match = re.fullmatch(r"([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)@([0-9a-f]{40})", source)
    if doc.get("kind") != "reference_bench" or not match:
        raise ValueError("Measured reference must be a reference_bench with a pinned official Hub source")
    scores = doc.get("scores") or {}
    measured = {}
    for suite in ("gifteval", "boom", "time", "fev"):
        pair = {metric: scores.get(f"{suite}_{metric}") for metric in ("mase", "crps")}
        if all(v is None for v in pair.values()):
            continue
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in pair.values()):
            raise ValueError(f"Invalid {suite} loss pair")
        measured[suite] = pair
    if not measured:
        raise ValueError("Reference has no measured loss pairs")
    metadata = model_metadata(match[1], match[2])
    preset = str(doc.get("preset", ""))
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", preset):
        raise ValueError("Reference needs a stable preset ID")
    entry = next((m for m in catalog["models"] if m["id"] == preset), None)
    if entry is None:
        entry = {"id": preset, "label": match[1].split("/")[-1], "scores": {}}
        catalog["models"].append(entry)
    entry.update(metadata, url=f"https://huggingface.co/{match[1]}")
    for pair in measured.values():
        pair.update(basis="same-pipeline reference", source=source,
                    data_revisions=doc.get("data_revisions"))
    entry["scores"].update(measured)


def cascade_presets():
    import torch

    from cascade.shared.config import load_chain_config
    from cascade.trainer.moe_model import CascadeMoEConfig, CascadeMoEModel, param_counts
    from cascade.trainer.toto2_model import Toto2Config, Toto2Model

    cfg = load_chain_config()
    result = {}
    for contract in (cfg.training.primary_size, cfg.training.v2_contract.primary_size):
        with torch.device("meta"):
            if contract.base_arch == "cascade_pyramid":
                from cascade.trainer.pyramid_model import CascadeMoEConfig as PyramidConfig
                from cascade.trainer.pyramid_model import CascadeMoEModel as PyramidModel
                from cascade.trainer.pyramid_model import param_counts as pyramid_counts

                model = PyramidModel(PyramidConfig.from_contract(contract))
                counts = pyramid_counts(model)
            elif contract.base_arch == "cascade_moe":
                model = CascadeMoEModel(CascadeMoEConfig.from_contract(contract))
                counts = param_counts(model)
            else:
                model = Toto2Model(Toto2Config.from_contract(contract))
                count = sum(p.numel() for p in model.parameters())
                counts = {"total": count, "active": count}
        result[contract.arch_preset] = {"parameters": int(counts["total"]),
                                       "active_parameters": int(counts["active"]),
                                       "base_arch_digest": contract.base_arch_digest,
                                       "source": "configured architecture, counted on meta device"}
    return result


def build():
    # Import the existing, tested GIFT-Eval aggregation rather than duplicating it.
    sys.path.insert(0, str(ROOT / "scripts"))
    import refresh_leaderboard_reference as gift

    reference = json.loads((ROOT / "cascade/website/data/gift-eval-reference.json").read_text())
    revision = reference["space_revision"]
    entries = list(reference["models"])
    smaller = ["Timestral-5M", "Timestral-28M", "Timestral-123M"]
    rows = gift.load_results(revision, smaller)
    for name in smaller:
        cfg = gift.model_config(revision, name)
        if cfg.get("testdata_leakage") != "No":
            raise ValueError(f"No clean leaderboard provenance for {name}")
        entries.append({"model": name, "model_link": f"https://huggingface.co/mistralai/{name}",
                        "cascade": gift.cascade_aggregate(rows, name)})
    models = []
    for entry in entries:
        repo = entry["model_link"].removeprefix("https://huggingface.co/")
        scores = entry["cascade"]
        if scores.get("n_scored") != 97:
            raise ValueError(f"Incomplete GIFT-Eval coverage: {entry['model']}")
        models.append({"id": entry["model"].lower(), "label": entry["model"],
                       "url": entry["model_link"], **model_metadata(repo),
                       "scores": {"gifteval": {k: scores[k] for k in ("mase", "crps")}},
                       "score_source": f"{gift.SPACE_URL}/tree/{revision}/results/{entry['model']}",
                       "basis": "published GIFT-Eval, Cascade aggregation, 97 configurations"})
    return {"schema": 1, "as_of": datetime.now(UTC).isoformat(timespec="seconds"),
            "benchmarks": {"gifteval": {"source": reference["source"], "source_revision": revision,
                "aggregation": "cascade_official", "n_configs": 97,
                "note": "Published leaderboard rows use Cascade aggregation; these models were not rerun by Cascade."}},
            "cascade_presets": cascade_presets(), "models": models}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--reference-doc", action="append", type=Path, default=[],
                        help="Merge an exported same-pipeline reference JSON, including measured BOOM/TIME/FEV scores.")
    args = parser.parse_args()
    doc = build()
    for path in args.reference_doc:
        add_measured_reference(doc, json.loads(path.read_text()))
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(doc['models'])} models to {args.out}; no benchmarks run or objects published.")


if __name__ == "__main__":
    main()
