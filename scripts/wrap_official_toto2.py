#!/usr/bin/env python
"""Wrap the OFFICIAL Datadog Toto2 weights in cascade's checkpoint layout.

The benchmark sidecar scores "a checkpoint dir in cascade's layout":
``weights.safetensors`` + ``config.json`` + ``model.py`` +
``forecast_wrapper.py``. To bench the official release through the identical
battery (see ``scripts/publish_reference_bench.py``), its weights must be
re-packaged in that layout.

Strategy — deliberately dumb and loud:

1. Take an existing CASCADE checkpoint of the SAME SIZE as the template
   (any round artifact fetched from the hub); its ``config.json`` /
   ``model.py`` / ``forecast_wrapper.py`` are copied through unchanged, so
   the inference path is bit-identical to what every round checkpoint uses.
2. Compare the official state dict against the template's, key by key and
   shape by shape. An exact match copies straight through. Any mismatch
   prints the full aligned diff and exits non-zero — this script NEVER
   guesses a mapping. If the official release uses different key names,
   write a ``--map-file`` (JSON: ``{"official.key": "cascade.key"}``) from
   the printed diff and re-run; renames are applied only when the shapes
   agree.
3. On success, writes the wrapped dir plus ``provenance.json`` recording the
   official source, the template used, and any key mapping — feed its
   ``source`` string to ``publish_reference_bench.py --source``.

Needs ``safetensors`` + ``numpy`` (the ``train`` extra provides both; run it
on the bench pod). The official weights can be a single ``*.safetensors``
file or a directory containing exactly one.

Example::

    python scripts/wrap_official_toto2.py \
        --template /ckpts/cascade-round-XXXX-king \
        --official /hf/Datadog/Toto-2.0-4m \
        --source "Datadog/Toto-2.0-4m@hf:<revision>" \
        --out /ckpts/official-toto2-4m
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

TEMPLATE_FILES = ("config.json", "model.py", "forecast_wrapper.py")


def _find_weights(path: Path) -> Path:
    if path.is_file():
        return path
    cands = sorted(path.glob("*.safetensors"))
    if len(cands) != 1:
        sys.exit(f"expected exactly one *.safetensors under {path}, found "
                 f"{[c.name for c in cands]} — pass the file directly")
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-package official Toto2 weights in cascade's checkpoint layout.")
    ap.add_argument("--template", type=Path, required=True,
                    help="A cascade checkpoint dir of the SAME SIZE (config/model/wrapper source).")
    ap.add_argument("--official", type=Path, required=True,
                    help="Official weights: a *.safetensors file or a dir holding one.")
    ap.add_argument("--source", required=True,
                    help="Provenance of the official weights, e.g. 'Datadog/Toto-2.0-4m@hf:<rev>'.")
    ap.add_argument("--out", type=Path, required=True, help="Output checkpoint dir.")
    ap.add_argument("--map-file", type=Path, default=None,
                    help="JSON {official_key: cascade_key} renames, written from a printed diff.")
    args = ap.parse_args()

    for name in TEMPLATE_FILES:
        if not (args.template / name).is_file():
            sys.exit(f"template is missing {name} — pass a full cascade checkpoint dir")

    from safetensors.numpy import load_file, save_file

    tmpl = load_file(str(_find_weights(args.template)))
    official = load_file(str(_find_weights(args.official)))
    mapping = json.loads(args.map_file.read_text()) if args.map_file else {}
    renamed = {mapping.get(k, k): v for k, v in official.items()}

    tset, oset = set(tmpl), set(renamed)
    missing, extra = sorted(tset - oset), sorted(oset - tset)
    shape_bad = sorted(k for k in tset & oset if tmpl[k].shape != renamed[k].shape)
    if missing or extra or shape_bad:
        print("STATE-DICT MISMATCH — refusing to guess. Build a --map-file from this diff:\n")
        for k in missing:
            print(f"  cascade-only : {k}  {tuple(tmpl[k].shape)}")
        for k in extra:
            print(f"  official-only: {k}  {tuple(renamed[k].shape)}")
        for k in shape_bad:
            print(f"  shape-differs: {k}  cascade{tuple(tmpl[k].shape)} vs "
                  f"official{tuple(renamed[k].shape)}")
        print(f"\n({len(missing)} missing, {len(extra)} extra, {len(shape_bad)} shape mismatches; "
              f"{len(tset & oset) - len(shape_bad)} keys already align)")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(renamed, str(args.out / "weights.safetensors"))
    for name in TEMPLATE_FILES:
        (args.out / name).write_text(
            (args.template / name).read_text(encoding="utf-8"), encoding="utf-8")
    (args.out / "provenance.json").write_text(json.dumps({
        "kind": "official_toto2_wrap",
        "source": args.source,
        "template": str(args.template),
        "key_mapping": mapping,
        "n_tensors": len(renamed),
        "wrapped_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }, indent=1), encoding="utf-8")
    print(f"wrapped → {args.out}  ({len(renamed)} tensors, identity-mapped "
          f"{len(renamed) - len(mapping)}, renamed {len(mapping)})")
    print("next: score it with the bench sidecar, then "
          "scripts/publish_reference_bench.py --source " + json.dumps(args.source))
    return 0


if __name__ == "__main__":
    sys.exit(main())
