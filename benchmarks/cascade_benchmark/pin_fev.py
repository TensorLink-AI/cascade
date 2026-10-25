"""Hash an explicitly staged local FEV task file and its datasets; no downloads."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .provenance import tree_hash, verify_fev_manifest


def pin_tasks(tasks_path, manifest_path):
    import fev

    tasks = Path(tasks_path).resolve()
    if not tasks.is_file():
        raise ValueError("stage a local task YAML before pinning FEV")
    roots = set()
    for task in fev.Benchmark.from_yaml(str(tasks)).tasks:
        root = Path(task.dataset_path)
        if task.dataset_config is not None or not root.is_absolute() or not root.exists():
            raise ValueError("FEV tasks must name existing absolute local datasets; no Hub refs or globs")
        roots.add(str(root))
    manifest = {"tasks": {"path": str(tasks), "sha256": tree_hash(tasks)},
                "datasets": [{"path": root, "sha256": tree_hash(root)} for root in sorted(roots)]}
    output = Path(manifest_path).resolve()
    if output == tasks or str(output) in roots:
        raise ValueError("the FEV manifest must not overwrite its task file or dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, sort_keys=True, indent=2)
        identity = verify_fev_manifest(temporary)
        os.replace(temporary, output)
        return identity
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tasks")
    p.add_argument("manifest")
    args = p.parse_args(argv)
    print(json.dumps({"manifest": str(Path(args.manifest).resolve()),
                      "revision": pin_tasks(args.tasks, args.manifest)}))


if __name__ == "__main__":
    main()
