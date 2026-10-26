"""Isolate the quick verifier's imports and enforce its wall in the parent."""
from __future__ import annotations

import contextlib
import json
import sys


def main():
    from ..shared.config.sections import GeneratorConfig
    from ..trainer.sandbox import run_in_sandbox

    try:
        request = json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr):
            cfg = GeneratorConfig(**request["generator"])
            cores = request.get("cpu_cores")
            cores = tuple(cores) if cores is not None else None
            first = run_in_sandbox(request["repo"], 0, cfg, cpu_cores=cores)
            second = run_in_sandbox(request["repo"], 0, cfg, cpu_cores=cores)
            if not first.n_series or first.digest != second.digest:
                raise ValueError("empty or non-deterministic generator corpus")
            digest = first.digest
        result = {"ok": True, "digest": digest}
    except Exception as exc:  # noqa: BLE001 — worker protocol
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
