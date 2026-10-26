"""Import the CPU generator libraries before admitting miner code; print versions."""
from __future__ import annotations

import importlib
import importlib.metadata
import json

LIBRARIES = {
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "torch": "torch",
    "numba": "numba", "scikit-learn": "sklearn.gaussian_process", "statsmodels": "statsmodels.api",
    "gpytorch": "gpytorch", "networkx": "networkx", "pyarrow": "pyarrow",
}


def check_runtime():
    versions = {}
    for package, module in LIBRARIES.items():
        importlib.import_module(module)
        versions[package] = importlib.metadata.version(package)
    return versions


if __name__ == "__main__":
    print(json.dumps({"generation_runtime": check_runtime()}, sort_keys=True, allow_nan=False))
