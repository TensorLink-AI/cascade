"""Actual training-budget points bound to an immutable checkpoint artifact.

This is trainer evidence, not hardware attestation. Never read the unsigned
dashboard summary to open the expert lane. Missing legacy evidence stays pending.
"""

import json
from pathlib import Path

from ..shared.atomic import write_text_atomic

MEASUREMENT_FILE = "training_measurement.json"


def write_measurement(out_dir: Path, *, metrics: dict, gen_ref: str,
                      corpus_digest: str, contract_digest: str, warm_start_ref: str,
                      token_budget: int) -> None:
    n = metrics.get("tokens_seen")
    if type(n) is not int or n < 0:
        # A test backend/old trainer may not implement measured points. Do not
        # manufacture them from the configured budget.
        (out_dir / MEASUREMENT_FILE).unlink(missing_ok=True)
        return
    anneal = metrics.get("anneal_tokens_seen", 0)
    if type(anneal) is not int or not 0 <= anneal <= n:
        raise ValueError("invalid anneal point count")
    # Fork-anneal produces scored weights but the stable checkpoint is what
    # the next lineage leg resumes. Do not credit its discarded anneal tail.
    n = min(n - anneal, token_budget)
    write_text_atomic(out_dir / MEASUREMENT_FILE, json.dumps({
        "version": 1, "tokens_seen": n, "gen_ref": gen_ref,
        "corpus_digest": corpus_digest, "contract_digest": contract_digest,
        "warm_start_ref": warm_start_ref or "",
    }, sort_keys=True))


def read_measurement(out_dir: Path, binding: dict) -> int:
    doc = json.loads((Path(out_dir) / MEASUREMENT_FILE).read_text())
    if not isinstance(doc, dict) or doc.get("version") != 1:
        raise ValueError("invalid lineage measurement version")
    for k in ("gen_ref", "corpus_digest", "contract_digest", "warm_start_ref"):
        if doc.get(k) != binding[k]:
            raise ValueError(f"lineage measurement has a different {k}")
    n = doc.get("tokens_seen")
    if type(n) is not int or n < 0:
        raise ValueError("lineage points must be a nonnegative integer")
    if n > binding["token_budget"]:
        raise ValueError("lineage points exceed the committed budget")
    return n
