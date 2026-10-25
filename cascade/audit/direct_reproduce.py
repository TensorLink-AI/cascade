"""Reproduce a direct-MoE journal's scores (DEC-CA-0052 trust model).

Validators only check that the trainer's signed journal replays from its own
recorded scores. Whether those scores are TRUE is settled here: anyone can
re-score the published checkpoints under the inputs each decision binds
(``scoring``: manifests, architecture, panel digests, samples, seed) and compare.

Per decision, three outcomes:

* PASS — every recorded per-case score reproduces within ``tolerance`` and the
  verdict re-derived from the reproduced scores equals the recorded one.
* FAIL — a score or verdict does not reproduce, the binding does not name the
  judged checkpoints, or a bound object is missing/corrupt.
* WARN — not reproducible YET: a decision without a binding (pre-binding
  journals), a private panel still unrevealed, or a public suite this auditor
  has no sidecar for. The half that can be checked is still checked.

Checkpoints are fetched credential-free from the public bucket the manifest
URIs name; a revealed panel comes from the trainer's journal bucket.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

from ..direct.evaluation import (
    PrivatePolicy,
    check_scoring_binding,
    export_checkpoint,
    private_verdict,
    score_private,
)
from ..direct.evaluator import CheckpointEvaluator, panel_windows
from ..direct.network import journal_decisions
from ..direct.storage import PodObjects, anonymous_objects, publish
from ..eval.direct_guard import GuardContract, judge_public
from ..shared.direct_moe import canonical, digest
from .checks import FAIL, PASS, WARN, CheckResult


def public_store(uri):
    if not uri.startswith("hippius://"):
        raise ValueError(f"object not publicly fetchable: {uri}")
    return anonymous_objects(uri)


class AuditObjects(PodObjects):
    """Local content-addressed cache in front of the public store each URI names.

    ``remote(uri)`` returns a store that can ``get`` it; every body is re-hashed
    against the URI's digest before it is cached or scored.
    """

    def __init__(self, root, *, remote: Callable = public_store):
        super().__init__(root)
        self.remote = remote

    def add(self, body):
        return publish(self, body)

    def get(self, uri):
        if not self._path(uri).exists():
            body = self.remote(uri).get(uri)
            if self.add(body)["sha256"] != self._path(uri).name:
                raise ValueError(f"object digest mismatch: {uri}")
        return super().get(uri)


def _close(a, b, tolerance):
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance * 1e-3)


def compare_private(recorded, reproduced, tolerance):
    """First divergence between two private row lists, or None."""
    def index(rows):
        # Through JSON, as the journal stores them (tuples become lists).
        return {(r["series_id"], r["channel"]): r for r in json.loads(canonical(rows))}

    old, new = index(recorded), index(reproduced)
    if old.keys() != new.keys():
        return "private case set differs"
    for key in sorted(old):
        a, b = old[key], new[key]
        for field in sorted(set(a) | set(b)):
            x, y = a.get(field), b.get(field)
            if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
                ok = all(_close(float(p), float(q), tolerance) for p, q in zip(x, y, strict=True))
            elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
                ok = _close(float(x), float(y), tolerance)
            else:
                ok = x == y
            if not ok:
                return f"private case {key[0]}/{key[1]} {field}: recorded {x!r}, reproduced {y!r}"
    return None


def compare_public(recorded, reproduced, tolerance):
    if recorded.keys() != reproduced.keys():
        return "public case set differs"
    for key in sorted(recorded):
        if not _close(float(recorded[key]), float(reproduced[key]), tolerance):
            return f"public case {key}: recorded {recorded[key]!r}, reproduced {reproduced[key]!r}"
    return None


def reproduce_decision(
    name, decision, *, objects, workdir, reveal, sidecar=None, device=None, tolerance=1e-3
) -> list[CheckResult]:
    """Re-score one decision. ``reveal(panel_digest)`` → (panel, body) or None."""
    if "scoring" not in decision:
        return [CheckResult(name, WARN, "decision predates the scoring binding; not reproducible")]
    try:
        scoring = check_scoring_binding(decision)
    except (KeyError, ValueError) as e:
        return [CheckResult(name, FAIL, f"scoring binding: {e}")]
    verdict = decision["public"]["verdict"]
    pointers = (verdict["reference"], verdict["candidate"])
    root = Path(workdir) / digest(decision)
    samples, seed = scoring["samples"], scoring["seed"]
    device = device or scoring["device"]

    def revealed(panel_digest):
        found = reveal(panel_digest)
        if found is None:
            return None
        panel, body = found
        objects.add(body)
        return panel

    try:
        paths = [
            export_checkpoint(objects, scoring["manifests"][p], scoring["architecture"],
                              root / ("checkpoint-" + digest(p)))
            for p in pointers
        ]
    except Exception as e:  # noqa: BLE001 — a missing/corrupt bound checkpoint fails the decision
        return [CheckResult(name, FAIL, f"bound checkpoint unavailable: {e}")]

    results = []
    panel = revealed(scoring["private_panel"])
    if panel is None:
        results.append(CheckResult(f"{name}/private", WARN, "private panel not revealed yet"))
    else:
        windows = panel_windows(objects, panel)
        rows = [score_private(path, windows, samples=samples, device=device, seed=seed)
                for path in paths]
        diverged = (compare_private(decision["reference_private"], rows[0], tolerance)
                    or compare_private(decision["candidate_private"], rows[1], tolerance))
        actual = private_verdict(*rows, PrivatePolicy(**decision["private"]["policy"]),
                                 target_domain=decision["private"]["target_domain"])
        recorded = decision["private"]
        if diverged:
            results.append(CheckResult(f"{name}/private", FAIL, diverged))
        elif (actual["status"], actual["passed"]) != (recorded["status"], recorded["passed"]):
            results.append(CheckResult(f"{name}/private", FAIL,
                                       f"verdict {actual['status']} != recorded {recorded['status']}"))
        else:
            results.append(CheckResult(f"{name}/private", PASS,
                                       f"{len(rows[1])} cases reproduce; {actual['status']}"))

    guard = GuardContract.from_dict(decision["public"]["guard"])
    public_panels, public_sidecar, missing = None, None, None
    if "panels" in scoring["public"]:
        public_panels = {suite: revealed(d) for suite, d in scoring["public"]["panels"].items()}
        if None in public_panels.values():
            missing = "public panels not revealed yet"
    elif sidecar is None:
        missing = "no public benchmark sidecar configured (--sidecar)"
    elif sidecar.revisions != scoring["public"]["sidecar"]:
        missing = "sidecar dataset revisions differ from the decision's; use the bound revisions"
    else:
        public_sidecar = sidecar
    if missing:
        results.append(CheckResult(f"{name}/public", WARN, missing))
        return results
    evaluator = CheckpointEvaluator(
        objects, root / "evaluation", guard=guard,
        private_policy=PrivatePolicy(**decision["private"]["policy"]), private_panel=None,
        public_panels=public_panels, sidecar=public_sidecar,
        samples=samples, seed=seed, device=device)
    try:
        rows = [evaluator._public_scores(p, path) for p, path in zip(pointers, paths, strict=True)]
    finally:
        evaluator.close()
    public = decision["public"]
    diverged = (compare_public(public["reference_rows"], rows[0], tolerance)
                or compare_public(public["candidate_rows"], rows[1], tolerance))
    actual = judge_public(guard, *rows, reference_id=pointers[0], candidate_id=pointers[1],
                          private_pass=verdict["private_pass"])
    if diverged:
        results.append(CheckResult(f"{name}/public", FAIL, diverged))
    elif actual["passed"] != verdict["passed"]:
        results.append(CheckResult(f"{name}/public", FAIL,
                                   f"verdict {actual['status']} != recorded {verdict['status']}"))
    else:
        results.append(CheckResult(f"{name}/public", PASS,
                                   f"{len(rows[1])} cases reproduce; {actual['status']}"))
    return results


def reproduce_journal(state, **kw) -> list[CheckResult]:
    """Re-score every decision in a replayed (already verified) journal state."""
    decisions = journal_decisions(state)
    if not decisions:
        return [CheckResult("decisions", WARN, "journal records no decision yet")]
    results = []
    for key, decision in decisions.items():
        results += reproduce_decision(f"decision:{key[:12]}", decision, **kw)
    return results

