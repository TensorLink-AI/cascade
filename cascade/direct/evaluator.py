"""Persistent evaluation worker using actual forecasts and pinned held-out panels."""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

import numpy as np

from ..eval.window import EvalWindow
from ..shared.direct_moe import SUITES, canonical, digest
from .evaluation import SCORER, decision_receipt, export_checkpoint, score_private


def panel_windows(objects, panel):
    """A checksum-verified held-out panel as evaluation windows."""
    import hashlib

    body = objects.get(panel["object"]["uri"])
    if hashlib.sha256(body).hexdigest() != panel["object"]["sha256"]:
        raise ValueError("evaluation panel checksum mismatch")
    with np.load(io.BytesIO(body), allow_pickle=False) as arrays:
        history, target = arrays["history"], arrays["target"]
    if len(history) != len(panel["cases"]) or len(target) != len(history):
        raise ValueError("panel/case count mismatch")
    if not np.isfinite(history).all() or not np.isfinite(target).all():
        raise ValueError("invalid held-out panel values")
    if len({row["id"] for row in panel["cases"]}) != len(panel["cases"]):
        raise ValueError("duplicate panel case")
    return [
        EvalWindow(
            row["id"],
            history[i],
            target[i],
            {
                "source": row["source"],
                "domain": row["domain"],
                "seasonal_period": row["seasonal_period"],
            },
        )
        for i, row in enumerate(panel["cases"])
    ]


class CheckpointEvaluator:
    def __init__(
        self,
        objects,
        root,
        *,
        guard,
        private_policy,
        private_panel,
        public_panels=None,
        sidecar=None,
        initial_artifacts=None,
        samples,
        seed,
        device="cpu",
        block_source=None,
    ):
        if (public_panels is None) == (sidecar is None):
            raise ValueError(
                "choose pinned-panel fixture evaluation OR real public benchmark sidecar"
            )
        if public_panels is not None and set(public_panels) != set(SUITES):
            raise ValueError("all four public panels required")
        self.objects, self.root = objects, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.guard, self.private_policy, self.private_panel = guard, private_policy, private_panel
        self.public_panels, self.sidecar = public_panels, sidecar
        self.initial_artifacts = initial_artifacts or {}
        self.samples, self.seed, self.device = samples, seed, device
        # Chain block at scoring time; an owner network config requires it to
        # pick the private panel active then (DEC-CA-0052 §9).
        self.block_source = block_source
        self.db = sqlite3.connect(str(self.root / "evaluation.sqlite"), isolation_level=None)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, body TEXT NOT NULL)"
        )

    def manifest(self, pointer, state):
        if pointer in self.initial_artifacts:
            return self.initial_artifacts[pointer]
        for merge in state["merged"]:
            if "manifest" in merge and digest(merge["manifest"]) == pointer:
                return merge["manifest"]
        for job in state["jobs"].values():
            if job["checkpoint"] is not None and digest(job["checkpoint"]) == pointer:
                return job["checkpoint"]
        raise ValueError("unknown immutable checkpoint reference")

    def windows(self, panel):
        return panel_windows(self.objects, panel)

    def _private_scores(self, pointer, checkpoint, panel):
        key = digest(
            {
                "checkpoint": pointer,
                "panel": panel,
                "samples": self.samples,
                "seed": self.seed,
                "device": self.device,
                "scorer": SCORER,
            }
        )
        old = self.db.execute("SELECT body FROM results WHERE key=?", (key,)).fetchone()
        if old:
            return json.loads(old[0])
        rows = score_private(
            checkpoint,
            self.windows(panel),
            samples=self.samples,
            device=self.device,
            seed=self.seed,
        )
        self.db.execute("INSERT INTO results VALUES (?,?)", (key, canonical(rows)))
        return rows

    def _public_scores(self, pointer, checkpoint):
        if self.sidecar is not None:
            key = digest(
                {
                    "checkpoint": pointer,
                    "guard": self.guard.to_dict(),
                    "revisions": self.sidecar.revisions,
                }
            )
            old = self.db.execute("SELECT body FROM results WHERE key=?", (key,)).fetchone()
            if old:
                return json.loads(old[0])
            rows = self.sidecar.score(checkpoint, self.guard)
            self.db.execute("INSERT INTO results VALUES (?,?)", (key, canonical(rows)))
            return rows
        output = {}
        for suite, panel in self.public_panels.items():
            if panel.get("fixture") is not True:
                raise ValueError(
                    "native suite integration requires PublicSidecar; panel path is fixture-only"
                )
            rows = self._private_scores(pointer, checkpoint, panel)
            grouped = {}
            for row in rows:
                grouped.setdefault(row["series_id"], []).append(row)
            for case in (c for c in self.guard.cases if c.suite == suite):
                group = grouped[case.case]
                if {row["source"] for row in group} != {case.cluster}:
                    raise ValueError("public source-cluster provenance mismatch")
                if case.metric == "mase":
                    output[case.key] = float(np.mean([r["mase"] for r in group]))
                elif case.metric == "crps":
                    output[case.key] = float(
                        np.mean([2 * np.mean(r["qloss_per_q"]) / r["abs_target"] for r in group])
                    )
                else:
                    raise ValueError("unsupported public metric")
        return output

    def evaluate(self, job, state, *, reference_id=None):
        j = state["jobs"][job]
        reference_id = reference_id or j["terms"]["reference"]
        candidate_id = digest(j["checkpoint"])
        return self.evaluate_artifacts(
            reference_id,
            candidate_id,
            state,
            j["work"]["architecture"],
            target_domain=j["work"].get("expert_domain"),
        )

    def evaluate_artifacts(
        self, reference_id, candidate_id, state, architecture, *, target_domain=None
    ):
        paths, manifests = [], {}
        for pointer in (reference_id, candidate_id):
            target = self.root / ("checkpoint-" + digest(pointer))
            manifests[pointer] = self.manifest(pointer, state)
            paths.append(export_checkpoint(self.objects, manifests[pointer], architecture, target))
        private = [
            self._private_scores(pointer, path, self.private_panel)
            for pointer, path in zip((reference_id, candidate_id), paths, strict=True)
        ]
        public = [
            self._public_scores(pointer, path)
            for pointer, path in zip((reference_id, candidate_id), paths, strict=True)
        ]
        return decision_receipt(
            self.guard,
            *public,
            *private,
            self.private_policy,
            reference_id=reference_id,
            candidate_id=candidate_id,
            target_domain=target_domain,
            scoring=self.scoring(manifests, architecture),
        )

    def scoring(self, manifests, architecture):
        if self.sidecar is not None:
            public = {"sidecar": self.sidecar.revisions}
        else:
            public = {"panels": {suite: digest(p) for suite, p in self.public_panels.items()}}
        stamp = {} if self.block_source is None else {"block": int(self.block_source())}
        return {
            **stamp,
            "scorer": SCORER,
            "manifests": manifests,
            "architecture": architecture,
            "private_panel": digest(self.private_panel),
            "public": public,
            "samples": self.samples,
            "seed": self.seed,
            "device": self.device,
        }

    def close(self):
        self.db.close()
