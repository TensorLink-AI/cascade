"""Real checkpoint scoring and strict public-suite adapters for direct competition."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from safetensors.torch import load

from ..eval.bootstrap import paired_bootstrap_lcb_aggregated
from ..eval.direct_guard import make_receipt
from ..shared.direct_moe import canonical, digest
from ..trainer.moe_model import QUANTILE_LEVELS, CascadeMoEConfig
from ..trainer.moe_trainer import _save_moe_checkpoint
from ..validator.evaluator import evaluate_checkpoint

SCORER = "direct-private/1"
SUITE_NAMES = {"GIFT": "gift-eval", "FEV": "fev", "TIME": "time", "BOOM": "boom"}


def export_checkpoint(objects, manifest, architecture, destination):
    """Materialize only on an evaluation/training worker, never the controller."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    weights = load(objects.get(manifest["ranks"]["0"]["weights"]["uri"]))
    # Shape/source contract verified by constructing the declared architecture.
    from ..trainer.direct_model import build_direct_model

    model = build_direct_model(architecture, device="meta")
    expected = model.state_dict()
    if weights.keys() != expected.keys() or any(
        weights[k].shape != expected[k].shape for k in weights
    ):
        raise ValueError("checkpoint differs from admitted architecture")
    cfg = CascadeMoEConfig.from_dict(architecture["config"])
    _save_moe_checkpoint(
        root,
        weights,
        cfg,
        QUANTILE_LEVELS[: cfg.num_quantiles],
        SimpleNamespace(
            arch_preset="direct-contract-" + digest(architecture), input_transform="arcsinh_causal"
        ),
    )
    return root


@dataclass(frozen=True)
class PrivatePolicy:
    alpha: float
    draws: int
    seed: str
    gain_lcb: float
    min_clusters: int
    domain_tolerance: float
    min_domain_clusters: int

    def __post_init__(self):
        if (
            not 0 < self.alpha < 0.5
            or self.draws < 2
            or self.min_clusters < 2
            or self.min_domain_clusters < 2
        ):
            raise ValueError("invalid private evaluation policy")
        if (
            not self.seed
            or not math.isfinite(self.gain_lcb)
            or not math.isfinite(self.domain_tolerance)
            or self.domain_tolerance < 0
        ):
            raise ValueError("explicit finite private gain/retention settings required")


def private_verdict(reference_rows, candidate_rows, policy, *, target_domain=None):
    def index(rows):
        result = {}
        for row in rows:
            key = (row["series_id"], row["channel"])
            if key in result:
                raise ValueError("duplicate private case")
            result[key] = row
        return result

    k, c = index(reference_rows), index(candidate_rows)
    result = {
        "passed": False,
        "status": "inconclusive",
        "bounds": {},
        "policy": asdict(policy),
        "target_domain": target_domain,
    }
    if not k or k.keys() != c.keys():
        return result
    keys = sorted(k)
    for key in keys:
        if (
            k[key]["source"] != c[key]["source"]
            or k[key]["domain"] != c[key]["domain"]
            or not k[key]["source"]
        ):
            return result
        for row in (k[key], c[key]):
            if (
                not np.isfinite([*row["qloss_per_q"], row["abs_target"], row["mase"]]).all()
                or row["abs_target"] <= 0
                or row["mase"] < 0
            ):
                return result
    groups = {
        "gain": [key for key in keys if target_domain is None or k[key]["domain"] == target_domain]
    }
    if target_domain:
        groups.update(
            {
                domain: [key for key in keys if k[key]["domain"] == domain]
                for domain in sorted({row["domain"] for row in k.values()} - {target_domain})
            }
        )
    for group, selected in groups.items():
        clusters = [k[key]["source"] for key in selected]
        minimum = policy.min_clusters if group == "gain" else policy.min_domain_clusters
        if len(set(clusters)) < minimum:
            return result
        arrays = []
        for rows in (k, c):
            arrays.extend(
                np.asarray([rows[key][field] for key in selected])
                for field in ("qloss_per_q", "abs_target", "mase")
            )
        lcb = paired_bootstrap_lcb_aggregated(
            *arrays,
            alpha=policy.alpha,
            B=policy.draws,
            seed=f"{policy.seed}|{group}",
            clusters=clusters,
        )
        if not math.isfinite(lcb):
            return result
        result["bounds"][group] = float(lcb)
    result["passed"] = result["bounds"]["gain"] >= policy.gain_lcb and all(
        value >= -policy.domain_tolerance
        for key, value in result["bounds"].items()
        if key != "gain"
    )
    result["status"] = "accepted" if result["passed"] else "rejected"
    return result


def score_private(checkpoint, windows, *, samples, device, seed):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    scores = evaluate_checkpoint(checkpoint, windows, num_samples=samples, device=device)
    return [{**asdict(row), "qloss_per_q": row.qloss_per_q.tolist()} for row in scores]


class PublicSidecar:
    """GIFT, FEV, TIME and BOOM through the existing pinned benchmark sidecar.

    Expected revisions and exact task IDs are supplied by the acceptance contract.
    Missing suites, partial suites, skipped tasks and extra/missing fixed cases are
    errors; a failed sidecar never becomes an empty passing benchmark.
    """

    def __init__(self, python, *, data_dir, revisions, timeout, device="cpu", fev_manifest=None):
        if set(revisions) != set(SUITE_NAMES) or not all(revisions.values()):
            raise ValueError("pin all four dataset revisions")
        self.python, self.data_dir, self.revisions = str(python), str(data_dir), revisions
        self.timeout, self.device = timeout, device
        self.fev_manifest = fev_manifest

    def rows_from_report(self, report, guard):
        by_suite = {}
        for suite in report["suites"]:
            name = suite["suite"]
            if name in by_suite:
                raise ValueError("duplicate suite")
            by_suite[name] = suite
        out = {}
        for logical, name in SUITE_NAMES.items():
            suite = by_suite.get(name)
            if (
                suite is None
                or suite["status"] != "ok"
                or suite.get("skipped")
                or suite.get("n_expected", 0) != len(suite.get("rows", []))
                or report["data_revisions"].get(name) != self.revisions[logical]
            ):
                raise ValueError(f"incomplete/unpinned public suite: {logical}")
            entries = {}
            for row in suite["rows"]:
                if row["full"] in entries:
                    raise ValueError("duplicate benchmark case")
                entries[row["full"]] = row
            expected = [case for case in guard.cases if case.suite == logical]
            if set(entries) != {case.case for case in expected}:
                raise ValueError("benchmark case contract mismatch")
            for case in expected:
                metric = {"crps": "CRPS", "mase": "MASE"}.get(case.metric)
                if metric is None:
                    raise ValueError("unsupported metric mapping")
                value = entries[case.case].get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid benchmark metric")
                out[case.key] = value
        return out

    def score(self, checkpoint, guard):
        if self.fev_manifest is None:
            raise ValueError("pinned local FEV task/data manifest required")
        with tempfile.TemporaryDirectory(prefix="direct-bench-") as temporary:
            output = Path(temporary) / "report.json"
            command = [
                self.python,
                "-m",
                "cascade_benchmark.direct_cli",
                "--fev-manifest",
                str(self.fev_manifest),
                str(checkpoint),
                str(output),
                "--suites",
                "gift-eval,fev,time,boom",
                "--data-dir",
                self.data_dir,
                "--device",
                self.device,
            ]
            subprocess.run(command, check=True, timeout=self.timeout, capture_output=True)
            return self.rows_from_report(json.loads(output.read_text()), guard)


def decision_receipt(
    guard,
    reference_public,
    candidate_public,
    reference_private,
    candidate_private,
    policy,
    *,
    reference_id,
    candidate_id,
    target_domain=None,
    scoring=None,
):
    private = private_verdict(
        reference_private, candidate_private, policy, target_domain=target_domain
    )
    public = make_receipt(
        guard,
        reference_public,
        candidate_public,
        reference_id=reference_id,
        candidate_id=candidate_id,
        private_pass=private["passed"] if private["status"] != "inconclusive" else None,
    )
    receipt = {
        "protocol": "direct-decision/1",
        "public": public,
        "private": private,
        "reference_private": reference_private,
        "candidate_private": candidate_private,
    }
    if scoring is not None:
        # What an auditor needs to reproduce every score above (cascade-audit
        # direct-moe --reproduce): the two checkpoint manifests, the architecture
        # and the content digests of the panels and scorer settings.
        receipt["scoring"] = scoring
    return receipt


def check_scoring_binding(receipt):
    scoring = receipt["scoring"]
    verdict = receipt["public"]["verdict"]
    manifests = scoring["manifests"]
    if set(manifests) != {verdict["reference"], verdict["candidate"]} or any(
        digest(manifest) != pointer for pointer, manifest in manifests.items()
    ):
        raise ValueError("scoring binding does not name the judged checkpoints")
    if "block" in scoring and (type(scoring["block"]) is not int or scoring["block"] < 0):
        raise ValueError("invalid scoring block")
    if scoring["scorer"] != SCORER or len(scoring["private_panel"]) != 64:
        raise ValueError("unsupported scoring binding")
    return scoring


def replay_decision(receipt):
    if receipt["protocol"] != "direct-decision/1":
        raise ValueError("unsupported decision receipt")
    from ..eval.direct_guard import replay_receipt

    private = receipt["private"]
    actual = private_verdict(
        receipt["reference_private"],
        receipt["candidate_private"],
        PrivatePolicy(**private["policy"]),
        target_domain=private["target_domain"],
    )
    if canonical(actual) != canonical(private):
        raise ValueError("private decision replay mismatch")
    if "scoring" in receipt:
        check_scoring_binding(receipt)
    expected = actual["passed"] if actual["status"] != "inconclusive" else None
    if receipt["public"]["verdict"]["private_pass"] is not expected:
        raise ValueError("private/public verdict binding mismatch")
    return replay_receipt(receipt["public"])
