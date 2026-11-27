"""Eval-pod cache dirs are placed on the pod's data mount (2026-11-27 RCA guard,
docs/launch-lessons-2026-11.md).

2026-11-27 16:30: the lane B settlement rebase failed "pod disk short … 6.8 GB
free < 26.0 GB needed" on valeval-blackwell4: ``<workdir>/_lane_b_offload``
sat on the 43 GB ``/`` overlay while a 159 GB ``/workspace`` was empty. #982
routes the eval-pod cache dirs through ``cascade.shared.pod_data_root`` (a host
``cache_root``, else a symlink onto the largest data mount).

This AST inventory (torch-free) finds every ``f"{<host>.workdir}/…"`` path the
eval-pod offload modules build and requires its function to go through the
placement (``base_dir`` / ``cache_root`` / ``_placement`` /
``place_on_data_mount_command``), or the site to be REVIEWED below with a
reason. A new pod cache dir written under the root overlay fails here.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The modules that stage checkpoints / caches on eval pods.
EVAL_POD_MODULES = (
    "cascade/trainer/lane_b_offload.py",
    "cascade/trainer/lane_b_bench.py",
    "cascade/validator/eval_offload.py",
    "cascade/validator/mkt_offload.py",
)
PLACEMENT_NAMES = {"base_dir", "cache_root", "_placement", "place_on_data_mount_command",
                   "_root"}
# "<path>::<function>" -> why the site may stay on <workdir>.
REVIEWED = {
    "cascade/validator/eval_offload.py::build_bench_remote_command":
        "the <workdir>/benchmarks path is the uv --project code dir, not a cache",
    "cascade/validator/mkt_offload.py::upload":
        "KNOWN GAP (2026-11-27 RCA, OPEN): MarketCast eval pods cache checkpoints under "
        "<workdir>/<CACHE_DIR> on the root overlay; route through pod_data_root",
}


def _workdir_fstrings(fn: ast.AST):
    for node in ast.walk(fn):
        if not isinstance(node, ast.JoinedStr) or not node.values:
            continue
        first = node.values[0]
        if (isinstance(first, ast.FormattedValue) and isinstance(first.value, ast.Attribute)
                and first.value.attr == "workdir"):
            yield node


def _names(fn: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def unplaced_sites(root: Path = ROOT, modules=EVAL_POD_MODULES) -> dict[str, int]:
    sites: dict[str, int] = {}
    for rel in modules:
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            hits = list(_workdir_fstrings(fn))
            if hits and not (PLACEMENT_NAMES & _names(fn)):
                sites.setdefault(f"{rel}::{fn.name}", hits[0].lineno)
    return sites


def test_every_eval_pod_cache_dir_is_placed_on_the_data_mount():
    sites = unplaced_sites()
    new = {k: v for k, v in sites.items() if k not in REVIEWED}
    assert not new, (
        "eval-pod path(s) under <workdir> with no data-mount placement (2026-11-27: "
        "_lane_b_offload filled the 43 GB root while /workspace sat empty): "
        + ", ".join(f"{k} (line {v})" for k, v in sorted(new.items()))
        + ". Use pod_data_root.base_dir(host) / place_on_data_mount_command, or add a "
        "REVIEWED reason here.")


def test_reviewed_sites_still_exist():
    for key, reason in REVIEWED.items():
        rel, fn = key.split("::")
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert f"def {fn}(" in src, f"REVIEWED names a function that is gone: {key}"
        assert len(reason) > 10


def test_the_pre_982_offload_root_shape_is_flagged(tmp_path):
    # lane_b_offload before #982: the cache root was <workdir>/_lane_b_offload, unplaced
    rel = "cascade/trainer/lane_b_offload.py"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_text(
        "class R:\n    def remote_dir(self, host):\n"
        "        return f\"{host.workdir}/_lane_b_offload\"\n")
    assert unplaced_sites(tmp_path, (rel,)) == {f"{rel}::remote_dir": 3}
    (tmp_path / rel).write_text(
        "class R:\n    def remote_dir(self, host):\n"
        "        return f\"{base_dir(host)}/_lane_b_offload\" or f\"{host.workdir}/x\"\n")
    assert unplaced_sites(tmp_path, (rel,)) == {}
