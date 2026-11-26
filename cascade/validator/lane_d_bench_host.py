"""The box's end of the lane D gate sweep (LANE_D_PLAN rev6.13, W3b6a;
``tests/accept/w3b6a/README.md``).

:class:`SSHBenchHost` is W3b2's ``Sweep`` (``sweep(digests, suites) ->
{digest: {suite: entry}}``) for :class:`~cascade.validator.lane_d_bench.
SweepSession`. Each call resolves the first ``final`` / ``any`` ``[[host]]``
of a hosts file re-read at every call (a ``bench_lease``-published file works
unchanged), sends the bench host the digests, suites, each digest's
effective context (set per settlement by the W8c source) and its presigned
GET grant (never weights, histories or targets: D12, §10.2), and runs
``python -m cascade.eval.lane_d_bench_sweep sweep`` there over SSH.

Every failure raises (no host, a digest without a context, a non-zero exit,
a malformed or unexpected reply): ``lane_d._run_gate`` turns it into
``bench_hold``. A row is never guessed.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

REQUEST_SCHEMA = "lane-d-bench-request/1"
REPLY_SCHEMA = "lane-d-bench-reply/1"
DEFAULT_TIMEOUT_S = 6 * 3600
SETTINGS_KEYS = ("num_samples", "max_series", "batch_size", "workers", "data_dir")
# The sidecar's batch size; lane A's DEC-CA-0063 sweep never overrides it
# (``eval_offload.run_bench_via_host_many`` / ``benchmarks.run_bench_reports``).
LANE_A_BATCH_SIZE = 512


def bench_settings(cfg) -> dict:
    """The sidecar settings lane A's DEC-CA-0063 sweep runs with
    (``validator_bench.ValidatorBenchMixin._vbench_run_sidecar_many``), read
    from ``cfg.eval`` so a lane D and a lane A bench of one checkpoint
    match, plus lane A's per-checkpoint allowance (box-side only)."""
    ec = cfg.eval
    return {"num_samples": int(ec.benchmark_num_samples or ec.num_samples),
            "max_series": int(ec.cascade_bench_max_series),
            "batch_size": LANE_A_BATCH_SIZE,
            "workers": str(getattr(ec, "bench_workers", "1")),
            "data_dir": ec.gift_gate_data_dir or None,
            "timeout_s_per_checkpoint": (int(ec.bench_hold_max_hours * 3600)
                                         if ec.cascade_bench_max_series == 0 else 1800)}


class LaneDBenchHostError(RuntimeError):
    """The bench host cannot answer this sweep (the gate holds)."""


class SSHBenchHost:
    """``host(digests, suites)``: one sweep on the bench host (module
    docstring). ``grants(digest)`` gives a presigned GET grant or ``None``;
    ``run`` is the ``subprocess.run``-shaped runner."""

    def __init__(self, hosts_path, *, cache_dir: str, budget_bytes: int, settings: Mapping,
                 grants: Callable[[str], Mapping | None] | None = None,
                 run: Callable | None = None):
        self.hosts_path = Path(hosts_path)
        self.cache_dir = str(cache_dir)
        self.budget_bytes = int(budget_bytes)
        self.settings = dict(settings)
        missing = [k for k in SETTINGS_KEYS if k not in self.settings]
        if missing:
            raise ValueError(f"SSHBenchHost settings lack {missing}")
        self.timeout_per_checkpoint_s = float(
            self.settings.get("timeout_s_per_checkpoint", DEFAULT_TIMEOUT_S))
        self._grants = grants
        self._run = run or subprocess.run
        self._contexts: dict[str, int] = {}

    def set_contexts(self, contexts: Mapping[str, int]) -> None:
        """Each digest's effective context for this settlement (W8c source)."""
        self._contexts = {str(d): int(c) for d, c in contexts.items()}

    def _resolve(self):
        from ..trainer.remote import load_hosts

        if not self.hosts_path.is_file():
            raise LaneDBenchHostError(f"{self.hosts_path}: no lane D bench hosts file")
        hosts = [h for h in load_hosts(self.hosts_path) if h.stage in ("any", "final")]
        if not hosts:
            raise LaneDBenchHostError(f"{self.hosts_path}: no final/any bench host")
        return hosts[0]

    def _request(self, digests: list[str], suites: list[str]) -> dict:
        missing = [d for d in digests if d not in self._contexts]
        if missing:
            raise LaneDBenchHostError(f"no effective context for {[d[:16] for d in missing]}")
        grants = {}
        for d in digests:
            g = None if self._grants is None else self._grants(d)
            grants[d] = None if g is None else dict(g)
        return {"schema": REQUEST_SCHEMA, "digests": digests, "suites": suites,
                "contexts": {d: self._contexts[d] for d in digests}, "grants": grants,
                "settings": {k: self.settings[k] for k in SETTINGS_KEYS}}

    @staticmethod
    def _checked(reply, digests: list[str]) -> dict:
        if not isinstance(reply, dict) or reply.get("schema") != REPLY_SCHEMA:
            raise LaneDBenchHostError(f"not a {REPLY_SCHEMA} reply")
        rows = reply.get("rows")
        if not isinstance(rows, dict):
            raise LaneDBenchHostError("reply rows is not an object")
        extra = sorted(set(rows) - set(digests))
        if extra:
            raise LaneDBenchHostError(f"reply names digests it was not asked for: {extra}")
        for dg, per in rows.items():
            if not isinstance(per, dict) or not all(
                    isinstance(s, str) and isinstance(e, dict) for s, e in per.items()):
                raise LaneDBenchHostError(f"reply rows for {dg[:16]} are malformed")
        return rows

    def __call__(self, digests, suites) -> dict:
        ds = [str(d) for d in digests]
        ss = [str(s) for s in suites]
        request = self._request(ds, ss)
        host = self._resolve()
        from ..trainer.remote import build_ssh_argv

        env = ""
        if host.cuda_device is not None:
            env = f"CUDA_VISIBLE_DEVICES={shlex.quote(host.cuda_device)} "
        args = [host.remote_python, "-m", "cascade.eval.lane_d_bench_sweep", "sweep",
                "--cache-dir", self.cache_dir, "--budget-bytes", str(self.budget_bytes)]
        if host.chain_toml:
            args += ["--chain-toml", host.chain_toml]
        cmd = f"cd {shlex.quote(host.workdir)} && {env}" + " ".join(shlex.quote(a) for a in args)
        payload = json.dumps(request, sort_keys=True).encode("utf-8")
        proc = self._run(build_ssh_argv(host, cmd), input=payload, capture_output=True,
                         timeout=self.timeout_per_checkpoint_s * len(ds), check=False)
        err = (proc.stderr or b"").decode("utf-8", "replace")[-600:]
        if proc.returncode != 0:
            raise LaneDBenchHostError(f"{host.name}: lane D bench sweep exited "
                                      f"{proc.returncode}: {err}")
        lines = [ln for ln in (proc.stdout or b"").decode("utf-8", "replace").splitlines()
                 if ln.strip()]
        if not lines:
            raise LaneDBenchHostError(f"{host.name}: lane D bench sweep answered nothing: {err}")
        try:
            reply = json.loads(lines[-1])
        except ValueError as e:
            raise LaneDBenchHostError(f"{host.name}: reply is not JSON ({e})") from e
        return self._checked(reply, ds)


__all__ = ["REPLY_SCHEMA", "REQUEST_SCHEMA", "SETTINGS_KEYS", "LaneDBenchHostError", "SSHBenchHost",
           "bench_settings"]
