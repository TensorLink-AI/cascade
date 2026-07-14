"""Pod health gate — a rented box must PROVE itself before it enters hosts.toml.

A marketplace pod is untrusted hardware behind a thin API: the SKU label can be
wrong (L40 sold as L40S), the image can have drifted off the runtime pin, disks
arrive full, and object storage may be unreachable from that datacenter. Any of
these discovered *mid-round* costs a training slot (or, on a final pod, the
round); discovered *here* it costs one replacement rental. So every check that
the trainer's contract depends on runs up front, and a pod joins the fleet only
when ALL of them pass:

1. **ssh echo** — the transport works in ``BatchMode`` (no interactive auth).
2. **gpu sku** — ``nvidia-smi`` reports the stage's exact device string on
   EVERY GPU (a multi-GPU pod prints one line per GPU). Exact, not substring:
   ``L40`` != ``L40S``, and the final's ``expected_gpu`` pin is byte-compared
   by the validator.
3. **runtime pin** — the pod venv's ``python``/``torch`` match the repo's
   pinned runtime (PR #75: python 3.11 + torch 2.4.1+cu124). An unpinned stack
   silently changes numerics and invalidates the KOTH comparison.
4. **worker import** — ``cascade.trainer.worker`` imports in the pod venv, so
   the first dispatch won't die on a missing dependency.
5. **image digest** — ``CASCADE_TRAIN_IMAGE_DIGEST`` matches the pinned image
   when ``[training] train_image_digest`` is set (the worker refuses a final
   run otherwise — better to bounce the pod now).
6. **hippius reachability** — an injected HEAD-probe callable, run from the
   orchestrator's viewpoint of the pod's storage path (checkpoint pushes and
   generator pulls both need it).
7. **disk headroom** — enough free GB under the workdir for a generator +
   corpus + checkpoint; full disks fail slowly and confusingly.

Everything here is a pure predicate over an injected ``run_ssh(argv) →
CompletedProcess``-style boundary; the replace-once policy on failure lives in
the service loop, which reads the per-check :class:`HealthReport` this module
returns.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .core import DEFAULT_REMOTE_PYTHON, DEFAULT_WORKDIR

__all__ = [
    "CheckResult",
    "EXPECTED_PYTHON",
    "EXPECTED_TORCH",
    "HealthGate",
    "HealthReport",
]

# The repo's pinned runtime (PR #75): .python-version pins the interpreter and
# pyproject pins torch==2.4.1 from the cu124 index — every pod must match, or
# king/challenger numerics drift.
EXPECTED_PYTHON = "3.11"
EXPECTED_TORCH = "2.4.1+cu124"

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")


class _ProcLike(Protocol):
    returncode: int
    stdout: str
    stderr: str


RunSSH = Callable[[Sequence[str]], _ProcLike]


@dataclass(frozen=True)
class CheckResult:
    """One health check's verdict: ``name``, pass/fail, and a human detail line."""

    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class HealthReport:
    """Every check's result for one pod. The pod is healthy iff ALL passed."""

    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.ok)

    def summary(self) -> str:
        """One log line: ``gpu_sku=FAIL(got 'NVIDIA L40'), disk=ok, …``."""
        return ", ".join(
            f"{c.name}={'ok' if c.ok else 'FAIL(' + c.detail + ')'}" for c in self.checks
        )


@dataclass(frozen=True)
class HealthGate:
    """The expectations one stage's pods are gated on.

    ``sku`` and ``gpus`` come from the stage policy; ``image_digest`` is the
    chain.toml ``[training] train_image_digest`` pin (empty ⇒ unpinned, check
    skipped — matching ``assert_train_image``); ``expected_python`` /
    ``expected_torch`` default to the repo's pinned runtime and exist as
    parameters only so a runtime re-pin is a config change here, not a code
    change. ``hippius_probe`` is the injected storage HEAD check (``None`` ⇒
    skipped, e.g. in ``--dry-run``).
    """

    sku: str
    gpus: int = 1
    remote_python: str = DEFAULT_REMOTE_PYTHON
    workdir: str = DEFAULT_WORKDIR
    image_digest: str = ""
    min_disk_gb: float = 20.0
    expected_python: str = EXPECTED_PYTHON
    expected_torch: str = EXPECTED_TORCH
    hippius_probe: Callable[[], bool] | None = field(default=None, compare=False)

    # ── the gate ─────────────────────────────────────────────────────────────

    def check(self, run_ssh: RunSSH) -> HealthReport:
        """Run every check over ``run_ssh`` and return the full report.

        All checks always run (a failed pod's report should show *everything*
        wrong with it, not just the first thing — the operator reads this when
        a provider keeps selling bad boxes). ``run_ssh`` receives the REMOTE
        argv; the caller binds the ssh transport (host/port/key/BatchMode).
        A transport exception fails that check rather than the gate itself.
        """
        checks = (
            ("ssh_echo", self._check_echo),
            ("gpu_sku", self._check_gpu_sku),
            ("runtime_pin", self._check_runtime),
            ("worker_import", self._check_worker_import),
            ("image_digest", self._check_image_digest),
            ("hippius", self._check_hippius),
            ("disk", self._check_disk),
        )
        results = []
        for name, fn in checks:
            try:
                ok, detail = fn(run_ssh)
            except Exception as e:  # noqa: BLE001 — a dead transport is a failed check
                ok, detail = False, f"check errored: {e}"
            results.append(CheckResult(name=name, ok=ok, detail=detail))
        return HealthReport(checks=tuple(results))

    # ── individual checks (pure over the injected boundary) ──────────────────

    def _check_echo(self, run_ssh: RunSSH) -> tuple[bool, str]:
        proc = run_ssh(["echo", "cascade-health-ok"])
        if proc.returncode != 0:
            return False, f"ssh rc={proc.returncode}: {(proc.stderr or '')[-200:]}"
        if proc.stdout.strip() != "cascade-health-ok":
            return False, f"unexpected echo output {proc.stdout.strip()!r}"
        return True, ""

    def _check_gpu_sku(self, run_ssh: RunSSH) -> tuple[bool, str]:
        proc = run_ssh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if proc.returncode != 0:
            return False, f"nvidia-smi rc={proc.returncode}"
        names = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        if len(names) < self.gpus:
            return False, f"{len(names)} GPU(s) visible, need {self.gpus}"
        # Exact match on EVERY line — L40 != L40S, and a pod mixing SKUs is
        # useless for a stage that must be homogeneous.
        bad = sorted({n for n in names if n != self.sku})
        if bad:
            return False, f"got {bad}, want {self.sku!r} on every GPU"
        return True, ""

    def _check_runtime(self, run_ssh: RunSSH) -> tuple[bool, str]:
        proc = run_ssh([
            self.remote_python, "-c",
            "import sys, torch; print('.'.join(map(str,sys.version_info[:2])), torch.__version__)",
        ])
        if proc.returncode != 0:
            return False, f"runtime probe rc={proc.returncode}: {(proc.stderr or '')[-200:]}"
        got = proc.stdout.strip()
        want = f"{self.expected_python} {self.expected_torch}"
        if got != want:
            return False, f"runtime {got!r} != pinned {want!r}"
        return True, ""

    def _check_worker_import(self, run_ssh: RunSSH) -> tuple[bool, str]:
        proc = run_ssh([self.remote_python, "-c", "import cascade.trainer.worker"])
        if proc.returncode != 0:
            return False, f"import failed rc={proc.returncode}: {(proc.stderr or '')[-200:]}"
        return True, ""

    def _check_image_digest(self, run_ssh: RunSSH) -> tuple[bool, str]:
        pinned = _sha256_of(self.image_digest)
        if not self.image_digest:
            return True, "unpinned"
        if pinned is None:
            return False, f"pin {self.image_digest!r} carries no sha256:<64hex>"
        proc = run_ssh(["printenv", "CASCADE_TRAIN_IMAGE_DIGEST"])
        if proc.returncode != 0:
            return False, "CASCADE_TRAIN_IMAGE_DIGEST unset on pod (inject at launch)"
        runtime = _sha256_of(proc.stdout)
        if runtime != pinned:
            return False, f"pod digest {runtime} != pinned {pinned}"
        return True, ""

    def _check_hippius(self, run_ssh: RunSSH) -> tuple[bool, str]:  # noqa: ARG002 — orchestrator-side probe
        if self.hippius_probe is None:
            return True, "no probe configured"
        return (True, "") if self.hippius_probe() else (False, "hippius HEAD probe failed")

    def _check_disk(self, run_ssh: RunSSH) -> tuple[bool, str]:
        # -P (POSIX) keeps one record per line; -k gives 1024-byte blocks.
        proc = run_ssh(["df", "-Pk", self.workdir])
        if proc.returncode != 0:
            return False, f"df rc={proc.returncode}"
        avail_gb = _df_avail_gb(proc.stdout or "")
        if avail_gb is None:
            return False, f"unparseable df output {(proc.stdout or '')[:100]!r}"
        if avail_gb < self.min_disk_gb:
            return False, f"{avail_gb:.1f} GB free < {self.min_disk_gb:g} GB required"
        return True, ""


def _sha256_of(value: str) -> str | None:
    """Extract ``sha256:<64hex>`` from a digest pin or env value — accepts a full
    digest-pinned ref or a bare digest (same normalisation as
    ``cascade.trainer.contract.assert_train_image``)."""
    m = _SHA256_RE.search((value or "").strip().lower())
    return m.group(0) if m else None


def _df_avail_gb(df_output: str) -> float | None:
    """The 'Available' column of ``df -Pk <path>``, in GB (``None`` if unparseable).

    POSIX format: header line, then one record per filesystem —
    ``Filesystem 1024-blocks Used Available Capacity Mounted on``.
    """
    lines = [ln for ln in df_output.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    fields = lines[-1].split()
    if len(fields) < 4:
        return None
    try:
        return int(fields[3]) / (1024.0 * 1024.0)
    except ValueError:
        return None
