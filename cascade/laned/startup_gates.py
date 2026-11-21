"""Shipped-config startup-gate harness (lane D R11, plan §4 T26, §13).

``python -P -m cascade.laned.startup_gates --config F --head-block N --role
validator|trainer [--head-timestamp S] [--state-dir D]`` runs a ``chain.toml``
through every REAL startup gate of the validator or trainer, in the order the
service runs them (``cascade/<role>/main.py:main`` then ``run_forever`` up to
its poll loop), at an injected chain head.

The 2026-11-13 incidents were configs that loaded fine but that a startup gate
refused at the live head (the 128-byte note crash-loop, ``v2b_note_problem``,
PR #457; the calendar-launch trial, ``check_calendar_launch``, PR #460). This
harness lets CI run the shipped file through those gates before a release.

Nothing here re-implements a gate. Each :class:`Gate` calls the service's own
function (or runner method) on the loaded config. The only stand-ins are for
external services: :class:`HeadChain` serves the injected head (no network,
empty commitments, no wallet / hotkey), and S3 / HF are never opened. The
node's persisted activation records are read from a private copy of
``--state-dir`` (the startup path may stamp records; the given directory is
never written).

Stdout carries one ``<PASS|FAIL|WARN|SKIP> <gate>[: <reason>]`` line per gate
of ``GATES[role]``; logging goes to stderr. A FAIL stops the sequence (later
gates print SKIP), as ``main.py`` does. An advisory gate (one whose refusal the
service logs and carries on past) prints WARN. Exit 0 = no FAIL, 1 = a FAIL,
2 = usage error. Contract: ``tests/accept/r11/README.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import logging
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("cascade.laned.startup_gates")

ROLES = ("validator", "trainer")

# The node's persisted activation records, under the names the services read
# (the validator beside its state db, the trainer in its work root).
STATE_FILES = (
    "activation_state.json",
    "activation_forfeit_state.json",
    "activation_margin_v2_state.json",
    "activation_prelaunch_burn_state.json",
    "activation_image_decoupled_state.json",
    "activation_shared_ladder_state.json",
)

# Seconds per block the injected chain clock steps back from the head.
BLOCK_SECONDS = 12

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


class GateRefused(Exception):
    """A gate refused; the message is the real refusal text."""


# ── the injected chain (the one external service on the startup path) ────────


class HeadChain:
    """The chain as seen at an injected head: ``current_block`` and
    ``finalized_block`` are the head, ``block_timestamp`` follows the head's
    chain time at ``BLOCK_SECONDS`` per block (and raises without one, as a
    node that cannot serve it does), commitments are empty and the node has no
    wallet or hotkey. Nothing here opens a connection."""

    def __init__(self, head_block: int, head_timestamp: float | None) -> None:
        self.head_block = int(head_block)
        self.head_timestamp = head_timestamp

    def current_block(self) -> int:
        return self.head_block

    def finalized_block(self) -> int:
        return self.head_block

    def block_timestamp(self, block: int) -> float:
        if self.head_timestamp is None:
            raise RuntimeError("chain time unavailable: the harness was given no --head-timestamp")
        return float(self.head_timestamp) - BLOCK_SECONDS * (self.head_block - int(block))

    def validator_stakes(self, block: int | None = None) -> list:
        return []

    def read_plain_commitments(self, block: int | None = None) -> dict[str, str]:
        return {}

    def poll_commitments(self, include_history: bool = False) -> list:
        return []

    def hotkey_ss58(self) -> str:
        return ""

    def wallet(self):
        raise RuntimeError("the startup-gate harness holds no wallet")


# ── run context ──────────────────────────────────────────────────────────────


@dataclass
class Context:
    """What one harness run shares between its gates."""

    role: str
    config: Path
    chain: HeadChain
    work_dir: Path                  # private copy of --state-dir (or empty)
    cfg: object | None = None
    runner: object | None = None
    # validator: the activation-startup gate a refusal was attributed to
    activation_refusal: tuple[str, str] | None = None
    activation_ran: bool = False

    def activation_store(self):
        from ..shared.activation import ActivationStore

        return ActivationStore(self.work_dir / "activation_state.json")

    def validator_runner(self):
        """The real ``ValidatorRunner`` over the loaded config, its activation
        store at the node's records (``build_runner`` places them beside the
        state db; here, in the private state copy)."""
        if self.runner is None:
            from ..validator.loop import ValidatorRunner

            self.runner = ValidatorRunner(cfg=self.cfg, activation_store=self.activation_store())
        return self.runner

    def trainer_runner(self):
        """The real ``TrainerRunner`` over the loaded config, its activation
        store at the node's records (``main`` places them in the work root)."""
        if self.runner is None:
            from ..trainer.loop import TrainerRunner

            self.runner = TrainerRunner(cfg=self.cfg, base_trainer=None, work_root=self.work_dir,
                                        activation_store=self.activation_store())
        return self.runner


@dataclass(frozen=True)
class Gate:
    """One startup gate: ``calls`` are the callee names it runs, exactly as the
    startup path spells them; ``run`` raises on a refusal, or returns the
    refusal text of an ``advisory`` gate (which the service logs and passes)."""

    name: str
    calls: tuple[str, ...]
    advisory: bool = False
    run: Callable[[Context], str | None] = field(default=None, compare=False, repr=False)


# ── gates shared by both roles ───────────────────────────────────────────────


def _code_path(ctx: Context) -> None:
    from ..shared.codepath import verify_release_code_path

    # main() exits the process on a mismatch; here the refusal surfaces as the
    # CodePathMismatch it re-raises once ``exit_fn`` returns.
    verify_release_code_path(f"cascade-{ctx.role}", exit_fn=lambda code: None)


def _loader(ctx: Context) -> None:
    from ..shared.config import load_chain_config
    from ..shared.config.loader import require_vote_pin

    require_vote_pin()
    ctx.cfg = load_chain_config(ctx.config)


def _launch_ready(ctx: Context) -> None:
    from ..shared.config import assert_launch_ready

    assert_launch_ready(ctx.cfg, role=ctx.role)


# ── validator gates ──────────────────────────────────────────────────────────


def _bench_signer(ctx: Context) -> None:
    from ..validator.validator_bench import check_main_bench_signer

    # main(): ``runner.bench_signer = client.wallet()``, None when the node
    # holds no wallet — which is this harness's node.
    try:
        signer = ctx.chain.wallet()
    except Exception as e:  # noqa: BLE001 — no wallet => never the main validator
        log.warning("bench signer unavailable (%s); this validator never sweeps", e)
        signer = None
    check_main_bench_signer(ctx.validator_runner().cfg, signer)


# The refusal checks ``ActivationMixin._activation_startup`` runs, in its
# order, and the gate each one's refusal belongs to.
_ACTIVATION_CHECKS = (
    ("v2b_note_problem", "activation_note"),
    ("v2b_typed_notice_problem", "v2b_typed_notice"),
    ("prelaunch_retirement_problem", "prelaunch_retirement"),
)


@contextlib.contextmanager
def _observed(module, names: tuple[str, ...], seen: list) -> Iterator[None]:
    """Record ``(name, result)`` of each call to ``module.<name>`` while the
    block runs. The real function runs and its result is returned unchanged:
    the startup method imports them at call time, so this is how one call of
    the real ``_activation_startup`` is split into its gates."""
    originals = {n: getattr(module, n) for n in names}

    def observe(name, fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            out = fn(*args, **kwargs)
            seen.append((name, out))
            return out
        return inner

    for n, fn in originals.items():
        setattr(module, n, observe(n, fn))
    try:
        yield
    finally:
        for n, fn in originals.items():
            setattr(module, n, fn)


def _run_activation_startup(ctx: Context) -> None:
    """Run the real ``_activation_startup`` once and attribute a refusal to
    the check that produced it (the gate after the last check that ran, for
    a refusal no check returned)."""
    if ctx.activation_ran:
        return
    ctx.activation_ran = True
    import cascade.shared.activation as act

    seen: list = []
    names = tuple(n for n, _g in _ACTIVATION_CHECKS)
    gate_of = dict(_ACTIVATION_CHECKS)
    try:
        with _observed(act, names, seen):
            ctx.validator_runner()._activation_startup(ctx.chain)
    except Exception as e:  # noqa: BLE001 — every startup refusal is a FAIL
        refused = [n for n, out in seen if out]
        if refused:
            gate = gate_of[refused[-1]]
        else:
            ran = {n for n, _o in seen}
            later = [g for n, g in _ACTIVATION_CHECKS if n not in ran]
            gate = later[0] if later else _ACTIVATION_CHECKS[-1][1]
        ctx.activation_refusal = (gate, _reason(e))


def _activation_gate(name: str) -> Callable[[Context], None]:
    def run(ctx: Context) -> None:
        _run_activation_startup(ctx)
        if ctx.activation_refusal is not None and ctx.activation_refusal[0] == name:
            raise GateRefused(ctx.activation_refusal[1])
    run.__name__ = f"_{name}"
    return run


def _validator_calendar_launch(ctx: Context) -> None:
    from ..shared.calendar_launch import check_calendar_launch

    check_calendar_launch(ctx.validator_runner()._base_cfg(), ctx.chain)


# ── trainer gates ────────────────────────────────────────────────────────────


def _trainer_calendar_launch(ctx: Context) -> None:
    from ..shared.calendar_launch import check_calendar_launch

    check_calendar_launch(ctx.trainer_runner()._base_cfg(), ctx.chain)


def _worker_release(ctx: Context) -> str | None:
    from ..trainer.launch_preflight import check_worker_release

    # run_forever logs a ValueError here and carries on (v2 rental stays
    # blocked); any other error ends the service.
    try:
        check_worker_release(ctx.trainer_runner()._base_cfg())
    except ValueError as e:
        return _reason(e)
    return None


def _trainer_v2b_typed_notice(ctx: Context) -> None:
    ctx.trainer_runner()._image_decoupled_startup(ctx.chain)


# ── the gate lists (main.py order) ───────────────────────────────────────────

_CODE_PATH = Gate("code_path", ("verify_release_code_path",), run=_code_path)
_LOADER = Gate("loader", ("require_vote_pin", "load_chain_config"), run=_loader)
_LAUNCH_READY = Gate("launch_ready", ("assert_launch_ready",), run=_launch_ready)

GATES: dict[str, tuple[Gate, ...]] = {
    "validator": (
        _CODE_PATH,
        _LOADER,
        _LAUNCH_READY,
        Gate("bench_signer", ("check_main_bench_signer",), run=_bench_signer),
        Gate("activation_note", ("_activation_startup", "v2b_note_problem"),
             run=_activation_gate("activation_note")),
        Gate("v2b_typed_notice", ("_v2b_typed_notice_startup", "v2b_typed_notice_problem"),
             run=_activation_gate("v2b_typed_notice")),
        Gate("prelaunch_retirement", ("prelaunch_retirement_problem",),
             run=_activation_gate("prelaunch_retirement")),
        Gate("calendar_launch", ("check_calendar_launch",), run=_validator_calendar_launch),
    ),
    "trainer": (
        _CODE_PATH,
        _LOADER,
        _LAUNCH_READY,
        Gate("calendar_launch", ("check_calendar_launch",), run=_trainer_calendar_launch),
        Gate("worker_release", ("check_worker_release",), advisory=True, run=_worker_release),
        Gate("v2b_typed_notice", ("_image_decoupled_startup", "v2b_typed_notice_problem"),
             run=_trainer_v2b_typed_notice),
    ),
}


# ── running ──────────────────────────────────────────────────────────────────


def _reason(exc: BaseException) -> str:
    text = " ".join(str(exc).splitlines()).strip()
    return text or type(exc).__name__


def run_gates(ctx: Context) -> tuple[list[str], int]:
    """Run ``GATES[ctx.role]`` in order; return the stdout lines and the exit code."""
    lines: list[str] = []
    failed = False
    for gate in GATES[ctx.role]:
        if failed:
            lines.append(f"SKIP {gate.name}")
            continue
        try:
            warning = gate.run(ctx)
        except (Exception, SystemExit) as e:  # noqa: BLE001 — a refusal of any kind
            failed = True
            log.error("startup gate %s refused: %s", gate.name, e)
            lines.append(f"FAIL {gate.name}: {_reason(e)}")
            continue
        if warning and gate.advisory:
            log.warning("startup gate %s (advisory) refused: %s", gate.name, warning)
            lines.append(f"WARN {gate.name}: {warning}")
        else:
            lines.append(f"PASS {gate.name}")
    return lines, EXIT_FAIL if failed else EXIT_OK


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -P -m cascade.laned.startup_gates",
        description="Run a chain.toml through every real startup gate at an injected head.")
    p.add_argument("--config", type=Path, required=True, help="the chain.toml to check")
    p.add_argument("--head-block", type=int, required=True, help="the injected chain head")
    p.add_argument("--role", required=True, choices=ROLES)
    p.add_argument("--head-timestamp", type=float, default=None,
                   help="chain time of the head, UNIX seconds (absent: unreadable)")
    p.add_argument("--state-dir", type=Path, default=None,
                   help="the node's persisted activation records (absent: fresh node)")
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as e:
        return EXIT_USAGE if e.code else EXIT_OK
    if not args.config.is_file():
        print(f"--config {args.config}: no such file", file=sys.stderr)
        return EXIT_USAGE
    if args.state_dir is not None and not args.state_dir.is_dir():
        print(f"--state-dir {args.state_dir}: no such directory", file=sys.stderr)
        return EXIT_USAGE
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = sys.stdout
    with tempfile.TemporaryDirectory(prefix="cascade-startup-gates-") as tmp:
        work = Path(tmp)
        if args.state_dir is not None:
            for name in STATE_FILES:
                src = args.state_dir / name
                if src.is_file():
                    shutil.copyfile(src, work / name)
        ctx = Context(role=args.role, config=args.config,
                      chain=HeadChain(args.head_block, args.head_timestamp), work_dir=work)
        # Only gate lines reach stdout; anything a gate prints is diagnostics.
        with contextlib.redirect_stdout(sys.stderr):
            lines, code = run_gates(ctx)
    for line in lines:
        print(line, file=out)
    out.flush()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
