"""Where miner generator code may run (CLAUDE.md operational invariant).

The orchestrator holds the private eval pool and the trainer wallet, and a
validator holds its own wallet and the pool windows. Neither may run miner
generator code unless ``[generator] sandbox_mode = "container"`` or
``sandbox_strict = true``. A ``subprocess`` sandbox that is not strict
degrades to a plain child on a host without network namespaces, and that
child can read the parent's environment. Two layers enforce this:

* a LOAD check (:func:`check_generator_sandbox_config`): any config knob
  that would run generator code in such a process is refused at load
  unless the sandbox is hardened. Today the only knob is the retired
  ``[round] dedup_probe_series``, which nothing reads; the check keeps a
  future re-wire from arming it on a weak sandbox;
* a RUNTIME guard (:func:`require_generator_sandbox`, called by every
  generator execution entry point in :mod:`cascade.trainer.sandbox` and
  :mod:`cascade.trainer.corpus`): a process that marked itself as holding
  secrets (:func:`mark_holds_secrets`: the trainer orchestrator and the
  validator mark themselves at start) refuses to run generator code. It
  refuses always IN-PROCESS, and in a sandbox child unless the sandbox is
  hardened. Payer / worker pods, the CPU generation tier and miner tools
  never mark themselves, so this changes nothing there.

The marker lives in this process's memory only. A sandbox child is a fresh
interpreter and never inherits it.
"""

from __future__ import annotations

_HOLDS_SECRETS: str = ""


class SandboxPolicyError(RuntimeError):
    """Refused: generator code in a process that holds the trainer wallet /
    the private eval pool without a hardened sandbox. Deliberately NOT a
    :class:`~cascade.trainer.corpus.CorpusError`, so no caller can book it
    as the miner's fault: it is an operator configuration problem."""


def mark_holds_secrets(reason: str) -> None:
    """Mark THIS process as holding the trainer wallet / private eval pool
    (``reason`` names it in the refusal)."""
    global _HOLDS_SECRETS
    _HOLDS_SECRETS = str(reason or "secrets")


def clear_holds_secrets() -> None:
    """Tests only."""
    global _HOLDS_SECRETS
    _HOLDS_SECRETS = ""


def holds_secrets() -> str:
    """Why this process holds secrets ("" = it does not)."""
    return _HOLDS_SECRETS


def sandbox_hardened(gen_cfg: object) -> bool:
    """``sandbox_mode = "container"`` or ``sandbox_strict = true`` (works on
    a :class:`GeneratorConfig` or a raw ``[generator]`` table)."""
    get = gen_cfg.get if isinstance(gen_cfg, dict) else (
        lambda k, d=None: getattr(gen_cfg, k, d))
    return (str(get("sandbox_mode", "subprocess")) == "container"
            or get("sandbox_strict", False) is True)


def require_generator_sandbox(gen_cfg: object, where: str, *, in_process: bool = False) -> None:
    """Runtime guard at a generator-execution entry point. It is a no-op in
    a process that holds no secrets. In one that does, it refuses
    in-process execution always, and sandboxed execution unless
    :func:`sandbox_hardened`. Raises :class:`SandboxPolicyError`."""
    why = _HOLDS_SECRETS
    if not why:
        return
    if in_process:
        raise SandboxPolicyError(
            f"refusing to run miner generator code in-process ({where}): this process holds "
            f"{why}. Generator code runs only in a sandbox child")
    if not sandbox_hardened(gen_cfg):
        raise SandboxPolicyError(
            f"refusing to run miner generator code ({where}): this process holds {why} and "
            "the sandbox is not hardened. Set [generator] sandbox_strict = true or "
            "sandbox_mode = \"container\" (CLAUDE.md operational invariant)")


def check_generator_sandbox_config(generator: dict, round_: dict) -> None:
    """LOAD check: refuse any config that runs generator code on the
    orchestrator / a validator unless the sandbox is hardened. The only
    such knob is ``[round] dedup_probe_series > 0`` (the behavioural dedup
    probe). It is retired and nothing reads it, but a positive value is
    still refused on a weak sandbox."""
    try:
        n = int(round_.get("dedup_probe_series", 0) or 0)
    except (TypeError, ValueError):
        n = 0
    if n > 0 and not sandbox_hardened(generator):
        raise ValueError(
            f"[round] dedup_probe_series = {n} would run miner generator code on the "
            "orchestrator (which holds the trainer wallet and the private eval pool). It "
            "needs [generator] sandbox_strict = true or sandbox_mode = \"container\"")
