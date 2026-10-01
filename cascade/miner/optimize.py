"""``cascade mine`` — the one-click optimisation loop around ``verify`` + ``score``.

A local hill-climb over generator variants. Every iteration:

1. **propose** — copy the current best into ``candidates/NNNN/`` and mutate it,
   with the built-in ``tune`` proposer (perturbs ``config.json`` mixture
   weights and float knobs; no LLM, no network), the ``agent`` proposer (runs
   a coding agent — Claude Code by default — in the candidate dir with the
   ``cascade-mine`` skill and the loop's history as context), or ``cmd`` —
   YOUR strategy, any executable (see :class:`CommandProposer`);
2. **verify** — the same checks the trainer runs (layout, import guard,
   hash-locked deps, determinism); a candidate that would be rejected on chain
   is never scored;
3. **score** — :func:`~cascade.miner.score.score_generator` on ONE fixed pool,
   seed set and init for the whole run, so every candidate is a paired
   comparison against the baseline (and the optional reference king);
4. **accept** — keep the candidate as the new best iff its mean geomean beats
   the best by ``min_improvement`` (relative). Lower is better.

Everything is on disk under ``workdir`` so the UI (``cascade mine-ui``), the
agent skill and a human can all watch or resume the same run:

``state.json``      live status (atomic rewrite each step)
``history.jsonl``   one record per scored/rejected candidate
``candidates/NNNN`` each candidate's generator tree (+ ``NNNN.note.md``)
``best/``           a copy of the current best generator — what you submit
``STOP``            touch it to stop after the current candidate

The score stays DIRECTIONAL (MINER.md §3): the validators score on a private
pool. Hill-climbing one local pool overfits it; rotate ``--pool-dir`` between
runs and read a win as "worth a submission", not as the verdict. Submitting is
never implicit: ``--auto-submit`` (plus an intake + wallet) is required, and
only a best that beats the reference king by ``--submit-margin`` goes out.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("cascade.miner.optimize")

PROPOSERS = ("tune", "agent", "cmd", "ralph")

STATE_FILE = "state.json"
SETUP_FILE = "setup.json"
LOCK_FILE = ".lock"
HISTORY_FILE = "history.jsonl"
STOP_FILE = "STOP"
BEST_DIR = "best"
CANDIDATES_DIR = "candidates"

# Generator trees are small; anything the loop itself writes stays out of them
# so a candidate verifies (and ZIPs for submit) exactly as the miner would ship it.
_COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".pytest_cache")

# Length bounds from INTERFACE.md — a tuned min/max_length never leaves them.
MIN_SERIES_LEN, MAX_SERIES_LEN = 64, 4096

# Headless Claude Code, edits confined to the candidate dir (its cwd) and the only
# shell command it may run is `cascade verify`. Overridable with --agent-cmd.
DEFAULT_AGENT_CMD = (
    'claude -p --permission-mode acceptEdits '
    '--allowedTools "Read,Edit,Write,Glob,Grep,Bash(cascade verify:*)"'
)
# The agent writes its one-line summary here (inside its cwd, where it may
# write); the loop moves it out before verify so it never ships.
AGENT_NOTE = ".mine-note.md"


@dataclass
class LoopConfig:
    workdir: Path
    start_dir: Path
    king_dir: Path | None = None
    proposer: str = "tune"                       # one of PROPOSERS
    iterations: int = 20
    seeds: tuple[int, ...] = (0,)
    min_improvement: float = 0.001               # relative; 0.001 = 0.1 %
    # Re-score best/ and the king when the scoring setup (init, pool, seeds,
    # budget, device) differs from the one this workdir's scores were taken under.
    rebaseline: bool = False
    # scorer knobs (forwarded to score_generator)
    pool_dir: Path | None = None
    pool_ref: str = ""
    train_hours: float = 0.25
    n_windows: int | None = None
    device: str = "auto"
    warm_start: str | None = None
    # tune proposer
    tune_sigma: float = 0.35                     # log-normal step size
    tune_max_keys: int = 3
    # agent proposer
    agent_cmd: str = DEFAULT_AGENT_CMD
    # cmd proposer (your own strategy; see CommandProposer)
    propose_cmd: str = ""
    propose_timeout: float = 1800.0
    agent_timeout: float = 1800.0
    agent_max_turns: int = 0                     # 0 = Claude Code's default
    # LLM backend for the agent/ralph proposers (see ralph.py): "anthropic" = Claude
    # Code's own login; "chutes" / "saygm" / "custom" = an Anthropic-compatible
    # endpoint. The key is read from the env var named by llm_key_env, never argv.
    llm_provider: str = "anthropic"
    llm_base_url: str = ""
    llm_model: str = ""
    llm_key_env: str = ""
    llm_auth: str = ""                           # "bearer" | "x-api-key" ("" = preset)
    # submission (never implicit)
    auto_submit: bool = False
    submit_margin: float = 0.01                  # best must beat king by this (relative)
    intake: str = ""
    wallet_name: str = ""
    wallet_hotkey: str = ""
    label: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        d["seeds"] = list(self.seeds)
        return d


@dataclass
class Candidate:
    iteration: int
    dir: Path
    parent: int | None
    note: str = ""
    status: str = "pending"          # scored | rejected | error | baseline | king
    score: float | None = None
    per_seed: list[float] = field(default_factory=list)
    accepted: bool = False
    detail: str = ""
    seconds: float = 0.0
    epoch: int = 0                   # scoring-setup generation; scores compare within one

    def record(self) -> dict:
        return {
            "iteration": self.iteration, "epoch": self.epoch, "dir": str(self.dir),
            "parent": self.parent,
            "note": self.note, "status": self.status, "score": self.score,
            "per_seed": self.per_seed, "accepted": self.accepted, "detail": self.detail,
            "seconds": round(self.seconds, 1), "ts": time.time(),
        }


# --------------------------------------------------------------------------- #
# disk state                                                                   #
# --------------------------------------------------------------------------- #

def _write_json_atomic(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def read_state(workdir: Path | str) -> dict:
    p = Path(workdir) / STATE_FILE
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def read_history(workdir: Path | str) -> list[dict]:
    p = Path(workdir) / HISTORY_FILE
    out: list[dict] = []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        with contextlib.suppress(ValueError):
            out.append(json.loads(line))
    return out


def request_stop(workdir: Path | str) -> None:
    (Path(workdir) / STOP_FILE).write_text(str(time.time()), encoding="utf-8")


class WorkdirBusy(RuntimeError):
    """Another loop holds this workdir."""


def _flock(fd: int, exclusive_nb: bool) -> bool:
    try:
        import fcntl
    except ImportError:          # non-POSIX: no locking, behave as before
        return True
    try:
        fcntl.flock(fd, (fcntl.LOCK_EX | fcntl.LOCK_NB) if exclusive_nb else fcntl.LOCK_UN)
    except BlockingIOError:
        return False
    return True


def workdir_lock_holder(workdir: Path | str) -> int | None:
    """PID of the loop holding ``workdir``'s lock, or None if no loop is running.

    The kernel releases a flock when its holder dies, however it dies, so this
    cannot be fooled by a stale ``state.json`` or a recycled PID."""
    p = Path(workdir) / LOCK_FILE
    if not p.is_file():
        return None
    fd = os.open(p, os.O_RDONLY)
    try:
        if _flock(fd, True):
            _flock(fd, False)
            return None
        try:
            return int(p.read_text(encoding="utf-8").strip() or 0) or -1
        except (OSError, ValueError):
            return -1
    finally:
        os.close(fd)


def format_history_rows(history: list[dict], n: int, *, with_reason: bool = False) -> str:
    """The recent-results table both agent prompts show (most recent last)."""
    rows = []
    for h in history[-n:]:
        s = "—" if h.get("score") is None else f"{h['score']:.5f}"
        mark = " ACCEPTED" if h.get("accepted") and h.get("status") == "scored" else ""
        why = ""
        if with_reason and h.get("status") not in ("scored", "baseline", "king"):
            first = (h.get("detail") or "").strip().splitlines()
            why = f" [{first[0][:120]}]" if first else ""
        rows.append(f"  #{h['iteration']:>3} {h['status']:<9} {s}{mark}  "
                    f"{h.get('note', '')[:140]}{why}")
    return "\n".join(rows) or "  (none yet)"


# --------------------------------------------------------------------------- #
# proposers                                                                    #
# --------------------------------------------------------------------------- #

def _is_num(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


# Named like sampling weights. Deliberately narrow: "prior", "mix" or "frac" also
# name hyperparameter dicts (a GP prior's lengthscale/variance) that must not be
# renormalised as if they were shares.
_MIXTURE_HINTS = ("weight", "prob", "mixture")


def _is_mixture(d: object, key: object = None) -> bool:
    """Family/sampling weights: a dict of >= 2 non-negative numbers that is either
    named like one (``family_weights``, ``probs``, ``mix``, …) or is all floats
    summing to ~1. A dict of unrelated knobs (``{"lam": 2.5, "k": 3}``) is not."""
    if not (isinstance(d, dict) and len(d) >= 2
            and all(_is_num(v) and v >= 0 for v in d.values())):
        return False
    total = float(sum(d.values()))
    if total <= 0:
        return False
    if key is not None and any(h in str(key).lower() for h in _MIXTURE_HINTS):
        return True
    return all(isinstance(v, float) for v in d.values()) and abs(total - 1.0) < 1e-3


def tunable_paths(cfg: object, prefix: tuple = ()) -> list[tuple]:
    """Leaves the ``tune`` proposer may move: members of a mixture dict, and
    float (not int, not bool) scalars. Ints are usually structural (lengths,
    batch sizes, prefetch depth) and are left alone, except the length bounds
    which are clamped into the interface range. Keys mentioning ``seed`` never move.
    """
    out: list[tuple] = []
    if isinstance(cfg, dict):
        mixture = _is_mixture(cfg, prefix[-1] if prefix else None)
        for k, v in cfg.items():
            path = (*prefix, k)
            if "seed" in str(k).lower():
                continue
            if mixture or (isinstance(v, float) and math.isfinite(v)) or (
                    k in ("min_length", "max_length") and _is_num(v)):
                out.append(path)
            elif isinstance(v, dict | list):
                out.extend(tunable_paths(v, path))
    elif isinstance(cfg, list):
        for i, v in enumerate(cfg):
            if isinstance(v, dict | list):
                out.extend(tunable_paths(v, (*prefix, i)))
    return out


def _get(cfg, path):
    for k in path[:-1]:
        cfg = cfg[k]
    return cfg


def tune_config(cfg: dict, rng: np.random.Generator, *, sigma: float, max_keys: int
                ) -> tuple[dict, str]:
    """Return a perturbed deep copy of ``cfg`` and a human-readable note.

    Each chosen leaf is multiplied by ``exp(N(0, sigma))`` (sign preserved; a
    zero mixture weight is revived to a small share). Mixtures are
    renormalised to their original total so the change is a pure reallocation.
    """
    new = json.loads(json.dumps(cfg))
    paths = tunable_paths(new)
    if not paths:
        return new, "no tunable keys in config.json (use --proposer agent)"
    k = int(rng.integers(1, min(max_keys, len(paths)) + 1))
    picks = [paths[i] for i in rng.choice(len(paths), size=k, replace=False)]
    notes: list[str] = []
    touched_mixtures: dict[int, tuple[dict, float]] = {}
    for path in picks:
        parent = _get(new, path)
        key = path[-1]
        old = parent[key]
        factor = float(np.exp(rng.normal(0.0, sigma)))
        if _is_mixture(parent, path[-2] if len(path) >= 2 else None):
            touched_mixtures.setdefault(id(parent), (parent, float(sum(parent.values()))))
            base = old if old > 0 else max(parent.values()) * 0.01
            val = base * factor
        elif isinstance(old, int):
            val = int(round(old * factor))
        else:
            val = old * factor
        if key in ("min_length", "max_length"):
            val = int(min(MAX_SERIES_LEN, max(MIN_SERIES_LEN, round(val))))
        parent[key] = val
        notes.append(f"{'.'.join(map(str, path))}: {_fmt(old)} -> {_fmt(val)}")
    for parent, total in touched_mixtures.values():
        s = float(sum(parent.values()))
        for kk in parent:
            parent[kk] = parent[kk] * total / s
    if _is_num(new.get("min_length")) and _is_num(new.get("max_length")) \
            and new["min_length"] > new["max_length"]:
        new["min_length"], new["max_length"] = new["max_length"], new["min_length"]
    return new, "tune: " + "; ".join(notes)


def _fmt(v) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


class TuneProposer:
    name = "tune"

    def __init__(self, cfg: LoopConfig, run_seed: int = 0) -> None:
        self.cfg = cfg
        self.run_seed = run_seed

    def propose(self, cand_dir: Path, iteration: int, history: list[dict]) -> str:
        p = cand_dir / "config.json"
        cfg = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        rng = np.random.default_rng([self.run_seed, iteration])
        new, note = tune_config(cfg, rng, sigma=self.cfg.tune_sigma, max_keys=self.cfg.tune_max_keys)
        p.write_text(json.dumps(new, indent=1) + "\n", encoding="utf-8")
        return note


def agent_prompt(cand_dir: Path, note_path: Path, history: list[dict], best: dict | None,
                 king: dict | None) -> str:
    """The instruction handed to the coding agent for one proposal."""
    rows = format_history_rows(history, 12, with_reason=True)
    best_s = "n/a" if not best else f"{best['score']:.5f} (#{best['iteration']})"
    king_s = "n/a" if not king or king.get("score") is None else f"{king['score']:.5f}"
    return f"""Use the cascade-mine skill (proposer mode).

You are ONE step of a cascade miner optimisation loop. The working directory
({cand_dir}) is a copy of the current best generator. Make ONE focused change
that you expect to LOWER the local score (geomean of CRPS and MASE of a fixed
forecaster trained on this generator's data; lower is better).

Current best: {best_s}. Reference king: {king_s}.
Recent attempts (most recent last):
{rows}

Rules:
- Edit files only inside {cand_dir} (the note file is removed before scoring). Keep generator.py / config.json /
  requirements.txt layout; keep it deterministic in the seed; no blocked imports,
  no packed code in strings; keep series finite, 64 <= L <= 4096, C <= 32.
- Do not repeat an attempt the history shows was rejected or scored worse.
- Do NOT run training or scoring yourself; the loop does that next.
  You may run `cascade verify {cand_dir}` to check your change.
- Finally write ONE line describing the change to {note_path}.
"""


class CommandProposer:
    """Bring-your-own strategy: any executable edits the candidate in place.

    The command (``--propose-cmd``, shlex-split, never run through a shell) runs
    with ``cwd`` = the candidate dir, which already holds a copy of the current
    best. It gets:

    * **stdin**: a JSON context document ``{iteration, candidate_dir, workdir,
      history, best, king}``. ``history`` is every record so far, the same
      rows as ``history.jsonl``.
    * **env**: ``CASCADE_CANDIDATE_DIR``, ``CASCADE_ITERATION``,
      ``CASCADE_WORKDIR``, ``CASCADE_HISTORY`` (path to ``history.jsonl``),
      ``CASCADE_NOTE_FILE``, ``CASCADE_BEST_SCORE``, ``CASCADE_KING_SCORE``.
    * ``{dir}`` in the command line is replaced with the candidate dir.

    It edits files in place, may write a one-line description to
    ``$CASCADE_NOTE_FILE`` (else the last line it prints is used), and exits 0.
    A non-zero exit is recorded as an ``error`` candidate, never scored. The loop
    then verifies, scores and accepts exactly as for the built-in proposers, so a
    strategy only has to answer "what should I try next?".
    """

    name = "cmd"
    label = "cmd"

    def __init__(self, cfg: LoopConfig, loop: OptimizationLoop, command: str | None = None,
                 timeout: float | None = None) -> None:
        self.cfg = cfg
        self.loop = loop
        self.command = command if command is not None else cfg.propose_cmd
        self.timeout = timeout if timeout is not None else cfg.propose_timeout
        if not self.command.strip():
            raise ValueError("--proposer cmd needs --propose-cmd (your strategy command)")

    # Hooks for presets (agent, ralph); no-ops for a plain strategy command.
    def base_env(self) -> dict[str, str]:
        """The environment the command starts from. A strategy command is the
        miner's own code, so it inherits everything; the agent presets narrow it."""
        return dict(os.environ)

    def extra_env(self) -> dict[str, str]:
        return {}

    def prepare(self, cand_dir: Path, iteration: int) -> None:
        pass

    def finish(self, cand_dir: Path, iteration: int) -> None:
        pass

    def stdin_for(self, cand_dir: Path, iteration: int, history: list[dict]) -> str:
        return json.dumps({
            "iteration": iteration, "candidate_dir": str(cand_dir),
            "workdir": str(self.loop.workdir), "history": history,
            "best": self.loop.best_record(), "king": self.loop.king_record(),
        })

    def propose(self, cand_dir: Path, iteration: int, history: list[dict]) -> str:
        note_path = cand_dir / AGENT_NOTE
        stdin = self.stdin_for(cand_dir, iteration, history)
        argv = [a.replace("{dir}", str(cand_dir)) for a in shlex.split(self.command)]
        best, king = self.loop.best_record(), self.loop.king_record()
        self.prepare(cand_dir, iteration)
        env = {
            **self.base_env(),
            **self.extra_env(),
            "CASCADE_CANDIDATE_DIR": str(cand_dir), "CASCADE_ITERATION": str(iteration),
            "CASCADE_WORKDIR": str(self.loop.workdir),
            "CASCADE_HISTORY": str(self.loop.workdir / HISTORY_FILE),
            "CASCADE_NOTE_FILE": str(note_path),
            "CASCADE_BEST_SCORE": "" if not best else repr(best["score"]),
            "CASCADE_KING_SCORE": "" if not king or king.get("score") is None
            else repr(king["score"]),
        }
        log.info("%s: %s (cwd=%s)", self.label, " ".join(argv), cand_dir)
        out_log = cand_dir.parent / f"{cand_dir.name}.{self.label}.log"
        note = ""
        try:
            with open(out_log, "w", encoding="utf-8") as out:
                proc = subprocess.run(
                    argv, input=stdin, text=True, cwd=cand_dir, stdout=out, env=env,
                    stderr=subprocess.STDOUT, timeout=self.timeout, check=False,
                )
        finally:
            # Even on a timeout: nothing the proposer writes for the loop ships.
            if note_path.is_file():
                note = note_path.read_text(encoding="utf-8").strip()
                os.replace(note_path, cand_dir.parent / f"{cand_dir.name}.note.md")
            self.finish(cand_dir, iteration)
        if proc.returncode != 0:
            raise RuntimeError(f"{self.label} exited {proc.returncode} (see {out_log})")
        if not note:
            lines = [ln.strip() for ln in out_log.read_text(encoding="utf-8").splitlines()]
            note = next((ln for ln in reversed(lines) if ln), "")
        return f"{self.label}: " + (note.splitlines()[0][:300] if note else "(no note written)")


class AgentProposer(CommandProposer):
    """The coding-agent preset of :class:`CommandProposer`: same contract, but
    stdin is a natural-language prompt (the cascade-mine skill, proposer mode)."""

    name = "agent"
    label = "agent"

    def __init__(self, cfg: LoopConfig, loop: OptimizationLoop) -> None:
        command = cfg.agent_cmd
        argv0 = (shlex.split(command) or [""])[0]
        # Provider routing, env isolation and --max-turns are Claude Code
        # features. Another agent (--agent-cmd "codex exec …", aider, …) is run
        # as the miner's own tool: its own env, its own flags.
        self.is_claude = Path(argv0).name == "claude"
        if self.is_claude and cfg.agent_max_turns > 0:
            command += f" --max-turns {int(cfg.agent_max_turns)}"
        super().__init__(cfg, loop, command=command, timeout=cfg.agent_timeout)
        from .ralph import resolve_provider
        if self.is_claude:
            self.provider = resolve_provider(cfg)
        elif (cfg.llm_provider or "anthropic") != "anthropic":
            raise ValueError("--llm-provider routes Claude Code only; with a non-claude "
                             "--agent-cmd, configure that agent's model itself")
        else:
            self.provider = None

    def base_env(self) -> dict[str, str]:
        if not self.is_claude:
            return dict(os.environ)
        from .ralph import agent_base_env
        return agent_base_env(self.provider, self.loop.workdir)

    def extra_env(self) -> dict[str, str]:
        return self.provider.claude_env() if self.is_claude else {}

    def stdin_for(self, cand_dir: Path, iteration: int, history: list[dict]) -> str:
        return agent_prompt(cand_dir, cand_dir / AGENT_NOTE, history,
                            self.loop.best_record(), self.loop.king_record())


# --------------------------------------------------------------------------- #
# the loop                                                                     #
# --------------------------------------------------------------------------- #

ScoreFn = Callable[[Path, int], float]
VerifyFn = Callable[[Path], tuple[bool, str]]


def _dir_digest_equal(a: Path, b: Path) -> bool:
    """True when two generator trees hold identical files (proposer was a no-op)."""
    def files(d: Path) -> dict[str, bytes]:
        return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts}
    return files(a) == files(b)


class OptimizationLoop:
    def __init__(self, cfg: LoopConfig, *, chain_cfg=None, score_fn: ScoreFn | None = None,
                 verify_fn: VerifyFn | None = None, proposer=None,
                 submit_fn: Callable[[Path], int] | None = None) -> None:
        self.cfg = cfg
        self.chain_cfg = chain_cfg
        self.workdir = Path(cfg.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        (self.workdir / CANDIDATES_DIR).mkdir(exist_ok=True)
        self._score_fn = score_fn or self._default_score
        self._verify_fn = verify_fn or self._default_verify
        self._submit_fn = submit_fn or self._default_submit
        if proposer is not None:
            self.proposer = proposer
        elif cfg.proposer == "agent":
            self.proposer = AgentProposer(cfg, self)
        elif cfg.proposer == "cmd":
            self.proposer = CommandProposer(cfg, self)
        elif cfg.proposer == "ralph":
            from .ralph import RalphProposer
            self.proposer = RalphProposer(cfg, self)
        elif cfg.proposer == "tune":
            self.proposer = TuneProposer(cfg, run_seed=int(time.time()))
        else:
            raise ValueError(f"unknown proposer {cfg.proposer!r} ({' | '.join(PROPOSERS)})")
        self.history: list[dict] = read_history(self.workdir)
        self.epoch = max((int(h.get("epoch", 0)) for h in self.history), default=0)
        self._lock_fd: int | None = None
        self._warm_dir: Path | None = None
        self._init_label = "random init"
        self._state: dict = {}

    # -- records ------------------------------------------------------------ #
    def _current(self) -> list[dict]:
        """History of the current scoring setup; earlier epochs are not comparable."""
        return [h for h in self.history if int(h.get("epoch", 0)) == self.epoch]

    def best_record(self) -> dict | None:
        acc = [h for h in self._current() if h.get("accepted") and h.get("score") is not None]
        return min(acc, key=lambda h: h["score"]) if acc else None

    def king_record(self) -> dict | None:
        ks = [h for h in self._current() if h.get("status") == "king"]
        return ks[-1] if ks else None

    def _next_iteration(self) -> int:
        return 1 + max((h["iteration"] for h in self.history), default=-1)

    def _append(self, c: Candidate) -> None:
        c.epoch = self.epoch
        rec = c.record()
        self.history.append(rec)
        with open(self.workdir / HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _set_state(self, **kw) -> None:
        self._state.update(kw)
        best, king = self.best_record(), self.king_record()
        self._state.update(
            pid=os.getpid(), updated=time.time(), config=self.cfg.to_json(),
            best=best, king=king, init=self._init_label,
            epoch=self.epoch,
            n_candidates=sum(1 for h in self._current()
                             if h["status"] not in ("baseline", "king")),
            n_accepted=sum(1 for h in self._current()
                           if h.get("accepted") and h["status"] == "scored"),
            beats_king=self._beats_king(best, king),
        )
        _write_json_atomic(self.workdir / STATE_FILE, self._state)

    def _beats_king(self, best: dict | None, king: dict | None) -> float | None:
        """Relative improvement of best over the reference king (positive = better)."""
        if not best or not king or king.get("score") in (None, 0):
            return None
        return (king["score"] - best["score"]) / king["score"]

    # -- default verify / score / submit ------------------------------------ #
    def _default_verify(self, d: Path) -> tuple[bool, str]:
        from .verify import verify_repo
        rep = verify_repo(d, self.chain_cfg, skip_runtime=False)
        return rep.ok, rep.render()

    def _resolve_device(self) -> str:
        if self.cfg.device != "auto":
            return self.cfg.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _default_score(self, d: Path, seed: int) -> float:
        from .score import score_generator
        r = score_generator(
            d, self.chain_cfg, pool_dir=self.cfg.pool_dir, pool_ref=self.cfg.pool_ref,
            train_hours=self.cfg.train_hours, n_windows=self.cfg.n_windows,
            device=self._resolve_device(), seed=seed, cache_dir=self.workdir / "_cache",
            warm_start=self._warm_dir,
        )
        return float(r.geomean)

    def _default_submit(self, d: Path) -> int:
        argv = [sys.executable, "-m", "cascade.miner.cli", "submit", str(d), self.cfg.intake,
                "--wallet-name", self.cfg.wallet_name, "--wallet-hotkey", self.cfg.wallet_hotkey]
        if self.cfg.label:
            argv += ["--label", self.cfg.label]
        log.info("submitting best: %s", " ".join(argv))
        return subprocess.run(argv, check=False).returncode

    # -- steps -------------------------------------------------------------- #
    def _evaluate(self, c: Candidate) -> Candidate:
        t0 = time.time()
        try:
            ok, report = self._verify_fn(c.dir)
            if not ok:
                c.status, c.detail = "rejected", report
                return c
            for s in self.cfg.seeds:
                self._set_state(phase=f"scoring #{c.iteration} (seed {s})")
                c.per_seed.append(float(self._score_fn(c.dir, s)))
            c.score = float(np.mean(c.per_seed))
            if not math.isfinite(c.score):
                c.status, c.score, c.detail = "error", None, "non-finite score"
                return c
            c.status = "scored"
        except Exception as e:  # noqa: BLE001 — one bad candidate never ends the run
            c.status, c.detail = "error", f"{type(e).__name__}: {e}"
            log.warning("candidate #%d failed: %s", c.iteration, c.detail)
        finally:
            c.seconds = time.time() - t0
        return c

    def _stop_requested(self) -> bool:
        return (self.workdir / STOP_FILE).exists()

    def _prepare_init(self) -> None:
        if not self.cfg.warm_start:
            return
        if self.chain_cfg is None:
            self._init_label = f"warm-start {self.cfg.warm_start}"
            self._warm_dir = Path(self.cfg.warm_start)
            return
        from .score import _resolve_warm_start
        # Resolve ONCE: every candidate trains from the identical init, and a
        # 'live' init that promotes mid-run cannot silently change the baseline.
        self._warm_dir, self._init_label = _resolve_warm_start(
            self.chain_cfg, self.cfg.warm_start, cache_dir=self.workdir / "_cache")

    def setup_fingerprint(self) -> dict:
        """Everything a score depends on besides the generator itself."""
        pool = str(Path(self.cfg.pool_dir).resolve()) if self.cfg.pool_dir else ""
        return {
            "init": self._init_label, "pool_dir": pool, "pool_ref": self.cfg.pool_ref,
            "seeds": list(self.cfg.seeds), "train_hours": self.cfg.train_hours,
            "n_windows": self.cfg.n_windows, "device": self._resolve_device(),
        }

    def _check_setup(self) -> None:
        """Scores only compare under one setup. A resumed workdir whose setup
        changed (a new live init, another pool, more seeds, a longer budget)
        either starts a new epoch (--rebaseline: best/ and the king are
        re-scored) or is refused, never silently mixed."""
        fp = self.setup_fingerprint()
        path = self.workdir / SETUP_FILE
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prev = None
        if prev is not None and self.history and prev.get("fingerprint") != fp:
            if not self.cfg.rebaseline:
                old = prev.get("fingerprint") or {}
                diff = ", ".join(f"{k}: {old.get(k)!r} -> {v!r}"
                                 for k, v in fp.items() if old.get(k) != v)
                raise RuntimeError(
                    f"scoring setup changed since this workdir was scored ({diff}); "
                    "scores would not be comparable. Use a new --workdir, or "
                    "--rebaseline to re-score best/ and the king under the new setup")
            self.epoch += 1
            log.info("setup changed: epoch %d, re-scoring best/ and the king", self.epoch)
        _write_json_atomic(path, {"epoch": self.epoch, "fingerprint": fp})

    def _baseline(self) -> None:
        """Score the start generator (and the reference king) once per epoch."""
        if self.best_record() is None:
            # A new epoch re-scores the existing best; a first run (or a retry
            # after a failed baseline) copies --start afresh.
            src = self.workdir / BEST_DIR if self.epoch > 0 and (
                self.workdir / BEST_DIR / "generator.py").is_file() else Path(self.cfg.start_dir)
            it = self._next_iteration()
            dst = self.workdir / CANDIDATES_DIR / f"{it:04d}"
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst, ignore=_COPY_IGNORE)
            label = "rebaseline" if self.epoch > 0 else "baseline"
            c = self._evaluate(Candidate(it, dst, None, note=f"{label}: {src}"))
            if c.status == "scored":
                c.status, c.accepted = "baseline", True
                self._promote(c)
            self._append(c)
            if not c.accepted:
                raise RuntimeError(f"baseline generator failed ({c.status}): {c.detail}")
        if self.cfg.king_dir and self.king_record() is None:
            self._set_state(phase="scoring reference king")
            k = self._evaluate(Candidate(self._next_iteration(), Path(self.cfg.king_dir), None,
                                         note=f"reference king: {self.cfg.king_dir}"))
            if k.status == "scored":
                k.status = "king"
            self._append(k)

    def _promote(self, c: Candidate) -> None:
        best = self.workdir / BEST_DIR
        tmp = self.workdir / (BEST_DIR + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(c.dir, tmp, ignore=_COPY_IGNORE)
        shutil.rmtree(best, ignore_errors=True)
        os.replace(tmp, best)

    def step(self) -> Candidate:
        best = self.best_record()
        assert best is not None
        it = self._next_iteration()
        cdir = self.workdir / CANDIDATES_DIR / f"{it:04d}"
        shutil.rmtree(cdir, ignore_errors=True)
        shutil.copytree(self.workdir / BEST_DIR, cdir, ignore=_COPY_IGNORE)
        c = Candidate(it, cdir, parent=best["iteration"])
        self._set_state(phase=f"proposing #{it} ({self.proposer.name})")
        try:
            c.note = self.proposer.propose(cdir, it, list(self.history))
        except Exception as e:  # noqa: BLE001
            c.status, c.detail = "error", f"proposer: {type(e).__name__}: {e}"
            self._append(c)
            self._report(c)
            return c
        if _dir_digest_equal(cdir, self.workdir / BEST_DIR):
            c.status, c.detail = "rejected", "proposal changed nothing"
            self._append(c)
            self._report(c)
            return c
        self._evaluate(c)
        if c.status == "scored" and c.score < best["score"] * (1.0 - self.cfg.min_improvement):
            c.accepted = True
            self._promote(c)
        self._append(c)
        self._report(c)
        log.info("#%d %s score=%s %s — %s", it, c.status,
                 "—" if c.score is None else f"{c.score:.5f}",
                 "ACCEPTED" if c.accepted else "", c.note)
        return c

    def _report(self, c: Candidate) -> None:
        """Tell a proposer that keeps its own memory (ralph) how its proposal did."""
        hook = getattr(self.proposer, "on_result", None)
        if hook is not None:
            with contextlib.suppress(Exception):
                hook(c.record(), self.best_record())

    def _acquire_lock(self) -> None:
        fd = os.open(self.workdir / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
        if not _flock(fd, True):
            os.close(fd)
            holder = workdir_lock_holder(self.workdir)
            raise WorkdirBusy(f"another loop (pid {holder}) is running in {self.workdir}; "
                              "stop it first or use another --workdir")
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._lock_fd = fd

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            _flock(self._lock_fd, False)
            os.close(self._lock_fd)
            self._lock_fd = None

    def run(self) -> dict:
        self._acquire_lock()           # before touching any state
        try:
            return self._run_locked()
        finally:
            self._release_lock()

    def _run_locked(self) -> dict:
        with contextlib.suppress(FileNotFoundError):
            (self.workdir / STOP_FILE).unlink()
        self._set_state(status="running", phase="starting", started=time.time(), error=None)
        try:
            self._prepare_init()
            self._check_setup()
            self._baseline()
            done = 0
            while done < self.cfg.iterations and not self._stop_requested():
                self.step()
                done += 1
                self._set_state(phase="idle", iterations_done=done)
            status = "stopped" if self._stop_requested() else "finished"
            self._set_state(status=status, phase="done")
            self._maybe_submit()
        except Exception as e:
            self._set_state(status="failed", phase="done", error=f"{type(e).__name__}: {e}")
            raise
        return self._state

    def _maybe_submit(self) -> None:
        if not self.cfg.auto_submit:
            return
        margin = self._beats_king(self.best_record(), self.king_record())
        if margin is None or margin < self.cfg.submit_margin:
            self._set_state(submit="skipped: best does not beat the reference king by "
                                   f"{self.cfg.submit_margin:.1%} (got "
                                   f"{'n/a' if margin is None else f'{margin:.2%}'})")
            return
        if not (self.cfg.intake and self.cfg.wallet_name and self.cfg.wallet_hotkey):
            self._set_state(submit="skipped: --intake / --wallet-name / --wallet-hotkey missing")
            return
        rc = self._submit_fn(self.workdir / BEST_DIR)
        self._set_state(submit=f"submit exited {rc}")
