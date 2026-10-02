"""Workers: Claude Code makes ONE edit per candidate, in a fresh context.

The judge hands a worker a copy of a population member and a prompt that
carries everything the worker may know: the operator's directives, the shared
notebook of lessons, and the stage OUTCOMES of earlier candidates. It never
carries pool data, receipts or G3+ numbers, and the worker's tree is the only
thing it can edit.

Two modes:

* ``inline`` — the judge runs ``claude -p`` itself (one machine, no Docker);
* ``queue``  — a file queue under ``<workdir>/queue`` that a SEPARATE worker
  container drains (``cascade gauntlet worker``). That container mounts only the
  queue, so a worker cannot read pools, receipts, the wallet or the Lium key
  even if it tried.

Provider routing and credential isolation are the Ralph loop's
(:mod:`cascade.miner.ralph`): with a third-party provider the agent starts from
a strict env allowlist and a fresh ``CLAUDE_CONFIG_DIR``.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from ..optimize import _COPY_IGNORE, AGENT_NOTE, DEFAULT_AGENT_CMD

log = logging.getLogger("cascade.miner.harness.workers")

LESSON_FILE = ".lesson.md"

PROMPT = """\
# Gauntlet worker: improve this cascade generator

You are ONE proposal in a long-running search. Your working directory is a copy
of a population member (`generator.py` + `config.json`), a cascade data
generator. The operator trains a fixed small forecaster (Toto2-4M) on its
synthetic series and scores it on REAL held-out windows at three horizons
(64, 256, 720 steps), split evenly across domains (energy, nature, sales, web,
transport, finance, epidemiology, sensors). Lower geomean of CRPS and MASE wins.

After you exit, a judge runs your change through a gauntlet you cannot see:
verify → throughput vs the king → a short training screen → a confirmation on
newer rounds → full-budget replays against real kings. Most changes die early.
You never see the eval data; do not try to find it.

## Your job
Make ONE focused change to the generator's CODE that you expect to make the
trained forecaster better on real data, and that survives on rounds you have
not seen. Good directions: realism of the observation process (resolution and
rounding, held values, missing bins, reporting cadence, count data, saturation);
a family covering behaviour the corpus lacks; fixing an unrealistic family;
longer-range structure that matters at the 256/720 horizons; SPEED (data is
streamed under a fixed wall clock, so a slower generator trains on less data —
a slower candidate fails the throughput stage).

The eval rewards BREADTH: past winners spread their gain over most domains and all
three horizons, while near misses packed a similar gain into one feed and lost.

## Hard rules (violations are rejected before any GPU is spent)
- Deterministic in `seed` only; no `hash()`, clock, `os.urandom`, network.
- No blocked imports (socket, subprocess, pickle, multiprocessing, …), no code
  packed in strings, only allowlisted deps; no shipped data or weights.
- Yields finite float arrays `(L,)` or `(C, L)`, 64 <= L <= 4096, C <= 32.
- Keep the layout; edit only files in your working directory.

## How
1. Read the directives and the lessons below. Do not retry what the outcome
   log shows failing unless you have a specific reason it differs now.
2. Grep, do not read huge files end to end.
3. Make the change; run `cascade verify .` and fix what it reports.
4. Write ONE line describing the change to `.mine-note.md`.
5. Optionally write ONE short lesson (something future workers should know)
   to `.lesson.md`.
Do not train or score.
"""


@dataclass
class Proposal:
    id: str
    parent_id: str
    tree: Path                 # the copy to edit (the worker's cwd)
    prompt: str                # may contain KNOWLEDGE_DIR, replaced at run time
    knowledge: Path | None = None   # read-only reference folder (never inside the tree)

    def to_json(self) -> dict:
        d = asdict(self)
        d["tree"] = str(self.tree)
        d["knowledge"] = str(self.knowledge) if self.knowledge else None
        return d


@dataclass
class Outcome:
    id: str
    ok: bool
    note: str = ""
    lesson: str = ""
    error: str = ""
    seconds: float = 0.0


KNOWLEDGE_DIR = "{KNOWLEDGE_DIR}"     # placeholder: the knowledge folder's real path

KNOWLEDGE_STEP = f"""
## Knowledge (read before you edit)
`{KNOWLEDGE_DIR}` holds the reference material (read-only; it is not your tree):
- `LINEAGE.md`: every past king, what it changed, the current king's anatomy, ranked edges.
- `DETHRONES.md`: why each king won, from the eval data (domains, horizons, sources).
- `RESEARCH.md`: 2025-26 literature on synthetic data for PFNs / time-series models.
- `attempts.jsonl`: one line per earlier candidate here: its change, how far it got,
  its screen number, and the exact reason it died.
Before editing: read the three briefs, then grep `attempts.jsonl` for your idea. If
it was tried, either pick another idea or say in your note why yours differs.
"""


def build_prompt(*, directives: str, notebook: str, outcomes: list[str], parent: str,
                 knowledge: bool = True) -> str:
    """The worker prompt: short by design. The briefs and the full attempt
    history live in the knowledge folder; the prompt carries the operator's
    directives, the last outcomes (with their full failure reasons) and the
    tail of the lessons notebook."""
    rows = "\n".join(outcomes[-10:]) or "(none yet)"
    return (PROMPT
            + (KNOWLEDGE_STEP if knowledge else "")
            + f"\n## Operator directives\n{directives.strip() or '(none)'}\n"
            + f"\n## Recent lessons\n{notebook.strip()[-2000:] or '(none yet)'}\n"
            + f"\n## Last outcomes (all of them: {KNOWLEDGE_DIR}/attempts.jsonl)\n{rows}\n"
            + f"\n## This proposal\nParent: {parent}. Your working directory is the copy.\n")


def _agent_argv(cfg, knowledge: Path | None = None) -> list[str]:
    argv = shlex.split(DEFAULT_AGENT_CMD)
    if cfg.agent_max_turns > 0:
        argv += ["--max-turns", str(int(cfg.agent_max_turns))]
    if knowledge is not None:
        argv += ["--add-dir", str(knowledge)]      # readable; edits stay in the tree
    return argv


def _agent_env(cfg, home: Path) -> dict[str, str]:
    from ..ralph import agent_base_env, resolve_provider
    provider = resolve_provider(cfg)
    return {**agent_base_env(provider, home), **provider.claude_env()}


def run_agent(p: Proposal, cfg, *, home: Path, runner=None) -> Outcome:
    """Run Claude Code in ``p.tree`` with ``p.prompt``; collect the note and lesson
    (both removed from the tree, so nothing the worker writes for us ships)."""
    t0 = time.time()
    log_path = p.tree.parent / f"{p.tree.name}.agent.log"
    try:
        env = _agent_env(cfg, home)
        with open(log_path, "w", encoding="utf-8") as out:
            if runner is not None:
                rc = runner(p, env)
            else:
                prompt = p.prompt.replace(KNOWLEDGE_DIR, str(p.knowledge or "(none)"))
                rc = subprocess.run(_agent_argv(cfg, p.knowledge), input=prompt, text=True,
                                    cwd=p.tree,
                                    stdout=out, stderr=subprocess.STDOUT, env=env,
                                    timeout=cfg.agent_timeout, check=False).returncode
        err = "" if rc == 0 else f"agent exited {rc} (see {log_path.name})"
    except subprocess.TimeoutExpired:
        err = f"agent timed out after {cfg.agent_timeout:.0f}s"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    note = lesson = ""
    tree_note = p.tree / AGENT_NOTE
    if err and tree_note.is_file() and tree_note.read_text(encoding="utf-8").strip():
        # It delivered (edit + note) and then hit --max-turns / a late error:
        # keep the candidate; G0 verify rejects anything it left broken.
        err = ""
    for name in (AGENT_NOTE, LESSON_FILE):
        f = p.tree / name
        if f.is_file():
            text = f.read_text(encoding="utf-8").strip()
            f.unlink()
            if name == AGENT_NOTE:
                note = text.splitlines()[0][:300] if text else ""
            else:
                lesson = text[:600]
    return Outcome(p.id, ok=not err, note=note, lesson=lesson, error=err,
                   seconds=time.time() - t0)


class InlineWorkers:
    def __init__(self, cfg, home: Path, *, parallel: int = 4, runner=None) -> None:
        self.cfg, self.home, self.parallel, self.runner = cfg, home, parallel, runner

    def run(self, proposals: list[Proposal]) -> list[Outcome]:
        with ThreadPoolExecutor(max_workers=max(1, self.parallel)) as pool:
            return list(pool.map(
                lambda p: run_agent(p, self.cfg, home=self.home, runner=self.runner),
                proposals))


class QueueWorkers:
    """Judge side of the file queue: ``pending/<id>/`` → ``done/<id>.json``."""

    def __init__(self, queue_dir: Path, *, timeout: float = 3600.0, poll: float = 5.0) -> None:
        self.q = Path(queue_dir)
        self.timeout, self.poll = timeout, poll
        for sub in ("pending", "running", "done"):
            (self.q / sub).mkdir(parents=True, exist_ok=True)

    def run(self, proposals: list[Proposal]) -> list[Outcome]:
        for p in proposals:
            stage = self.q / "tmp" / p.id
            shutil.rmtree(stage, ignore_errors=True)
            shutil.copytree(p.tree, stage / "tree", ignore=_COPY_IGNORE)
            if p.knowledge is not None and Path(p.knowledge).is_dir():
                # The worker container sees only the queue: ship the knowledge
                # next to the tree (never inside it).
                shutil.copytree(p.knowledge, stage / "knowledge")
            (stage / "proposal.json").write_text(
                json.dumps({"id": p.id, "parent_id": p.parent_id, "prompt": p.prompt}),
                encoding="utf-8")
            os.replace(stage, self.q / "pending" / p.id)      # atomic publish
        out: dict[str, Outcome] = {}
        deadline = time.time() + self.timeout
        while len(out) < len(proposals) and time.time() < deadline:
            for p in proposals:
                done = self.q / "done" / f"{p.id}.json"
                if p.id in out or not done.is_file():
                    continue
                doc = json.loads(done.read_text(encoding="utf-8"))
                tree = self.q / "done" / p.id / "tree"
                agent_log = tree.parent / "tree.agent.log"
                if agent_log.is_file():
                    shutil.copyfile(agent_log, p.tree.parent / f"{p.tree.name}.agent.log")
                if tree.is_dir():
                    shutil.rmtree(p.tree, ignore_errors=True)
                    shutil.copytree(tree, p.tree, ignore=_COPY_IGNORE)
                    shutil.rmtree(tree.parent, ignore_errors=True)
                done.unlink()
                out[p.id] = Outcome(**doc)
            if len(out) < len(proposals):
                time.sleep(self.poll)
        for p in proposals:
            if p.id not in out:
                for sub in ("pending", "running"):
                    shutil.rmtree(self.q / sub / p.id, ignore_errors=True)
                out[p.id] = Outcome(p.id, ok=False, error="no worker finished it in time")
        return [out[p.id] for p in proposals]


def serve_queue(queue_dir: Path, cfg, *, home: Path, once: bool = False, poll: float = 5.0,
                runner=None) -> int:
    """Worker side: claim ``pending/<id>`` (atomic rename), edit, publish ``done``."""
    q = Path(queue_dir)
    for sub in ("pending", "running", "done"):
        (q / sub).mkdir(parents=True, exist_ok=True)
    handled = 0
    while True:
        for item in sorted((q / "pending").iterdir()):
            run_dir = q / "running" / item.name
            try:
                os.replace(item, run_dir)                     # claim
            except OSError:
                continue                                      # another worker took it
            meta = json.loads((run_dir / "proposal.json").read_text(encoding="utf-8"))
            kdir = run_dir / "knowledge"
            p = Proposal(meta["id"], meta["parent_id"], run_dir / "tree", meta["prompt"],
                         knowledge=kdir if kdir.is_dir() else None)
            o = run_agent(p, cfg, home=home, runner=runner)
            dest = q / "done" / p.id
            shutil.rmtree(dest, ignore_errors=True)
            os.replace(run_dir, dest)
            tmp = q / "done" / f"{p.id}.tmp"
            tmp.write_text(json.dumps(asdict(o)), encoding="utf-8")
            os.replace(tmp, q / "done" / f"{p.id}.json")
            handled += 1
        if once:
            return handled
        time.sleep(poll)
