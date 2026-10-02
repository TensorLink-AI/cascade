"""The mining gauntlet — a long-running, multi-stage harness around ``cascade score``.

Three roles, one rule: **LLMs decide what to try; deterministic code decides what
counts, what gets spent, and what gets promoted.**

* **Operator** (Hermes Agent, any OpenAI-compatible model): steers. It reads the
  gauntlet's status files and writes directives; it holds no keys and cannot
  promote or submit anything (``deploy/harness/hermes``).
* **Workers** (Claude Code on Chutes / SayGM / Anthropic): each makes ONE edit to
  a copy of a population member, in a fresh context (:mod:`.workers`). Workers
  never see pool data or receipts — only stage outcomes.
* **Judge** (this package): runs every candidate through the stages below on a
  local GPU or rented Lium pods (:mod:`.executor`), under a spend cap.

Stages (cost increases, only survivors advance — :mod:`.gauntlet`):

``G0`` verify + exact-identity dedup (local) → ``G1`` throughput vs the king →
``G2`` short screen on one pool-A round (a fresh round per candidate) →
``G3`` confirm on the pool-B rounds (pass/fail only) → ``G4`` full-contract
replays judged against the kings' signed receipt scores → ``G4.5`` one-shot on
pool C (built AFTER the finalist froze) → ``G5`` submit (approval or
guard-railed autonomous, :mod:`.submit`).

Pools are REPLAYED ROUNDS (:mod:`.rounds`): revealed eval snapshots + signed
receipts, slid forward once a day as reveals land; every slide is a new epoch
(scores never mix across epochs) and re-baselines the population.
"""


def install_logging(level: int = 20) -> None:
    """(Re)attach the gauntlet's handler to the ``cascade`` logger. Idempotent.

    Importing bittensor (the live king, receipt signatures) strips the handlers
    from EVERY logger and raises them to CRITICAL, so the judge calls this again
    after each phase change instead of trusting a one-time setup."""
    import logging

    lg = logging.getLogger("cascade")
    if not any(getattr(h, "_gauntlet", False) for h in lg.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        h._gauntlet = True
        lg.addHandler(h)
    lg.setLevel(level)
    lg.propagate = False
    # bittensor also pins every EXISTING logger to CRITICAL (and may disable
    # it); hand the module loggers back to the "cascade" parent.
    for name, child in list(logging.root.manager.loggerDict.items()):
        if name.startswith("cascade.") and isinstance(child, logging.Logger):
            child.setLevel(logging.NOTSET)
            child.disabled = False
            child.handlers = [h for h in child.handlers if getattr(h, "_gauntlet", False)]
            child.propagate = True
