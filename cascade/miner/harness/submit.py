"""G5 — submitting a finalist: off, human approval, or guard-railed autonomous.

Submitting spends a hotkey (one submission each) and funds a GPU leg from the
miner's Lium account, so it is the one irreversible step. Three modes
(``[submit] mode``):

``off``         finalists are recorded, nothing else happens;
``approval``    (default) the finalist and its evidence go to
                ``submit/pending/<id>.json`` and the notify webhook; a person runs
                ``cascade gauntlet approve <id> --hotkey HK --confirm SUBMIT``;
``autonomous``  the judge submits by itself, but ONLY when every guardrail holds:

  1. no kill switch (``<workdir>/submit/HOLD`` or ``<workdir>/STOP``);
  2. the one-shot pool-C stage passed (or, with ``g45_enabled = false``, G4) with
     relative improvement >= ``[submit] margin`` (>= 0.005, the duel's floor);
  3. fewer than ``max_per_day`` submissions in the last 24h;
  4. an UNUSED hotkey remains in ``[submit] hotkeys`` (used ones are recorded and
     never reused);
  5. this exact generator tree was never submitted before;
  6. ``LIUM_API_KEY`` is present (the submission funds its own leg).

Any failed guardrail downgrades that finalist to ``approval``: it lands in
``pending/`` for a person, with the reason.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .rounds import tree_digest

log = logging.getLogger("cascade.miner.harness.submit")

DAY = 86400.0


class Submitter:
    def __init__(self, cfg, workdir: Path, *, runner: Callable[[list[str]], int] | None = None,
                 now: Callable[[], float] = time.time) -> None:
        """``cfg`` is the ``[submit]`` section (:class:`SubmitConfig`)."""
        self.cfg = cfg
        self.workdir = Path(workdir)
        self.dir = self.workdir / "submit"
        (self.dir / "pending").mkdir(parents=True, exist_ok=True)
        self._runner = runner or (lambda argv: subprocess.run(argv, check=False).returncode)
        self._now = now

    # -- records -----------------------------------------------------------
    def history(self) -> list[dict]:
        p = self.dir / "history.jsonl"
        if not p.is_file():
            return []
        return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def _record(self, row: dict) -> None:
        with open(self.dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({**row, "ts": self._now()}) + "\n")

    def used_hotkeys(self) -> set[str]:
        # Any ATTEMPT spends the hotkey for our purposes: a submit that failed
        # after its commit landed must never be retried on the same hotkey.
        return {r["hotkey"] for r in self.history()
                if r.get("hotkey") and r.get("action") in ("submitted", "failed")}

    def notify(self, event: str, doc: dict) -> None:
        if not self.cfg.notify_url:
            return
        body = json.dumps({"event": event, **doc}).encode()
        req = urllib.request.Request(self.cfg.notify_url, data=body, method="POST",
                                     headers={"content-type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=15).close()
        except Exception as e:  # noqa: BLE001 — a dead webhook never blocks the judge
            log.warning("notify %s failed: %s", event, e)

    # -- guardrails ----------------------------------------------------------
    def guardrail_failures(self, tree: Path, evidence: dict) -> list[str]:
        why = []
        if (self.dir / "HOLD").exists() or (self.workdir / "STOP").exists():
            why.append("kill switch present (submit/HOLD or STOP)")
        final = evidence.get("g45") if evidence.get("g45_required", True) else evidence.get("g4")
        if not final or not final.get("pass"):
            why.append("the final stage did not pass")
        elif not final.get("rel", -1.0) >= self.cfg.margin:
            why.append(f"improvement {final.get('rel', float('nan')):+.4f} < margin "
                       f"{self.cfg.margin:.4f}")
        recent = [r for r in self.history()
                  if r.get("rc") == 0 and self._now() - r.get("ts", 0) < DAY]
        if len(recent) >= self.cfg.max_per_day:
            why.append(f"{len(recent)} submission(s) in the last 24h (max {self.cfg.max_per_day})")
        if self.next_hotkey() is None:
            why.append("no unused hotkey left in [submit] hotkeys")
        digest = tree_digest(tree)
        if any(r.get("digest") == digest and r.get("rc") == 0 for r in self.history()):
            why.append("this exact tree was already submitted")
        if not os.environ.get("LIUM_API_KEY"):
            why.append("LIUM_API_KEY is not set (the submission funds its own leg)")
        return why

    def next_hotkey(self) -> str | None:
        used = self.used_hotkeys()
        return next((h for h in self.cfg.hotkeys if h not in used), None)

    # -- the stage ---------------------------------------------------------
    def offer(self, cand_id: str, tree: Path, evidence: dict) -> dict:
        """Hand a finalist to G5. Returns ``{"action": off|pending|submitted|failed}``."""
        if self.cfg.mode == "off":
            return {"action": "off"}
        frozen = self.dir / "frozen" / cand_id
        if not frozen.is_dir():
            shutil.copytree(tree, frozen)
        doc = {"id": cand_id, "tree": str(frozen), "digest": tree_digest(frozen),
               "evidence": evidence}
        if self.cfg.mode == "autonomous":
            why = self.guardrail_failures(frozen, evidence)
            if not why:
                return self._submit(cand_id, frozen, self.next_hotkey(), auto=True)
            doc["autonomous_refused"] = why
            log.warning("autonomous submit of %s refused: %s", cand_id, "; ".join(why))
        (self.dir / "pending" / f"{cand_id}.json").write_text(json.dumps(doc, indent=1),
                                                              encoding="utf-8")
        self.notify("finalist_pending", doc)
        return {"action": "pending", **({"refused": doc["autonomous_refused"]}
                                        if "autonomous_refused" in doc else {})}

    def approve(self, cand_id: str, hotkey: str) -> dict:
        """A person approved a pending finalist (``cascade gauntlet approve``)."""
        p = self.dir / "pending" / f"{cand_id}.json"
        if not p.is_file():
            raise ValueError(f"no pending finalist {cand_id}")
        if hotkey in self.used_hotkeys():
            raise ValueError(f"hotkey {hotkey} already submitted once; use a fresh one")
        doc = json.loads(p.read_text(encoding="utf-8"))
        out = self._submit(cand_id, Path(doc["tree"]), hotkey, auto=False)
        if out["action"] == "submitted":
            p.unlink()
        return out

    def reject(self, cand_id: str) -> None:
        p = self.dir / "pending" / f"{cand_id}.json"
        if p.is_file():
            p.unlink()
            self._record({"id": cand_id, "action": "rejected"})

    def pending(self) -> list[dict]:
        return [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((self.dir / "pending").glob("*.json"))]

    def _submit(self, cand_id: str, tree: Path, hotkey: str | None, *, auto: bool) -> dict:
        if not (self.cfg.intake and self.cfg.wallet_name and hotkey):
            raise ValueError("[submit] intake, wallet_name and a hotkey are required")
        argv = ["cascade", "submit", str(tree), self.cfg.intake,
                "--wallet-name", self.cfg.wallet_name, "--wallet-hotkey", hotkey,
                "--label", self.cfg.label]
        log.warning("SUBMITTING %s with hotkey %s (%s)", cand_id, hotkey,
                    "autonomous" if auto else "approved")
        rc = self._runner(argv)
        row = {"id": cand_id, "action": "submitted" if rc == 0 else "failed",
               "hotkey": hotkey, "digest": tree_digest(tree), "rc": rc, "auto": auto}
        self._record(row)
        self.notify("submitted" if rc == 0 else "submit_failed", row)
        return row
