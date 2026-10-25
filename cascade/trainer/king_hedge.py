"""Delayed king replica: first valid completion wins, with a durable choice.

The callbacks train the same pinned experiment on distinct allocation names.
Selection observes completion only; it never inspects evaluation scores.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

from ..shared.atomic import path_lock, write_text_atomic
from .continuation import canonical


class KingHedge:
    def __init__(self, path, binding, *, clock=time.time):
        self.path, self.binding, self.clock = Path(path), binding, clock
        self.cancel = {name: threading.Event() for name in ("primary", "backup")}

    def _read(self):
        state = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "binding": self.binding,
                "started_at": self.clock(),
                "winner": None,
                "key_failed": False,
            }
        )
        if canonical(state["binding"]) != canonical(self.binding):
            raise ValueError("king replica state belongs to an incompatible experiment")
        return state

    def note_key_failure(self):
        with path_lock(self.path):
            state = self._read()
            state["key_failed"] = True
            write_text_atomic(self.path, canonical(state).decode())

    def run(self, primary, backup, *, delay, deadline, cleanup=None):
        with path_lock(self.path):
            state = self._read()
            if state["winner"] is not None:
                self.cancel[
                    "backup" if state["winner"]["replica"] == "primary" else "primary"
                ].set()
                if cleanup is not None:
                    cleanup("backup" if state["winner"]["replica"] == "primary" else "primary")
                return state["winner"]
            write_text_atomic(self.path, canonical(state).decode())
        completions = queue.Queue()
        started = set()
        failures = []
        decision = threading.Event()
        selected = [None]

        def launch(name, callback):
            started.add(name)

            def work():
                try:
                    result = callback(self.cancel[name])
                    completions.put((name, result, None))
                except Exception as exc:
                    completions.put((name, None, exc))
                finally:
                    decision.wait(max(0, deadline - self.clock()) + 1)
                    if cleanup is not None and selected[0] != name:
                        cleanup(name)

            threading.Thread(target=work, name=f"king-{name}", daemon=True).start()

        launch("primary", primary)
        try:
            while self.clock() < deadline:
                elapsed = self.clock() - state["started_at"]
                if "backup" not in started and (elapsed >= delay or failures):
                    launch("backup", backup)
                try:
                    name, result, failure = completions.get(
                        timeout=min(1, max(0.001, deadline - self.clock()))
                    )
                except queue.Empty:
                    continue
                if failure is not None:
                    failures.append(failure)
                    if len(failures) == 2:
                        raise RuntimeError("both king replicas failed") from failure
                    continue
                winner = {"replica": name, **result}
                with path_lock(self.path):
                    current = self._read()
                    if current["winner"] is None:
                        if name == "backup":
                            winner["funded_by"] = (
                                "operator" if current["key_failed"] else "operator_fault"
                            )
                        current["winner"] = winner
                        write_text_atomic(self.path, canonical(current).decode())
                    winner = current["winner"]
                selected[0] = winner["replica"]
                self.cancel["backup" if winner["replica"] == "primary" else "primary"].set()
                return winner
            raise TimeoutError("no king replica completed within the pinned era deadline")
        finally:
            # A winning replica has already finished. Both events also cover
            # timeout/failed persistence; late callbacks cannot publish a choice.
            for event in self.cancel.values():
                event.set()
            decision.set()


def king_account_status(runner, hotkey):
    """Read-only proof: distinguish rejected credentials from an API outage."""
    import hashlib

    from ..funding.faults import classify_rent_failure
    from ..funding.providers import decode_credentials
    from ..provision.funded import provider_for_key

    vault = runner._payer_vault()
    raw = (vault.get_shared(hotkey) if vault is not None else None) or ""
    digest = hashlib.sha256(raw.encode()).hexdigest() if raw else ""
    if not raw:
        return "failed", digest
    unknown = False
    for credential in decode_credentials(raw):
        try:
            provider_for_key(credential.provider, credential.api_key).list_tagged("cascade-")
        except Exception as exc:
            if classify_rent_failure(str(exc)) != "auth":
                unknown = True
        else:
            return "ok", digest
    return ("unknown" if unknown else "failed"), digest
