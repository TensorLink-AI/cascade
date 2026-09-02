"""Per-pod Hub credentials: a Harbor robot account minted for ONE funded leg.

The Hippius Hub is a Harbor OCI registry. A funded pod (DEC-CA-0036) lives on
the MINER's Lium account — its payer has console access to the box, so
anything in the worker's environment is readable by them. Forwarding the
operator's Hub login (or S3 keys, or an HF token) to such a pod hands those
credentials to the miner. PRISM's answer is a credential-free pod with a
master-side SSH harvest; ours can't do that without a worker-image rebuild
(the pinned worker uploads its own checkpoint), so this module gives the pod
the LEAST credential that still works:

* a Harbor **robot account** scoped to the checkpoint project with
  ``repository:push`` only — no pull of anything private (the repos the
  worker reads are public and pull anonymously), no delete, no other project;
* minted **per pod at rent time** and **revoked at teardown**, with a
  day-granularity Harbor expiry as the backstop, so a leaked secret is
  push-only, single-project, and dead within the leg;
* fail-closed: if minting fails the leg does NOT fall back to forwarding
  operator credentials — it skips as an unburned infra fault.

The worker sees an ordinary ``HIPPIUS_HUB_USERNAME`` / ``HIPPIUS_HUB_PASSWORD``
pair (``robot$<project>+<name>`` / secret) and pushes exactly as before. The
name embeds the netuid, round, and payer slug so an operator listing Harbor's
robots can attribute every one; a random suffix keeps retries from colliding.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = ["HarborRobots", "RobotCredential", "RobotError", "robot_name"]

log = logging.getLogger("cascade.funding.robots")

_NAME_RE = re.compile(r"[^a-z0-9]")


class RobotError(RuntimeError):
    """Harbor refused (or could not reach) a robot create/delete."""


@dataclass(frozen=True)
class RobotCredential:
    """One minted robot: the id Harbor addresses it by, and the login pair."""

    id: int
    username: str               # "robot$<project>+<name>"
    secret: str = field(repr=False)
    project: str = ""

    def as_env(self) -> tuple[tuple[str, str], ...]:
        """The pod-side env pairs (`RemoteHost.static_env` shape)."""
        return (("HIPPIUS_HUB_USERNAME", self.username),
                ("HIPPIUS_HUB_PASSWORD", self.secret))


def robot_name(netuid: int, round_id: str, hotkey: str) -> str:
    """``funded-n<netuid>-<round>-<slug>-<rand>``: Harbor-legal and attributable."""
    slug = _NAME_RE.sub("", hotkey.lower())[:12] or "anon"
    rnd = _NAME_RE.sub("", str(round_id).lower())[:24] or "r"
    return f"funded-n{int(netuid)}-{rnd}-{slug}-{uuid.uuid4().hex[:6]}"


class HarborRobots:
    """Mint / revoke project-scoped push-only robots against a Harbor registry.

    ``auth_header`` is the OPERATOR's Harbor authorization (Basic/Bearer —
    ``hippius_hub.auth.resolve_auth_header(None)`` resolves it from the
    orchestrator's env or cached login); it never leaves this process.
    ``http`` is the ``(method, url, json) -> (status, body_dict)`` seam the
    tests inject; production uses httpx.
    """

    def __init__(self, registry_url: str, auth_header: str, *,
                 timeout: float = 30.0,
                 http: Callable[[str, str, dict | None], tuple[int, dict]] | None = None) -> None:
        if not auth_header:
            raise RobotError("no Hub credential to mint robots with (operator "
                             "login missing on the orchestrator)")
        self.base = registry_url.rstrip("/")
        self.auth_header = auth_header
        self.timeout = timeout
        self._http = http or self._httpx

    def _httpx(self, method: str, url: str, body: dict | None) -> tuple[int, dict]:
        import httpx

        headers = {"Authorization": self.auth_header, "Accept": "application/json"}
        with httpx.Client(timeout=self.timeout) as client:
            r = client.request(method, url, json=body, headers=headers)
        try:
            data = r.json() if r.content else {}
        except ValueError:
            data = {"raw": r.text[:300]}
        return r.status_code, (data if isinstance(data, dict) else {"raw": data})

    def create(self, name: str, project: str, *, duration_days: int = 1,
               actions: tuple[str, ...] = ("push",)) -> RobotCredential:
        """A project-level robot with ``repository:<action>`` grants only.

        ``duration_days`` is Harbor's expiry backstop (day granularity; the
        real lifetime bound is :meth:`delete` at teardown). Harbor returns the
        secret exactly once, in this response.
        """
        body = {
            "name": name,
            "description": "cascade funded leg (auto-minted; revoked at teardown)",
            "duration": int(duration_days),
            "level": "project",
            "disable": False,
            "permissions": [{
                "kind": "project",
                "namespace": project,
                "access": [{"resource": "repository", "action": a} for a in actions],
            }],
        }
        status, data = self._http("POST", f"{self.base}/api/v2.0/robots", body)
        if status not in (200, 201):
            raise RobotError(f"robot create for project {project!r} failed: "
                             f"HTTP {status} {_short(data)}")
        try:
            cred = RobotCredential(id=int(data["id"]), username=str(data["name"]),
                                   secret=str(data["secret"]), project=project)
        except (KeyError, TypeError, ValueError) as e:
            raise RobotError(f"robot create returned an unexpected body: "
                             f"{_short(data)}") from e
        if not cred.secret:
            raise RobotError("robot create returned an empty secret")
        log.info("minted Hub robot %s (id=%d) push-only on project %s",
                 cred.username, cred.id, project)
        return cred

    def delete(self, robot_id: int) -> bool:
        """Revoke a robot; True when gone (404 counts — idempotent)."""
        status, data = self._http("DELETE", f"{self.base}/api/v2.0/robots/{int(robot_id)}", None)
        if status in (200, 204, 404):
            log.info("revoked Hub robot id=%d (HTTP %d)", robot_id, status)
            return True
        raise RobotError(f"robot delete id={robot_id} failed: HTTP {status} {_short(data)}")


def _short(data: object) -> str:
    text = str(data)
    return text if len(text) <= 200 else text[:200] + "…"
