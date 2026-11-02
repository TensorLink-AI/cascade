"""DEC-CA-0067 automatic image repin: the image pins a LEG runs under.

Legs of eras starting at or after the resolved ``[v2] image_decoupled_from_block``
run the ``[v2] *_after`` pins; earlier eras keep the current ones. The leg's
contract already carries the decision (``TrainingContractConfig.at_block`` of
the era start pins ``train_image_digest_after`` from the switch), so the rest of
the pins follow it:

* :func:`leg_round` — the ``[round]`` view (pod image, worker code
  fingerprint, measured-wall image) for the leg on this thread, used by every
  rent and the rent-time code check;
* :func:`dispatch_host_hook` — installed on the dispatcher ONLY while a repin
  is configured: before each dispatch it checks the host's code fingerprint
  against the leg's era pin (a lane or pod on the other image never takes the
  leg) and, for a post-switch leg, pushes a deployed chain.toml whose pins ARE
  the after pins (:func:`render_after_toml`) so any worker release resolves
  the same image and contract digest as the orchestrator.

Nothing here runs while ``[v2] train_image_digest_after`` is empty: every pin,
file and dispatch is byte-identical to the release without it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shlex
import threading
import tomllib
from pathlib import Path

from ..shared.config import image_digest
from ..shared.era import image_after_armed, image_decoupled_era, round_pins_at
from .faults import _OperatorFault

log = logging.getLogger(__name__)

AFTER_TOML_NAME = "chain.deployed.after.toml"
# host identity -> (fingerprint, measured at). Keyed on the endpoint AND the
# pinned host key / allocation id, so a reused ip:port (a new container) never
# returns another pod's value; short TTL, and a mismatch is always re-measured
# before refusing, so a lane re-bootstrapped onto the other image is accepted
# without a trainer restart.
_FINGERPRINTS: dict[tuple, tuple[str, float]] = {}
_FP_LOCK = threading.Lock()
FINGERPRINT_TTL_SECONDS = 300.0


def _local(runner):
    return runner.__dict__.get("_leg_local_obj")


def leg_contract(runner):
    """The contract of the leg running on this thread, or ``None``."""
    loc = _local(runner)
    return getattr(loc, "funded_contract", None) or getattr(loc, "leg_contract", None)


def contract_runs_after(cfg, contract) -> bool:
    """Whether ``contract`` (an era's effective contract) pins the after image."""
    if contract is None or not image_after_armed(cfg):
        return False
    after = image_digest(str(cfg.v2.train_image_digest_after))
    return bool(after) and image_digest(str(getattr(contract, "train_image_digest", ""))) == after


def leg_round(runner):
    """The ``[round]`` pins for the leg on this thread: the after view for a
    post-switch leg, else ``runner.cfg.round`` itself (same object). A
    post-switch leg whose after release is NOT qualified raises
    :class:`ImageAfterUnqualified` — every rent reads its pins here, so no pod
    is ever rented on an unqualified image (and never on the old one for a
    post-switch leg)."""
    cfg = runner.cfg
    if not contract_runs_after(cfg, leg_contract(runner)):
        return cfg.round
    require_after_qualified(cfg, "rent")
    return round_pins_at(cfg, int(cfg.v2.image_decoupled_from_block))


# ── the owner's qualification rule (enforced) ───────────────────────────────


def after_image_refusal(cfg) -> str:
    """Why the ``[v2] *_after`` pins must NOT be used, or "" (qualified, or
    the owner's emergency override is on, or no after pins). See
    :mod:`cascade.shared.image_qualification`."""
    from ..shared.image_qualification import qualification_status

    if not image_after_armed(cfg):
        return ""
    st = qualification_status(cfg)
    return "" if st.usable else st.reason


def require_after_qualified(cfg, what: str) -> None:
    """Raise :class:`ImageAfterUnqualified` (loud ERROR) when the after
    release is not qualified. Never a fallback to the current image: the leg
    fails as an operator fault (unburned) and waits for a record."""
    why = after_image_refusal(cfg)
    if why:
        log.error("DEC-CA-0067: REFUSING %s on the after image %s — %s. Commit the "
                  "qualification record (scripts/write_image_qualification.py) or, in an "
                  "emergency only, set [v2] image_after_unqualified_override = true",
                  what, str(cfg.v2.train_image_digest_after)[:19], why)
        raise ImageAfterUnqualified(f"after image not qualified ({what}): {why}")


@contextlib.contextmanager
def leg_contract_scope(runner, contract):
    """Name the leg's contract for :func:`leg_round` on this thread (paths that
    do not enter ``funded_leg_scope``: operator legs, the king's backup)."""
    loc = runner.__dict__.setdefault("_leg_local_obj", threading.local())
    missing = object()
    saved = getattr(loc, "leg_contract", missing)
    loc.leg_contract = contract
    try:
        yield
    finally:
        if saved is missing:
            with contextlib.suppress(AttributeError):
                delattr(loc, "leg_contract")
        else:
            loc.leg_contract = saved


class LegImageMismatch(_OperatorFault):
    """The host runs (or cannot prove it runs) another worker image than the
    leg's era pins. Never a lane fault (the lane breaker does not count it)
    and never the miner's: an operator fault, requeued unburned."""


class ImageAfterUnqualified(LegImageMismatch):
    """A post-switch leg whose ``[v2] *_after`` release has no valid
    qualification record (the owner's bit-identical rule)."""


# ── the deployed chain.toml a post-switch leg's worker loads ────────────────


_HEADER = re.compile(r"^\s*\[\[?\s*([A-Za-z0-9_.\-]+)\s*\]\]?\s*(#.*)?$")


def _set_keys(text: str, table: str, values: dict) -> str:
    """``text`` with ``key = value`` lines of ``[table]`` replaced (or appended
    right after the table header when absent). Values are str or int."""
    lines = text.splitlines(keepends=True)
    out, current, header_at, done = [], None, None, set()
    for line in lines:
        m = _HEADER.match(line)
        if m:
            if current == table and header_at is not None:
                out[header_at + 1:header_at + 1] = [
                    f"{k} = {json.dumps(v)}\n" for k, v in values.items() if k not in done]
                done.update(values)
            current = m.group(1)
            out.append(line)
            if current == table:
                header_at = len(out) - 1
            continue
        if current == table:
            key = line.split("=", 1)[0].strip() if "=" in line else ""
            if key in values and not line.lstrip().startswith("#"):
                out.append(f"{key} = {json.dumps(values[key])}\n")
                done.add(key)
                continue
        out.append(line)
    if current == table and header_at is not None:
        out[header_at + 1:header_at + 1] = [
            f"{k} = {json.dumps(v)}\n" for k, v in values.items() if k not in done]
        done.update(values)
    if set(values) - done:
        raise ValueError(f"deployed chain.toml has no [{table}] table")
    return "".join(out)


def render_after_toml(text: str, cfg) -> str:
    """The deployed chain.toml with the after pins IN PLACE of the current ones
    (``[v2.training] train_image_digest``, ``[round] funded_pod_image`` /
    ``worker_code_fingerprint`` / ``funded_sku_wall_measured_image``) and the
    resolved switch block typed (``[v2] image_decoupled_from_block``) — so a
    worker of any release resolves the post-switch contract the orchestrator
    expects. Verified by re-parsing."""
    v2 = cfg.v2
    rnd = round_pins_at(cfg, int(v2.image_decoupled_from_block))
    out = _set_keys(text, "v2.training", {"train_image_digest": str(v2.train_image_digest_after)})
    round_vals = {"funded_pod_image": rnd.funded_pod_image,
                  "worker_code_fingerprint": rnd.worker_code_fingerprint}
    if rnd.funded_sku_wall_measured_image:
        round_vals["funded_sku_wall_measured_image"] = rnd.funded_sku_wall_measured_image
    out = _set_keys(out, "round", round_vals)
    out = _set_keys(out, "v2", {"image_decoupled_from_block": int(v2.image_decoupled_from_block)})
    raw = tomllib.loads(out)
    want = {("v2", "training", "train_image_digest"): str(v2.train_image_digest_after),
            ("v2", "image_decoupled_from_block"): int(v2.image_decoupled_from_block),
            **{("round", k): v for k, v in round_vals.items()}}
    for path, value in want.items():
        node = raw
        for part in path:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        if node != value:
            raise ValueError(f"rendered post-switch chain.toml: {'.'.join(path)} = {node!r}, "
                             f"expected {value!r}")
    return out


def _push_after_toml(runner, host):
    from dataclasses import replace as dc_replace

    from ..shared.atomic import write_text_atomic

    src = getattr(runner, "deployed_chain_toml", None)
    if not src:
        # the pod would resolve its baked / hosts.toml config: never the
        # post-switch pins the orchestrator expects — refuse, never guess
        raise LegImageMismatch("post-switch leg needs the trainer's deployed_chain_toml "
                               "(--chain-toml) to render its after-image config")
    text = render_after_toml(Path(src).read_text(encoding="utf-8"), runner.cfg)
    local = Path(getattr(runner, "work_root", ".")) / "_deployed" / AFTER_TOML_NAME
    local.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(local, text)
    pushed = runner._push_deployed_chain_toml(host, src=local, dest_name=AFTER_TOML_NAME)
    return dc_replace(pushed, chain_toml=f"{host.workdir}/{AFTER_TOML_NAME}")


def _fp_key(host) -> tuple:
    return (str(getattr(host, "host", "")), int(getattr(host, "port", 0) or 0),
            str(getattr(host, "workdir", "")), str(getattr(host, "pinned_host_key", "") or ""),
            str(getattr(host, "allocation_id", "") or ""))


def host_code_fingerprint(host, *, fresh: bool = False, clock=None) -> str:
    """The host's worker ``cascade/**/*.py`` fingerprint (cached per host
    identity for :data:`FINGERPRINT_TTL_SECONDS`; ``fresh`` re-measures)."""
    import time

    from ..provision.codeprint import CHECK_TIMEOUT_SECONDS, remote_code_fingerprint
    from .remote import build_ssh_argv, run_ssh

    now = (clock or time.monotonic)()
    key = _fp_key(host)
    with _FP_LOCK:
        hit = _FINGERPRINTS.get(key)
        if hit is not None and not fresh and now - hit[1] < FINGERPRINT_TTL_SECONDS:
            return hit[0]

    def _run(argv):
        return run_ssh(build_ssh_argv(host, shlex.join(list(argv))),
                       timeout=CHECK_TIMEOUT_SECONDS)

    got, _n, err = remote_code_fingerprint(_run, workdir=host.workdir)
    if err:
        raise LegImageMismatch(
            f"code fingerprint unavailable on {getattr(host, 'name', host)}: {err}")
    with _FP_LOCK:
        _FINGERPRINTS[key] = (got, now)
    return got




class _HostHook:
    """``hook(host, block) -> host`` for the dispatcher (``block`` = the leg's
    ``--contract-block``, its era start). Pre-switch leg: the host must run
    the CURRENT worker code; post-switch leg: the AFTER code, and the host
    gets the rendered post-switch chain.toml. :meth:`matches` lets a lane
    pool pick a lane on the leg's image instead of taking one and refusing."""

    def __init__(self, runner, fingerprint_fn):
        self.runner, self.fingerprint_fn = runner, fingerprint_fn

    def _want(self, block) -> str:
        return str(round_pins_at(self.runner.cfg, int(block)).worker_code_fingerprint
                   or "").lower()

    def _fingerprint(self, host, *, fresh=False) -> str:
        return self.fingerprint_fn(host, fresh=fresh)

    def require_qualified(self, block) -> None:
        """Raise :class:`ImageAfterUnqualified` for a post-switch leg whose
        after release is not qualified (before any lane is taken)."""
        cfg = self.runner.cfg
        if image_after_armed(cfg) and block is not None and image_decoupled_era(cfg, int(block)):
            require_after_qualified(cfg, f"dispatch (contract block {int(block)})")

    def matches(self, host, block) -> bool:
        """Whether ``host`` runs the leg's era image (re-measured on a miss)."""
        if not image_after_armed(self.runner.cfg) or block is None:
            return True
        want = self._want(block)
        if not want:
            return True
        try:
            return (self._fingerprint(host) == want
                    or self._fingerprint(host, fresh=True) == want)
        except LegImageMismatch:
            return False

    def __call__(self, host, block):
        cfg = self.runner.cfg
        if not image_after_armed(cfg) or block is None:
            return host
        post = image_decoupled_era(cfg, int(block))
        if post:
            self.require_qualified(block)
        want = self._want(block)
        if want:
            got = self._fingerprint(host)
            if got != want:
                got = self._fingerprint(host, fresh=True)   # a re-bootstrapped lane
            if got != want:
                raise LegImageMismatch(
                    f"{getattr(host, 'name', host)} runs worker code {got[:12]}… but this "
                    f"leg's era (start {block}) pins {want[:12]}… "
                    f"({'after' if post else 'current'} image) — not dispatched here")
        return _push_after_toml(self.runner, host) if post else host


def dispatch_host_hook(runner, *, fingerprint_fn=host_code_fingerprint):
    """The dispatcher's per-leg image hook (:class:`_HostHook`)."""
    return _HostHook(runner, fingerprint_fn)


# ── the per-tick signal: ERROR log + operator alert ─────────────────────────


def image_qualification_condition(cfg) -> dict | None:
    """The operator-watchdog condition for the after release, or ``None``:
    ``image_after_unqualified`` (armed without a valid record — every
    post-switch leg is refused) or ``image_after_override`` (used unqualified
    under the owner's emergency override)."""
    from ..shared.image_qualification import qualification_status

    if not image_after_armed(cfg):
        return None
    st = qualification_status(cfg)
    if not st.usable:
        return {"code": "image_after_unqualified", "scope": "image", "reason": st.reason[:300]}
    if st.override and not st.qualified:
        return {"code": "image_after_override", "scope": "image", "reason": st.reason[:300]}
    return None


def log_image_qualification(cfg) -> dict | None:
    """One trainer tick: a loud ERROR while the after release is unqualified
    (refused) or used under the override. Returns the condition."""
    cond = image_qualification_condition(cfg)
    if cond is None:
        return None
    if cond["code"] == "image_after_unqualified":
        log.error("DEC-CA-0067: [v2] *_after release %s is NOT QUALIFIED — every post-switch "
                  "rent/dispatch on it is refused (legs wait as operator faults): %s",
                  str(cfg.v2.train_image_digest_after)[:19], cond["reason"])
    else:
        log.error("DEC-CA-0067: [v2] image_after_unqualified_override = true — the after "
                  "release %s is used WITHOUT a qualification record (%s). Emergency only: "
                  "qualify it and turn the override off",
                  str(cfg.v2.train_image_digest_after)[:19], cond["reason"])
    return cond
