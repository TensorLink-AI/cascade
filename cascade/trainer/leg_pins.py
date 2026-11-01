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

log = logging.getLogger(__name__)

AFTER_TOML_NAME = "chain.deployed.after.toml"
_FINGERPRINTS: dict[tuple, str] = {}
_FP_LOCK = threading.Lock()


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
    post-switch leg, else ``runner.cfg.round`` itself (same object)."""
    cfg = runner.cfg
    if not contract_runs_after(cfg, leg_contract(runner)):
        return cfg.round
    return round_pins_at(cfg, int(cfg.v2.image_decoupled_from_block))


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
    if (raw["v2"]["training"]["train_image_digest"] != v2.train_image_digest_after
            or raw["round"]["funded_pod_image"] != rnd.funded_pod_image
            or raw["round"]["worker_code_fingerprint"] != rnd.worker_code_fingerprint):
        raise ValueError("rendered post-switch chain.toml does not carry the after pins")
    return out


def _push_after_toml(runner, host):
    from dataclasses import replace as dc_replace

    src = getattr(runner, "deployed_chain_toml", None)
    if not src:
        return host
    text = render_after_toml(Path(src).read_text(encoding="utf-8"), runner.cfg)
    local = Path(getattr(runner, "work_root", ".")) / "_deployed" / AFTER_TOML_NAME
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(text, encoding="utf-8")
    pushed = runner._push_deployed_chain_toml(host, src=local, dest_name=AFTER_TOML_NAME)
    return dc_replace(pushed, chain_toml=f"{host.workdir}/{AFTER_TOML_NAME}")


def host_code_fingerprint(host) -> str:
    """The host's worker ``cascade/**/*.py`` fingerprint (cached per endpoint)."""
    from ..provision.codeprint import CHECK_TIMEOUT_SECONDS, remote_code_fingerprint
    from .remote import build_ssh_argv, run_ssh

    key = (str(getattr(host, "host", "")), int(getattr(host, "port", 0) or 0),
           str(getattr(host, "workdir", "")), str(getattr(host, "pinned_host_key", "") or ""))
    with _FP_LOCK:
        if key in _FINGERPRINTS:
            return _FINGERPRINTS[key]

    def _run(argv):
        return run_ssh(build_ssh_argv(host, shlex.join(list(argv))),
                       timeout=CHECK_TIMEOUT_SECONDS)

    got, _n, err = remote_code_fingerprint(_run, workdir=host.workdir)
    if err:
        raise RuntimeError(f"code fingerprint unavailable on {getattr(host, 'name', host)}: {err}")
    with _FP_LOCK:
        _FINGERPRINTS[key] = got
    return got


class LegImageMismatch(RuntimeError):
    """The host runs another worker image than the leg's era pins."""


def dispatch_host_hook(runner, *, fingerprint_fn=host_code_fingerprint):
    """``hook(host, block) -> host`` for the dispatcher (``block`` = the leg's
    ``--contract-block``, its era start). Pre-switch leg: the host must run the
    CURRENT worker code; post-switch leg: the AFTER code, and the host gets the
    rendered post-switch chain.toml."""

    def hook(host, block):
        cfg = runner.cfg
        if not image_after_armed(cfg) or block is None:
            return host
        post = image_decoupled_era(cfg, int(block))
        pins = round_pins_at(cfg, int(block))
        want = str(pins.worker_code_fingerprint or "").lower()
        if want:
            got = fingerprint_fn(host)
            if got != want:
                raise LegImageMismatch(
                    f"{getattr(host, 'name', host)} runs worker code {got[:12]}… but this "
                    f"leg's era (start {block}) pins {want[:12]}… "
                    f"({'after' if post else 'current'} image) — not dispatched here")
        return _push_after_toml(runner, host) if post else host

    return hook
