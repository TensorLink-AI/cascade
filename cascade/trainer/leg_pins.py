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


def proven_runtime_terms(cfg, contract, digest: str) -> tuple[str, str] | None:
    """DEC-CA-0067: the PROVEN ``(train_image_digest, continuation_protocol)``
    of a leg that recorded the full ``digest`` against ``contract``, else
    None (:func:`cascade.shared.manifest.prove_runtime_terms`). Tried: the
    contract's own pin, every ``[v2] prior_train_image_digests`` entry and —
    with an automatic repin — the pre-switch pin."""
    from ..shared.manifest import prove_runtime_terms

    priors = tuple(getattr(getattr(cfg, "v2", None), "prior_train_image_digests", ()) or ())
    base = getattr(cfg.training, "v2_contract", None)
    if base is not None and getattr(base, "train_image_digest_after", ""):
        priors = (*priors, str(base.train_image_digest))
    return prove_runtime_terms(contract, str(digest or ""), priors)


def receipt_contract_ok(cfg, contract, digest: str, contract_block) -> bool:
    """A worker receipt's ``training_contract_digest`` is the dispatched
    contract's — or, in an era under the DEC-CA-0067 decoupled rule, provably
    that contract with other runtime terms (an old-image leg re-attached
    across a repin + restart)."""
    from ..shared.manifest import contract_digest

    if str(digest or "") == contract_digest(contract):
        return True
    return bool(contract_block is not None
                and image_decoupled_era(cfg, int(contract_block))
                and proven_runtime_terms(cfg, contract, digest) is not None)


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
    and never the miner's: an operator fault, requeued unburned.

    ``code_evidence``: the trainer MEASURED the pod's worker code fingerprint
    and it is another release's — the only evidence a host / executor is ever
    blamed on (never our injected image env or our deployed config)."""

    def __init__(self, *args, code_evidence: bool = False) -> None:
        super().__init__(*args)
        self.code_evidence = bool(code_evidence)


class LegImageUnverified(LegImageMismatch):
    """The pod's release could not be measured right now (the probe's remote
    command failed: python, OOM, disk, a missing package dir). Unknown, never
    "wrong image": no blame, no retry-cap count; the lane / leg retries."""


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

    import subprocess

    from .remote import RemoteDispatchError

    transport: list[str] = []

    def _run(argv):
        try:
            proc = run_ssh(build_ssh_argv(host, shlex.join(list(argv))),
                           timeout=CHECK_TIMEOUT_SECONDS)
        except (subprocess.TimeoutExpired, OSError) as e:
            transport.append(f"{type(e).__name__}: {e}")
            return subprocess.CompletedProcess(argv, 255, "", str(e))
        if getattr(proc, "returncode", 0) == 255:
            transport.append((getattr(proc, "stderr", "") or "ssh exit 255")[-200:])
        return proc

    got, _n, err = remote_code_fingerprint(_run, workdir=host.workdir)
    if err and transport:
        # the POD is unreachable, not on another image: the dispatcher's
        # transport path drains it (mark_dead), never an image refusal
        raise RemoteDispatchError(
            f"pod unreachable during the code fingerprint probe of "
            f"{getattr(host, 'name', host)}: {transport[-1]}", returncode=255)
    if err:
        raise LegImageUnverified(
            f"code fingerprint unavailable on {getattr(host, 'name', host)}: {err}")
    with _FP_LOCK:
        _FINGERPRINTS[key] = (got, now)
    return got




_RUNTIME_IMAGES: dict[tuple, tuple[str, float]] = {}
_IMAGE_RE = re.compile(r"sha256:[0-9a-f]{64}")


def host_runtime_image(host, *, fresh: bool = False, clock=None) -> str:
    """The ``CASCADE_TRAIN_IMAGE_DIGEST`` a worker dispatched to ``host`` will
    see — exactly its launch env: the host's own ``static_env`` (which the
    dispatcher sources into the worker's session), else what a plain ssh
    session on the host exports (the worker runs in one:
    ``cascade.trainer.remote.build_remote_command`` = ``cd <workdir> && [set -a
    && . /dev/stdin && set +a &&] <worker>``; sshd's env + the login rc files,
    the same session ``printenv`` runs in). Cached like
    :func:`host_code_fingerprint`. "" = unset (the worker refuses every FINAL).
    Unreachable ⇒ ``RemoteDispatchError(255)``; a failed probe command ⇒
    :class:`LegImageUnverified`."""
    import subprocess
    import time

    from ..provision.codeprint import CHECK_TIMEOUT_SECONDS
    from .remote import RemoteDispatchError, build_ssh_argv, run_ssh

    for k, v in tuple(getattr(host, "static_env", ()) or ()):
        if k == "CASCADE_TRAIN_IMAGE_DIGEST":
            m = _IMAGE_RE.search(str(v).lower())
            return m.group(0) if m else ""
    now = (clock or time.monotonic)()
    key = _fp_key(host)
    with _FP_LOCK:
        hit = _RUNTIME_IMAGES.get(key)
        if hit is not None and not fresh and now - hit[1] < FINGERPRINT_TTL_SECONDS:
            return hit[0]
    try:
        proc = run_ssh(build_ssh_argv(host, shlex.join(["printenv", "CASCADE_TRAIN_IMAGE_DIGEST"])),
                       timeout=CHECK_TIMEOUT_SECONDS)
    except (subprocess.TimeoutExpired, OSError) as e:
        raise RemoteDispatchError(f"pod unreachable during the runtime image probe of "
                                  f"{getattr(host, 'name', host)}: {e}", returncode=255) from e
    rc = getattr(proc, "returncode", 0)
    if rc == 255:
        raise RemoteDispatchError(f"pod unreachable during the runtime image probe of "
                                  f"{getattr(host, 'name', host)}", returncode=255)
    if rc not in (0, 1):          # printenv: 1 = unset (a real answer)
        raise LegImageUnverified(f"runtime image probe failed on "
                                 f"{getattr(host, 'name', host)} (rc={rc})")
    m = _IMAGE_RE.search(str(getattr(proc, "stdout", "") or "").lower()) if rc == 0 else None
    got = m.group(0) if m else ""
    with _FP_LOCK:
        _RUNTIME_IMAGES[key] = (got, now)
    return got


class _HostHook:
    """``hook(host, block) -> host`` for the dispatcher (``block`` = the leg's
    ``--contract-block``, its era start). It acts ONLY on legs of eras under
    the resolved switch (:meth:`filters`): such a leg needs a host whose
    worker code AND runtime image identity are the after release, and gets
    the rendered post-switch chain.toml. Every other leg — a pre-switch era,
    or any era while the vote is unresolved — is handed through untouched
    (lane handout and dispatch exactly as without the repin; the worker's own
    FINAL image check still guards it). :meth:`match_state` lets a lane pool
    pick a matching lane instead of taking one and refusing."""

    # a lane known to run another image is re-measured at most this often
    REPROBE_SECONDS = 60.0

    def __init__(self, runner, fingerprint_fn, image_fn=None, clock=None):
        import time

        self.runner, self.fingerprint_fn = runner, fingerprint_fn
        self.image_fn = image_fn if image_fn is not None else host_runtime_image
        self.clock = clock or time.monotonic
        self._reprobed: dict[tuple, float] = {}

    def filters(self, block) -> bool:
        cfg = self.runner.cfg
        return bool(image_after_armed(cfg) and block is not None
                    and image_decoupled_era(cfg, int(block)))

    def _want(self, block) -> tuple[str, str]:
        cfg = self.runner.cfg
        fp = str(round_pins_at(cfg, int(block)).worker_code_fingerprint or "").lower()
        img = image_digest(str(cfg.training.at_block(int(block)).train_image_digest or ""))
        return fp, img

    def _fingerprint(self, host, *, fresh=False) -> str:
        return self.fingerprint_fn(host, fresh=fresh)

    def _image(self, host, *, fresh=False) -> str:
        return self.image_fn(host, fresh=fresh)

    def _observe(self, host, *, fresh: bool) -> tuple[str, str]:
        return self._fingerprint(host, fresh=fresh), self._image(host, fresh=fresh)

    def require_qualified(self, block) -> None:
        """Raise :class:`ImageAfterUnqualified` for a post-switch leg whose
        after release is not qualified (before any lane is taken)."""
        if self.filters(block):
            require_after_qualified(self.runner.cfg, f"dispatch (contract block {int(block)})")

    def match_state(self, host, block):
        """True: ``host`` runs the leg's era release (code and runtime image);
        False: it runs another one; None: unknown right now (unreachable) —
        wait / retry, never "no lane". A host seen on another release is
        re-measured at most every :attr:`REPROBE_SECONDS`."""
        from .remote import RemoteDispatchError

        if not self.filters(block):
            return True
        want = self._want(block)
        try:
            got = self._observe(host, fresh=False)
            if got == want:
                return True
            key = _fp_key(host)
            now = self.clock()
            if now - self._reprobed.get(key, float("-inf")) < self.REPROBE_SECONDS:
                return False
            self._reprobed[key] = now
            return self._observe(host, fresh=True) == want
        except (RemoteDispatchError, LegImageUnverified):
            return None           # unknown right now: wait / retry, never "wrong"
        except LegImageMismatch:
            return False

    def matches(self, host, block) -> bool:
        """Whether ``host`` may take the leg (:meth:`match_state` is True)."""
        return self.match_state(host, block) is True

    def __call__(self, host, block):
        if not self.filters(block):
            return host
        self.require_qualified(block)
        want_fp, want_img = self._want(block)
        got = self._observe(host, fresh=False)
        if got != (want_fp, want_img):
            got = self._observe(host, fresh=True)        # a re-bootstrapped lane
        if got[0] != want_fp:
            raise LegImageMismatch(
                f"{getattr(host, 'name', host)} runs worker code {got[0][:12]}… but this "
                f"leg's era (start {block}) pins {want_fp[:12]}… (after image) — not "
                "dispatched here", code_evidence=True)
        if got[1] != want_img:
            raise LegImageMismatch(
                f"{getattr(host, 'name', host)} reports runtime image "
                f"{got[1][:19] or '<unset>'} but this leg's era (start {block}) pins "
                f"{want_img[:19]} — its worker would refuse FINAL; not dispatched here")
        return _push_after_toml(self.runner, host)


def dispatch_host_hook(runner, *, fingerprint_fn=host_code_fingerprint, image_fn=None,
                       clock=None):
    """The dispatcher's per-leg image hook (:class:`_HostHook`)."""
    return _HostHook(runner, fingerprint_fn, image_fn, clock)


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


# ── owner rule 2026-11-02: never the wrong image ────────────────────────────
#
# Post-switch legs only (pre-switch behaviour is exactly the release before
# it). A pod / lane the leg cannot use because of its image is released at
# once; its host + executor are marked bad ONLY on evidence the trainer
# measured on the pod itself (the worker code fingerprint) — never on our
# injected image env or our deployed config. The retry is unburned, keeps the
# leg's billing and is capped (IMAGE_RETRY_CAP per leg per IMAGE_RETRY_WINDOW_S,
# persisted across restarts); a capped leg WAITS (no_capacity), it is not
# dropped.

IMAGE_RETRY_CAP = 2
IMAGE_RETRY_WINDOW_S = 3600.0
WRONG_IMAGE_BAD_SECONDS = 24 * 3600.0
IMAGE_REFUSALS_FILE = "image_refusals.json"


class ImageRetryCapped(LegImageMismatch):
    """The leg already lost :data:`IMAGE_RETRY_CAP` pods to the wrong image in
    the window: it waits (unburned, no_capacity) instead of renting again."""


def worker_refused_image(exc: BaseException) -> bool:
    """The worker's own structured FINAL image refusal (exit 2 + the stdout
    sentinel, :func:`cascade.trainer.remote.worker_image_refusal`) — set by
    the dispatcher from the exit code and the worker's stdout, never from
    message text."""
    return bool(getattr(exc, "image_refusal", False)) and getattr(exc, "returncode", None) == 2


def leg_key(hotkey: str, role: str = "", era_seed: object = "") -> str:
    return f"{hotkey}|{role}|{era_seed}"


def _refusals_path(runner):
    root = getattr(runner, "work_root", None)
    return Path(root) / IMAGE_REFUSALS_FILE if root else None


def _load_refusals(runner) -> dict:
    mem = runner.__dict__.setdefault("_image_refusals", None)
    if mem is not None:
        return mem
    data: dict = {}
    path = _refusals_path(runner)
    if path is not None and path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            data = {str(k): [float(t) for t in v] for k, v in dict(raw).items()}
        except (OSError, ValueError, TypeError) as e:
            log.warning("image refusal counts unreadable (%s); starting empty", e)
    runner.__dict__["_image_refusals"] = data
    return data


def _window(seen, now):
    return [t for t in seen if now - t < IMAGE_RETRY_WINDOW_S]


def note_image_refusal(runner, key: str, *, now: float | None = None) -> int:
    """Count one wrong-image pod for leg ``key``; persisted next to the
    trainer state. Returns the count inside the window."""
    import time

    from ..shared.atomic import write_text_atomic

    now = time.time() if now is None else float(now)
    with _FP_LOCK:
        data = _load_refusals(runner)
        for k in list(data):
            data[k] = _window(data[k], now)
            if not data[k]:
                data.pop(k)
        data.setdefault(key, []).append(now)
        n = len(data[key])
        path = _refusals_path(runner)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                write_text_atomic(path, json.dumps(data, sort_keys=True))
            except OSError as e:
                log.warning("image refusal counts not persisted: %s", e)
        return n


def image_retry_exhausted(runner, key: str, *, now: float | None = None) -> bool:
    import time

    now = time.time() if now is None else float(now)
    with _FP_LOCK:
        return len(_window(_load_refusals(runner).get(key, []), now)) >= IMAGE_RETRY_CAP


def mark_wrong_image_host(runner, host, pod, reason: str) -> list[str]:
    """Never rent this host / executor again for the image: quarantine the
    host IP and mark every executor the trainer claimed for ``pod`` bad
    (with its learned GPU / site identity, :mod:`cascade.provision.lium_identity`).
    Best-effort; returns the executors marked."""
    from ..provision.core import record_host_quarantine
    from ..provision.lium_identity import mark_executor_bad

    why = f"wrong runtime image: {reason}"[:300]
    ip = str(getattr(host, "host", "") or "")
    if ip and ip not in ("127.0.0.1", "localhost"):
        try:
            record_host_quarantine(ip, why, seconds=WRONG_IMAGE_BAD_SECONDS, host_fault=True)
        except Exception as e:  # noqa: BLE001 — never mask the refusal
            log.warning("wrong-image host %s not quarantined: %s", ip, e)
    marked: list[str] = []
    iid = str(getattr(pod, "instance_id", "") or "")
    if iid:
        meta = getattr(runner, "_funded_claim_meta", None) or {}
        for machine, (_t, owner) in list(meta.items()):
            if owner == iid:
                mark_executor_bad(machine, why, seconds=WRONG_IMAGE_BAD_SECONDS)
                marked.append(machine)
    log.error("DEC-CA-0067: %s booted the wrong runtime image — released; host %s and "
              "executor(s) %s marked bad (%s)", iid or getattr(host, "name", "?"),
              ip or "?", ", ".join(m[:8] for m in marked) or "none known", reason)
    return marked
