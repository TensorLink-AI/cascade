"""``cascade-provisioner`` console script — the per-round pod-rental service.

Runs BESIDE the trainer (same box or same work-root mount), never inside it:
the trainer holds the wallet and the round logic; this service holds only
provider API keys and an SSH key, and its whole job is to make the trainer's
``--remote-hosts`` file point at healthy rented GPUs at the right time and to
make the bill stop the moment each stage is done.

Configuration is a small TOML (see ``scripts/provision.example.toml``) for the
rental policy plus the trainer's own ``chain.toml`` for the round shape
(``[round] epoch_blocks``, ``[training] target_train_hours``, the image-digest
pin). ``--dry-run`` walks the full trigger→count→size→budget pipeline and logs
what it WOULD rent, renting nothing.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from ..shared.config import load_chain_config
from .core import (
    DEFAULT_FORWARD_ENV,
    DEFAULT_REMOTE_PYTHON,
    DEFAULT_WORKDIR,
    ProvisionError,
    _read_pubkey,
    build_providers,
    validate_digest_pinned,
    wait_ssh_reachable,
)
from .health import HealthGate, HealthReport
from .loop import PodProfile, ProvisionerLoop, RenderSettings, parse_plan_output
from .policy import ProvisionPolicy, SkuCandidate, StagePolicy

log = logging.getLogger("cascade.provision.main")


# ── config (pure parse + validation; tested) ─────────────────────────────────


def build_stage_policy(raw: dict, stage: str) -> StagePolicy:
    """One ``[provisioner.heat|final|eval]`` table → a validated :class:`StagePolicy`."""
    sku = str(raw.get("sku", "")).strip()
    if not sku:
        raise ProvisionError(f"[provisioner.{stage}] sku must be non-empty (the exact "
                             "nvidia-smi device string, e.g. 'NVIDIA L40S')")
    gpus = int(raw.get("gpus_per_pod", 1))
    if gpus < 1:
        raise ProvisionError(f"[provisioner.{stage}] gpus_per_pod must be >= 1; got {gpus}")
    max_pods = int(raw.get("max_pods", 1))
    if max_pods < 0:
        raise ProvisionError(f"[provisioner.{stage}] max_pods must be >= 0 "
                             f"(0 = stage unmanaged, served by static hosts); got {max_pods}")
    price = float(raw.get("max_price_hr", 0))
    if price <= 0:
        raise ProvisionError(f"[provisioner.{stage}] max_price_hr must be > 0; got {price}")
    providers = tuple(str(p) for p in raw.get("providers", ("lium", "shadeform")))
    if not providers:
        raise ProvisionError(f"[provisioner.{stage}] providers must list at least one adapter")
    overhead = float(raw.get("slot_overhead", 1.3))
    if overhead < 1.0:
        raise ProvisionError(f"[provisioner.{stage}] slot_overhead must be >= 1.0; got {overhead}")
    candidates = []
    for i, c in enumerate(raw.get("candidate", ())):
        csku = str(c.get("sku", "")).strip()
        if not csku:
            raise ProvisionError(f"[[provisioner.{stage}.candidate]] #{i}: sku required")
        cgpus = int(c.get("gpus_per_pod", 1))
        cprice = float(c.get("max_price_hr", price))
        if cgpus < 1 or cprice <= 0:
            raise ProvisionError(f"[[provisioner.{stage}.candidate]] #{i} ({csku}): "
                                 f"gpus_per_pod >= 1 and max_price_hr > 0 required")
        candidates.append(SkuCandidate(sku=csku, market_sku=str(c.get("market_sku", "")).strip(),
                                       gpus_per_pod=cgpus, max_price_hr=cprice))
    return StagePolicy(sku=sku, gpus_per_pod=gpus, max_pods=max_pods,
                       providers=providers, max_price_hr=price, slot_overhead=overhead,
                       market_sku=str(raw.get("market_sku", "")).strip(),
                       candidates=tuple(candidates))


def build_policy(raw: dict, *, epoch_blocks: int) -> ProvisionPolicy:
    """The ``[provisioner]`` tree → a validated :class:`ProvisionPolicy`.

    ``epoch_blocks`` comes from chain.toml so the margin check runs against the
    REAL round cadence: a margin >= the epoch would trigger permanently.
    """
    top = raw.get("provisioner", raw)
    for stage in ("heat", "final"):
        if stage not in top:
            raise ProvisionError(f"provision config needs a [provisioner.{stage}] table")
    # [provisioner.eval] is OPTIONAL — the elastic validator eval pod. Absent
    # (or max_pods = 0) the stage does not exist: pre-eval configs keep their
    # exact behaviour, which is why it is not in the required loop above.
    eval_sp = build_stage_policy(top["eval"], "eval") if "eval" in top else None
    margin = int(top.get("trigger_margin_blocks", 25))
    if not 0 < margin < epoch_blocks:
        raise ProvisionError(
            f"trigger_margin_blocks={margin} must be in (0, epoch_blocks={epoch_blocks})")
    max_spend = float(top.get("max_spend_per_round", 0))
    if max_spend <= 0:
        raise ProvisionError(f"max_spend_per_round must be > 0 USD; got {max_spend}")
    ttl_epochs = int(top.get("ttl_epochs", 1))
    if ttl_epochs < 1:
        raise ProvisionError(f"ttl_epochs must be >= 1; got {ttl_epochs}")
    return ProvisionPolicy(
        heat=build_stage_policy(top["heat"], "heat"),
        final=build_stage_policy(top["final"], "final"),
        eval=eval_sp,
        trigger_margin_blocks=margin,
        max_spend_per_round=max_spend,
        ttl_epochs=ttl_epochs,
    )


# ── real boundaries (adapter surface, not unit-tested) ───────────────────────


def plan_argv(chain_toml: Path | None, work_root: Path, network: str | None) -> list[str]:
    """The exact ``--plan-only`` invocation. The network MUST be forwarded:
    the trainer CLI defaults to finney, so an unforwarded testnet provisioner
    counts the field on MAINNET's netuid — someone else's subnet — and plans
    eligible=0 every round (observed live: three consecutive windows)."""
    argv = ["uv", "run", "cascade-trainer", "--plan-only", "--work-root", str(work_root)]
    if chain_toml is not None:
        argv += ["--chain-toml", str(chain_toml)]
    if network is not None:
        argv += ["--network", network]
    return argv


def make_plan_fn(chain_toml: Path | None, work_root: Path,
                 network: str | None = None) -> callable:
    """The COUNT boundary: run ``cascade-trainer --plan-only`` and parse its JSON.

    Through ``uv run`` so the trainer resolves in the project venv regardless
    of how the provisioner itself was launched (systemd's PATH is minimal).
    The subprocess needs no wallet and no GPU — it only counts the field.
    """
    argv = plan_argv(chain_toml, work_root, network)

    def plan() -> dict:
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired as e:
            raise ProvisionError("--plan-only timed out after 600s") from e
        if proc.returncode != 0:
            raise ProvisionError(
                f"--plan-only failed (rc={proc.returncode}): {(proc.stderr or '')[-500:]}")
        return parse_plan_output(proc.stdout or "")

    return plan


def make_health_check(policy: ProvisionPolicy, render: RenderSettings, *,
                      image_digest: str, min_disk_gb: float,
                      hippius_probe) -> callable:
    """Bind the pure :class:`HealthGate` to a real ``ssh`` transport per pod.

    Gates are built lazily per ``(stage, provider)``: the pod's user/workdir/
    python differ by provider (lium=root, shadeform=the ``shadeform`` user)."""
    from ..trainer.remote import RemoteHost, build_ssh_argv, run_ssh

    gates: dict = {}

    def check(addr, stage: str, provider: str = "", *,
              sku: str = "", gpus: int = 0, attested_digest: str = "") -> HealthReport:
        prof = render.profile_for(provider)
        sp = {"heat": policy.heat, "final": policy.final, "eval": policy.eval}.get(stage)
        # The gate asserts what was ACTUALLY rented — with SKU fallback the
        # round's device can be any configured candidate, not just the primary.
        # The loop always passes the rented candidate's sku/gpus, so the stage
        # policy is only a fallback (and eval's may legitimately be None).
        gate_sku = sku or (sp.sku if sp is not None else "")
        gate_gpus = gpus or (sp.gpus_per_pod if sp is not None else 1)
        key = (stage, provider, gate_sku, gate_gpus)
        if key not in gates:
            gates[key] = HealthGate(
                sku=gate_sku, gpus=gate_gpus,
                remote_python=prof.remote_python, workdir=prof.workdir,
                image_digest=image_digest, min_disk_gb=min_disk_gb,
                hippius_probe=hippius_probe,
            )
        host = RemoteHost(name="health-probe", host=addr.ip, port=addr.ssh_port,
                          user=prof.user, key_path=render.key_path, workdir=prof.workdir)

        def run(remote_argv: Sequence[str]):
            return run_ssh(build_ssh_argv(host, shlex.join(list(remote_argv))), timeout=120)

        # Per-pod provider attestation on the (stage-cached) gate — see
        # HealthGate.attested_digest for why pod env alone can't be trusted
        # to exist on sshd-as-PID-1 images.
        gate = replace(gates[key], attested_digest=attested_digest)
        return gate.check(run)

    return check


def make_bootstrap(script: Path, render: RenderSettings, *,
                   timeout_s: float, pod_user: str,
                   auth_wait_s: float = 900.0) -> callable:
    """The BOOTSTRAP boundary: run an operator-supplied script against a fresh pod.

    No digest-pinned worker image is published yet, so pods rent bare and get
    provisioned over SSH — the script (run ON the orchestrator) receives the
    pod's coordinates via env (``POD_IP``/``POD_PORT``/``POD_USER``/``POD_KEY``/
    ``POD_STAGE``/``POD_WORKDIR``) and typically rsyncs the source tree and
    ``uv sync``s the pinned lock. Exit 0 = ready for the health gate — which
    then independently verifies whatever the script claims to have built.
    """
    import os

    def _auth_ready(addr, user: str) -> bool:
        """Marketplace pods inject SSH keys well AFTER sshd answers TCP — lium
        ~30-60s, hyperstack VMs (shadeform VM-mode) 7-8+ MINUTES of cloud-init
        (observed live 2026-07-15: eval pod killed at t+27s pre-gate; then a
        healthy 4xA6000s burned at 180s AND 420s — auth landed ~1 min after
        the 420s expiry). Poll a no-op ssh
        until auth lands or the wait expires."""
        import time as _t

        argv = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                "-o", "BatchMode=yes", "-i", os.path.expanduser(render.key_path),
                "-p", str(addr.ssh_port), f"{user}@{addr.ip}", "true"]
        # Two distinct pod failures wear different stderr:
        #  * "Permission denied (publickey)" = port UP, key still injecting —
        #    wait the full auth_wait_s (marketplace key lag can be minutes).
        #  * "Connection refused" = sshd/port DOWN. The pod already passed the
        #    TCP reachability gate before bootstrap, so a port that is now
        #    refused is a lemon that won't recover; fail fast (dead_port_cap)
        #    so the replacement rents while the market still has offers
        #    (a 15-min wait on a dead port dried up lium's 2x pool, 2026-07-15).
        dead_port_cap = min(150.0, auth_wait_s)
        t0 = _t.monotonic()
        last_err = ""
        while True:
            try:
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
                if proc.returncode == 0:
                    return True
                last_err = (proc.stderr or "").strip()[-200:]
            except subprocess.TimeoutExpired:
                last_err = "ssh probe timed out"
            refused = "connection refused" in last_err.lower()
            cap = dead_port_cap if refused else auth_wait_s
            if _t.monotonic() - t0 >= cap:
                log.warning("auth probe giving up on %s@%s:%s after %.0fs (%s): %s",
                            user, addr.ip, addr.ssh_port, _t.monotonic() - t0,
                            "dead port" if refused else "auth lag", last_err or "(none)")
                return False
            _t.sleep(10)

    def bootstrap(addr, stage: str, provider: str = "") -> bool:
        prof = render.profile_for(provider)
        user = prof.user if provider else pod_user
        if not _auth_ready(addr, user):
            log.error("bootstrap %s:%s: ssh auth never came up within %.0fs "
                      "(key injection lag exceeded)", addr.ip, addr.ssh_port, auth_wait_s)
            return False
        env = dict(os.environ)
        env.update({
            "POD_IP": addr.ip, "POD_PORT": str(addr.ssh_port),
            "POD_USER": user,
            "POD_KEY": render.key_path, "POD_STAGE": stage, "POD_WORKDIR": prof.workdir,
        })
        try:
            proc = subprocess.run(["bash", str(script)], env=env, timeout=timeout_s,
                                  capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            log.error("bootstrap %s:%s timed out after %.0fs", addr.ip, addr.ssh_port, timeout_s)
            return False
        if proc.returncode != 0:
            log.error("bootstrap %s:%s failed (rc=%d): %s", addr.ip, addr.ssh_port,
                      proc.returncode, (proc.stderr or proc.stdout or "")[-800:])
            return False
        log.info("bootstrap %s:%s done", addr.ip, addr.ssh_port)
        return True

    return bootstrap


def make_hippius_probe(storage) -> callable:
    """A reachability probe for the manifest store (health check #6).

    Reads the tiny ``latest.json`` pointer: if that round-trips, the pods'
    checkpoint pushes and generator pulls have a live storage path. Built
    lazily so a dry-run without Hippius credentials still works.
    """
    from ..shared.hippius import MANIFEST_LATEST_KEY, open_manifest_store

    store = open_manifest_store(storage)

    def probe() -> bool:
        try:
            store.get_text(MANIFEST_LATEST_KEY)
            return True
        except Exception:  # noqa: BLE001 — unreachable or unauthorised: same verdict
            return False

    return probe


# ── CLI ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cascade-provisioner",
        description="Per-round GPU pod provisioner for the cascade trainer.",
    )
    p.add_argument("--config", type=Path, required=True,
                   help="Provisioner TOML (see scripts/provision.example.toml).")
    p.add_argument("--chain-toml", type=Path, default=None,
                   help="The trainer's chain.toml (round cadence, budgets, image pin).")
    p.add_argument("--work-root", type=Path, default=Path("./_train_work"),
                   help="The TRAINER's work root (shared): heat_complete.json is watched here.")
    p.add_argument("--network", default="finney")
    p.add_argument("--dry-run", action="store_true",
                   help="Walk trigger→count→size→budget and log intended rentals; rent nothing.")
    p.add_argument("--once", action="store_true", help="Run a single cycle and exit.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    from ..shared.env import load_env_files

    load_env_files()
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # UNMUTABLE service logging: bittensor's in-process logging machinery
    # reconfigures the root logger when the chain client first connects, which
    # silently swallowed every provisioner log after cycle 1 (observed live
    # 2026-07-14: 3h of invisible-but-working loop, then an invisible trigger
    # failure). Give the cascade tree its OWN handler and stop propagating —
    # nothing bittensor does to root can mute us again.
    # bittensor's logging init STRIPS handlers and raises the level to
    # CRITICAL on named loggers whenever a chain client connects (verified
    # live 2026-07-14: same logger object, handlers=[] and level=50 after
    # ChainClient.from_config). A one-time setup therefore cannot survive —
    # ensure_service_logging() re-asserts idempotently and the loop calls it
    # at EVERY cycle start (ProvisionerLoop.on_cycle).
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    level = args.log_level
    state: dict = {"stream": None, "file": None}

    def _alive(h) -> bool:
        stream = getattr(h, "stream", None)
        return h is not None and stream is not None and not getattr(stream, "closed", True)

    def ensure_service_logging() -> None:
        """Re-assert AND, when necessary, REBUILD the service handlers.

        bittensor's logging init doesn't just strip named loggers' handlers —
        it can leave previously-attached handler objects with CLOSED streams,
        so re-adding the same object emits into a ValueError that the logging
        module swallows. Broken handlers are therefore reconstructed from
        scratch, not re-attached.
        """
        lg = logging.getLogger("cascade")
        if not _alive(state["stream"]):
            state["stream"] = logging.StreamHandler()
            state["stream"].setFormatter(fmt)
        if not _alive(state["file"]):
            state["file"] = logging.FileHandler("provisioner-service.log")
            state["file"].setFormatter(fmt)
        for h in (state["stream"], state["file"]):
            if h not in lg.handlers:
                lg.addHandler(h)
        # drop dead strays so emits don't raise-and-swallow on them
        for h in list(lg.handlers):
            if h not in (state["stream"], state["file"]):
                lg.removeHandler(h)
        lg.setLevel(level)
        lg.propagate = False
        lg.disabled = False
        # bittensor sets level=CRITICAL on EVERY existing named logger — the
        # children too ("cascade.provision.loop" etc.), and a child's own
        # level beats any parent fix. Sweep the whole cascade.* tree back to
        # NOTSET (defer to the parent) and re-enable.
        for name, obj in list(logging.root.manager.loggerDict.items()):
            if name.startswith("cascade.") and isinstance(obj, logging.Logger):
                obj.setLevel(logging.NOTSET)
                obj.disabled = False

    ensure_service_logging()
    globals()["_ensure_service_logging"] = ensure_service_logging
    try:
        return _run(args)
    except ProvisionError as e:
        log.error("%s", e)
        return 2
    except KeyboardInterrupt:
        log.warning("interrupted — rented pods stay in the ledger; restart to resume teardown")
        return 130


def _run(args) -> int:
    import tomllib

    cfg = load_chain_config(args.chain_toml)
    raw = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    top = raw.get("provisioner", {})
    policy = build_policy(raw, epoch_blocks=cfg.round.epoch_blocks)

    image = str(top.get("image", ""))
    if not top.get("bootstrap_script"):
        # Image-boot mode: the pod IS the image, so a moving tag breaks the
        # expected_gpu re-derivation contract — digest pin required.
        validate_digest_pinned(image)
    # Bootstrap mode: image may be EMPTY — lium then boots its default SSH
    # template (a template name is not a valid docker ref and would 400), and
    # shadeform VM-mode ignores the image entirely.
    pubkey_arg = str(top.get("ssh_pubkey", ""))
    if not pubkey_arg:
        raise ProvisionError("[provisioner] ssh_pubkey is required (inline key or .pub path)")
    render = RenderSettings(
        image=image,
        ssh_pubkey=_read_pubkey(pubkey_arg),
        key_path=str(top.get("ssh_key_path",
                             pubkey_arg[:-4] if pubkey_arg.endswith(".pub") else "")),
        forward_env=tuple(top.get("forward_env", DEFAULT_FORWARD_ENV)),
        remote_python=str(top.get("remote_python", DEFAULT_REMOTE_PYTHON)),
        workdir=str(top.get("workdir", DEFAULT_WORKDIR)),
        chain_toml=(str(top["chain_toml"]) if top.get("chain_toml") else None),
        ssh_port=int(top.get("ssh_port", 22)),
    )
    if not render.key_path:
        raise ProvisionError("[provisioner] ssh_key_path is required (private key for hosts.toml)")
    # Per-provider pod profiles MUST attach before anything closes over render:
    # RenderSettings is frozen, so replace() makes a NEW object — a consumer
    # built earlier keeps the profile-less one. That exact bug ran every
    # shadeform bootstrap as root@ (default profile) while the config said
    # user="shadeform": three healthy 4xA6000s burned across two rental
    # windows before the probe's stderr exposed the wrong user (2026-07-15).
    profiles = {
        name: PodProfile(
            user=str(t.get("user", "root")),
            workdir=str(t.get("workdir", render.workdir)),
            remote_python=str(t.get("remote_python", render.remote_python)),
        )
        for name, t in (top.get("pods", {}) or {}).items()
    }
    if profiles:
        render = replace(render, profiles=profiles)

    hosts_path = Path(top.get("hosts_path", "hosts.toml"))
    work_root = Path(args.work_root)
    state_path = Path(top.get("state_path", work_root / "provisioner_state.json"))

    # Elastic validator eval pod: BOTH the [provisioner.eval] policy and
    # eval_hosts_path must be present for the stage to exist. Half a config is
    # an operator mistake — fail loudly now, not silently never-rent.
    eval_hosts_path = Path(top["eval_hosts_path"]) if top.get("eval_hosts_path") else None
    receipt_prefix = str(top.get("receipt_prefix", ""))
    if policy.eval is not None and policy.eval.max_pods > 0 and eval_hosts_path is None:
        raise ProvisionError("[provisioner.eval] is configured but eval_hosts_path is "
                             "not set — the eval pod would have nowhere to publish")
    if eval_hosts_path is not None and eval_hosts_path == hosts_path:
        raise ProvisionError("eval_hosts_path must differ from hosts_path — the "
                             "validator's eval file must never clobber the trainer's fleet")

    # Operator's static [[host]] entries (e.g. a long-lived final pod) — appended
    # verbatim to every hosts.toml publish so provisioner activity never drops them.
    static_hosts_text = ""
    if top.get("static_hosts"):
        static_path = Path(top["static_hosts"])
        static_hosts_text = static_path.read_text(encoding="utf-8")
        from ..trainer.remote import load_hosts  # validate NOW, not mid-round
        try:
            load_hosts(static_path)
        except Exception as e:
            raise ProvisionError(f"static_hosts {static_path} does not parse as a "
                                 f"hosts.toml fragment: {e}") from e

    bootstrap = None
    if top.get("bootstrap_script"):
        script = Path(top["bootstrap_script"])
        if not script.is_file():
            raise ProvisionError(f"bootstrap_script not found: {script}")
        bootstrap = make_bootstrap(
            script, render,
            timeout_s=float(top.get("bootstrap_timeout_s", 1800.0)),
            pod_user=str(top.get("pod_user", "root")),
        )

    provider_names = list(dict.fromkeys([
        *policy.heat.providers, *policy.final.providers,
        *(policy.eval.providers if policy.eval is not None else ()),
    ]))
    provider_opts: dict[str, dict] = {}
    if top.get("shadeform_ssh_key_id"):
        provider_opts["shadeform"] = {"ssh_key_id": str(top["shadeform_ssh_key_id"])}
    providers = {p.name: p for p in build_providers(provider_names, provider_opts)}

    hippius_probe = None
    manifest_store = None
    if not args.dry_run:
        from ..shared.hippius import open_manifest_store

        manifest_store = open_manifest_store(cfg.storage)
        hippius_probe = make_hippius_probe(cfg.storage)

    from ..shared.chain import ChainClient

    loop = ProvisionerLoop(
        policy=policy,
        providers=providers,
        chain_client=ChainClient.from_config(cfg, network=args.network),
        chain_client_factory=lambda: ChainClient.from_config(cfg, network=args.network),
        plan_fn=make_plan_fn(args.chain_toml, work_root, args.network),
        render=render,
        hosts_path=hosts_path,
        work_root=work_root,
        state_path=state_path,
        epoch_blocks=cfg.round.epoch_blocks,
        final_hours=cfg.training.target_train_hours,
        manifest_store=manifest_store,
        eval_hosts_path=eval_hosts_path,
        receipt_prefix=receipt_prefix,
        health_check=make_health_check(
            policy, render,
            image_digest=cfg.training.train_image_digest,
            min_disk_gb=float(top.get("min_disk_gb", 20.0)),
            hippius_probe=hippius_probe,
        ),
        bootstrap=bootstrap,
        static_hosts_text=static_hosts_text,
        ssh_probe=lambda ip, port: wait_ssh_reachable(ip, port, timeout=300.0),
        poll_seconds=float(top.get("poll_seconds", 30.0)),
        dry_run=bool(args.dry_run),
        on_cycle=globals().get("_ensure_service_logging"),
    )
    eval_desc = ("off" if policy.eval is None or policy.eval.max_pods == 0
                 or eval_hosts_path is None
                 else f"{policy.eval.max_pods}×{policy.eval.sku}"
                      f"({policy.eval.gpus_per_pod}x)→{eval_hosts_path}")
    log.info("provisioner up: heat=%d×%s(%dx) final=%d×%s(%dx) eval=%s margin=%d blocks "
             "cap=$%.2f/round ttl=%d epoch(s)%s",
             policy.heat.max_pods, policy.heat.sku, policy.heat.gpus_per_pod,
             policy.final.max_pods, policy.final.sku, policy.final.gpus_per_pod,
             eval_desc, policy.trigger_margin_blocks, policy.max_spend_per_round,
             policy.ttl_epochs, " [DRY RUN]" if args.dry_run else "")
    if args.once:
        loop.run_once()
        return 0
    loop.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
