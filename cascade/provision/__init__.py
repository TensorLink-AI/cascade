"""Ephemeral GPU-pod provisioning for the cascade trainer's remote data plane.

``cascade.provision`` is the trainer's *sibling* service: it rents GPU pods per
round (sized off ``cascade-trainer --plan-only``), health-checks them, publishes
the trainer-local ``hosts.toml``, and tears the pods back down per stage. It
consumes the trainer's contract (``cascade.trainer.remote.load_hosts``, the
``heat_complete.json`` marker, the published manifest) and never modifies it.

The public surface re-exported here is the provider/rendering core lifted from
``deploy/provision.py`` (which remains as a thin CLI shim for the one-shot
manual flow).
"""

from .core import (
    DEFAULT_FORWARD_ENV,
    DEFAULT_PROVIDER_PRIORITY,
    DEFAULT_READY_TIMEOUT,
    DEFAULT_REMOTE_PYTHON,
    DEFAULT_SKU,
    DEFAULT_SSH_OPTIONS,
    DEFAULT_SSH_PORT,
    DEFAULT_WORKDIR,
    LaunchSpec,
    LiumProvider,
    PodAddress,
    Provider,
    ProvisionError,
    RenderOpts,
    ShadeformProvider,
    build_providers,
    lium_pod_address,
    lium_pod_ready,
    parse_lium_executors,
    parse_lium_pods,
    parse_ssh_host,
    parse_ssh_port,
    pick_shadeform_offer,
    provision_and_run,
    render_hosts_toml,
    select_provider,
    shadeform_create_body,
    shadeform_pod_address,
    teardown,
    validate_digest_pinned,
    wait_ssh_reachable,
)

__all__ = [
    "DEFAULT_FORWARD_ENV",
    "DEFAULT_PROVIDER_PRIORITY",
    "DEFAULT_READY_TIMEOUT",
    "DEFAULT_REMOTE_PYTHON",
    "DEFAULT_SKU",
    "DEFAULT_SSH_OPTIONS",
    "DEFAULT_SSH_PORT",
    "DEFAULT_WORKDIR",
    "LaunchSpec",
    "LiumProvider",
    "PodAddress",
    "Provider",
    "ProvisionError",
    "RenderOpts",
    "ShadeformProvider",
    "build_providers",
    "lium_pod_address",
    "lium_pod_ready",
    "parse_lium_executors",
    "parse_lium_pods",
    "parse_ssh_host",
    "parse_ssh_port",
    "pick_shadeform_offer",
    "provision_and_run",
    "render_hosts_toml",
    "select_provider",
    "shadeform_create_body",
    "shadeform_pod_address",
    "teardown",
    "validate_digest_pinned",
    "wait_ssh_reachable",
]
