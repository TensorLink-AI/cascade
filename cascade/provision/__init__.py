"""GPU-pod provisioning for cascade.

``cascade.provision`` holds the provider adapters (lium, shadeform; Targon/Shadeform CPU VMs
for the DEC-CA-0053 generation tier in :mod:`.cpu`), the pod
health gate, the per-payer rentals the trainer's funded legs use
(:mod:`cascade.provision.funded`), and the ``cascade-provisioner`` service: the
validator's elastic eval pod plus the trainer's ``hosts.toml`` (static
fragment + operator lanes). It consumes the trainer's contract
(``cascade.trainer.remote.load_hosts``) and never modifies it.

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
    pick_shadeform_cpu_offer,
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
from .cpu import (
    CpuProvider,
    CpuShape,
    ShadeformCpuProvider,
    cpu_providers,
    select_cpu_offer,
    select_tiered_offer,
)
from .targon import TargonCpuProvider, parse_cpu_inventory

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
    "pick_shadeform_cpu_offer",
    "pick_shadeform_offer",
    "provision_and_run",
    "render_hosts_toml",
    "select_provider",
    "shadeform_create_body",
    "shadeform_pod_address",
    "teardown",
    "validate_digest_pinned",
    "wait_ssh_reachable",
    "CpuProvider",
    "CpuShape",
    "ShadeformCpuProvider",
    "TargonCpuProvider",
    "cpu_providers",
    "parse_cpu_inventory",
    "select_cpu_offer",
    "select_tiered_offer",
]
