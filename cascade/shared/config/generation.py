"""``[generation]`` — the CPU generation tier (DEC-CA-0053).

Operator-side, never part of ``contract_digest``: where and on what a leg's
corpus is materialised before the GPU trains on it. The scored half (the
``shards`` corpus mode, ``long_series``, the context schedule) lives in
``[training]``. ``enabled = false`` keeps generation on the GPU pod. The
mainnet template enables this tier, but it runs only for a shards contract;
the testnet template leaves it off.
"""

from __future__ import annotations

from dataclasses import dataclass

GENERATION_PROVIDERS = ("targon", "shadeform")
KING_CORPUS_PUBLISH = ("full", "recipe")


@dataclass(frozen=True)
class GenerationConfig:
    # Master switch. False ⇒ the tier never rents, generates or stores.
    enabled: bool = False
    # CPU providers, preference-ordered. Lium is not one (no CPU executors).
    providers: tuple[str, ...] = GENERATION_PROVIDERS
    # The pinned machine shape every leg gets: vCPUs (the budget unit; the
    # owner pays) and RAM per sandbox child. ``children`` = 0 ⇒ one per vCPU.
    vcpus: int = 32
    child_memory_mb: int = 2048
    children: int = 0
    # Shared CPU VM limit; additional corpora wait before provisioning.
    max_parallel_jobs: int = 2
    # Hard ceiling on the hourly price of a rented CPU VM (USD).
    max_price_per_hour: float = 1.25
    # 90-minute generation window per leg: the CPU budget is vcpus × this.
    window_seconds: int = 5400
    # Unique points materialised per leg (pre-patch), and points per shard.
    target_points: int = 50_000_000_000
    shard_points: int = 16_777_216
    # Longest series a generator may yield once the tier is live.
    max_length: int = 32768
    # Hippius S3 prefix of the corpus dataset, and how long non-king corpora
    # are kept before deletion.
    storage_prefix: str = "corpora/"
    # The S3 bucket corpora live in ("" = [storage] manifest_bucket); private
    # by default, a king's published objects carry the public-read ACL.
    bucket: str = ""
    # The provider VM image the tier boots (Targon needs one; Shadeform VM
    # mode uses its own). Bootstrapped over ssh like a bare GPU pod.
    vm_image: str = ""
    retention_days: int = 14
    # How a crowned king's corpus is published: every shard, or the recipe
    # (generator ref + seed + digest) plus ``sample_shards`` shards.
    king_publish: str = "recipe"
    sample_shards: int = 4

    @property
    def n_children(self) -> int:
        return self.children if self.children > 0 else self.vcpus

    @property
    def cpu_budget_seconds(self) -> int:
        return self.vcpus * self.window_seconds


def parse_generation(raw: dict | None) -> GenerationConfig:
    """Parse ``[generation]``; an absent table is the inert default."""
    g = raw or {}
    providers = tuple(str(p) for p in g.get("providers", GENERATION_PROVIDERS))
    bad = [p for p in providers if p not in GENERATION_PROVIDERS]
    if bad:
        raise ValueError(f"[generation] providers {bad} unknown; expected {GENERATION_PROVIDERS}")
    cfg = GenerationConfig(
        enabled=bool(g.get("enabled", False)),
        providers=providers,
        vcpus=int(g.get("vcpus", 32)),
        child_memory_mb=int(g.get("child_memory_mb", 2048)),
        children=int(g.get("children", 0)),
        max_parallel_jobs=int(g.get("max_parallel_jobs", 2)),
        max_price_per_hour=float(g.get("max_price_per_hour", 1.25)),
        window_seconds=int(g.get("window_seconds", 5400)),
        target_points=int(g.get("target_points", 50_000_000_000)),
        shard_points=int(g.get("shard_points", 16_777_216)),
        max_length=int(g.get("max_length", 32768)),
        storage_prefix=str(g.get("storage_prefix", "corpora/")),
        bucket=str(g.get("bucket", "")),
        vm_image=str(g.get("vm_image", "")),
        retention_days=int(g.get("retention_days", 14)),
        king_publish=str(g.get("king_publish", "recipe")),
        sample_shards=int(g.get("sample_shards", 4)),
    )
    if cfg.king_publish not in KING_CORPUS_PUBLISH:
        raise ValueError(f"[generation] king_publish must be one of {KING_CORPUS_PUBLISH}")
    if min(cfg.vcpus, cfg.child_memory_mb, cfg.max_parallel_jobs, cfg.window_seconds,
           cfg.target_points, cfg.shard_points, cfg.max_length) <= 0:
        raise ValueError("[generation] vcpus, child_memory_mb, max_parallel_jobs, "
                         "window_seconds, target_points, "
                         "shard_points and max_length must be positive")
    if cfg.children < 0 or cfg.retention_days < 0 or cfg.sample_shards < 0:
        raise ValueError("[generation] children, retention_days and sample_shards must be >= 0")
    if not cfg.storage_prefix.endswith("/"):
        raise ValueError("[generation] storage_prefix must end with '/'")
    if cfg.enabled and not providers:
        raise ValueError("[generation] enabled needs at least one provider")
    return cfg
