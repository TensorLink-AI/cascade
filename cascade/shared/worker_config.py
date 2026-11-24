"""The WORKER CONFIG a leg pod loads: the deployed chain.toml cut down to what
the pod's pinned worker release can parse (owner launch blocker 2026-11-24).

Leg pods run ``cascade.trainer.worker``, which calls ``load_chain_config`` on
the chain.toml the trainer / provisioner pushes (``chain.deployed.toml``). A
pinned worker image runs the loader of the commit it was BUILT from, and that
loader validates every section it knows — including sections the worker never
reads. Main's chain.toml therefore kills every leg on an older image at worker
start (rc15 ``5d41d94b`` raises on main's ``[marketcast] suites`` and then on
``[lane_c] targets = ["mkt:market"]``; mkt-rc2 ``0205f761`` raises
``[marketcast] unscorable`` on ``verdict_wait_blocks = 3900``). Repinning the
lane A image is a consensus event (seed-epoch boundary, five pins, measured
walls), so the fix is trainer-side: the pod gets a config its release parses.

A :class:`WorkerConfigProfile` per legacy image names the top-level tables the
pod keeps — an ALLOWLIST derived from that release's worker code (the tables its
loader requires plus every table the worker / ``TrainerRunner`` / the leg code
reads) — and any down-conversion of a key that only the box's verdict code
reads. Every kept table is copied byte-for-byte from the deployed file
(``[training]``, ``[v2]`` / ``[v2.training]`` and therefore ``contract_digest``
/ ``base_arch_digest`` / every pin are untouched); the derivation re-parses its
output and refuses unless the kept tables are exactly the deployed ones.

An image with NO profile gets the deployed file unchanged (exactly the
behaviour before this module): a release built from current code parses its
own chain.toml. A future release that needs a table a legacy profile drops
declares it in its own profile.
"""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass

from .config import image_digest

# Header of a table / array-of-tables line: ``[a.b]`` / ``[[a.b]]`` (+ comment).
_HEADER = re.compile(r"^\s*\[\[?\s*([A-Za-z0-9_.\-]+)\s*\]\]?\s*(#.*)?$")

# The tables a v2-era worker reads (or its loader requires) on a lane A / lane
# B / king / heat leg — audited against worker.py, loop.TrainerRunner,
# runner_rolling's on-pod init build and the loaders at 5e3fade / 6c9cab9:
# required by load_chain_config: generator training eval scoring dependencies
# static_guard manifest validator; read by the leg: training (+ v2.training via
# the [v2] table), round (heat guard, continuation), storage (Hub), generator
# (sandbox), lane_b (leg_hours), wandb / telemetry (leg logging), generation
# (shards corpus), subnet / activation / netconfig / direct_moe (loader cross
# checks that feed the round grid and the v2 contract).
LEG_TABLES: tuple[str, ...] = (
    "subnet", "generator", "training", "round", "eval", "scoring", "dependencies",
    "static_guard", "storage", "manifest", "validator", "wandb", "telemetry",
    "activation", "v2", "lane_b", "generation", "netconfig", "direct_moe",
)


@dataclass(frozen=True)
class WorkerConfigProfile:
    """What one pinned worker release loads. ``keep``: the top-level tables
    copied verbatim (top-level scalar keys such as ``schema_version`` always
    are). ``down_convert``: ``(table, key, fn)`` — ``fn(raw)`` returns the
    value the release must see for a key ONLY the box's verdict code reads
    (``None`` = keep the deployed value)."""

    name: str
    source_commit: str
    keep: tuple[str, ...]
    down_convert: tuple[tuple[str, str, Callable[[dict], object]], ...] = ()


def _rc2_verdict_wait_blocks(raw: dict) -> int | None:
    """``[marketcast] verdict_wait_blocks`` that mkt-rc2's loader accepts.

    mkt-rc2 (6c9cab9) still judges every horizon on FORWARD windows and so
    refuses a wait that holds no aligned window of the longest path
    (``scorability_problems``: wait_s = blocks × 12 − start_margin_seconds ≥ 2 ×
    horizon, and ≥ min_time_blocks × cluster_block_hours). Main (DEC-CA-0074)
    draws every window from the embargo instead, so 3900 blocks is valid there.
    The key is read only by the box's verdict code (validator / lane C
    driver) — never by a MarketCast leg (worker.run_mkt_worker,
    marketcast_leg, marketcast_recipe, pool.marketcast fetch / snapshot) — so
    the pod is handed the smallest wait rc2 accepts. ``None`` when the
    deployed value already passes."""
    from .config.marketcast import BLOCK_SECONDS, duration_minutes

    mc = raw.get("marketcast")
    if not isinstance(mc, dict):
        return None
    suites = list(mc.get("suites", ()) or ())
    hz = mc.get("horizons", {}) or {}
    longest = 0
    for s in suites:
        for h in (hz.get(s, ()) if isinstance(hz, dict) else ()):
            longest = max(longest, 60 * int(duration_minutes(str(h))))
    tb_s = int(mc.get("min_time_blocks", 8)) * int(mc.get("cluster_block_hours", 6)) * 3600
    need_s = max(2 * longest, tb_s) + int(mc.get("start_margin_seconds", 3600))
    need = int(math.ceil(need_s / BLOCK_SECONDS))
    have = int(mc.get("verdict_wait_blocks", 43_200))
    return None if have >= need else need


# image digest -> profile. NEVER remove an entry while a [round] / [v2] /
# [marketcast] pin (or a seed epoch's legs) can still name that image.
LEGACY_WORKER_PROFILES: dict[str, WorkerConfigProfile] = {
    # lane A + lane B + king legs: [round] funded_pod_image, [v2.training]
    # train_image_digest. Never reads [marketcast] / [lane_c] (only the
    # mkt_finetune / finetune roles do, and those never run on this image).
    "sha256:5d41d94bc0c0d3ae3dfd042b0cc7e4d900c7d1f48173808436c12bf764a67641":
        WorkerConfigProfile(name="worker-v2.0.0-rc15", source_commit="5e3fade",
                            keep=LEG_TABLES),
    # MarketCast (lane C mkt:) legs: [marketcast] pod_image. The leg reads
    # [marketcast] (base, suites, frequencies, horizons, assets, sources,
    # knots, recipe limits, event_times, crps_scale, micro_batch_tokens,
    # native banks, continuation_protocol) and [lane_c] leg_hours.
    "sha256:0205f761fde14da71b9a90b4f1ecdc862d657f982accb11450e0ce22c61d2f12":
        WorkerConfigProfile(
            name="worker-v2.1.0-mkt-rc2", source_commit="6c9cab9",
            keep=(*LEG_TABLES, "lane_c", "marketcast"),
            down_convert=(("marketcast", "verdict_wait_blocks", _rc2_verdict_wait_blocks),)),
}


def profile_for_image(image: str) -> WorkerConfigProfile | None:
    """The profile of the release ``image`` (a ref or bare digest), or None."""
    return LEGACY_WORKER_PROFILES.get(image_digest(str(image or "")))


def _top(name: str) -> str:
    return name.split(".", 1)[0].strip().strip('"')


def derive_worker_toml(text: str, profile: WorkerConfigProfile | None) -> str:
    """The deployed chain.toml ``text`` as ``profile``'s release must load it
    (``text`` itself when ``profile`` is None). Raises ``ValueError`` unless
    the result re-parses to exactly the deployed kept tables (+ the declared
    down-conversions)."""
    if profile is None:
        return text
    raw = tomllib.loads(text)
    keep = set(profile.keep)
    out: list[str] = []
    current: str | None = None          # None = before the first header
    for line in text.splitlines(keepends=True):
        m = _HEADER.match(line)
        if m:
            current = _top(m.group(1))
        if current is None or current in keep:
            out.append(line)
    derived = "".join(out)
    converted: dict[tuple[str, str], object] = {}
    for table, key, fn in profile.down_convert:
        if table not in keep:
            raise ValueError(f"{profile.name}: down-converts [{table}] which it drops")
        value = fn(raw)
        if value is not None:
            from ..trainer.leg_pins import _set_keys

            derived = _set_keys(derived, table, {key: value})
            converted[(table, key)] = value
    got = tomllib.loads(derived)
    want = {k: v for k, v in raw.items() if not isinstance(v, dict) or k in keep}
    for (table, key), value in converted.items():
        want[table] = {**want[table], key: value}
    if got != want:
        diff = sorted(set(got) ^ set(want)) or sorted(
            k for k in want if got.get(k) != want[k])
        raise ValueError(f"worker config for {profile.name} does not re-parse to the "
                         f"deployed kept tables (differs at {diff})")
    return derived


def worker_toml_for_image(text: str, image: str) -> tuple[str, str]:
    """``(toml text, profile name or "")`` for a pod booting ``image``."""
    prof = profile_for_image(image)
    return derive_worker_toml(text, prof), (prof.name if prof is not None else "")


__all__ = ["LEGACY_WORKER_PROFILES", "LEG_TABLES", "WorkerConfigProfile",
           "derive_worker_toml", "profile_for_image", "worker_toml_for_image"]
