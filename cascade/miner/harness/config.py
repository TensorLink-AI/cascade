"""``harness.toml`` — the gauntlet's configuration (one file, every knob).

Everything has a default except the ones that spend money: a Lium executor
refuses to start without ``[compute] daily_usd_cap``, and autonomous submission
refuses to start without an explicit hotkey pool and intake.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

SUBMIT_MODES = ("off", "approval", "autonomous")
EXECUTORS = ("local", "lium")
WORKER_MODES = ("inline", "queue")


@dataclass(frozen=True)
class RoundsConfig:
    snapshot_root: Path = Path("eval-pool")       # local copy of the revealed pool
    hf_repo: str = "Tensor-Link/cascade-eval-pool"
    sync_reveals: bool = True                     # download missing snapshot folders
    # "local": this machine downloads from Hugging Face. "executor": a job on
    # the executor (a Lium pod) downloads and the folders are copied back, for
    # hosts that cannot reach huggingface.co.
    sync_via: str = "local"
    validator: str = ""                           # receipts namespace ("" = any)
    n_a: int = 6                                  # pool A: search rounds
    n_b: int = 2                                  # pool B: the newest replayable rounds
    receipt_scan: int = 60                        # newest index rows considered


@dataclass(frozen=True)
class StagesConfig:
    g1_throughput_tol: float = 0.10       # may be this much slower than the king
    g1_seconds: float = 60.0              # generation sampled per throughput probe
    # What G2/G3 candidates are compared against:
    #   "cached" (default): the king trained at the stage's own budget on the same
    #     round and seeds, ONCE per (king, round, budget) and reused across epochs
    #     (trained lazily, in the same batch as the first candidate needing it);
    #   "receipt": the king's signed full-budget scores, no king training at all,
    #     but a short leg differs from the receipt by ~±0.5% per round (measured),
    #     which is larger than the screen margin;
    #   "trained": legacy: every window round re-trained each epoch, plus σ legs.
    reference: str = "cached"
    g2_hours: float = 0.25                # short screen budget
    g2_margin_floor: float = 0.002        # relative; the calibrated margin never goes below
    g2_noise_z: float = 1.0               # screen margin = max(floor, z × σ)
    g2_explore_frac: float = 0.1          # G2 losers promoted anyway (calibration)
    g3_hours: float = 1.0
    g3_noise_z: float = 2.0               # confirm margin = max(floor, z × σ / √n_b)
    calib_salts: int = 2                  # extra king seeds measured per epoch for σ
    g4_every_cycles: int = 3              # run G4 every N cycles when finalists exist
    g4_finalists: int = 2                 # population members sent to G4
    g4_rounds: int = 3                    # newest replayable rounds replayed at full budget
    g4_min_wins: int = 2                  # rounds that must clear the round's own rule
    g45_enabled: bool = True
    g45_sources: str = ""                 # `cascade-pool build --sources` ("" = defaults)


@dataclass(frozen=True)
class SearchConfig:
    population_k: int = 4
    proposals_per_cycle: int = 4
    # The improvement over the king a submission needs (relative). Sets the
    # dethrone progress score's 100 (gauntlet.progress); ~ the live margin.
    target_improvement: float = 0.01
    max_cycles: int = 0                   # 0 = run until STOP


@dataclass(frozen=True)
class ComputeConfig:
    executor: str = "local"
    device: str = "auto"                  # local executor torch device
    max_parallel: int = 1                 # concurrent jobs (local: GPUs; lium: pods)
    daily_usd_cap: float = 0.0            # REQUIRED for lium (0 refuses to start)
    total_usd_cap: float = 0.0            # lifetime ceiling across days (0 = none)
    sku: str = "L40S"
    # Any of these, cheapest first (DEC-CA-0036 open-market launch). All are
    # Ampere/Ada/Hopper: the worker image's pinned torch (cu124) has no kernels
    # for Blackwell (RTX 5090, RTX PRO 6000, B200), which the pod health check
    # rejects anyway. Under points+mv billing the budget, not the GPU, sets the
    # token count, so mixing these keeps legs comparable.
    sku_choices: tuple[str, ...] = ("L40S", "RTX6000", "RTX4090", "A100", "H100")
    max_price_per_hour: float = 0.0
    image: str = ""                       # worker image, digest-pinned (default: chain.toml)
    ssh_key: Path = Path("~/.ssh/cascade_gauntlet")
    pod_prefix: str = "cascade-gauntlet"
    idle_minutes: float = 20.0            # tear a pod down after this long unused
    boot_timeout: float = 1200.0
    job_timeout: float = 8 * 3600.0


@dataclass(frozen=True)
class WorkersConfig:
    mode: str = "inline"                  # inline subprocess, or a file queue (separate container)
    queue_dir: Path = Path("queue")       # relative to workdir (shared volume in compose)
    llm_provider: str = "anthropic"       # anthropic | chutes | saygm | custom
    llm_model: str = ""
    llm_base_url: str = ""
    llm_key_env: str = ""
    llm_auth: str = ""
    agent_max_turns: int = 120
    agent_timeout: float = 1800.0


@dataclass(frozen=True)
class SubmitConfig:
    mode: str = "approval"                # off | approval | autonomous
    intake: str = ""
    wallet_name: str = ""
    hotkeys: tuple[str, ...] = ()         # pool of UNUSED hotkeys (one submission each)
    margin: float = 0.01                  # G4.5 relative improvement required
    max_per_day: int = 1
    notify_url: str = ""                  # optional webhook (JSON POST) on every G5 event
    label: str = "gauntlet"


@dataclass(frozen=True)
class HarnessConfig:
    workdir: Path = Path("gauntlet")
    start_dir: Path = Path("champions/king")
    king_dir: Path = Path("champions/king")
    # "dir": king_dir as given. "live": the king named by the anchor validator's
    # latest signed receipt, re-read at every window refresh (king.py);
    # king_dir is then only the fallback until the first fetch succeeds.
    king_source: str = "dir"
    chain_toml: Path | None = None
    rounds: RoundsConfig = field(default_factory=RoundsConfig)
    stages: StagesConfig = field(default_factory=StagesConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    workers: WorkersConfig = field(default_factory=WorkersConfig)
    submit: SubmitConfig = field(default_factory=SubmitConfig)

    def validate(self) -> None:
        """Refuse configurations that would spend without a bound."""
        c, s = self.compute, self.submit
        if self.king_source not in ("dir", "live"):
            raise ValueError("king_source must be 'dir' or 'live'")
        if c.executor not in EXECUTORS:
            raise ValueError(f"[compute] executor must be one of {EXECUTORS}")
        if c.executor == "lium" and c.daily_usd_cap <= 0:
            raise ValueError("[compute] daily_usd_cap is required (> 0) for the lium executor")
        if c.executor == "lium" and c.max_price_per_hour <= 0:
            raise ValueError("[compute] max_price_per_hour is required (> 0) for the lium "
                             "executor: it filters rentals AND bounds the spend ledger")
        if c.max_parallel < 1:
            raise ValueError("[compute] max_parallel must be >= 1")
        if self.workers.mode not in WORKER_MODES:
            raise ValueError(f"[workers] mode must be one of {WORKER_MODES}")
        if s.mode not in SUBMIT_MODES:
            raise ValueError(f"[submit] mode must be one of {SUBMIT_MODES}")
        if s.mode == "autonomous" and not (s.intake and s.wallet_name and s.hotkeys):
            raise ValueError("[submit] autonomous needs intake, wallet_name and a hotkeys pool")
        if s.mode == "autonomous" and s.margin < 0.005:
            raise ValueError("[submit] autonomous margin must be >= 0.005 (the duel's floor)")
        if self.stages.reference not in ("cached", "receipt", "trained"):
            raise ValueError("[stages] reference must be 'cached', 'receipt' or 'trained'")
        if self.rounds.sync_via not in ("local", "executor"):
            raise ValueError("[rounds] sync_via must be 'local' or 'executor'")
        if self.rounds.n_a < 1 or self.rounds.n_b < 1:
            raise ValueError("[rounds] n_a and n_b must be >= 1")
        if self.stages.g4_min_wins > self.stages.g4_rounds:
            raise ValueError("[stages] g4_min_wins cannot exceed g4_rounds")


def _section(cls, doc: dict, base: Path):
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(doc) - set(known))
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {unknown}")
    kw = {}
    for k, v in doc.items():
        default = getattr(cls(), k)
        if isinstance(default, Path) or k in ("snapshot_root", "ssh_key", "queue_dir"):
            p = Path(str(v)).expanduser()
            kw[k] = p if p.is_absolute() or k == "queue_dir" else base / p
        elif isinstance(default, tuple):
            kw[k] = tuple(v)
        else:
            kw[k] = v
    return cls(**kw)


def load_harness_config(path: Path | str) -> HarnessConfig:
    """Load ``harness.toml``; relative paths resolve against the file's directory."""
    path = Path(path)
    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    base = path.resolve().parent
    sections = {"rounds": RoundsConfig, "stages": StagesConfig, "search": SearchConfig,
                "compute": ComputeConfig, "workers": WorkersConfig, "submit": SubmitConfig}
    kw = {}
    for name, cls in sections.items():
        kw[name] = _section(cls, doc.pop(name, {}), base)
    for k in ("workdir", "start_dir", "king_dir", "chain_toml"):
        if k in doc:
            p = Path(str(doc.pop(k))).expanduser()
            kw[k] = p if p.is_absolute() else base / p
    if "king_source" in doc:
        kw["king_source"] = str(doc.pop("king_source"))
    if doc:
        raise ValueError(f"unknown harness.toml keys: {sorted(doc)}")
    cfg = HarnessConfig(**kw)
    cfg.validate()
    return cfg


def with_overrides(cfg: HarnessConfig, **sections) -> HarnessConfig:
    """``cfg`` with per-section field overrides, e.g. ``compute={"max_parallel": 2}``."""
    out = cfg
    for name, kv in sections.items():
        out = replace(out, **{name: replace(getattr(out, name), **kv)})
    return out
