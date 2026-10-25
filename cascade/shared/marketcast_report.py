"""MarketCast per-window evidence (DEC-CA-0064), signed by the main validator.

The main validator forecasts the pinned base and each fine-tune on forward-only
snapshot windows. Other validators replay its signed rows and verdict inputs.

Availability comes from a later trainer-signed manifest's mkt_reports digest
or "absent" acknowledgement, so missing reports do not hold other lanes.
Every eligible window is represented, including non-finite losses as null.
Content-addressed float32 quantile sidecars let the audit recompute losses;
infra_deferred entries remain pending for retry.

Gain uses class/horizon-balanced ratios of pooled mean losses with clustered
bootstrap and recorded zero-scale exclusions. Calibration checks returns and
volatility, including tail coverage. Near-duplicate checks bind the senior
reference forecasts by digest. JSON floats round-trip for deterministic replay.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from .config.marketcast import COVERAGE_GATE_SUITES, duration_minutes

MKT_REPORT_VERSION = 3
# the entry status of a row set the main validator could not compute
INFRA_DEFERRED = "infra_deferred"
# a lane-C manifest's ``mkt_reports`` value when no report appeared in time
REPORT_ABSENT = "absent"


def mkt_report_key(round_id: str) -> str:
    return f"marketcast/round-{round_id}.json"


def mkt_quantiles_key(round_id: str, hotkey: str, pointer: str, phase: str) -> str:
    """The sidecar of one entry's per-window quantiles (``.npy``)."""
    tag = hashlib.sha256(f"{hotkey}|{pointer}|{phase}".encode()).hexdigest()[:24]
    return f"marketcast/round-{round_id}/q-{tag}.npy"


def mkt_ref_quantiles_key(round_id: str, hotkey: str, pointer: str, phase: str) -> str:
    """Red-team F4: the sidecar of the closest SENIOR entry's quantiles on
    one entry's windows (``.npy``, the fine-tune plane only)."""
    tag = hashlib.sha256(f"{hotkey}|{pointer}|{phase}|ref".encode()).hexdigest()[:24]
    return f"marketcast/round-{round_id}/r-{tag}.npy"


def mkt_text_digest(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MktWindowRow:
    wid: str                          # "<class>:<asset>|<suite>|<horizon>|<start>"
    base: float                       # per-window loss of the untouched base (NaN = none)
    ft: float                         # per-window loss of the fine-tune (NaN = none)
    cov_in: tuple[int, ...] = ()      # fine-tune: steps inside each coverage interval
    cov_n: int = 0                    # finite target steps
    # review M13: the SECOND, non-gating baseline — the base given the
    # fine-tune's covariates (NaN = not computed: the fine-tune takes none)
    bc: float = float("nan")
    # red-team F1b: the fine-tune's tail tallies — per ``tail_levels`` level
    # the finite steps beyond it (below q_l for l < 0.5, above for l > 0.5)
    tail: tuple[int, ...] = ()

    @property
    def cluster(self) -> str:
        """The asset (``<class>:<asset>``)."""
        return self.wid.split("|", 1)[0]

    @property
    def asset_class(self) -> str:
        return self.wid.split(":", 1)[0]

    def time_block(self, hours: int) -> int:
        """The UTC time block of ``hours`` the window STARTS in."""
        return self.start // (3600 * int(hours))

    def time_cluster(self, hours: int) -> str:
        """The bootstrap cluster (review M10): asset class × time block —
        windows of one class in one block share a regime, so they are never
        independent evidence; different classes in one block are."""
        return f"{self.asset_class}|{self.time_block(hours)}"

    @property
    def start(self) -> int:
        return int(self.wid.rsplit("|", 1)[1])

    @property
    def horizon(self) -> str:
        return self.wid.split("|")[2]

    @property
    def end(self) -> int:
        return self.start + 60 * duration_minutes(self.horizon)


@dataclass(frozen=True)
class MktEntryRows:
    hotkey: str
    trained_pointer: str
    target: str
    phase: str                        # "judge" | "confirm"
    after: float                      # windows start strictly after this unix time
    fault: str = ""                   # deterministic failure on the main validator
    windows: tuple[MktWindowRow, ...] = ()
    status: str = ""                  # "" = computed | INFRA_DEFERRED
    infra_reason: str = ""
    quantiles_key: str = ""           # the per-window quantile sidecar (computed rows)
    quantiles_sha256: str = ""
    # red-team F4: {"hotkey", "pointer", "commit_block", "mean_rel", "max_rel",
    # "n_windows", "quantiles_key", "quantiles_sha256"} of the closest senior
    # entry measured on these windows ({} = none measured)
    near_dup: dict = field(default_factory=dict)

    @property
    def deferred(self) -> bool:
        return self.status == INFRA_DEFERRED


@dataclass(frozen=True)
class MktReport:
    round_id: str
    created_block: int
    snapshot: dict                    # key, sha256, digest, t0, end, anchor_*, block_seconds
    entries: tuple[MktEntryRows, ...]
    coverage_levels: tuple[float, ...] = (0.5, 0.8, 0.98)
    tail_levels: tuple[float, ...] = (0.05, 0.95)
    version: int = MKT_REPORT_VERSION
    signature: str = ""

    def canonical_body(self) -> bytes:
        d = asdict(self)
        d.pop("signature")
        for e in d["entries"]:
            for k in ("status", "infra_reason", "quantiles_key", "quantiles_sha256", "near_dup"):
                if not e.get(k):
                    e.pop(k, None)
            for w in e["windows"]:
                for k in ("base", "ft"):
                    if not math.isfinite(float(w[k])):
                        w[k] = None
                if not math.isfinite(float(w["bc"])):
                    w.pop("bc")
        return json.dumps(d, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")

    def to_json(self) -> str:
        d = json.loads(self.canonical_body())
        d["signature"] = self.signature
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    def entry_for(self, hotkey: str, pointer: str, phase: str) -> MktEntryRows | None:
        for e in self.entries:
            if (e.hotkey, e.trained_pointer, e.phase) == (hotkey, pointer, phase):
                return e
        return None


def _loss(v) -> float:
    return float("nan") if v is None else float(v)


def load_mkt_report(text: str) -> MktReport:
    o = json.loads(text)
    if not isinstance(o, dict) or int(o.get("version", 0)) != MKT_REPORT_VERSION:
        raise ValueError("unsupported MarketCast report")
    ents = tuple(MktEntryRows(
        hotkey=str(e["hotkey"]), trained_pointer=str(e["trained_pointer"]),
        target=str(e["target"]), phase=str(e["phase"]), after=float(e["after"]),
        fault=str(e.get("fault", "") or ""),
        windows=tuple(MktWindowRow(str(w["wid"]), _loss(w["base"]), _loss(w["ft"]),
                                   tuple(int(x) for x in w.get("cov_in", ())),
                                   int(w.get("cov_n", 0)), _loss(w.get("bc")),
                                   tuple(int(x) for x in w.get("tail", ())))
                      for w in e.get("windows", ())),
        status=str(e.get("status", "") or ""), infra_reason=str(e.get("infra_reason", "") or ""),
        quantiles_key=str(e.get("quantiles_key", "") or ""),
        quantiles_sha256=str(e.get("quantiles_sha256", "") or ""),
        near_dup=dict(e.get("near_dup") or {}))
        for e in o["entries"])
    return MktReport(round_id=str(o["round_id"]), created_block=int(o["created_block"]),
                     snapshot=dict(o.get("snapshot") or {}), entries=ents,
                     coverage_levels=tuple(float(c) for c in o.get("coverage_levels", ())),
                     tail_levels=tuple(float(c) for c in o.get("tail_levels", ())),
                     version=int(o["version"]), signature=str(o.get("signature", "") or ""))


# ── the per-window quantile sidecar + the ONE loss rule (validator + audit) ──


def published_quantiles(q) -> np.ndarray:
    """Quantiles as published: rounded to float32 (the sidecar's dtype) and
    back, so a loss computed from them is exactly what the audit recomputes
    from the sidecar."""
    return np.asarray(q, dtype=np.float32).astype(np.float64)


def encode_quantiles(base_q, ft_q) -> bytes:
    """The sidecar bytes of one entry: per window (in row order) its
    ``(steps, K)`` base and fine-tune quantiles, stacked into ONE float32
    array ``(Σ steps, 2, K)`` and written with ``np.save`` (deterministic)."""
    blocks = [np.stack([np.asarray(b, dtype=np.float32), np.asarray(f, dtype=np.float32)],
                       axis=1) for b, f in zip(base_q, ft_q, strict=True)]
    arr = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 2, 0), np.float32)
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr, dtype=np.float32), allow_pickle=False)
    return buf.getvalue()


def decode_quantiles(data: bytes, steps, n_knots: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """``[(base (steps_i, K), ft (steps_i, K)), ...]`` from sidecar bytes and
    the windows' step counts (in row order). Raises ``ValueError`` on any
    shape mismatch."""
    arr = np.load(io.BytesIO(bytes(data)), allow_pickle=False)
    total = int(sum(int(s) for s in steps))
    if arr.dtype != np.float32 or arr.ndim != 3 or arr.shape != (total, 2, int(n_knots)):
        raise ValueError(f"quantile sidecar is {arr.dtype}{arr.shape}, expected float32"
                         f"({total}, 2, {n_knots})")
    out, i = [], 0
    for s_ in steps:
        blk = arr[i:i + int(s_)].astype(np.float64)
        out.append((blk[:, 0, :], blk[:, 1, :]))
        i += int(s_)
    return out


def score_windows(targets, base_q, ft_q, *, knots, coverage_levels, gated: bool,
                  tail_levels=()) -> list[tuple[float, float, tuple[int, ...], int,
                                                tuple[int, ...]]]:
    """``[(base loss, ft loss, cov_in, cov_n, tail), ...]`` per window (row
    order) from the PUBLISHED quantiles: CRPS per window (:func:`cascade.
    eval.marketcast.window_crps`) and, on a calibration-gated suite, the
    fine-tune's interval tallies and tail tallies (``tail_levels``). A window
    with a non-finite loss carries no tallies (it never enters the verdict).
    Windows are scored grouped by their step count in row order — the main
    validator and the audit call exactly this."""
    from ..eval.marketcast import window_coverage_counts, window_crps, window_tail_counts

    n = len(targets)
    out: list = [None] * n
    groups: dict[int, list[int]] = {}
    for i, y in enumerate(targets):
        groups.setdefault(int(np.asarray(y).size), []).append(i)
    for _steps, idx in sorted(groups.items()):
        y = np.stack([np.asarray(targets[i], dtype=np.float64) for i in idx])
        qb = np.stack([published_quantiles(base_q[i]) for i in idx])
        qf = np.stack([published_quantiles(ft_q[i]) for i in idx])
        lb, lf = window_crps(qb, y, knots), window_crps(qf, y, knots)
        for j, i in enumerate(idx):
            b, f = float(lb[j]), float(lf[j])
            ci, cn, tl = (), 0, ()
            if gated and math.isfinite(b) and math.isfinite(f):
                ins, cn = window_coverage_counts(qf[j], y[j], knots, coverage_levels)
                ci = tuple(int(x) for x in ins)
                tl = tuple(int(x) for x in window_tail_counts(qf[j], y[j], knots,
                                                                tuple(tail_levels)))
            out[i] = (b, f, ci, int(cn), tl)
    return out


def sign_mkt_report(report: MktReport, wallet) -> MktReport:
    hotkey = getattr(wallet, "hotkey", wallet)
    sig = hotkey.sign(report.canonical_body())
    return replace(report, signature=sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig))


def verify_mkt_report(report: MktReport, signer_hotkey: str) -> bool:
    if not report.signature or not signer_hotkey:
        return False
    from .sr25519 import verify_hotkey

    return verify_hotkey(signer_hotkey, report.canonical_body(), report.signature)


def window_weights(rows) -> list[float]:
    """Review M13: the in-horizon weight of each row — ``1 / n(class,
    horizon)`` over the rows with a finite base loss, so every asset CLASS
    weighs the same inside a horizon (28 crypto perps never outvote 4
    metals); rows without a finite base loss weigh 0."""
    n: dict = {}
    ok = [math.isfinite(float(w.base)) for w in rows]
    for w, good in zip(rows, ok, strict=True):
        if good:
            k = (w.asset_class, w.horizon)
            n[k] = n.get(k, 0) + 1
    return [1.0 / n[(w.asset_class, w.horizon)] if good else 0.0
            for w, good in zip(rows, ok, strict=True)]


def _class_rates(ws, pick, expected: dict) -> tuple[dict, dict]:
    """Class-weighted rates over one horizon's rows: per level the MEAN over
    asset classes of the class's ratio of summed tallies (``pick(w)`` → the
    row's per-level integers) to its summed finite steps — exactly the
    gain's weighting (:func:`window_weights`) — and the binomial standard
    error at the EXPECTED rate, ``sqrt(Σ_c p(1 − p) / n_c) / C`` with ``n_c``
    the class's finite steps (every scored horizon but ``4h`` is one step
    per window). NaN rate when no class has a finite step."""
    by: dict = {}
    for w in ws:
        if int(w.cov_n) <= 0:
            continue
        k, n, m = by.setdefault(w.asset_class, ([0] * len(expected), 0, 0))
        got = pick(w)
        k = [a + (int(got[j]) if len(got) > j else 0) for j, a in enumerate(k)]
        by[w.asset_class] = (k, n + int(w.cov_n), m + 1)
    rates, se = {}, {}
    C = len(by)
    for j, (lv, p) in enumerate(expected.items()):
        if not C:
            rates[lv], se[lv] = float("nan"), 0.0
            continue
        rates[lv] = float(sum(k[j] / n for k, n, _m in by.values()) / C)
        se[lv] = float(math.sqrt(sum(p * (1.0 - p) / n for _k, n, _m in by.values())) / C)
    return rates, se


def calibration_gate(rows, mc, levels, tail_levels=None):
    """The fine-tune's calibration gate (review M14, red-team F1b / F2):

    * per HORIZON, central coverage at ``coverage_gate_levels`` two-sided
      (``coverage_tolerance`` below, ``coverage_tolerance_over`` above), the
      other levels (98 %) report-only;
    * per horizon AND pooled over horizons (equal horizon weight, the gain's),
      the TAIL hit rate beyond each ``tail_levels`` level within
      ``tail_tolerance`` of ``min(l, 1 − l)`` (collapsed 1 / 5 / 95 / 99 %
      knots fail here; the pooled row has the power a thin horizon lacks);
    * every rate is class-weighted like the gain (:func:`_class_rates`), and
      every tolerance widened to ``calibration_z ×`` its standard error.

    Tallies re-aggregated exactly from the signed integers."""
    from ..eval.marketcast import combine_gates, coverage_gate

    z = float(mc.calibration_z)
    tls = tuple(mc.tail_levels if tail_levels is None else tail_levels)
    hz = sorted({w.horizon for w in rows}, key=lambda h: (duration_minutes(h), h))
    gates = []
    pooled: dict = {lv: [] for lv in tls}
    for h in hz:
        ws = [w for w in rows if w.horizon == h]
        emp, se = _class_rates(ws, lambda w: w.cov_in, {float(c): float(c) for c in levels})
        gates.append(coverage_gate(emp, tolerance=float(mc.coverage_tolerance),
                                   tolerance_over=float(mc.coverage_tolerance_over),
                                   horizon=h, gate_levels=tuple(mc.coverage_gate_levels),
                                   slack={c: z * x for c, x in se.items()}))
        if tls:
            want = {float(lv): min(float(lv), 1.0 - float(lv)) for lv in tls}
            rate, tse = _class_rates(ws, lambda w: w.tail, want)
            gates.append(coverage_gate(rate, tolerance=float(mc.tail_tolerance),
                                       tolerance_over=float(mc.tail_tolerance), horizon=h,
                                       kind="tail", expected=want,
                                       slack={c: z * x for c, x in tse.items()}))
            for lv in want:
                pooled[lv].append((rate[lv], tse[lv]))
    if tls and hz:
        want = {float(lv): min(float(lv), 1.0 - float(lv)) for lv in tls}
        rate = {lv: float(np.mean([r for r, _s in pooled[lv]])) for lv in want}
        sl = {lv: z * float(math.sqrt(sum(s_ * s_ for _r, s_ in pooled[lv]))) / len(pooled[lv])
              for lv in want}
        gates.append(coverage_gate(rate, tolerance=float(mc.tail_tolerance),
                                   tolerance_over=float(mc.tail_tolerance), horizon="",
                                   kind="tail", expected=want, slack=sl))
    return combine_gates(gates)


def verdict_from_rows(rows: MktEntryRows, mc, *, suite: str, k: int, seed: str,
                      coverage_levels=None, tail_levels=None):
    """The suite verdict (:func:`cascade.eval.marketcast.judge_suite`) from
    one entry's signed rows: paired base / fine-tune losses over the rows
    whose BASE loss is finite (a non-finite base forecast excludes the
    window for everyone; a non-finite fine-tune one is a fault upstream);
    clusters = asset class × ``cluster_block_hours`` time block, at least
    ``min_time_blocks`` distinct blocks; the gain a geomean over (class ×
    horizon) of MEAN-loss ratios (red-team F1), horizons weighed equally and
    asset classes equally inside a horizon;
    level ``alpha / k`` (``k`` = 1 on a confirm); for a calibration-gated
    suite :func:`calibration_gate`."""
    from ..eval.marketcast import judge_suite

    levels = tuple(coverage_levels if coverage_levels is not None else mc.coverage_levels)
    ws = rows.windows
    cov = (calibration_gate(ws, mc, levels, tail_levels)
           if suite in COVERAGE_GATE_SUITES else None)
    hrs = int(mc.cluster_block_hours)
    return judge_suite(
        suite, [w.base for w in ws], [w.ft for w in ws],
        alpha=float(mc.alpha) / max(1, int(k)), B=int(mc.bootstrap_B),
        gain_margin=float(mc.gain_margin), seed=seed,
        clusters=[w.time_cluster(hrs) for w in ws], min_windows=int(mc.min_windows),
        min_clusters=int(mc.min_clusters), coverage=cov,
        horizons=[w.horizon for w in ws], weights=window_weights(ws),
        time_blocks=[w.time_block(hrs) for w in ws], min_time_blocks=int(mc.min_time_blocks),
        zero_scale_frac=float(mc.zero_scale_frac), ratio_cap=float(mc.cell_ratio_cap))


def behaviour_distance(base_q, ft_q, ref_q) -> tuple[float, float]:
    """Red-team F4: ``(mean, max)`` over every step × knot of
    ``|q_ft − q_ref| / w`` with ``w`` the BASE's width between its second
    and second-to-last knots (5–95 % on Timestral's grid) at that step (the
    base is fixed, so the scale is the same for every candidate; a step
    whose base width is not > 0 is skipped). Inputs are per-window lists of
    ``(steps, K)`` arrays in row order (the published float32 values).
    ``(nan, nan)`` when nothing is comparable."""
    num, mx, cnt = 0.0, 0.0, 0
    for b, f, r in zip(base_q, ft_q, ref_q, strict=True):
        b, f, r = (np.asarray(x, dtype=np.float64) for x in (b, f, r))
        k = b.shape[-1]
        lo, hi = (1, k - 2) if k >= 5 else (0, k - 1)
        wid = b[..., hi] - b[..., lo]
        ok = np.isfinite(wid) & (wid > 0)
        if not ok.any():
            continue
        d = np.abs(f[ok] - r[ok]) / wid[ok][:, None]
        num += float(d.sum())
        cnt += int(d.size)
        mx = max(mx, float(d.max()))
    if not cnt:
        return float("nan"), float("nan")
    return num / cnt, mx


def base_cov_gain(rows) -> float | None:
    """Review M13 (reported, never gating): the point gain over the base
    GIVEN THE FINE-TUNE'S COVARIATES — ``1 − exp(mean ln ft − mean ln bc)``
    over the rows where both are finite; None when not computed."""
    pairs = [(w.ft, w.bc) for w in rows.windows
             if math.isfinite(float(w.bc)) and math.isfinite(float(w.ft))]
    if not pairs:
        return None
    lf = np.log(np.maximum([p[0] for p in pairs], 1e-12))
    lb = np.log(np.maximum([p[1] for p in pairs], 1e-12))
    return float(1.0 - np.exp(lf.mean() - lb.mean()))


@dataclass
class MktReportState:
    """One settlement's report as a validator handles it (in memory)."""
    round_id: str
    key: str
    report: MktReport | None = None
    text: str = ""
    sha256: str = ""
    verified: bool = False
    computed: dict = field(default_factory=dict)   # (hotkey, pointer, phase) -> MktEntryRows


__all__ = [
    "INFRA_DEFERRED", "MKT_REPORT_VERSION", "MktEntryRows", "MktReport", "MktReportState",
    "MktWindowRow", "REPORT_ABSENT", "base_cov_gain", "behaviour_distance", "calibration_gate",
    "decode_quantiles", "encode_quantiles", "load_mkt_report", "mkt_ref_quantiles_key",
    "mkt_quantiles_key", "mkt_report_key", "mkt_text_digest", "published_quantiles",
    "score_windows", "sign_mkt_report", "verdict_from_rows", "verify_mkt_report",
    "window_weights",
]
