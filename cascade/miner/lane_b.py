"""``cascade lane-b`` — fund and track a domain-expert leg (DEC-CA-0059 item 8).

A lane-B entry is committed with ``cascade deploy --expert-domain <targets>``
(``metro-v1:expert:<domain|codes>:hippius:<ref>``). Every lane is miner-
funded: once the commitment reveals, fund ONE leg with your Lium key and the
trainer normally rents it on YOUR key. A capacity miss waits unburned;
operator-caused make-good work can use configured operator capacity without
charging the miner for a second rental.

* ``fund <intake> --targets T[,T2[,T3]] --ref R --yes`` — ``POST
  /v1/lane-b/fund``, signed by your hotkey over the key digest, the targets
  (exactly as listed) and the ref. ``--target T`` (repeatable, rev6.22)
  names the targets one per flag instead of ``--targets`` — e.g. a suite
  target (WB18, while ``[lane_b] suite_targets`` is on) with domains.
  ``--declared-rows <file>`` sends your expert rows (WB14) as
  ``X-Lane-B-Declared-Rows``, checked locally with the intake's own parser
  and signed (``|rows=<sha256>``). ``--withdraw`` takes a still-queued entry
  back out. ``--gpu`` / ``--allow-fallback`` / ``--max-price`` as ``cascade
  fund`` (DEC-CA-0060). ``--train-hours`` is refused locally until the
  intake accepts a training-hours request (WB23).
* ``status <intake> [--hotkey H]`` — the public lane-B queue
  (``GET /v1/lane-b/queue``), optionally just your entry, with the lane B
  deferral and gate reasons of the public ``status/lane_b.json``.
* ``verify-repo <dir>`` — the intake's WB13 contamination scan, run locally
  against the pinned ``[lane_b] contamination_hashset``: the tree is read,
  never executed, and nothing is uploaded.

The targets are validated locally against ``chain.toml [lane_b]`` (the rule
``cascade deploy --expert-domain`` uses) before anything is sent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .output import NETWORK, OK, REJECTED, USAGE, reporter

# The intake's reply codes for a refused WB13 scan (cascade/funding/intake.py).
SCAN_REFUSAL_CODES = {"embedded": "benchmark_series_embedded",
                      "budget_exceeded": "contamination_budget_exceeded"}
# The repo root: the released benchmarks/contamination/ artefacts live here.
_RELEASE_ROOT = Path(__file__).resolve().parents[2]


LANE_B_EPILOG = """\
examples (commit first with `cascade deploy --expert-domain energy,h720`):
  cascade lane-b verify-repo ./my-generator
  cascade lane-b test ./my-generator --targets energy,h720 --base live --pool-dir ./pool
  cascade lane-b fund https://submissions.cascadesub.net --targets energy,h720 --ref REPO@DIGEST --declared-rows rows.json --gpu 8xH100 --wallet-name W --wallet-hotkey H --yes
  cascade lane-b status https://submissions.cascadesub.net --hotkey SS58
rules: `cascade rules b` prints the live [lane_b] values (targets, gain bar, guards,
GPU list, leg hours, pay tiers); guide: docs/MINER_LANE_B.md."""


def add_lane_b(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("lane-b", help="Lane B (domain experts): fund and track a leg.",
                       epilog=LANE_B_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    lsub = p.add_subparsers(dest="lane_b_cmd", required=True)
    from .lane_tests import add_lane_test
    add_lane_test(lsub, "b")

    f = lsub.add_parser("fund", help="Fund one domain-expert leg with YOUR provider key (paid).")
    f.add_argument("intake_url", help="Operator intake base URL (https://…).")
    which = f.add_mutually_exclusive_group(required=True)
    which.add_argument("--targets",
                       help="The committed targets, comma-separated (as for `cascade deploy "
                            "--expert-domain`, e.g. energy or energy,h720).")
    which.add_argument("--target", metavar="T", action="append",
                       help="One committed target; repeat it to list several (e.g. "
                            "--target suite:gift --target energy), up to [lane_b] "
                            "max_domains_per_leg. suite:<s> needs [lane_b] suite_targets on.")
    f.add_argument("--ref", required=True, help="The committed generator ref (repo@digest).")
    f.add_argument("--declared-rows", type=Path, default=None, metavar="FILE",
                   help="JSON file of the expert rows you train, keyed by MoE ordinal "
                        '(e.g. {"0": [3, 17]}); checked locally with the intake\'s parser '
                        "and signed.")
    f.add_argument("--train-hours", type=str, default=None, metavar="H",
                   help="Hour limit of the leg on your chosen GPU (0.5 h steps, up to "
                        "[lane_b] max_leg_hours = 12). Needs --gpu.")
    f.add_argument("--train-tokens", type=str, default=None, metavar="N",
                   help="Token limit of the leg, e.g. 20e9. "
                        "The leg stops at the token limit OR the hour limit, whichever comes "
                        "first; tokens only = the max_leg_hours hour limit. Needs --gpu. A GPU "
                        "with no measured rate is AT YOUR OWN RISK: no refund if the leg OOMs, "
                        "cannot boot or crashes.")
    # WB27 (plan rev6.21): the miner-chosen leg levers (X-Lane-B-Levers)
    f.add_argument("--lr-scale", type=str, default=None, metavar="X",
                   help="Multiply the leg's base LR by X ([lane_b] lr_scale_min..max, "
                        "shipped 0.5..2.0; default 1.0).")
    f.add_argument("--curriculum", type=str, default=None, metavar="NAME:FRAC,...",
                   help="Up to 3 ordered stages, e.g. easy:0.3,hard:0.7 (fractions sum to 1, "
                        "at most 2 decimals). Your Generator.__init__ must take the "
                        "curriculum_stage keyword (check: verify-repo --curriculum).")
    f.add_argument("--context-length", type=int, default=None, metavar="N",
                   help="Training context of the leg (4096..[lane_b] max_context_length = "
                        "32768, powers of two); must fit the chosen GPU's measured memory.")
    f.add_argument("--ema", type=float, default=None, metavar="SIGMA",
                   help="Scored power-EMA width from [lane_b] sigma_rel_choices (shipped 0.03, "
                        "0.05, 0.10, 0.15, 0.20; default 0.10); only that EMA is computed.")
    from ..funding.providers import add_provider_args

    add_provider_args(f)
    f.add_argument("--lium-key-env", default="LIUM_API_KEY",
                   help="Env var holding your Lium API key (never on the command line).")
    f.add_argument("--label", default="", help="Display name for the public queue (cosmetic).")
    f.add_argument("--withdraw", action="store_true",
                   help="Withdraw a still-queued lane-B entry instead of funding.")
    f.add_argument("--yes", action="store_true",
                   help="Confirm funding: the leg rents a pod billed to your selected provider key.")
    f.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    from .cli import add_gpu_args

    add_gpu_args(f, lane_b=True)
    f.add_argument("--wallet-name", required=True, help="Bittensor wallet (coldkey) name.")
    f.add_argument("--wallet-hotkey", required=True, help="Bittensor wallet hotkey name.")
    f.add_argument("--wallet-path", default=None, help="Optional non-default wallet root.")
    f.set_defaults(func=cmd_fund, command="lane-b fund")

    s = lsub.add_parser("status", help="The public lane-B queue (optionally your entry).")
    s.add_argument("intake_url", help="Operator intake base URL.")
    s.add_argument("--hotkey", default=None, help="Show only this hotkey's entry.")
    s.add_argument("--chain-toml", type=Path, default=None,
                   help="Override chain.toml path (the [storage] public bucket of "
                        "status/lane_b.json).")
    s.set_defaults(func=cmd_status, command="lane-b status")

    v = lsub.add_parser("verify-repo",
                        help="Run the intake's contamination scan on a generator tree locally "
                             "(read only; nothing is executed or uploaded).")
    v.add_argument("repo_dir", type=Path, help="The generator tree to scan.")
    v.add_argument("--curriculum", action="store_true",
                   help="WB27: only check that Generator.__init__ declares the "
                        "curriculum_stage keyword a multi-stage curriculum needs (static; "
                        "never runs the code).")
    v.add_argument("--chain-toml", type=Path, default=None,
                   help="Override chain.toml path ([lane_b] contamination_hashset and "
                        "contamination_min_hits).")
    v.set_defaults(func=cmd_verify_repo, command="lane-b verify-repo")


def lane_b_fund_targets(cfg, spec: str) -> tuple[tuple[str, ...], str | None]:
    """``(targets in the [lane_b] slot order, None)`` or ``((), why)`` — the
    ``--expert-domain`` rule (``h<h>`` = ``horizon:<h>``), so a fund names
    exactly what the commitment declared."""
    from .cli import expert_targets

    declared, why = expert_targets(cfg, spec)
    if why:
        return (), why
    slots = tuple(getattr(cfg.lane_b, "targets", None) or cfg.lane_b.domains)
    return tuple(t for t in slots if t in declared), None


def build_lane_b_headers(action: str, hotkey_ss58: str, targets, ref: str, api_key: str,
                         sign_fn, *, now=None, label: str = "", gpu: str = "",
                         allow_fallback: bool = False,
                         max_price: float | None = None,
                         declared_rows=None, levers_json: str = "",
                         length_headers: dict | None = None) -> dict[str, str]:
    """The signed header set of a lane-B intake request (pure; testable):
    the canonical message binds the action, hotkey, ``"<targets>|<ref>"``
    (targets exactly as sent), the timestamp and the key digest — and, with
    ``gpu`` (fund only), the GPU headers through the v3 message. With
    ``declared_rows`` (parsed WB14 rows, fund only) the request also carries
    ``X-Lane-B-Declared-Rows`` (the canonical rows JSON) and the signed ref
    gains the ``|rows=<sha256>`` tag; without them it is today's request.
    With ``levers_json`` (WB27, fund only) the request carries
    ``X-Lane-B-Levers`` and the signed ref gains ``|levers=<sha256>``."""
    from ..funding.intake import (
        LANE_B_DECLARED_ROWS_HEADER,
        LANE_B_TARGETS_HEADER,
        lane_b_signed_ref,
    )
    from .funding import build_intake_headers

    target_headers = {LANE_B_TARGETS_HEADER: ",".join(targets)}
    if declared_rows is not None:
        from ..trainer.declared_rows import canonical_declared_rows

        target_headers[LANE_B_DECLARED_ROWS_HEADER] = canonical_declared_rows(declared_rows)
    if levers_json:
        from ..shared.lane_b_levers import LEVERS_HEADER

        target_headers[LEVERS_HEADER] = levers_json
    from ..shared.lane_b_leg import TRAIN_HOURS_HEADER, TRAIN_TOKENS_HEADER

    lh = dict(length_headers or {})
    target_headers.update(lh)
    return build_intake_headers(action, hotkey_ss58, ref, api_key, sign_fn,
                                now=now, label=label, gpu=gpu,
                                allow_fallback=allow_fallback, max_price=max_price,
                                fund_action="lane-b-fund",
                                target_headers=target_headers,
                                signed_ref=lane_b_signed_ref(
                                    targets, ref, declared_rows, levers=levers_json,
                                    train_hours=lh.get(TRAIN_HOURS_HEADER, ""),
                                    train_tokens=lh.get(TRAIN_TOKENS_HEADER, "")))


def read_declared_rows(path: Path, lane_b) -> dict:
    """The parsed rows of a ``--declared-rows`` file under the ``[lane_b]``
    clamp — the intake's own parser, so a local refusal carries the code the
    intake would answer. Raises ``DeclaredRowsError`` (a declaration the
    intake refuses) or ``OSError`` / ``UnicodeDecodeError`` (unreadable)."""
    from ..trainer.declared_rows import model_experts_per_layer, parse_declared_rows

    raw = Path(path).read_text(encoding="utf-8")
    return parse_declared_rows(raw, experts_per_layer=model_experts_per_layer(),
                               min_frac=float(lane_b.min_experts_frac),
                               max_frac=float(lane_b.max_experts_frac))


def _target_spec(args, out) -> str | int:
    """The target spec of ``--targets`` or the repeated ``--target T`` flags
    (rev6.22: one leg may target several slots — domains, horizons and
    suites mixed), or a USAGE exit. The list is validated as one
    ``--expert-domain`` value (configured, unique, at most ``[lane_b]
    max_domains_per_leg``)."""
    if args.target is None:
        return args.targets
    targets = [str(t).strip() for t in args.target]
    bad = [t for t in targets if not t or "," in t]
    if bad:
        return out.fail("bad_targets", f"--target takes one target per flag, got {bad[0]!r}; "
                        "repeat --target (or use --targets T1,T2)", USAGE)
    return ",".join(targets)


def _receipt_lines(out, receipt) -> None:
    """The WB13 contamination receipt, for people (stderr)."""
    from .submission_status import safe_text

    if not isinstance(receipt, dict):
        return
    parts = [f"result={safe_text(receipt.get('result'))}"]
    for field in ("suite", "window_id", "hashset_version"):
        if receipt.get(field) is not None:
            parts.append(f"{field}={safe_text(receipt.get(field))}")
    out.warn("  contamination check: " + " ".join(parts))


def cmd_fund(args: argparse.Namespace) -> int:
    import urllib.request

    from ..shared.config import load_chain_config
    from .cli import post_intake, report_gpu_reply
    from .funding import prepare_funding_request, redact_secrets

    out = reporter(args)
    action = "lane-b-withdraw" if args.withdraw else "lane-b-fund"
    cfg = load_chain_config(args.chain_toml)
    length_headers, rc = _leg_length_headers(args, cfg, out)
    if rc is not None:
        return rc
    spec = _target_spec(args, out)
    if isinstance(spec, int):
        return spec
    targets, why = lane_b_fund_targets(cfg, spec)
    if why:
        flag = "--target" if args.target is not None else "--targets"
        return out.fail("bad_targets", f"{flag} refused: {why}", USAGE)
    declared_rows = None
    if args.declared_rows is not None:
        if args.withdraw:
            return out.fail("conflicting_flags", "--declared-rows applies to a fund, not "
                            "--withdraw", USAGE)
        from ..trainer.declared_rows import DeclaredRowsError

        try:
            declared_rows = read_declared_rows(args.declared_rows, cfg.lane_b)
        except DeclaredRowsError as e:
            return out.fail(e.code, f"--declared-rows refused: {e}", USAGE)
        except (OSError, UnicodeDecodeError) as e:
            return out.fail("declared_rows_unreadable",
                            f"--declared-rows: cannot read {args.declared_rows}: "
                            f"{type(e).__name__}", USAGE)
    levers_json, rc = _levers_json(args, cfg, out)
    if rc is not None:
        return rc
    prepared = prepare_funding_request(
        args, out, funding=action == "lane-b-fund",
        confirmation="funding rents a domain-expert pod billed to your selected provider key")
    if isinstance(prepared, int):
        return prepared
    headers = build_lane_b_headers(
        action, prepared.hotkey, targets, args.ref, prepared.api_key, prepared.sign_fn,
        label=prepared.label, gpu=prepared.choice[0], allow_fallback=prepared.choice[1],
        max_price=prepared.choice[2], declared_rows=declared_rows, levers_json=levers_json,
        length_headers=length_headers)
    path = "/v1/lane-b/withdraw" if args.withdraw else "/v1/lane-b/fund"
    req = urllib.request.Request(f"{prepared.base}{path}", method="POST", headers=headers)
    answer = post_intake(req, out, timeout=30)
    if answer is None:
        return NETWORK
    status, body = answer
    # An intake reply is never trusted not to echo the provider key back.
    body = redact_secrets(body, prepared.api_key)
    receipt = body.get("contamination") if isinstance(body.get("contamination"), dict) else None
    if 200 <= status < 300:
        out.result(action=action, status=body.get("status", "ok"), hotkey=prepared.hotkey,
                   targets=list(targets), ref=args.ref)
        if declared_rows is not None:
            out.result(declared_rows={str(k): list(v) for k, v in declared_rows.items()})
        if levers_json:
            out.result(levers=json.loads(levers_json))
        for k in ("train_hours", "train_tokens", "steps", "guard_seconds", "gpu_measured"):
            if k in body:
                out.result(**{k: body[k]})
        if body.get("gpu_measured") is False:
            out.info("WARNING: this GPU type has no measured rate — AT YOUR OWN RISK: no "
                     "refund if the leg fails (OOM, no boot, crash); hitting the hour limit "
                     "is a normal stop and the leg is judged on what it trained")
        if receipt is not None:
            out.result(contamination=receipt)
        out.info(f"{action}: {body.get('status', 'ok')} (hotkey {prepared.hotkey}, targets "
                 f"{','.join(targets)})")
        report_gpu_reply(out, body)
        if action == "lane-b-fund":
            from .submission_status import tracking_result

            tracking_result(out, args, hotkey=prepared.hotkey, ref=args.ref, lane="b", targets=targets)
        return OK
    extra = {"contamination": receipt} if receipt is not None else {}
    if receipt is not None:
        out.result(contamination=receipt)
    rc = out.fail(str(body.get("code") or f"{action}_rejected"),
                  f"{action} rejected ({status}): {body.get('code', '?')} — "
                  f"{body.get('message', '')}", http_status=status, **extra)
    _receipt_lines(out, receipt)
    return rc


def lane_b_levers_json(cfg, *, lr_scale=None, curriculum=None, context_length=None,
                       ema=None) -> str:
    """WB27: the ``X-Lane-B-Levers`` value (canonical JSON) of the CLI flags,
    pre-checked with the intake's parser and codes (raises ``LeversError``);
    ``""`` when no lever flag is given (today's request, byte for byte)."""
    from ..shared.lane_b_levers import BAD_CURRICULUM, LeversError, parse_levers

    d: dict = {}
    if lr_scale is not None:
        d["lr_scale"] = str(lr_scale)
    if curriculum is not None:
        stages = []
        for part in str(curriculum).split(","):
            name, sep, frac = part.strip().partition(":")
            if not sep:
                raise LeversError(BAD_CURRICULUM, f"--curriculum stage {part!r} is not "
                                                  "NAME:FRACTION")
            stages.append([name.strip(), frac.strip()])
        d["curriculum"] = stages
    if context_length is not None:
        d["context_length"] = int(context_length)
    if ema is not None:
        d["ema"] = float(ema)
    if not d:
        return ""
    levers = parse_levers(json.dumps(d), cfg.lane_b)
    return levers.to_json() if levers is not None else ""


def _leg_length_headers(args, cfg, out) -> tuple[dict, int | None]:
    """WB23 + owner 2026-11-24: the ``X-Lane-B-Train-Hours`` /
    ``X-Lane-B-Train-Tokens`` headers of ``--train-hours`` / ``--train-tokens``,
    pre-checked with the intake's parsers; ``{}`` when neither is given."""
    from ..shared import lane_b_leg as lbl

    hours = getattr(args, "train_hours", None)
    tokens = getattr(args, "train_tokens", None)
    if hours is None and tokens is None:
        return {}, None
    if args.withdraw:
        return {}, out.fail("conflicting_flags", "--train-hours / --train-tokens apply to a "
                            "fund, not --withdraw", USAGE)
    lb = cfg.lane_b
    out_h: dict = {}
    try:
        if hours is not None:
            lbl.parse_train_hours(str(hours), max_leg_hours=lbl.leg_caps(lb)[0])
            out_h[lbl.TRAIN_HOURS_HEADER] = str(hours).strip()
        if tokens is not None:
            lbl.parse_train_tokens(str(tokens))
            out_h[lbl.TRAIN_TOKENS_HEADER] = str(tokens).strip()
    except lbl.LegLengthError as e:
        flag = "--train-tokens" if e.code == lbl.BAD_TRAIN_TOKENS else "--train-hours"
        return {}, out.fail(e.code, f"{flag} refused: {e.message}", USAGE)
    return out_h, None


def _levers_json(args, cfg, out) -> tuple[str, int | None]:
    from ..shared.lane_b_levers import LeversError

    flags = (getattr(args, "lr_scale", None), getattr(args, "curriculum", None),
             getattr(args, "context_length", None), getattr(args, "ema", None))
    if all(f is None for f in flags):
        return "", None
    if args.withdraw:
        return "", out.fail("conflicting_flags", "lever flags apply to a fund, not "
                            "--withdraw", USAGE)
    try:
        return lane_b_levers_json(cfg, lr_scale=flags[0], curriculum=flags[1],
                                  context_length=flags[2], ema=flags[3]), None
    except LeversError as e:
        return "", out.fail(e.code, f"lever refused: {e.message}", USAGE)


def generator_declares_curriculum_stage(repo: Path) -> bool:
    """WB27: does ``generator.py``'s ``Generator.__init__`` declare the
    ``curriculum_stage`` keyword (or ``**kwargs``)? Static AST read only."""
    import ast

    try:
        tree = ast.parse((Path(repo) / "generator.py").read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Generator":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    a = fn.args
                    names = {x.arg for x in (*a.args, *a.kwonlyargs)}
                    return "curriculum_stage" in names or a.kwarg is not None
    return False


def cmd_status(args: argparse.Namespace) -> int:
    from .submission_status import lane_b_public_status, lane_queue_status, render_lane_b_status

    rc = lane_queue_status(args, "b")
    out = reporter(args)
    view = lane_b_public_status(args, hotkey=args.hotkey)
    if view is not None:
        out.result(lane_b_status=view)
        render_lane_b_status(out, view)
    return rc


def cmd_verify_repo(args: argparse.Namespace) -> int:
    """The WB13 scan of ``repo_dir`` against the pinned hash set, offline.

    ``load_pinned_hashset`` and ``scan_repo`` are looked up on the
    ``contamination_scan`` module at call time (the intake's own entry
    points). The tree is only read; no wallet, key or connection is used."""
    from ..interface import contamination_scan
    from ..shared.config import load_chain_config

    out = reporter(args)
    repo = Path(args.repo_dir)
    if not repo.is_dir():
        return out.fail("not_a_directory", f"verify-repo: {repo} is not a directory", USAGE)
    if getattr(args, "curriculum", False):
        ok = generator_declares_curriculum_stage(repo)
        out.result(repo=str(repo), curriculum_stage_keyword=ok)
        if not ok:
            return out.fail("generator_curriculum_unsupported",
                            "verify-repo: Generator.__init__ does not declare the "
                            "curriculum_stage keyword a multi-stage curriculum needs", REJECTED)
        out.info(f"verify-repo: {repo} declares curriculum_stage")
        return OK
    cfg = load_chain_config(args.chain_toml)
    lane_b = cfg.lane_b
    pin = str(getattr(lane_b, "contamination_hashset", "") or "")
    if not pin:
        return out.fail("contamination_hashset_unpinned",
                        "verify-repo: [lane_b] contamination_hashset is not pinned in this "
                        "chain.toml; there is no hash set to scan against", USAGE)
    try:
        hashset = contamination_scan.load_pinned_hashset(pin, _RELEASE_ROOT)
    except (OSError, ValueError) as e:
        return out.fail("contamination_hashset_unavailable",
                        f"verify-repo: the pinned hash set {pin} could not be loaded: {e}", USAGE)
    if hashset is None:
        return out.fail("contamination_hashset_unavailable",
                        f"verify-repo: no hash set loaded for pin {pin}", USAGE)
    try:
        receipt = dict(contamination_scan.scan_repo(
            repo, hashset, min_hits=lane_b.contamination_min_hits))
    except Exception as e:  # noqa: BLE001 — an unscannable tree is refused, never "clean"
        return out.fail("contamination_scan_failed",
                        f"verify-repo: the scan of {repo} could not complete: "
                        f"{type(e).__name__}: {e}", REJECTED)
    out.result(repo=str(repo), hashset=pin, contamination=receipt)
    code = SCAN_REFUSAL_CODES.get(receipt.get("result"))
    if code is None and receipt.get("result") != contamination_scan.RESULT_CLEAN:
        code = "contamination_scan_failed"
    if code is not None:
        rc = out.fail(code, f"verify-repo: {repo} would be refused by the intake ({code})",
                      REJECTED)
        _receipt_lines(out, receipt)
        return rc
    out.info(f"verify-repo: {repo} is clean against hash set v{receipt.get('hashset_version')} "
             f"({pin})")
    return OK


__all__ = ["add_lane_b", "build_lane_b_headers", "cmd_verify_repo",
           "generator_declares_curriculum_stage", "lane_b_fund_targets", "lane_b_levers_json",
           "read_declared_rows"]
