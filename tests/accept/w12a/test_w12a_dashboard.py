"""W12a: the main miner dashboard (``cascade/website/index.html``) covers all
four competitions and the repo ``llms.txt`` is an llmstxt.org agent file
(owner 2026-11-27: "do an llms.txt agent file and update our main miner
dashboard to include all comps ... make sure it updates correctly, everything
updates and is correctly formatted").

Dashboard contract:

- one tab and one page per lane (A king, B experts, C MarketCast, D open
  weights), each lane page rendering every panel from fixture data built by
  the REAL writers (``build_v2_status``, ``build_lane_d_status``, the lane-B
  driver's ``status()`` keys) and rendering an explicit empty state when
  nothing is published (no exceptions in either case);
- every lane source is polled by ``poll()`` (so a refresh updates it): the
  keys are the writers' own constants;
- the lane D page reads only fields the writers emit (status doc, signed
  report schema), the lane B queue only the driver's status keys;
- times are UTC minutes, shares are percentages, money is whole dollars.

The node render tests need ``node`` (present on the CI runners); the static
checks need nothing but the files.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
WEB = REPO / "cascade" / "website"
INDEX = WEB / "index.html"
LLMS = REPO / "llms.txt"
REPORT_SCHEMA = REPO / "cascade" / "validator" / "schemas" / "lane_d_report_1.json"
LANE_B_DRIVER = REPO / "cascade" / "trainer" / "lane_b_driver.py"

LANE_TABS = ("main", "laneb", "lanec", "laned")
LANE_PANELS = {
    "main": ("verdict", "era-settlements", "subs"),
    "laneb": ("lb-gate", "lb-domains", "lb-queue"),
    "lanec": ("lc-status", "lc-targets", "lc-queue", "lc-settle"),
    "laned": ("ld-throne", "ld-queue", "ld-settle"),
}
UTC_MINUTE = re.compile(r"\b20\d\d-\d\d-\d\d \d\d:\d\d UTC\b")

HOUSE = hashlib.sha256(b"w12a dashboard house").hexdigest()
ENTRY = hashlib.sha256(b"w12a dashboard entry").hexdigest()
CHALLENGER = hashlib.sha256(b"w12a dashboard challenger").hexdigest()
BOUNDARY = 9_593_100
BOUNDARY_TS = 1_795_721_268          # a live report's boundary stamp (UTC seconds)


def _html() -> str:
    return INDEX.read_text(encoding="utf-8")


def _fn(name: str) -> str:
    """The source of the page's function ``name`` (to the next top-level
    function or section rule)."""
    m = re.search(r"\nfunction " + re.escape(name) + r"\(.*?(?=\nfunction |\n// ──|\n\(function|\nvar |\n\}\n)",
                  _html(), re.S)
    assert m, f"index.html has no function {name}()"
    return m.group(0)


def _reads(body: str, var: str) -> set[str]:
    return set(re.findall(r"(?<![\w.])" + re.escape(var) + r"\.([a-z_][a-z0-9_]*)", body))


def _schema() -> dict:
    return json.loads(REPORT_SCHEMA.read_text())


def _props(defn: dict) -> set[str]:
    return set(defn.get("properties", {}))


# ── the lane D fixtures, built by the real writer ────────────────────────────


def _cfg():
    from cascade.shared.config import load_chain_config

    cfg = load_chain_config(REPO / "chain.toml")
    return dataclasses.replace(cfg, lane_d=dataclasses.replace(cfg.lane_d, from_block=9_586_800))


def _report(*, crowned: bool) -> dict:
    """A lane-d-report/1 body with the fields the page reads (the live
    report's shape; every key is a schema property)."""
    params = {"share": 0.5, "contexts": [4096, 8192, 16384, 32768], "d_margin": 0.01,
              "d_margin_start": 0.03, "d_margin_warmup_settlements": 32,
              "court_weights": [0.8, 0.15, 0.05]}
    house = {"digest": HOUSE, "hotkey": None, "house": True, "reveal_ts": 1_794_914_412}
    king = ({"digest": CHALLENGER, "hotkey": "5Champ" + "x" * 42, "house": False,
             "reveal_ts": 1_795_600_000} if crowned else house)
    verdicts = ([{"digest": CHALLENGER, "king_digest": HOUSE, "stage": "first", "outcome": "crowned",
                  "lcb": 0.031, "gain": 0.0425, "halves": {}, "sources": {}}] if crowned else [])
    court = ([{"digest": CHALLENGER, "hotkey": king["hotkey"], "uid": 42, "burn_reason": None},
              {"digest": HOUSE, "hotkey": None, "uid": None, "burn_reason": "house_king"}]
             if crowned else
             [{"digest": HOUSE, "hotkey": None, "uid": None, "burn_reason": "house_king"}])
    return {"schema": "lane-d-report/1", "boundary": BOUNDARY, "boundary_ts": BOUNDARY_TS,
            "armed_contexts": [4096, 8192, 16384], "params": params,
            "crown_bar": {"tenure": 0 if crowned else 2, "margin": 0.0275},
            "crowned": CHALLENGER if crowned else None, "house": house,
            "king_before": house, "king_after": king,
            "prior_kings": [house] if crowned else [], "entries": [],
            "judged": [CHALLENGER] if crowned else [],
            "skipped": [] if crowned else [{"digest": ENTRY, "reason": "pending_floor"}],
            "verdicts": verdicts, "pay": {"share": 0.5, "court": court}}


def _state(report: dict, *, crowned: bool, entries: bool) -> dict:
    from cascade.validator import lane_d_state

    st = lane_d_state.genesis({"digest": HOUSE, "reveal_ts": 1_794_914_412})
    st["boundary"] = BOUNDARY
    if crowned:
        st["king"] = {"digest": CHALLENGER, "hotkey": report["king_after"]["hotkey"],
                      "house": False, "since_boundary": BOUNDARY, "reveal_ts": 1_795_600_000}
        st["prior_kings"] = [{"digest": HOUSE, "hotkey": None, "reveal_ts": 1_794_914_412}]
    if entries:
        st["entries"] = {ENTRY: {
            "status": "pending", "reason": "pending_floor", "hotkey": "5Entry" + "y" * 42,
            "coldkey_at_ready": "5Cold" + "z" * 43, "context": 8192, "ready_block": 9_592_000,
            "reveal_block": 9_592_100, "reveal_ts": 1_795_700_000, "confirm": None,
            "frozen_set": {"window_ids": ["w1", "w2", "w3"],
                           "sources": {"w1": "s1", "w2": "s2", "w3": "s2"}, "frozen": False}}}
    st["last_report"] = {"report_sha256": lane_d_state.report_digest(report)}
    return st


def lane_d_fixtures(*, crowned: bool = False, entries: bool = True) -> dict:
    from cascade.validator.lane_d_status import LANE_D_STATUS_KEY, build_lane_d_status
    from cascade.validator.lane_d_wiring import RECEIPT_LATEST_KEY, report_key

    rep = _report(crowned=crowned)
    doc = build_lane_d_status(_cfg(), state=_state(rep, crowned=crowned, entries=entries),
                              report=rep, block=BOUNDARY)
    assert doc is not None
    receipt = {"schema": "lane-d-receipt/1", "boundary": BOUNDARY + 900, "status": "absent",
               "reason": "infra_deferred"}
    return {LANE_D_STATUS_KEY: doc, report_key(BOUNDARY): rep, RECEIPT_LATEST_KEY: receipt}


def lane_b_status(block: int) -> dict:
    """``status/lane_b.json`` as ``publish_lane_status`` writes it: the
    envelope plus the driver's ``status()`` keys (the live shape)."""
    return {"lane": "b", "block": block, "as_of": datetime.now(UTC).isoformat(),
            "leg_funding": "miner", "leg_hours": 3.0, "legs_per_domain": 1,
            "gate_opened_block": block - 7000, "open_rule": "tokens", "gate_rule": "tokens",
            "gate_closed_reason": "", "pending": 0, "finished": 1, "pending_merges": 0,
            "submission_progress": [], "deferred": [],
            "queue": {"as_of": "2026-11-27T10:03:49+00:00", "lane": "b", "queued_depth": 1,
                      "pending_reveal_count": 0, "entries": [
                          {"hotkey": "5Queued" + "q" * 41, "targets": ["sales", "horizon:64"],
                           "status": "queued", "phase": "waiting_admission", "gpu": "8xH100",
                           "funded_at": 1_795_652_216.2, "spend_usd": 33.4}]},
            "open": [{"hotkey": "5GMzsPiUhy6gtqU5GxeUgzJ4yyQnJ5iyx8EUdSgZD2riSDgn", "uid": 171,
                      "targets": ["energy"], "ref": "ns/gen@sha256:" + "7" * 8 + "2e8" * 18 + "f3",
                      "base_era": 2665, "experts": 60, "gpu": "8xH200", "train_hours": 3.0,
                      "steps": 0, "guard_seconds": 0, "funded": True}],
            "outcomes": [{"hotkey": "5Old" + "o" * 44, "uid": 9, "domain": "nature",
                          "outcome": "requeued", "round_id": "",
                          "detail": {"class": "operator_blocked", "error": "paused"}}]}


# ── static: every lane has a tab, a page, panels and a polled source ─────────


def test_dashboard_has_a_tab_page_and_panels_for_every_lane():
    html = _html()
    for tab in LANE_TABS:
        assert f'data-ptab="{tab}"' in html, f"no tab for {tab}"
        assert f'id="page-{tab}"' in html, f"no page-{tab}"
        for panel in LANE_PANELS[tab]:
            assert f'id="{panel}"' in html, f"{tab}: no #{panel} panel"
    set_page = _fn("setPage")
    for tab in LANE_TABS:
        assert f'"page-{tab}"' in set_page, f"setPage() never shows page-{tab}"
    for render in ("renderLaneB", "renderLaneC", "renderLaneD"):
        assert render + "()" in _fn("renderAll"), f"renderAll() never calls {render}()"


def test_every_lane_source_is_polled_by_its_writer_key():
    from cascade.trainer.lane_b_driver import LANE_B_STATUS_KEY
    from cascade.validator.lane_d_status import LANE_D_STATUS_KEY
    from cascade.validator.lane_d_wiring import RECEIPT_LATEST_KEY, report_key

    poll = _fn("poll")
    html = _html()
    for key in ("status/v2.json", "status/lane_c.json", LANE_B_STATUS_KEY):
        assert f'fetchJSON("{key}")' in poll, f"poll() never refreshes {key}"
    assert "loadLaneD()" in poll, "poll() never refreshes lane D"
    load = _fn("loadLaneD") + _fn("loadLdReports")
    assert f'fetchJSON("{LANE_D_STATUS_KEY}")' in load
    assert f'fetchJSON("{RECEIPT_LATEST_KEY}")' in load
    prefix = report_key(BOUNDARY).split(str(BOUNDARY))[0]
    assert f'fetchJSON("{prefix}"+' in load, f"the lane D reports ({prefix}<b>.json) are never read"
    vercel = json.loads((WEB / "vercel.json").read_text())
    assert any(r["source"] == "/lane-d/:path*" for r in vercel["rewrites"]), \
        "the same-origin fallback has no /lane-d/ rewrite"
    assert "setInterval(function(){ if(!document.hidden) poll(); }, POLL_MS);" in html


def test_lane_d_page_reads_only_written_fields():
    from cascade.validator import lane_d_status

    fx = lane_d_fixtures(entries=True)
    doc = fx[lane_d_status.LANE_D_STATUS_KEY]
    body = _fn("renderLdThrone") + _fn("renderLdQueue") + _fn("renderLdSettle")
    assert _reads(body, "d") <= set(doc), _reads(body, "d") - set(doc)
    assert _reads(body, "k") <= set(doc["king"]), _reads(body, "k") - set(doc["king"])
    assert _reads(body, "fl") <= set(doc["floors"])
    row = doc["queue"][0]
    assert _reads(_fn("renderLdQueue"), "e") <= set(row), _reads(_fn("renderLdQueue"), "e") - set(row)
    assert _reads(_fn("renderLdQueue"), "pr") <= set(row["progress"])
    assert _reads(_fn("renderLdSettle"), "board") <= set(doc["board"])

    schema = _schema()
    defs = schema["$defs"]
    top = _props(schema)
    for fn, var in (("renderLdThrone", "rep"), ("renderLdSettle", "r")):
        got = _reads(_fn(fn), var)
        assert got and got <= top, (fn, got - top)
    assert _reads(_fn("renderLdThrone"), "p") <= _props(defs["params"])
    assert _reads(_fn("renderLdThrone"), "s") <= _props(defs["court_slot"])
    assert _reads(_fn("renderLdSettle"), "v") <= _props(defs["verdict"])
    assert _reads(_fn("renderLdSettle"), "s") <= _props(defs["skip"])
    assert _reads(_fn("ldKingLabel"), "k") <= _props(defs["king"]) | set(doc["king"])


def _driver_status_keys() -> set[str]:
    """The keys of ``LaneBDriver.status()``'s returned dict literal (+ the
    ``publish_lane_status`` envelope), read from the source."""
    tree = ast.parse(LANE_B_DRIVER.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "status":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                    keys = {k.value for k in sub.value.keys if isinstance(k, ast.Constant)}
                    open_rows = [v for k, v in zip(sub.value.keys, sub.value.values, strict=True)
                                 if isinstance(k, ast.Constant) and k.value == "open"]
                    leg = set()
                    for comp in open_rows:
                        for d in ast.walk(comp):
                            if isinstance(d, ast.Dict):
                                leg |= {k.value for k in d.keys if isinstance(k, ast.Constant)}
                    return keys | {"lane", "block", "as_of"}, leg
    raise AssertionError("LaneBDriver.status() not found")


def test_lane_b_queue_reads_only_the_driver_status_keys():
    keys, leg = _driver_status_keys()
    body = _fn("renderLbQueue")
    assert _reads(body, "d") <= keys, _reads(body, "d") - keys
    assert _reads(body, "l") <= leg, _reads(body, "l") - leg


# ── node renders: every lane from fixtures, and every empty state ────────────


def _render(fx: dict, calls: list[str], tmp_path: Path) -> dict:
    from tests.unit.test_website_render import NODE
    from tests.unit.test_website_render import _render as render

    if NODE is None:
        pytest.skip("node not installed")
    return render("index.html", fx, "dashboard.casdesub.net", calls, tmp_path)


def _text(els: dict, el: str) -> str:
    return els.get(el, {}).get("html", "") + els.get(el, {}).get("text", "")


LANE_CALLS = ["renderAll()", "setPage('laneb')", "renderLaneB()", "setPage('lanec')",
              "renderLaneC()", "setPage('laned')", "renderLaneD()", "setPage('main')", "renderAll()"]


def _full_bucket() -> dict:
    from tests.unit.test_website_lanes import _with_lanes
    from tests.unit.test_website_render import _bucket

    fx = _with_lanes(_bucket(v2_from=1000, block=50_000, lane_b=True), lane_c=True)
    fx["status/lane_c.json"]["settlements"] = [
        {"round_id": "12135123300179706659", "entries": 2, "digest": "24" * 32,
         "key": "lane_c/round-12135123300179706659.json"},
        {"round_id": "15935462151605169871", "entries": 0, "digest": "", "key": ""}]
    fx["status/lane_b.json"] = lane_b_status(50_000)
    return fx


def test_every_lane_renders_from_published_fixtures(tmp_path):
    fx = {**_full_bucket(), **lane_d_fixtures(entries=True)}
    els = _render(fx, LANE_CALLS, tmp_path)

    # lane A: the king, the era and the verdict render as before
    assert _text(els, "verdict") and _text(els, "v2-state")
    # lane B: the queue panel from status/lane_b.json
    lb = _text(els, "lb-queue")
    for want in ("Legs in flight", "uid 171", "8xH200", "60", "energy", "Funded queue",
                 "horizon 64", "$33", "Recent outcomes", "requeued", "operator_blocked"):
        assert want in lb, want
    assert UTC_MINUTE.search(lb), "the funded time is not a UTC minute"
    assert "status/lane_b.json at block 50000" in _text(els, "lb-queue-when")
    # lane C: targets, queue and the settlement records
    assert "● live" in _text(els, "lc-status")
    settle = _text(els, "lc-settle")
    assert "lane_c/round-12135123300179706659.json" in settle and "no lane C record" in settle
    # lane D: throne, entries, settlements
    throne = _text(els, "ld-throne")
    for want in ("● open", "the house", "2.75%", "3.00%", "1.00%", "32 settlements", "50%",
                 "80/15/5", "4096 · 8192 · 16384", "1000", "house king"):
        assert want in throne, want
    queue = _text(els, "ld-queue")
    for want in (ENTRY[:12], "8192", "pending_floor", "3 / 1000", "2 / 20", "block 9692900"):
        assert want in queue, want
    assert UTC_MINUTE.search(queue), "the reveal time is not a UTC minute"
    assert "1 open entry" in _text(els, "ld-queue-when")
    settle = _text(els, "ld-settle")
    for want in ("block 9594000", "absent", "infra_deferred", "Recent signed reports",
                 "9593100", "pending_floor", "4096, 8192, 16384"):
        assert want in settle, want
    assert UTC_MINUTE.search(settle), "the settlement time is not a UTC minute"
    assert "1 signed report" in _text(els, "ld-settle-when")


def test_lane_d_crowned_king_court_and_verdicts(tmp_path):
    fx = {**_full_bucket(), **lane_d_fixtures(crowned=True, entries=False)}
    els = _render(fx, LANE_CALLS, tmp_path)
    throne = _text(els, "ld-throne")
    assert "the house" not in throne.split("Court")[0], "a crowned miner still reads as the house"
    for want in ("5Champ", CHALLENGER[:12], "crowned at block 9593100", "tenure 0", "uid 42",
                 "prior 1"):
        assert want in throne, want
    settle = _text(els, "ld-settle")
    for want in ("crowned", "3.10%", "4.25%", HOUSE[:12]):
        assert want in settle, want
    assert "no entries yet" in _text(els, "ld-queue")


def test_nothing_published_renders_every_empty_state(tmp_path):
    from tests.unit.test_website_render import _bucket

    fx = _bucket(v2_from=1000, block=50_000, lane_b=False)
    els = _render(fx, LANE_CALLS, tmp_path)
    assert "No lane B queue published" in _text(els, "lb-queue")
    assert "Lane C not live" in _text(els, "lc-settle")
    assert "Lane D not live" in _text(els, "ld-throne")
    assert "No lane D entries" in _text(els, "ld-queue")
    assert "No lane D settlements" in _text(els, "ld-settle")
    for panel in ("ld-throne", "ld-queue", "ld-settle", "lb-queue", "lc-settle"):
        assert "undefined" not in _text(els, panel) and "NaN" not in _text(els, panel), panel


def test_published_but_empty_lanes_say_no_entries_yet(tmp_path):
    fx = {**_full_bucket(), **lane_d_fixtures(entries=False)}
    doc = fx["status/lane_b.json"]
    doc.update(open=[], outcomes=[], finished=0)
    doc["queue"].update(entries=[], queued_depth=0)
    fx["status/lane_c.json"]["settlements"] = []
    els = _render(fx, LANE_CALLS, tmp_path)
    lb = _text(els, "lb-queue")
    assert "no legs in flight" in lb and "no entries yet" in lb and "no outcomes yet" in lb
    assert "no settlements yet" in _text(els, "lc-settle")
    assert "no entries yet" in _text(els, "ld-queue")
    assert "0 open entries" in _text(els, "ld-queue-when")
    for panel in ("ld-throne", "ld-queue", "ld-settle", "lb-queue", "lc-settle"):
        assert "undefined" not in _text(els, panel) and "NaN" not in _text(els, panel), panel


# ── llms.txt: the llmstxt.org agent file ─────────────────────────────────────


def _llms_sections() -> tuple[list[str], dict[str, list[str]]]:
    lines = LLMS.read_text(encoding="utf-8").splitlines()
    head, sections, cur = [], {}, None
    for line in lines:
        if line.startswith("## "):
            cur = line[3:].strip()
            sections[cur] = []
        elif cur is None:
            head.append(line)
        else:
            sections[cur].append(line)
    return head, sections


def test_llms_txt_follows_the_llmstxt_format():
    head, sections = _llms_sections()
    assert head[0] == "# Cascade", "the first line must be the H1 title"
    body = [x for x in head[1:] if x.strip()]
    assert body and body[0].startswith("> "), "the H1 must be followed by a blockquote summary"
    assert not any(re.match(r"#{1,6} ", x) for x in head[1:]), "headings before the first H2"
    assert len(sections) >= 5, sorted(sections)
    link_item = re.compile(r"^- \[[^\]]+\]\([^)\s]+\): \S")
    for name, lines in sections.items():
        items = [x for x in lines if x.strip()]
        assert items, f"## {name} is empty"
        bad = [x for x in items if not link_item.match(x)]
        assert not bad, f"## {name}: every line must be '- [name](url): description' ({bad[:2]})"
    assert list(sections)[-1] == "Optional", "the secondary links belong in a final ## Optional"


def test_llms_txt_covers_every_lane_submission_scoring_history_and_dashboard():
    text = LLMS.read_text(encoding="utf-8")
    _, sections = _llms_sections()
    assert "netuid 91" in text
    for doc in ("MINER_LANE_A.md", "MINER_LANE_B.md", "MINER_LANE_C_MARKETCAST.md",
                "MINER_LANE_D.md", "AGENT_MINING.md"):
        assert f"](docs/{doc}" in text, doc
    submit = "\n".join(sections.get("Submit", []))
    for cmd in ("cascade submit", "cascade lane-b fund", "cascade lane-c commit",
                "cascade lane-d submit"):
        assert cmd in submit, f"## Submit does not name {cmd}"
    past = "\n".join(sections.get("Past evaluations", []))
    for doc in ("MINER_LANE_A.md", "MINER_LANE_B.md", "MINER_LANE_C_MARKETCAST.md",
                "MINER_LANE_D.md"):
        assert f"](docs/{doc}" in past, f"## Past evaluations does not link {doc}"
    assert "receipts/index.json" in past
    dash = "\n".join(sections.get("Dashboard and live data", []))
    base = "https://s3.hippius.com/cascade-manifests/"
    for key in ("index.html", "status/v2.json", "status/lane_b.json", "status/lane_c.json",
                "status/lane_d.json", "lane-d/receipts/latest.json"):
        assert f"]({base}{key})" in dash, f"## Dashboard and live data does not link {key}"
    assert "Scoring and rewards" in sections
