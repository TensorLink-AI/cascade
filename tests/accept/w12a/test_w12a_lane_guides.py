"""One miner guide per competition (owner 2026-11-24: "we need miner docs for
each competition — tell the intent of the competition, strategies they might
try and practical useful information").

docs/MINER_LANE_A.md, MINER_LANE_B.md, MINER_LANE_C_MARKETCAST.md and
MINER_LANE_D.md each carry: intent, what you submit, how it is scored, how
you are paid (with a worked example at the owner-proposed split), strategies,
costs, refusals and a FAQ. Every `cascade ...` snippet must name a real
command and real flags (scripts/check_docs.py's own checker), links resolve,
and each guide points at `cascade rules <lane>`. The values the guides list
come from chain.toml, checked here against the loader.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GUIDES = {
    "a": REPO / "docs" / "MINER_LANE_A.md",
    "b": REPO / "docs" / "MINER_LANE_B.md",
    "c": REPO / "docs" / "MINER_LANE_C_MARKETCAST.md",
    "d": REPO / "docs" / "MINER_LANE_D.md",
}
# heading word -> the section it marks (case-insensitive, any heading level)
SECTIONS = {
    "intent": r"intent",
    "submit": r"what you submit",
    "scored": r"how it is scored",
    "paid": r"how you are paid",
    "strategies": r"strateg",
    "costs": r"cost",
    "refusals": r"refus",
    "faq": r"\bfaq\b|frequently asked",
}


def _text(lane: str) -> str:
    path = GUIDES[lane]
    assert path.is_file(), f"{path.relative_to(REPO)} is missing"
    return path.read_text()


def _headings(text: str) -> list[str]:
    body = re.sub(r"^```[^\n]*\n.*?^```", "", text, flags=re.M | re.S)
    return [h.strip().lower() for h in re.findall(r"^#{1,6}\s+(.+)$", body, re.M)]


def _cfg():
    from cascade.shared.config import load_chain_config

    return load_chain_config(REPO / "chain.toml")


def test_each_guide_exists_and_its_title_names_the_lane():
    for lane in GUIDES:
        title = _text(lane).splitlines()[0]
        assert title.startswith("# "), title
        assert re.search(rf"\blane {lane}\b", title, re.I), title


def test_each_guide_has_every_required_section():
    for lane in GUIDES:
        heads = _headings(_text(lane))
        for name, pattern in SECTIONS.items():
            assert any(re.search(pattern, h) for h in heads), f"lane {lane}: no {name} section"


def test_each_guide_points_at_cascade_rules_for_its_lane():
    for lane in GUIDES:
        assert f"cascade rules {lane}" in _text(lane), lane


def test_each_pay_section_has_a_worked_example_at_the_proposed_split():
    for lane in GUIDES:
        text = _text(lane)
        assert "A0 / B35 / C15 / D50" in text, lane
        assert re.search(r"owner-proposed|owner's proposal", text), lane
        assert re.search(r"worked example", text, re.I), lane


def test_every_cli_snippet_names_a_real_command_and_real_flags():
    from scripts.check_docs import cli_issues

    paths = [GUIDES[lane] for lane in GUIDES]
    for lane in GUIDES:
        _text(lane)
    issues, examples, *_ = cli_issues(paths, root=REPO)
    real = [i for i in issues if not i.startswith(("Undocumented miner command",
                                                   "Undocumented console entrypoint"))]
    assert not real, real
    assert examples >= 20, f"only {examples} CLI snippets in the lane guides"


def test_links_resolve_and_the_guides_are_linked_from_miner_md_and_llms_txt():
    from scripts.check_docs import link_issues

    for lane in GUIDES:
        _text(lane)
    assert not link_issues(list(GUIDES.values()), root=REPO)
    miner = (REPO / "docs" / "MINER.md").read_text()
    llms = (REPO / "llms.txt").read_text()
    for path in GUIDES.values():
        assert f"]({path.name})" in miner, f"MINER.md does not link {path.name}"
        assert f"](docs/{path.name})" in llms, f"llms.txt does not link {path.name}"
    assert "cascade rules" in miner and "cascade rules" in llms


def test_listed_values_match_chain_toml():
    cfg = _cfg()
    d = _text("d")
    for context in cfg.lane_d.contexts:
        assert str(int(context)) in d, f"lane D guide omits context {context}"
    assert "BF16" in d and "MIT" in d
    b = _text("b")
    for domain in cfg.lane_b.domains:
        assert domain in b, f"lane B guide omits domain {domain}"
    for suite in cfg.lane_b.require_public:
        assert f"suite:{suite}" in b, f"lane B guide omits suite:{suite}"
    c = _text("c")
    for suite in cfg.marketcast.suites:
        assert f"`{suite}`" in c, f"lane C guide omits suite {suite}"
    for _, bars in cfg.marketcast.frequencies:
        for bar in bars:
            assert re.search(rf"\b{bar}\b", c), f"lane C guide omits timeframe {bar}"
    for target in cfg.lane_c.targets:
        assert target in c
    a = _text("a")
    for horizon in cfg.lane_b.horizons:
        assert str(horizon) in a, f"lane A guide omits horizon {horizon}"


def test_guides_quote_the_measured_gpu_rates_where_miners_pay_for_training():
    for lane in ("b", "d"):
        text = _text(lane)
        for token in ("8×H100", "0.404", "847k", "8×RTX4090", "0.137", "286k", "4×H100", "0.212"):
            assert token in text, f"lane {lane} costs omit {token}"
