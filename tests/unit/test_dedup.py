"""Content-level duplicate screen — fingerprints, pairwise verdicts, and the
trainer wiring (fetch faked; no Hub, no GPU)."""

from __future__ import annotations

import json

import pytest

from cascade.interface.dedup import (
    KING_UID,
    fingerprint_dir,
    normalized_tokens,
    screen_duplicates,
)

# A generator body long enough that a one-line edit stays above a 0.99 token
# ratio (the near-duplicate tier) while structural rewrites fall well below.
BASE_SOURCE = "\n".join(
    ["import numpy as np", "", "class Generator:", "    def __init__(self, cfg):",
     "        self.cfg = cfg"]
    + [f"        self.w{i} = np.float64({i}) * 0.5 + {i % 7}" for i in range(120)]
    + ["", "    def generate(self, seed):", "        rng = np.random.default_rng(seed)"]
    + [f"        x{i} = rng.normal(0.0, 1.0, size=64) * self.w{i}" for i in range(120)]
    + ["        return sum([" + ", ".join(f"x{i}" for i in range(120)) + "])"]
)


def _repo(tmp_path, name: str, source: str, extra: dict[str, str] | None = None):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "generator.py").write_text(source)
    for fname, content in (extra or {"requirements.txt": "numpy\n"}).items():
        (d / fname).write_text(content)
    return d


def _entry(tmp_path, name, uid, source, extra=None):
    return (name, uid, fingerprint_dir(_repo(tmp_path, name, source, extra)))


# ── fingerprints ─────────────────────────────────────────────────────────────

def test_comment_and_whitespace_shuffle_is_token_identical(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE))
    shuffled = "# totally new comment\n" + BASE_SOURCE.replace(
        "import numpy as np", "import numpy as np   # renamed upload"
    ).replace("    def generate", "\n\n    def generate")
    b = fingerprint_dir(_repo(tmp_path, "b", shuffled))
    assert a.tree_sha256 != b.tree_sha256
    assert a.token_sha256 == b.token_sha256


def test_rename_only_copy_is_masked_identical(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE))
    renamed = BASE_SOURCE.replace("Generator", "MyGen").replace("rng", "prng").replace(
        "cfg", "conf")
    b = fingerprint_dir(_repo(tmp_path, "b", renamed))
    assert a.token_sha256 != b.token_sha256
    assert a.masked_sha256 == b.masked_sha256


def test_identical_tree_reupload_matches(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE))
    b = fingerprint_dir(_repo(tmp_path, "b", BASE_SOURCE))
    assert a.tree_sha256 == b.tree_sha256


def test_tokenize_fallback_on_syntax_error():
    toks = normalized_tokens("def broken(:\n  pass")
    assert toks  # stable fingerprint even for unparsable source


# ── pairwise screen ──────────────────────────────────────────────────────────

def test_near_copy_with_a_real_token_delta_is_kept(tmp_path):
    # POLICY: there is no similarity threshold — a ratio bar is gameable by
    # spacing edits under it and it false-positived a round's finalist. A
    # 1-token edit is a different program; it is kept. (Zero-delta re-rolls
    # are caught by the identical tiers / the behavioral probe instead.)
    tweaked = BASE_SOURCE.replace("size=64", "size=65", 1)
    entries = [
        _entry(tmp_path, "orig", 10, BASE_SOURCE),
        _entry(tmp_path, "twk", 42, tweaked),
    ]
    result = screen_duplicates(entries, None)
    assert result.kept_hotkeys == ("orig", "twk")
    assert not result.dropped


def test_copy_of_king_dropped(tmp_path):
    king = fingerprint_dir(_repo(tmp_path, "king", BASE_SOURCE))
    entries = [_entry(tmp_path, "c", 7, "# defend the throne\n" + BASE_SOURCE)]
    result = screen_duplicates(entries, king)
    assert result.kept_hotkeys == ()
    (v,) = result.dropped
    assert v.matched_uid == KING_UID and v.tier == "token_identical"


def test_template_variants_are_kept(tmp_path):
    # Same scaffold, genuinely different data process — kept, no verdicts.
    variant = BASE_SOURCE
    for i in range(0, 120, 17):
        variant = variant.replace(f"* 0.5 + {i % 7}", f"* 1.5 - {i % 5}")
        variant = variant.replace(f"rng.normal(0.0, 1.0, size=64) * self.w{i}",
                                  f"rng.laplace(0.0, 2.0, size=32) * self.w{i}")
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE),
        _entry(tmp_path, "variant", 2, variant),
    ]
    result = screen_duplicates(entries, None)
    assert result.kept_hotkeys == ("orig", "variant")
    assert not result.dropped and not result.shadow


def test_distinct_generators_kept_silently(tmp_path):
    other = "\n".join(f"value_{i} = {i} ** 2" for i in range(200))
    entries = [
        _entry(tmp_path, "a", 1, BASE_SOURCE),
        _entry(tmp_path, "b", 2, other),
    ]
    result = screen_duplicates(entries, None)
    assert result.kept_hotkeys == ("a", "b")
    assert not result.dropped and not result.shadow


def test_dropped_copies_never_become_rivals(tmp_path):
    # b and c are both copies of a. Each must be judged against KEPT entries
    # (a) — never chained through another dropped copy.
    entries = [
        _entry(tmp_path, "a", 1, BASE_SOURCE),
        _entry(tmp_path, "b", 2, "# reup one\n" + BASE_SOURCE),
        _entry(tmp_path, "c", 3, "# reup two\n" + BASE_SOURCE),
    ]
    result = screen_duplicates(entries, None)
    assert result.kept_hotkeys == ("a",)
    assert [(v.hotkey, v.matched_hotkey) for v in result.dropped] == [
        ("b", "a"), ("c", "a")]


def test_shadow_mode_drops_nothing(tmp_path):
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE),
        _entry(tmp_path, "copy", 2, BASE_SOURCE),
    ]
    result = screen_duplicates(entries, None, enforce=False)
    assert result.kept_hotkeys == ("orig", "copy")  # kept…
    assert [v.hotkey for v in result.dropped] == ["copy"]  # …but the verdict logs


# ── trainer wiring ───────────────────────────────────────────────────────────

@pytest.fixture
def dedup_runner(cfg, tmp_path, monkeypatch):
    """A TrainerRunner whose fetch resolves refs to local fixture dirs."""
    from dataclasses import replace

    from cascade.trainer import loop as loop_mod
    from cascade.trainer.loop import ResolvedGenerator, TrainerRunner

    repos = {}
    transport_fail: set[str] = set()

    def add(name, uid, source, reveal_block=0):
        ref = f"{name}/gen@sha256:{name[0] * 64}"
        repos[ref] = _repo(tmp_path / "repos", name, source)
        return ResolvedGenerator(hotkey=name, uid=uid, ref=ref,
                                 reveal_block=reveal_block)

    def fake_fetch(ref, dest, hub=None):
        from types import SimpleNamespace

        from cascade.shared.hippius import StorageError

        ref = str(ref)
        if ref in transport_fail:
            raise StorageError(f"fetch of {ref}: connection reset by peer")
        if ref not in repos:
            cause = RuntimeError("not found")
            cause.response = SimpleNamespace(status_code=404)
            raise StorageError(f"fetch of {ref}") from cause
        return repos[ref]

    class _FakeLogs:
        def __init__(self):
            self.puts = {}

        def put_text(self, key, text, **kwargs):
            self.puts[key] = text

    fake_logs = _FakeLogs()
    add.transport_fail = transport_fail
    add.logs = fake_logs
    monkeypatch.setattr(loop_mod, "fetch_from_hub", fake_fetch)
    monkeypatch.setattr(loop_mod.TrainerRunner, "logs_store", lambda self: fake_logs)
    dedup_cfg = replace(cfg, round=replace(cfg.round, dedup_mode="enforce",
                                           dedup_probe_mode="off",
                                           dedup_probe_series=0))
    runner = TrainerRunner(cfg=dedup_cfg, base_trainer=object(),
                           work_root=tmp_path / "work", use_sandbox=False)
    monkeypatch.setattr(TrainerRunner, "hub", lambda self: None)
    return runner, add


def test_runner_screen_drops_copy_and_writes_report(dedup_runner, tmp_path):
    runner, add = dedup_runner
    king = add("king", 0, BASE_SOURCE)
    orig = add("alice", 3, "\n".join(f"v{i} = {i} + 1" for i in range(200)))
    copy = add("mallory", 9, "# resubmit\n" + BASE_SOURCE)  # copy of the king

    kept = runner._screen_duplicate_entrants(king, [orig, copy], base_seed=77)
    assert [c.hotkey for c in kept] == ["alice"]

    report = json.loads((runner.work_root / "77" / "dedup_report.json").read_text())
    assert report["mode"] == "enforce"
    assert [d["hotkey"] for d in report["dropped"]] == ["mallory"]
    assert report["dropped"][0]["matched_hotkey"] == "king"
    # fetched trees are cleaned up after screening
    assert not (runner.work_root / "77" / "dedup").exists()
    # the report is also published to the logs store (shadow evidence must
    # not depend on orchestrator disk)
    assert "logs/round-77/dedup_report.json" in add.logs.puts


def test_runner_denied_or_missing_ref_dropped_in_enforce(dedup_runner):
    from cascade.trainer.loop import ResolvedGenerator

    runner, add = dedup_runner
    ok = add("alice", 3, BASE_SOURCE)
    # The fake fetch raises a StorageError chained to an HTTP 404 for unknown
    # refs — the miner's fault (missing/denied), so enforce drops it.
    ghost = ResolvedGenerator(hotkey="ghost", uid=8,
                              ref="ghost/gen@sha256:" + "9" * 64)
    kept = runner._screen_duplicate_entrants(None, [ok, ghost], base_seed=78)
    assert [c.hotkey for c in kept] == ["alice"]


def test_runner_transport_fetch_failure_fails_open(dedup_runner):
    runner, add = dedup_runner
    ok = add("alice", 3, BASE_SOURCE)
    flaky = add("frank", 8, OTHER_SOURCE)
    add.transport_fail.add(flaky.ref)

    # Pods fetch refs themselves — an orchestrator-side transport failure
    # must NOT cost the entrant its slot (or its burn), even in enforce.
    kept = runner._screen_duplicate_entrants(None, [ok, flaky], base_seed=85)
    assert [c.hotkey for c in kept] == ["alice", "frank"]
    report = json.loads((runner.work_root / "85" / "dedup_report.json").read_text())
    assert [u["hotkey"] for u in report["unscreened"]] == ["frank"]
    assert report["fetch_failed"] == []


def test_runner_mode_off_is_a_no_op(dedup_runner):
    from dataclasses import replace

    runner, add = dedup_runner
    a = add("alice", 3, BASE_SOURCE)
    b = add("bobby", 4, BASE_SOURCE)
    runner.cfg = replace(runner.cfg, round=replace(runner.cfg.round, dedup_mode="off"))
    assert runner._screen_duplicate_entrants(None, [a, b], base_seed=79) == [a, b]


# ── config files in the fingerprint ──────────────────────────────────────────

OTHER_SOURCE = "\n".join(f"value_{i} = {i} ** 2" for i in range(200))


def test_json_reformat_and_requirements_order_are_cosmetic(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE, {
        "config.json": '{"alpha": 0.5, "beta": 2}',
        "requirements.txt": "numpy\nscipy\n",
    }))
    b = fingerprint_dir(_repo(tmp_path, "b", BASE_SOURCE, {
        "config.json": '{\n  "beta": 2,\n  "alpha": 0.5\n}',
        "requirements.txt": "scipy\n# pinned deps\nnumpy\n",
    }))
    assert a.tree_sha256 != b.tree_sha256
    assert a.token_sha256 == b.token_sha256


def test_deeply_nested_json_does_not_crash_the_fingerprinter(tmp_path):
    # json.loads on attacker-chosen bytes raises RecursionError (not a
    # ValueError) past ~stack-depth nesting; a 4KB file of brackets must fall
    # back to whitespace normalization, not escape and sink the screen.
    depth = 2000
    hostile = "[" * depth + "]" * depth
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE,
                              {"config.json": hostile}))
    b = fingerprint_dir(_repo(tmp_path, "b", BASE_SOURCE,
                              {"config.json": hostile}))
    assert a.token_sha256 == b.token_sha256  # still screened, still identical


def test_config_value_delta_is_a_real_delta_not_identical(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE,
                              {"config.json": '{"alpha": 0.095}'}))
    b = fingerprint_dir(_repo(tmp_path, "b", BASE_SOURCE,
                              {"config.json": '{"alpha": 0.10}'}))
    # A config sweep must never collapse into the identical tiers.
    assert a.token_sha256 != b.token_sha256
    assert a.masked_sha256 != b.masked_sha256
    assert a.py_sha256 == b.py_sha256  # …the config_only tier claims it instead


def test_different_requirements_are_not_rename_identical(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE,
                              {"requirements.txt": "numpy\n"}))
    b = fingerprint_dir(_repo(tmp_path, "b", BASE_SOURCE,
                              {"requirements.txt": "torch\n"}))
    # Requirement pins are data, not identifiers — the masked digest must not
    # erase a dependency swap (deps can carry generator logic).
    assert a.masked_sha256 != b.masked_sha256


# ── cache/VCS junk exclusion ─────────────────────────────────────────────────

def test_cache_junk_does_not_change_the_tree_digest(tmp_path):
    a_dir = _repo(tmp_path, "a", BASE_SOURCE)
    b_dir = _repo(tmp_path, "b", BASE_SOURCE)
    # Hub-cache metadata churns per upload: content hashes + timestamps.
    cache = b_dir / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "generator.py.metadata").write_text("etag: abc\ntimestamp: 1e9\n")
    (cache / "state.lock").write_text("")
    (b_dir / ".gitattributes").write_text("*.bin filter=lfs\n")
    (b_dir / "__pycache__").mkdir()
    (b_dir / "__pycache__" / "generator.cpython-311.pyc").write_bytes(b"\x00junk")

    a, b = fingerprint_dir(a_dir), fingerprint_dir(b_dir)
    assert a.tree_sha256 == b.tree_sha256
    assert a.token_sha256 == b.token_sha256


# ── absolute token-delta floor ───────────────────────────────────────────────

# ── config_only tier ─────────────────────────────────────────────────────────

def test_config_only_shadow_label_by_default(tmp_path):
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE, {"config.json": '{"alpha": 0.095}'}),
        _entry(tmp_path, "swp", 2, BASE_SOURCE, {"config.json": '{"alpha": 0.10}'}),
    ]
    result = screen_duplicates(entries, None)
    # Identical code + differing configs: kept, labeled, with the absolute
    # config delta recorded so the log separates a version-string A/B from a
    # real re-parameterization.
    assert result.kept_hotkeys == ("orig", "swp")
    assert not result.dropped
    (v,) = [s for s in result.shadow if s.tier == "config_only"]
    assert v.hotkey == "swp" and v.matched_hotkey == "orig"
    assert v.abs_delta is not None and v.abs_delta >= 1


def test_config_only_large_rewrite_survives_with_label(tmp_path):
    # A config rewrite big enough to fall under the ratio bar: identical code,
    # genuinely different parameterization — kept, with the label recorded.
    big_a = json.dumps({f"w{i}": i * 0.095 for i in range(400)})
    big_b = json.dumps({f"w{i}": (i % 7) * 1.31 + 40 for i in range(400)})
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE, {"config.json": big_a}),
        _entry(tmp_path, "rew", 2, BASE_SOURCE, {"config.json": big_b}),
    ]
    result = screen_duplicates(entries, None)
    assert result.kept_hotkeys == ("orig", "rew")
    assert not result.dropped
    assert [s.tier for s in result.shadow if s.hotkey == "rew"].count("config_only") == 1


def test_config_only_enforced_drops(tmp_path):
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE, {"config.json": '{"alpha": 0.095}'}),
        _entry(tmp_path, "swp", 2, BASE_SOURCE, {"config.json": '{"alpha": 0.10}'}),
    ]
    result = screen_duplicates(entries, None, config_only_enforce=True)
    assert result.kept_hotkeys == ("orig",)
    (v,) = result.dropped
    assert v.tier == "config_only" and v.hotkey == "swp"


# ── behavioral probe ─────────────────────────────────────────────────────────

def test_collapse_identical_behavior_pure():
    from cascade.interface.dedup import collapse_identical_behavior

    kept, dropped = collapse_identical_behavior(
        [("carol", 9, "dg1"), ("alice", 3, "dg1"), ("bob", 5, "dg2"),
         ("kcopy", 7, "KING")],
        king_digest="KING",
    )
    assert kept == ("alice", "bob")
    assert {(v.hotkey, v.matched_hotkey, v.tier) for v in dropped} == {
        ("carol", "alice", "behavior_identical"),
        ("kcopy", "king", "behavior_identical"),
    }


@pytest.fixture
def probe_runner(dedup_runner, monkeypatch):
    """dedup_runner with the probe on and build_round_corpus faked: the digest
    comes from a BEHAVIOR file in the repo, and a NONDET file makes every draw
    unique (a generator seeding from entropy)."""
    from dataclasses import replace
    from itertools import count

    from cascade.trainer import loop as loop_mod
    from cascade.trainer.corpus import CorpusError

    runner, add = dedup_runner
    # build_round_corpus is faked below, but the probe gate cannot know that:
    # use_sandbox=False means "run generator code in this process", the weakest
    # posture there is, so the fixture has to accept it out loud.
    runner.cfg = replace(runner.cfg,
                         round=replace(runner.cfg.round, dedup_probe_series=4,
                                       dedup_probe_mode="enforce",
                                       dedup_probe_allow_weak_sandbox=True))
    ticks = count()

    class _Result:
        def __init__(self, digest):
            self.digest = digest

    captured_cfgs = []

    def fake_build(repo_dir, seed, cfg, mode, *, use_sandbox=True, blocked=(),
                   allow_netns=True):
        captured_cfgs.append(cfg)
        repo = repo_dir
        if (repo / "BROKEN").exists():
            raise CorpusError("generator_import_failed")
        if (repo / "INFRA").exists():
            raise CorpusError("sandbox_crashed (rc=137): oom or missing image")
        if (repo / "NONDET").exists():
            return _Result(f"entropy-{next(ticks)}")
        marker = repo / "BEHAVIOR"
        return _Result(marker.read_text() if marker.exists() else f"seeded-{seed}-{repo.name}")

    add.captured_cfgs = captured_cfgs
    monkeypatch.setattr(loop_mod, "build_round_corpus", fake_build)
    return runner, add


def test_probe_drops_nondeterministic_generator(probe_runner, tmp_path):
    runner, add = probe_runner
    ok = add("alice", 3, BASE_SOURCE)
    bad = add("edgar", 5, OTHER_SOURCE)
    (tmp_path / "repos" / "edgar" / "NONDET").write_text("")

    kept = runner._screen_duplicate_entrants(None, [ok, bad], base_seed=80)
    assert [c.hotkey for c in kept] == ["alice"]
    report = json.loads((runner.work_root / "80" / "dedup_report.json").read_text())
    assert report["probe_dropped"][0]["hotkey"] == "edgar"
    assert report["probe_dropped"][0]["tier"] == "nondeterministic"


def test_probe_infra_fault_is_never_the_miners_problem(probe_runner, tmp_path):
    # A sandbox_crashed / missing-image CorpusError is OUR infrastructure, not
    # a generator verdict — under probe enforce it must keep the entrant, or a
    # bad image on the orchestrator burns the whole field.
    runner, add = probe_runner
    ok = add("alice", 3, BASE_SOURCE)
    unlucky = add("edgar", 5, OTHER_SOURCE)
    (tmp_path / "repos" / "edgar" / "INFRA").write_text("")

    kept = runner._screen_duplicate_entrants(None, [ok, unlucky], base_seed=82)
    assert sorted(c.hotkey for c in kept) == ["alice", "edgar"]
    report = json.loads((runner.work_root / "82" / "dedup_report.json").read_text())
    assert report["probe_dropped"] == []


def test_probe_collapses_same_behavior_different_code(probe_runner, tmp_path):
    runner, add = probe_runner
    a = add("alice", 3, BASE_SOURCE)
    b = add("bobby", 6, OTHER_SOURCE)  # different code (sim < 0.99) …
    (tmp_path / "repos" / "alice" / "BEHAVIOR").write_text("same-bytes")
    (tmp_path / "repos" / "bobby" / "BEHAVIOR").write_text("same-bytes")

    kept = runner._screen_duplicate_entrants(None, [a, b], base_seed=81)
    assert [c.hotkey for c in kept] == ["alice"]  # … but the same process
    report = json.loads((runner.work_root / "81" / "dedup_report.json").read_text())
    behav = [d for d in report["dropped"] if d["tier"] == "behavior_identical"]
    assert behav and behav[0]["hotkey"] == "bobby" and behav[0]["matched_hotkey"] == "alice"


def test_probe_matches_king_behavior(probe_runner, tmp_path):
    runner, add = probe_runner
    king = add("king", 0, BASE_SOURCE)
    c = add("carla", 4, OTHER_SOURCE)
    (tmp_path / "repos" / "king" / "BEHAVIOR").write_text("royal-bytes")
    (tmp_path / "repos" / "carla" / "BEHAVIOR").write_text("royal-bytes")

    kept = runner._screen_duplicate_entrants(king, [c], base_seed=82)
    assert kept == []


def test_probe_uses_its_own_wall_clock(probe_runner, tmp_path):
    from dataclasses import replace

    runner, add = probe_runner
    runner.cfg = replace(runner.cfg, round=replace(
        runner.cfg.round, dedup_probe_generate_seconds=90))
    a = add("alice", 3, BASE_SOURCE)
    runner._screen_duplicate_entrants(None, [a], base_seed=86)

    # Every probe draw runs under the probe's OWN generation deadline — never
    # the full-corpus 1800s budget (execution is paid even in shadow mode, so
    # a hostile generator must not be able to stall the orchestrator).
    assert add.captured_cfgs, "probe never ran"
    assert all(c.max_generate_seconds == 90 for c in add.captured_cfgs)
    assert all(c.corpus_n_series == 4 for c in add.captured_cfgs)


def test_probe_failure_drops_in_enforce(probe_runner, tmp_path):
    runner, add = probe_runner
    ok = add("alice", 3, BASE_SOURCE)
    broken = add("brock", 5, OTHER_SOURCE)
    (tmp_path / "repos" / "brock" / "BROKEN").write_text("")

    kept = runner._screen_duplicate_entrants(None, [ok, broken], base_seed=83)
    assert [c.hotkey for c in kept] == ["alice"]


def test_probe_shadow_mode_drops_nothing(probe_runner, tmp_path):
    from dataclasses import replace

    runner, add = probe_runner
    runner.cfg = replace(runner.cfg,
                         round=replace(runner.cfg.round, dedup_probe_mode="shadow"))
    a = add("alice", 3, BASE_SOURCE)
    b = add("bobby", 6, OTHER_SOURCE)
    (tmp_path / "repos" / "alice" / "BEHAVIOR").write_text("same-bytes")
    (tmp_path / "repos" / "bobby" / "BEHAVIOR").write_text("same-bytes")

    # Static tiers still ENFORCE; the probe observes: the behavior_identical
    # verdict is logged in the report but bobby keeps its heat slot.
    kept = runner._screen_duplicate_entrants(None, [a, b], base_seed=84)
    assert [c.hotkey for c in kept] == ["alice", "bobby"]
    report = json.loads((runner.work_root / "84" / "dedup_report.json").read_text())
    behav = [d for d in report["dropped"] if d["tier"] == "behavior_identical"]
    assert behav and behav[0]["enforced"] is False


@pytest.mark.parametrize("dedup_mode", ["off", "shadow", "enforce"])
@pytest.mark.parametrize("probe_mode", ["off", "shadow", "enforce"])
def test_mode_matrix_static_and_probe_gate_independently(
        probe_runner, tmp_path, dedup_mode, probe_mode):
    """All 9 (dedup_mode × dedup_probe_mode) combinations: static drops apply
    iff dedup_mode == enforce, probe drops iff dedup_probe_mode == enforce,
    and dedup_mode == off disables the whole screen."""
    from dataclasses import replace

    runner, add = probe_runner
    runner.cfg = replace(runner.cfg, round=replace(
        runner.cfg.round, dedup_mode=dedup_mode, dedup_probe_mode=probe_mode))
    alice = add("alice", 3, BASE_SOURCE)
    copy = add("copyc", 7, "# reupload\n" + BASE_SOURCE)   # token-identical → static tier
    nondet = add("nondt", 9, OTHER_SOURCE)                 # entropy-seeded → probe tier
    (tmp_path / "repos" / "nondt" / "NONDET").write_text("")

    kept = [c.hotkey for c in runner._screen_duplicate_entrants(
        None, [alice, copy, nondet], base_seed=90)]

    expected = {"alice", "copyc", "nondt"}
    if dedup_mode == "enforce":
        expected -= {"copyc"}
    if dedup_mode != "off" and probe_mode == "enforce":
        expected -= {"nondt"}
    assert set(kept) == expected


# ── cost caps: the screen must not become the cheapest way to stall a round ──

def _big_repo(tmp_path, name, nlines, extra=""):
    """A repo whose token stream is far past any sane cap."""
    src = "def build():\n" + "".join(
        f"    x_{i} = compute(a_{i}, b_{i}) + 12345  # pad\n" for i in range(nlines))
    return _repo(tmp_path, name, src + extra)


def test_oversize_stream_is_not_scoreable_and_keeps_exact_digests(tmp_path):
    # difflib is O(n²) in tokens, so a stream past the cap must not reach it —
    # but the digests are fed incrementally and stay exact regardless.
    a = fingerprint_dir(_big_repo(tmp_path, "a", 2000), max_tokens=500)
    b = fingerprint_dir(_big_repo(tmp_path, "b", 2000), max_tokens=500)
    c = fingerprint_dir(_big_repo(tmp_path, "c", 2000, extra="    y = 1\n"),
                        max_tokens=500)
    assert not a.scoreable and a.tokens == ()      # nothing retained
    assert a.n_tokens > 500                        # but the count is real
    assert a.token_sha256 == b.token_sha256        # identical trees still collide
    assert a.token_sha256 != c.token_sha256        # a real edit still differs


def test_padding_past_the_cap_does_not_evade_the_exact_tiers(tmp_path):
    # Digests are streamed, never taken from the retained prefix — so a copy
    # inflated past the token cap still collides on the token tier, while a
    # padded repo with a REAL edit is kept (policy: no similarity threshold;
    # an edit is a different program).
    orig = ("orig", 1, fingerprint_dir(_big_repo(tmp_path, "orig", 2000),
                                       max_tokens=500))
    copy = ("copy", 2, fingerprint_dir(          # comment shuffle: new tree bytes
        _big_repo(tmp_path, "copy", 2000, extra="# reupload comment\n"),
        max_tokens=500))
    edit = ("edit", 3, fingerprint_dir(
        _big_repo(tmp_path, "edit", 2000, extra="    y = 1\n"), max_tokens=500))
    result = screen_duplicates([orig, copy, edit], None)
    assert result.kept_hotkeys == ("orig", "edit")
    (v,) = result.dropped
    assert v.hotkey == "copy" and v.tier == "token_identical"


def test_undecoded_files_enter_the_digests_as_opaque_content(tmp_path):
    # Past the decode budget the tokenizer is skipped — but a file's content
    # must still reach the digests, or two repos differing only inside a large
    # file would read as token_identical and one would be dropped as a copy.
    big = "x = [" + ", ".join(str(i) for i in range(200_000)) + "]\n"
    a = _repo(tmp_path, "a", BASE_SOURCE, extra={"data.py": big})
    b = _repo(tmp_path, "b", BASE_SOURCE, extra={"data.py": big + "y = 1\n"})
    fa = fingerprint_dir(a, max_text_mb=0)     # 0 ⇒ decode nothing
    fb = fingerprint_dir(b, max_text_mb=0)
    assert fa.token_sha256 != fb.token_sha256
    assert fa.tree_sha256 != fb.tree_sha256
    # …and with both budgets high enough, the same repo tokenizes normally
    assert fingerprint_dir(a, max_text_mb=64, max_tokens=0).scoreable


# ── shadow mode is a true counterfactual of enforce ──────────────────────────

def test_shadow_verdicts_name_the_rival_enforce_would_have_used(tmp_path):
    # A dropped entry must not become a rival for later entries: otherwise the
    # shadow log measures a chain enforce would never produce, and it is the
    # log that would calibrate any future enforcement.
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE),
        _entry(tmp_path, "cop2", 2, "# a\n" + BASE_SOURCE),
        _entry(tmp_path, "cop3", 3, "# b\n" + BASE_SOURCE),
    ]
    shadow = screen_duplicates(entries, None, enforce=False)
    enforced = screen_duplicates(entries, None, enforce=True)
    assert shadow.kept_hotkeys == ("orig", "cop2", "cop3")   # nothing drops
    assert enforced.kept_hotkeys == ("orig",)
    # same verdicts, same rivals, in both modes
    assert ([(v.hotkey, v.matched_hotkey, v.tier) for v in shadow.dropped]
            == [(v.hotkey, v.matched_hotkey, v.tier) for v in enforced.dropped]
            == [("cop2", "orig", "token_identical"),
                ("cop3", "orig", "token_identical")])


def test_config_only_needs_actual_python(tmp_path):
    # Two repos with no .py at all share an empty code digest; collapsing them
    # on that basis would be a false positive.
    a = ("a", 1, fingerprint_dir(_repo_no_py(tmp_path, "a", '{"lr": 1}')))
    b = ("b", 2, fingerprint_dir(_repo_no_py(tmp_path, "b", '{"lr": 2}')))
    result = screen_duplicates([a, b], None, config_only_enforce=True)
    assert result.kept_hotkeys == ("a", "b")
    assert not [v for v in result.dropped + result.shadow if v.tier == "config_only"]


def _repo_no_py(tmp_path, name, config):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(config)
    return d


# ── trainer wiring: the screen's own cost and blast radius ───────────────────

def test_runner_skips_oversize_repos_without_fingerprinting(dedup_runner, monkeypatch):
    # A repo over [generator] max_repo_mb fails its heat run anyway, so the
    # screen must not spend fingerprint time on input the miner chooses.
    from dataclasses import replace

    from cascade.interface import dedup as dedup_mod

    runner, add = dedup_runner
    runner.cfg = replace(runner.cfg,
                         generator=replace(runner.cfg.generator, max_repo_mb=0))
    seen = []
    real = dedup_mod.fingerprint_dir
    monkeypatch.setattr(dedup_mod, "fingerprint_dir",
                        lambda d, **kw: seen.append(d) or real(d, **kw))

    alice = add("alice", 3, BASE_SOURCE)
    copy = add("mallory", 9, BASE_SOURCE)
    kept = runner._screen_duplicate_entrants(None, [alice, copy], base_seed=91)

    assert [c.hotkey for c in kept] == ["alice", "mallory"]   # kept, not dropped
    assert seen == []                                         # and never hashed
    report = json.loads((runner.work_root / "91" / "dedup_report.json").read_text())
    assert [o["hotkey"] for o in report["oversize"]] == ["alice", "mallory"]


def test_runner_screen_deadline_fails_open(dedup_runner, monkeypatch):
    # The screen's inputs are attacker-chosen (repo bytes, generator runtime),
    # so it runs under a wall clock — and an expired screen must not sink the
    # round it exists to protect.
    from cascade.trainer.loop import TrainerRunner

    runner, add = dedup_runner
    budget = []

    def boom(fn, seconds):
        budget.append(seconds)
        raise TimeoutError("screen overran")

    monkeypatch.setattr(TrainerRunner, "_with_deadline", staticmethod(boom))
    alice = add("alice", 3, BASE_SOURCE)
    copy = add("mallory", 9, BASE_SOURCE)          # would drop if the screen ran

    kept = runner._screen_duplicate_entrants(None, [alice, copy], base_seed=92)
    assert [c.hotkey for c in kept] == ["alice", "mallory"]
    assert budget == [runner.cfg.round.dedup_phase_seconds]


def test_probe_refuses_a_weak_sandbox(probe_runner, tmp_path):
    # The probe executes untrusted generator code on the box that holds the
    # eval pool and the wallet. Shadow mode gates the DROPS, not the execution,
    # so a soft sandbox must disable the probe outright.
    from dataclasses import replace

    runner, add = probe_runner
    runner.use_sandbox = True         # the real path, not the in-process one
    runner.cfg = replace(
        runner.cfg,
        generator=replace(runner.cfg.generator, sandbox_mode="subprocess",
                          sandbox_strict=False),
        round=replace(runner.cfg.round, dedup_probe_allow_weak_sandbox=False))
    ok = add("alice", 3, BASE_SOURCE)
    bad = add("edgar", 5, OTHER_SOURCE)
    (tmp_path / "repos" / "edgar" / "NONDET").write_text("")

    kept = runner._screen_duplicate_entrants(None, [ok, bad], base_seed=93)
    assert [c.hotkey for c in kept] == ["alice", "edgar"]   # probe never ran
    report = json.loads((runner.work_root / "93" / "dedup_report.json").read_text())
    assert report["probe_mode"] == "off"
    assert report["probe_dropped"] == []
    assert add.captured_cfgs == []


@pytest.mark.parametrize("posture", [
    {"sandbox_mode": "container"},                    # kernel-enforced
    {"sandbox_mode": "subprocess", "sandbox_strict": True},   # netns or refuse
])
def test_probe_runs_under_hard_isolation(probe_runner, tmp_path, posture):
    from dataclasses import replace

    runner, add = probe_runner
    runner.use_sandbox = True
    runner.cfg = replace(runner.cfg,
                         generator=replace(runner.cfg.generator, **posture))
    ok = add("alice", 3, BASE_SOURCE)
    bad = add("edgar", 5, OTHER_SOURCE)
    (tmp_path / "repos" / "edgar" / "NONDET").write_text("")

    kept = runner._screen_duplicate_entrants(None, [ok, bad], base_seed=94)
    assert [c.hotkey for c in kept] == ["alice"]


def test_probe_weak_sandbox_opt_in_is_explicit(probe_runner, tmp_path):
    # Testnet's escape hatch: run the probe on a soft box, deliberately.
    from dataclasses import replace

    runner, add = probe_runner
    runner.use_sandbox = True
    runner.cfg = replace(
        runner.cfg,
        generator=replace(runner.cfg.generator, sandbox_mode="subprocess",
                          sandbox_strict=False),
        round=replace(runner.cfg.round, dedup_probe_allow_weak_sandbox=True))
    ok = add("alice", 3, BASE_SOURCE)
    bad = add("edgar", 5, OTHER_SOURCE)
    (tmp_path / "repos" / "edgar" / "NONDET").write_text("")

    kept = runner._screen_duplicate_entrants(None, [ok, bad], base_seed=95)
    assert [c.hotkey for c in kept] == ["alice"]


def test_probe_per_draw_clock_is_derived_from_the_stage_budget(probe_runner):
    # N hostile entrants each burning the per-draw clock must not stretch the
    # stage without bound: the share shrinks with the number of waves.
    from dataclasses import replace

    from cascade.trainer.loop import _PROBE_WORKERS

    runner, add = probe_runner
    runner.cfg = replace(runner.cfg, round=replace(
        runner.cfg.round, dedup_probe_generate_seconds=120,
        dedup_probe_budget_seconds=240))
    # structurally distinct bodies: all 8 must SURVIVE the static tiers, or the
    # wave count (and so the derived clock) would not be what this asserts
    entrants = [add(f"m{i:03d}", i,
                    "\n".join(f"z{i}_{j} = {j * (i + 1)} ** {i + 2} + {j % (i + 3)}"
                              for j in range(40 + 25 * i)))
                for i in range(_PROBE_WORKERS * 2)]     # 2 waves → 240/(2*2) = 60s

    runner._screen_duplicate_entrants(None, entrants, base_seed=96)
    # 240s / (2 waves * 2 draws) = 60s share, minus the 30s subprocess spawn
    # grace the sandbox spends ON TOP of max_generate_seconds — the stage must
    # fund it or it overruns its own budget and trips the phase deadline.
    assert {c.max_generate_seconds for c in add.captured_cfgs} == {30}
    report = json.loads((runner.work_root / "96" / "dedup_report.json").read_text())
    assert report["probe_seconds_per_draw"] == 30


def test_probe_skips_when_the_budget_cannot_fund_the_floor(probe_runner):
    # A field too large for the stage budget must cost the PROBE, never the
    # screen: overrunning would trip the outer dedup_phase_seconds deadline
    # and discard the static verdicts with it.
    from dataclasses import replace

    from cascade.trainer.loop import _PROBE_WORKERS

    runner, add = probe_runner
    runner.cfg = replace(runner.cfg, round=replace(
        runner.cfg.round, dedup_probe_generate_seconds=120,
        dedup_probe_budget_seconds=100))   # 100/(2*2) - 30 < 30s floor
    entrants = [add(f"m{i:03d}", i,
                    "\n".join(f"z{i}_{j} = {j * (i + 1)} ** {i + 2} + {j % (i + 3)}"
                              for j in range(40 + 25 * i)))
                for i in range(_PROBE_WORKERS * 2)]

    kept = runner._screen_duplicate_entrants(None, entrants, base_seed=97)
    assert not add.captured_cfgs           # no sandbox was ever spawned
    assert len(kept) == len(entrants)      # static tiers judged, probe skipped
    report = json.loads((runner.work_root / "97" / "dedup_report.json").read_text())
    assert report["probe_seconds_per_draw"] == 0


def test_decode_budget_keeps_the_digests_exact(tmp_path):
    # Past the decode budget the tokenizer stops, but every file's content
    # still reaches the digests as an opaque hash — so an exact copy still
    # collides, and an edited copy still differs, however large the repo.
    body = "\n".join(f"    q{i} = weigh(p{i}, {i}) * 3.5" for i in range(4000))
    orig = _repo(tmp_path, "orig", "def run():\n" + body)
    copy = _repo(tmp_path, "copy", "def run():\n" + body)
    edit = _repo(tmp_path, "edit", "def run():\n" + body + "\n    extra = 1\n")
    fo = fingerprint_dir(orig, max_text_mb=0.05, max_tokens=500)
    fc = fingerprint_dir(copy, max_text_mb=0.05, max_tokens=500)
    fe = fingerprint_dir(edit, max_text_mb=0.05, max_tokens=500)

    assert fo.truncated and not fo.scoreable      # the file was cut short…
    assert fo.token_sha256 == fc.token_sha256     # …an exact copy still collides
    assert fo.token_sha256 != fe.token_sha256     # …and a real edit still differs
    result = screen_duplicates(
        [("orig", 1, fo), ("copy", 2, fc), ("edit", 3, fe)], None)
    assert result.kept_hotkeys == ("orig", "edit")
    assert result.dropped[0].tier in ("tree_identical", "token_identical")


def test_over_cap_entrants_are_named_in_the_report(dedup_runner):
    # Honest submissions run ~100KB; bulk past the token cap (exact digest
    # tiers only) is a pattern worth being able to see build up over rounds.
    from dataclasses import replace

    runner, add = dedup_runner
    runner.cfg = replace(runner.cfg,
                         round=replace(runner.cfg.round, dedup_max_tokens=100))
    alice = add("alice", 3, BASE_SOURCE)

    runner._screen_duplicate_entrants(None, [alice], base_seed=97)
    report = json.loads((runner.work_root / "97" / "dedup_report.json").read_text())
    assert report["over_token_cap"][0]["hotkey"] == "alice"
    assert report["over_token_cap"][0]["n_tokens"] > 100


# ── commit order: who submitted a generator FIRST ────────────────────────────

class _WitnessClient:
    """A chain client whose sealed-commit view the test drives directly."""

    def __init__(self, sealed=None):
        self.sealed = dict(sealed or {})

    def poll_pending_commits(self):
        return dict(self.sealed)


def test_witness_freezes_the_commit_block_when_the_seal_disappears(dedup_runner):
    # The chain deletes a commit's block at reveal, so the witness has to catch
    # it while sealed and hold it: seen-sealed → pending, gone → committed.
    runner, _ = dedup_runner
    client = _WitnessClient({"alice": 100, "bob": 140})

    w = runner.witness_commits(client)
    assert w["alice"] == {"pending": 100, "committed": None}

    del client.sealed["alice"]                     # alice's reveal lands
    w = runner.witness_commits(client)
    assert w["alice"] == {"pending": None, "committed": 100}
    assert w["bob"] == {"pending": 140, "committed": None}   # bob still sealed


def test_witness_survives_a_restart(dedup_runner, tmp_path):
    # The record only exists on our disk; losing it on restart would silently
    # drop the whole round back to reveal order.
    from cascade.trainer.loop import TrainerRunner

    runner, _ = dedup_runner
    client = _WitnessClient({"alice": 100})
    runner.witness_commits(client)
    client.sealed.clear()
    runner.witness_commits(client)

    fresh = TrainerRunner(cfg=runner.cfg, base_trainer=object(),
                          work_root=runner.work_root, use_sandbox=False)
    assert fresh.witness_commits(_WitnessClient())["alice"]["committed"] == 100


def test_a_new_seal_replaces_the_previous_pending_commit(dedup_runner):
    # Re-committing claims a NEW submission; the old commit's priority must not
    # follow the hotkey onto it.
    runner, _ = dedup_runner
    client = _WitnessClient({"alice": 100})
    runner.witness_commits(client)
    client.sealed["alice"] = 500                   # re-commit, still sealed
    w = runner.witness_commits(client)
    assert w["alice"]["pending"] == 500


def test_witness_tolerates_a_client_without_the_read(dedup_runner):
    runner, _ = dedup_runner
    assert runner.witness_commits(object()) == {}


def test_commit_priority_falls_back_to_reveal_block(dedup_runner):
    # An entrant we merely failed to observe must not be expropriated for it:
    # reveal order is the same units and still puts a copyist behind the repo
    # they copied (copying a revealed repo means revealing after it).
    runner, add = dedup_runner
    seen = add("alice", 3, BASE_SOURCE, reveal_block=900)
    unseen = add("bobby", 4, OTHER_SOURCE, reveal_block=880)
    runner.witness_commits(_WitnessClient({"alice": 100}))
    runner.witness_commits(_WitnessClient())       # alice reveals

    prio = runner._commit_priority([seen, unseen])
    assert prio == {"alice": 100, "bobby": 880}


def test_earliest_commit_keeps_the_slot_over_a_lower_uid(tmp_path):
    # The expropriation this exists to stop: Bittensor recycles deregistered
    # UIDs, so a newcomer can hold a LOWER uid than the miner they copy. Order
    # by when each was submitted, not by which slot they inherited.
    entries = [
        _entry(tmp_path, "victim", 200, BASE_SOURCE),        # high uid, committed first
        _entry(tmp_path, "copyist", 12, "# mine now\n" + BASE_SOURCE),
    ]
    by_uid = screen_duplicates(entries, None)
    assert by_uid.kept_hotkeys == ("copyist",)               # the old rule

    by_commit = screen_duplicates(entries, None,
                                  priority={"victim": 100, "copyist": 400})
    assert by_commit.kept_hotkeys == ("victim",)
    (v,) = by_commit.dropped
    assert v.hotkey == "copyist" and v.matched_hotkey == "victim"


def test_uid_still_breaks_a_tie_at_the_same_block(tmp_path):
    entries = [
        _entry(tmp_path, "aaa", 9, BASE_SOURCE),
        _entry(tmp_path, "bbb", 4, "# copy\n" + BASE_SOURCE),
    ]
    result = screen_duplicates(entries, None, priority={"aaa": 100, "bbb": 100})
    assert result.kept_hotkeys == ("bbb",)


def test_runner_screen_orders_on_the_witnessed_commit(dedup_runner):
    # End to end: the copyist holds the lower UID, but committed later.
    runner, add = dedup_runner
    victim = add("victim", 200, BASE_SOURCE, reveal_block=980)
    copyist = add("copyst", 12, "# resubmit\n" + BASE_SOURCE, reveal_block=995)
    runner.witness_commits(_WitnessClient({"victim": 100, "copyst": 400}))
    runner.witness_commits(_WitnessClient())       # both reveal

    kept = runner._screen_duplicate_entrants(None, [victim, copyist], base_seed=98)
    assert [c.hotkey for c in kept] == ["victim"]
    report = json.loads((runner.work_root / "98" / "dedup_report.json").read_text())
    assert report["dropped"][0]["hotkey"] == "copyst"


# ── audit regressions ────────────────────────────────────────────────────────

def test_repos_with_no_functional_content_do_not_collide(tmp_path):
    # Every repo with no .py and no config shares the empty-stream digest, so
    # the token tiers would call two unrelated junk repos copies of each other
    # — and the loser burns its one lifetime submission on a false verdict.
    def _docs(name, text):
        d = tmp_path / name
        d.mkdir()
        (d / "README.md").write_text(text)
        return fingerprint_dir(d)

    a, b = _docs("a", "one thing\n"), _docs("b", "a totally different thing\n")
    assert a.token_sha256 == b.token_sha256      # the collision is real…
    result = screen_duplicates([("a", 1, a), ("b", 2, b)], None)
    assert result.kept_hotkeys == ("a", "b")     # …and must not be a verdict
    assert not result.dropped

    # byte-identical junk is still a genuine duplicate (the tree tier)
    c, d = _docs("c", "same\n"), _docs("d", "same\n")
    assert screen_duplicates([("c", 1, c), ("d", 2, d)],
                             None).dropped[0].tier == "tree_identical"


def test_unknown_submission_block_sorts_last_not_first(dedup_runner):
    # Mapping "we know nothing" to block 0 would sort it FIRST and let that
    # entrant win every collision it appears in.
    runner, add = dedup_runner
    known = add("alice", 200, BASE_SOURCE, reveal_block=900)
    nothing = add("ghost", 12, "# copy\n" + BASE_SOURCE)    # no reveal block

    prio = runner._commit_priority([known, nothing])
    assert prio == {"alice": 900}                            # ghost omitted
    kept = runner._screen_duplicate_entrants(None, [known, nothing], base_seed=99)
    assert [c.hotkey for c in kept] == ["alice"]


def test_witness_does_not_freeze_commits_on_a_failed_read(dedup_runner):
    # "I could not look" is not evidence that every pending commit revealed.
    runner, _ = dedup_runner

    class _Failing:
        def poll_pending_commits(self):
            return None          # read failed, per ChainClient's contract

    class _Raising:
        def poll_pending_commits(self):
            raise RuntimeError("websocket died")

    runner.witness_commits(_WitnessClient({"alice": 100}))
    for client in (_Failing(), _Raising()):
        w = runner.witness_commits(client)
        assert w["alice"] == {"pending": 100, "committed": None}, client
