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
    similarity,
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
    assert similarity(a, b) == 1.0


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

def test_near_duplicate_dropped_lowest_uid_kept(tmp_path):
    tweaked = BASE_SOURCE.replace("size=64", "size=65", 1)
    entries = [
        _entry(tmp_path, "orig", 10, BASE_SOURCE),
        _entry(tmp_path, "copy", 42, tweaked),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90)
    assert result.kept_hotkeys == ("orig",)
    (v,) = result.dropped
    assert v.hotkey == "copy" and v.matched_hotkey == "orig"
    assert v.tier == "near_duplicate" and v.score >= 0.99


def test_copy_of_king_dropped(tmp_path):
    king = fingerprint_dir(_repo(tmp_path, "king", BASE_SOURCE))
    entries = [_entry(tmp_path, "c", 7, "# defend the throne\n" + BASE_SOURCE)]
    result = screen_duplicates(entries, king, threshold=0.99, shadow_floor=0.90)
    assert result.kept_hotkeys == ()
    (v,) = result.dropped
    assert v.matched_uid == KING_UID and v.tier == "token_identical"


def test_template_band_is_shadow_logged_not_dropped(tmp_path):
    # Rewrite ~5% of the weight lines: same template, genuinely different data
    # process — must land in [floor, threshold) and survive.
    variant = BASE_SOURCE
    for i in range(0, 120, 17):
        variant = variant.replace(f"* 0.5 + {i % 7}", f"* 1.5 - {i % 5}")
        variant = variant.replace(f"rng.normal(0.0, 1.0, size=64) * self.w{i}",
                                  f"rng.laplace(0.0, 2.0, size=32) * self.w{i}")
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE),
        _entry(tmp_path, "variant", 2, variant),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90)
    assert result.kept_hotkeys == ("orig", "variant")
    assert not result.dropped
    (s,) = result.shadow
    assert s.hotkey == "variant" and 0.90 <= s.score < 0.99


def test_distinct_generators_kept_silently(tmp_path):
    other = "\n".join(f"value_{i} = {i} ** 2" for i in range(200))
    entries = [
        _entry(tmp_path, "a", 1, BASE_SOURCE),
        _entry(tmp_path, "b", 2, other),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90)
    assert result.kept_hotkeys == ("a", "b")
    assert not result.dropped and not result.shadow


def test_no_transitive_merging(tmp_path):
    # b is a near-copy of a (dropped). c sits in the shadow band vs a. c must
    # be judged against KEPT entries only — never chained through b.
    near = BASE_SOURCE.replace("size=64", "size=63", 1)
    variant = BASE_SOURCE
    for i in range(0, 120, 17):
        variant = variant.replace(f"* 0.5 + {i % 7}", f"* 2.5 - {i % 3}")
        variant = variant.replace(f"rng.normal(0.0, 1.0, size=64) * self.w{i}",
                                  f"rng.gumbel(1.0, 3.0, size=16) * self.w{i}")
    entries = [
        _entry(tmp_path, "a", 1, BASE_SOURCE),
        _entry(tmp_path, "b", 2, near),
        _entry(tmp_path, "c", 3, variant),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90)
    assert result.kept_hotkeys == ("a", "c")
    assert [v.hotkey for v in result.dropped] == ["b"]


def test_shadow_mode_drops_nothing(tmp_path):
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE),
        _entry(tmp_path, "copy", 2, BASE_SOURCE),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90,
                               enforce=False)
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

    def add(name, uid, source):
        ref = f"{name}/gen@sha256:{name[0] * 64}"
        repos[ref] = _repo(tmp_path / "repos", name, source)
        return ResolvedGenerator(hotkey=name, uid=uid, ref=ref)

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
                                           dedup_threshold=0.99,
                                           dedup_shadow_floor=0.90,
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


def test_config_value_delta_is_a_real_delta_not_identical(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE,
                              {"config.json": '{"alpha": 0.095}'}))
    b = fingerprint_dir(_repo(tmp_path, "b", BASE_SOURCE,
                              {"config.json": '{"alpha": 0.10}'}))
    # A config sweep must never collapse into the identical tiers …
    assert a.token_sha256 != b.token_sha256
    assert a.masked_sha256 != b.masked_sha256
    # … it is measured like any code delta (tiny sweep ⇒ near-duplicate tier).
    assert similarity(a, b) >= 0.99


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

def test_abs_delta_floor_boundary(tmp_path):
    # Three single-token edits: sim stays >= 0.99, absolute delta is exactly 3.
    tweaked = BASE_SOURCE.replace("size=64", "size=65", 3)
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE),
        _entry(tmp_path, "twk", 2, tweaked),
    ]
    # Cap below the delta: over the ratio bar but a "large" edit — shadow-logged.
    spared = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90,
                               max_abs_delta=2)
    assert spared.kept_hotkeys == ("orig", "twk")
    (s,) = [v for v in spared.shadow if v.tier == "near_duplicate_large_delta"]
    assert s.hotkey == "twk" and s.abs_delta == 3 and s.score >= 0.99
    # Cap at the delta: dropped as before.
    at_cap = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90,
                               max_abs_delta=3)
    assert at_cap.kept_hotkeys == ("orig",)
    assert at_cap.dropped[0].tier == "near_duplicate"
    assert at_cap.dropped[0].abs_delta == 3
    # Cap disabled (0): current behavior unchanged.
    disabled = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90)
    assert disabled.kept_hotkeys == ("orig",)


# ── config_only tier ─────────────────────────────────────────────────────────

def test_config_only_label_does_not_exempt_tiny_sweeps(tmp_path):
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE, {"config.json": '{"alpha": 0.095}'}),
        _entry(tmp_path, "swp", 2, BASE_SOURCE, {"config.json": '{"alpha": 0.10}'}),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90)
    # With enforcement off, config_only is a LABEL, not an exemption: the
    # byte-identical-code A/B sweep is shadow-logged AND still drops on the
    # similarity tier (its ratio is >= 0.99).
    assert result.kept_hotkeys == ("orig",)
    (d,) = result.dropped
    assert d.hotkey == "swp" and d.tier == "near_duplicate"
    (v,) = [s for s in result.shadow if s.tier == "config_only"]
    assert v.hotkey == "swp" and v.matched_hotkey == "orig" and v.score == 1.0


def test_config_only_large_rewrite_survives_with_label(tmp_path):
    # A config rewrite big enough to fall under the ratio bar: identical code,
    # genuinely different parameterization — kept, with the label recorded.
    big_a = json.dumps({f"w{i}": i * 0.095 for i in range(400)})
    big_b = json.dumps({f"w{i}": (i % 7) * 1.31 + 40 for i in range(400)})
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE, {"config.json": big_a}),
        _entry(tmp_path, "rew", 2, BASE_SOURCE, {"config.json": big_b}),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90)
    assert result.kept_hotkeys == ("orig", "rew")
    assert not result.dropped
    assert [s.tier for s in result.shadow if s.hotkey == "rew"].count("config_only") == 1


def test_config_only_enforced_drops(tmp_path):
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE, {"config.json": '{"alpha": 0.095}'}),
        _entry(tmp_path, "swp", 2, BASE_SOURCE, {"config.json": '{"alpha": 0.10}'}),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90,
                               config_only_enforce=True)
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
    runner.cfg = replace(runner.cfg,
                         round=replace(runner.cfg.round, dedup_probe_series=4,
                                       dedup_probe_mode="enforce"))
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


def test_similarity_refuses_an_uncapped_pair(tmp_path):
    from cascade.interface.dedup import Unscoreable

    a = fingerprint_dir(_big_repo(tmp_path, "a", 500), max_tokens=100)
    b = fingerprint_dir(_big_repo(tmp_path, "b", 500, extra="    y = 1\n"),
                        max_tokens=100)
    with pytest.raises(Unscoreable):
        similarity(a, b)


def test_padding_past_the_cap_does_not_evade_the_screen(tmp_path):
    # The evasion a naive "too big to score, keep it" cap would open: inflate a
    # copy past the cap and it is judged by nobody. The sketch tier judges it.
    orig = ("orig", 1, fingerprint_dir(_big_repo(tmp_path, "orig", 2000),
                                       max_tokens=500))
    # a near-copy, not a byte-copy: the identical tiers must not be what saves us
    copy = ("copy", 2, fingerprint_dir(
        _big_repo(tmp_path, "copy", 2000, extra="    y = 1\n"), max_tokens=500))
    enforced = screen_duplicates([orig, copy], None, sketch_mode="enforce",
                                 sketch_threshold=0.99)
    assert enforced.kept_hotkeys == ("orig",)
    assert enforced.dropped[0].tier == "oversize_sketch"

    # shadow (the shipped default): logged with its Jaccard, never dropped
    observed = screen_duplicates([orig, copy], None, sketch_mode="shadow")
    assert observed.kept_hotkeys == ("orig", "copy")
    assert [v.tier for v in observed.shadow] == ["oversize_sketch"]
    assert observed.shadow[0].score >= 0.99

    # off: nothing judges the pair at all — the padding hole, made explicit
    assert screen_duplicates([orig, copy], None,
                             sketch_mode="off").kept_hotkeys == ("orig", "copy")


def test_sketch_separates_copies_from_distinct_repos(tmp_path):
    from cascade.interface.dedup import sketch_jaccard

    a = fingerprint_dir(_big_repo(tmp_path, "a", 400), max_tokens=50)
    same = fingerprint_dir(_big_repo(tmp_path, "same", 400), max_tokens=50)
    other = fingerprint_dir(_repo(tmp_path, "other", BASE_SOURCE), max_tokens=50)
    assert sketch_jaccard(a, same) == 1.0
    assert sketch_jaccard(a, other) < 0.05


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
    # log that calibrates the thresholds.
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
    runner.cfg = replace(runner.cfg, generator=replace(
        runner.cfg.generator, sandbox_mode="subprocess", sandbox_strict=False))
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
    assert {c.max_generate_seconds for c in add.captured_cfgs} == {60}
    report = json.loads((runner.work_root / "96" / "dedup_report.json").read_text())
    assert report["probe_seconds_per_draw"] == 60


def test_a_near_copy_padded_past_the_decode_budget_is_still_judged(tmp_path):
    # The gap a stub-on-overflow would leave: if an over-budget file
    # contributed no shingles, inflating a copy past the DECODE budget (not
    # just the token cap) would evade every similarity tier. The decoded
    # prefix keeps it judgeable.
    from cascade.interface.dedup import sketch_jaccard

    body = "\n".join(f"    q{i} = weigh(p{i}, {i}) * 3.5" for i in range(4000))
    orig = _repo(tmp_path, "orig", "def run():\n" + body)
    copy = _repo(tmp_path, "copy", "def run():\n" + body + "\n    extra = 1\n")
    fo = fingerprint_dir(orig, max_text_mb=0.05, max_tokens=500)
    fc = fingerprint_dir(copy, max_text_mb=0.05, max_tokens=500)

    assert fo.truncated and not fo.scoreable      # the file was cut short…
    assert fo.token_sha256 != fc.token_sha256     # …but the digests still differ
    assert sketch_jaccard(fo, fc) >= 0.99         # …and the sketch sees the copy
    result = screen_duplicates([("orig", 1, fo), ("copy", 2, fc)], None,
                               sketch_mode="enforce", sketch_threshold=0.99)
    assert result.kept_hotkeys == ("orig",)


def test_over_cap_entrants_are_named_in_the_report(dedup_runner):
    # Honest submissions run ~100KB; bulk that only the sketch tier can judge
    # is a pattern worth being able to see build up over rounds.
    from dataclasses import replace

    runner, add = dedup_runner
    runner.cfg = replace(runner.cfg,
                         round=replace(runner.cfg.round, dedup_max_tokens=100))
    alice = add("alice", 3, BASE_SOURCE)

    runner._screen_duplicate_entrants(None, [alice], base_seed=97)
    report = json.loads((runner.work_root / "97" / "dedup_report.json").read_text())
    assert report["over_token_cap"][0]["hotkey"] == "alice"
    assert report["over_token_cap"][0]["n_tokens"] > 100
