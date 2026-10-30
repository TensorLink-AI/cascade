"""DEC-CA-0053: the orchestrator's generation stage and pre-signed transfer."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.generation import shards as S
from cascade.generation import transfer as T
from cascade.generation.cli import upload_corpus_dir
from cascade.generation.stage import GenerationFault, GenerationInfraError, GenerationStage
from cascade.provision.core import PodAddress
from cascade.provision.cpu import CpuShape
from cascade.shared.config import GenerationConfig, load_chain_config
from cascade.shared.hippius import ObjectNotFound
from cascade.trainer.remote import RemoteHost, push_corpus_files, worker_argv
from cascade.trainer.runner_generation import leg_corpus

HK = "5Fminer"


def _corpus(d: Path, seed: int = 3) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    kids = []
    for i in range(2):
        sh = [S.write_shard(d / f"c{i}-{n:05d}.parquet",
                            [rng.standard_normal((1, 64 + 8 * j)) for j in range(5)])
              for n in range(3)]
        kids.append({"index": i, "seed": S.child_seed(seed, i), "shards": sh})
    return S.finalize_index(d, seed=seed, children=kids, meta={"generator_ref": "g@sha256:1"})


# ── remote corpus: download links, verified, deleted once read ──────────────


def _remote_copy(src: Path, dst: Path) -> None:
    dst.mkdir()
    shutil.copy(src / S.INDEX_NAME, dst / S.INDEX_NAME)
    idx = S.read_index(src)
    urls = {sh["name"]: (src / sh["name"]).as_uri()
            for c in idx["children"] for sh in c["shards"]}
    (dst / S.URLS_NAME).write_text(json.dumps(urls))


def test_remote_corpus_streams_the_same_series(tmp_path):
    _corpus(tmp_path / "src")
    _remote_copy(tmp_path / "src", tmp_path / "pod")
    local = list(S.iter_corpus(tmp_path / "src"))
    remote = list(S.iter_corpus(tmp_path / "pod"))
    assert len(local) == len(remote) == 30
    assert all(np.array_equal(a, b) for a, b in zip(local, remote, strict=True))
    assert not list((tmp_path / "pod").glob("*.parquet")), "shards are deleted once read"


def test_remote_corpus_digest_matches_the_index(tmp_path):
    from cascade.trainer.stream import open_round_stream

    idx = _corpus(tmp_path / "src")
    _remote_copy(tmp_path / "src", tmp_path / "pod")
    cfg = replace(load_chain_config().generator, min_channels=1)
    with open_round_stream("shards", tmp_path, 0, cfg, token_budget=10**9,
                           corpus_dir=tmp_path / "pod") as st:
        for _ in st.series():
            pass
        assert st.digest == idx["corpus_digest"]


def test_a_tampered_remote_shard_is_refused(tmp_path):
    idx = _corpus(tmp_path / "src")
    _remote_copy(tmp_path / "src", tmp_path / "pod")
    victim = idx["children"][1]["shards"][1]["name"]
    (tmp_path / "src" / victim).write_bytes(b"not the shard")
    with pytest.raises(S.ShardError, match="sha256"):
        list(S.iter_corpus(tmp_path / "pod"))


def test_shard_names_cover_every_child_shard():
    g = GenerationConfig(vcpus=4, target_points=1000, shard_points=100)
    names = T.shard_names(g)
    assert len(names) == 4 * (3 + 1)            # ceil(250/100) + 1 per child
    assert names[0] == "c0-00000.parquet" and "c3-00003.parquet" in names


def test_upload_puts_shards_then_index_and_cleans_up(tmp_path, monkeypatch):
    idx = _corpus(tmp_path / "out")
    names = [sh["name"] for c in idx["children"] for sh in c["shards"]]
    urls = {n: f"https://up/{n}" for n in [*names, S.INDEX_NAME]}
    (tmp_path / "urls.json").write_text(json.dumps(urls))
    order: list[str] = []
    monkeypatch.setattr(T, "put_file", lambda url, path: order.append(Path(path).name))
    n = upload_corpus_dir(tmp_path / "out", idx, tmp_path / "urls.json", workers=2)
    assert n == len(names) + 1 and order[-1] == S.INDEX_NAME
    assert sorted(order[:-1]) == sorted(names)
    assert not list((tmp_path / "out").glob("*.parquet"))


def test_upload_refuses_a_missing_link(tmp_path, monkeypatch):
    idx = _corpus(tmp_path / "out")
    (tmp_path / "urls.json").write_text(json.dumps({S.INDEX_NAME: "x"}))
    monkeypatch.setattr(T, "put_file", lambda url, path: None)
    with pytest.raises(RuntimeError, match="no upload link"):
        upload_corpus_dir(tmp_path / "out", idx, tmp_path / "urls.json")


# ── the stage ───────────────────────────────────────────────────────────────


class FakeProvider:
    name = "targon"
    ssh_user = "ubuntu"

    def __init__(self, offers=True, ready=True):
        self._offers, self._ready = offers, ready
        self.launched, self.terminated = [], []

    def offers(self, vcpus, price):
        return [CpuShape(vcpus=32, memory_gb=419, price_per_hour=1.0, provider="targon",
                         type_id="cpu-xlarge-amd")] if self._offers else []

    def launch(self, shape, *, name, ssh_pubkey, image=""):
        self.launched.append(name)
        return "vm-1"

    def wait_ready(self, vm, *, timeout):
        return self._ready

    def get_ip(self, vm):
        return PodAddress("10.0.0.9", 2222)

    def terminate(self, vm):
        self.terminated.append(vm)


class FakeStore:
    def __init__(self):
        self.objs: dict[str, str] = {}
        self.cfg = SimpleNamespace(bucket="b")

    def client(self):
        return SimpleNamespace(generate_presigned_url=lambda m, Params, ExpiresIn:
                               f"https://s3/{m}/{Params['Key']}?exp={ExpiresIn}")

    def put_text(self, key, text, content_type="", acl=None):
        self.objs[key] = text

    def get_text(self, key):
        if key not in self.objs:
            raise ObjectNotFound(key)
        return self.objs[key]


def _stage(tmp_path, provider, store, *, gen_rc=0, digest=None):
    cfg = load_chain_config()
    cfg = replace(cfg, generation=replace(cfg.generation, enabled=True),
                  generator=replace(cfg.generator, min_channels=1))
    idx = _corpus(tmp_path / "made")
    calls = []

    def runner(argv, timeout, stdin=None):
        cmd = argv[-1]
        calls.append(cmd)
        if "cascade-generate" in cmd:
            base = "corpora/round-77/5Fminer/"
            store.put_text(base + S.INDEX_NAME, json.dumps(idx))
            summary = {"ok": gen_rc == 0, "corpus_digest": digest or idx["corpus_digest"],
                       "faults": [{"error": "CorpusError: boom"}] if gen_rc == 3 else []}
            return subprocess.CompletedProcess(argv, gen_rc, json.dumps(summary) + "\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def shell(argv, **kw):
        calls.append(" ".join(map(str, argv)))
        return subprocess.CompletedProcess(argv, 0, "", "")

    st = GenerationStage(cfg=cfg, store=store, providers=[provider], ssh_key_path="/k",
                         ssh_pubkey="ssh-ed25519 AAA", runner=runner, shell=shell)
    return st, idx, calls


def _go(st, tmp_path):
    (tmp_path / "repo").mkdir(exist_ok=True)
    return st.generate(round_id="77", hotkey=HK, uid=4, era=9, gen_ref="g@sha256:1",
                       repo_dir=tmp_path / "repo", seed=123)


def test_stage_generates_registers_and_terminates(tmp_path):
    prov, store = FakeProvider(), FakeStore()
    st, idx, calls = _stage(tmp_path, prov, store)
    res = _go(st, tmp_path)
    assert prov.terminated == ["vm-1"]
    assert res.index["corpus_digest"] == idx["corpus_digest"]
    assert res.row["uid"] == 4 and res.row["provider"] == "targon"
    assert "corpora/round-77/5Fminer/meta.json" in store.objs
    gen_cmd = next(c for c in calls if "cascade-generate" in c)
    assert "--upload-urls _gen/urls.json" in gen_cmd and "https://" not in gen_cmd
    assert any("pod_bootstrap.sh" in c for c in calls)
    files = st.pod_files(res)
    urls = json.loads(files[S.URLS_NAME])
    assert set(urls) == {sh["name"] for c in idx["children"] for sh in c["shards"]}
    assert all("get_object" in u for u in urls.values())


def test_generator_fault_is_the_miners_and_still_terminates(tmp_path):
    prov = FakeProvider()
    st, _, _ = _stage(tmp_path, prov, FakeStore(), gen_rc=3)
    with pytest.raises(GenerationFault, match="miner submission rejected") as ei:
        _go(st, tmp_path)
    assert ei.value.returncode == 3 and prov.terminated == ["vm-1"]


@pytest.mark.parametrize("kw,match", [
    ({"provider": FakeProvider(offers=False)}, "no CPU capacity"),
    ({"provider": FakeProvider(ready=False)}, "not ready"),
    ({"gen_rc": 4}, "failed"),
    ({"digest": "f" * 64}, "does not match"),
])
def test_infra_failures_requeue(tmp_path, kw, match):
    prov = kw.pop("provider", FakeProvider())
    st, _, _ = _stage(tmp_path, prov, FakeStore(), **kw)
    with pytest.raises(GenerationInfraError, match=match):
        _go(st, tmp_path)
    assert prov.terminated == (["vm-1"] if prov.launched else [])


# ── dispatch plumbing ───────────────────────────────────────────────────────


def test_corpus_files_go_over_stdin_and_the_worker_gets_the_dir():
    host = RemoteHost(name="p", host="1.2.3.4", workdir="/w")
    seen = []

    def runner(argv, timeout, stdin=None):
        seen.append((argv[-1], stdin))
        return subprocess.CompletedProcess(argv, 0, "", "")

    d = push_corpus_files(host, {"corpus.json": "{}", "urls.json": '{"a": "https://x"}'},
                          "challenger-5F-1", runner=runner)
    assert d == "/w/_corpus/challenger-5F-1"
    assert all("https://" not in cmd for cmd, _ in seen)
    assert seen[1][1] == '{"a": "https://x"}'
    argv = worker_argv(host, gen_ref="g", uid=1, hotkey="h", role="challenger", base_seed=1,
                       block=1, trainer_spec="m:C", corpus_dir=d)
    assert argv[-2:] == ["--corpus-dir", d]
    assert "--corpus-dir" not in worker_argv(host, gen_ref="g", uid=1, hotkey="h",
                                             role="challenger", base_seed=1, block=1,
                                             trainer_spec="m:C")


def test_leg_corpus_only_for_shards_legs():
    calls = []
    runner = SimpleNamespace(_leg_corpus=lambda *a, **k: calls.append(a) or {"x": "y"})
    assert leg_corpus(runner, None, None, SimpleNamespace(corpus_mode="stream_cpu")) == {}
    assert leg_corpus(runner, None, None, SimpleNamespace(corpus_mode="shards")) == {
        "corpus_files": {"x": "y"}}
    assert len(calls) == 1


def test_an_unplanned_shard_name_is_the_miners(tmp_path):
    prov, store = FakeProvider(), FakeStore()
    st, idx, _ = _stage(tmp_path, prov, store)
    idx["children"][0]["shards"][0]["name"] = "../../etc/x.parquet"
    with pytest.raises(GenerationFault, match="unplanned shards"):
        _go(st, tmp_path)
    assert prov.terminated == ["vm-1"]


def test_isolation_unavailable_is_infra(tmp_path):
    from cascade.generation import tier

    src = Path(tier.__file__).read_text(encoding="utf-8")
    assert '"sandbox_isolation_unavailable" in str(e)' in src


def test_a_second_ref_of_one_hotkey_gets_its_own_corpus(tmp_path):
    from cascade.trainer.runner_generation import _GenerationMixin

    made = []

    class Stage:
        def generate(self, **kw):
            made.append(kw)
            return kw

        def pod_files(self, result):
            return {"corpus.json": result["round_id"]}

    cfg = load_chain_config()
    me = SimpleNamespace(cfg=cfg, work_root=tmp_path, hub=lambda: None,
                         _generation_stage=lambda: Stage())
    me._build_leg_corpus = lambda g, s, e: _GenerationMixin._build_leg_corpus(me, g, s, e)
    import cascade.shared.hippius as H

    orig, H.fetch_from_hub = H.fetch_from_hub, lambda ref, repo, hub: Path(repo).mkdir(
        parents=True, exist_ok=True)
    try:
        seeds = SimpleNamespace(base_seed=9, generation_seed=1)
        c = SimpleNamespace(corpus_mode="shards")
        a = _GenerationMixin._leg_corpus(me, SimpleNamespace(hotkey=HK, uid=1, ref="r@1"),
                                         seeds, c)
        b = _GenerationMixin._leg_corpus(me, SimpleNamespace(hotkey=HK, uid=1, ref="r@2"),
                                         seeds, c)
        again = _GenerationMixin._leg_corpus(me, SimpleNamespace(hotkey=HK, uid=1, ref="r@1"),
                                             seeds, c)
    finally:
        H.fetch_from_hub = orig
    assert a != b and again == a and len(made) == 2
    assert all(str(m["round_id"]).startswith("9-") for m in made)
    assert not list((tmp_path / "_generation").iterdir())      # repo dirs removed


def test_cached_corpus_gets_fresh_download_links_on_retry(tmp_path):
    from cascade.generation.stage import StageResult
    from cascade.trainer.runner_generation import _GenerationMixin

    now = [0]
    signed = []

    class Client:
        def generate_presigned_url(self, method, *, Params, ExpiresIn):
            signed.append((now[0], ExpiresIn))
            return f"https://example.invalid/{Params['Key']}?signed={now[0]}"

    store = SimpleNamespace(client=lambda: Client(), cfg=SimpleNamespace(bucket="test"))
    stage = SimpleNamespace(store=store)
    stage.pod_files = lambda result: GenerationStage.pod_files(stage, result)
    corpus = StageResult(index={"children": [{"shards": [{"name": "c0-0.parquet"}]}]},
                         base="test/", provider="fake", shape="cpu", row={})
    builds = []

    def build(*args):
        builds.append(args)
        return corpus

    runner = SimpleNamespace(cfg=load_chain_config(), _generation_stage=lambda: stage, _build_leg_corpus=build)
    gen = SimpleNamespace(hotkey=HK, ref="r@1")
    seeds, contract = SimpleNamespace(base_seed=9), SimpleNamespace(corpus_mode="shards")
    first = _GenerationMixin._leg_corpus(runner, gen, seeds, contract, era_index=1)
    now[0] = 25 * 3600
    retry = _GenerationMixin._leg_corpus(runner, gen, seeds, contract, era_index=2)
    assert len(builds) == 1
    assert first[S.INDEX_NAME] == retry[S.INDEX_NAME]
    assert first[S.URLS_NAME] != retry[S.URLS_NAME]
    assert signed == [(0, T.DOWNLOAD_EXPIRES_S), (25 * 3600, T.DOWNLOAD_EXPIRES_S)]


def test_vm_known_hosts_is_fresh_per_vm():
    from cascade.generation.stage import vm_known_hosts

    p = vm_known_hosts("vm/../1")
    assert p.name == "cascade-genvm-vm_.._1.known_hosts" and p.read_text() == ""
    p.write_text("[1.2.3.4]:22 ssh-ed25519 OLD\n")       # a previous VM on a recycled IP
    assert vm_known_hosts("vm/../1").read_text() == ""   # never trusted by the next VM
    assert vm_known_hosts("vm-2") != p
