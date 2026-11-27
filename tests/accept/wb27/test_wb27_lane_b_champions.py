"""2026-11-27 owner ("if they become king it should be published, probably in
the repo, but handle the complexity of being associated with different
repos"): a lane B leg that WINS its slot — its experts accepted into the
king's init (``LaneBDriverState.merged_inits``) — publishes its generator the
way a lane A king does: the SAME ``ChampionPublisher`` and ``[round]
champion_publish`` / ``champion_publish_delay_rounds`` policy on the trainer,
the SAME scan gate and PROVENANCE/SCAN writer as ``scripts/sync_king.py`` in
the repo (king-sync workflow). Layout ``champions/lane_b/<domain>/<slot>/``,
a re-won slot's previous holder under ``champions/lane_b/archive/<domain>/
<slot>/<round>/``, one hashed ``champions/lane_b/index.json``. Torch-free."""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from .helpers import torch_blocked

REPO = Path(__file__).resolve().parents[3]
HK1, HK2, HK3 = ("5LaneBMinerOne" + "A" * 34, "5LaneBMinerTwo" + "B" * 34,
                 "5LaneBMinerThree" + "C" * 32)
GEN_FILES = {"generator.py": b"class Generator:\n    def generate(self):\n        return [1.0]\n",
             "config.json": b"{}\n", "requirements.txt": b"numpy\n"}


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in sorted(files.items()):
            zf.writestr(name, data)
    return buf.getvalue()


class FakePublic:
    """S3Store-shaped public bucket (missing key = ObjectNotFound)."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.fail_text = False

    def put_bytes(self, key, data, content_type="", acl=None):
        self.objects[key] = bytes(data)

    def put_text(self, key, text, content_type="", acl=None):
        if self.fail_text:
            raise OSError("bucket 500")
        self.objects[key] = text.encode()

    def get_text(self, key):
        from cascade.shared.hippius import ObjectNotFound

        if key not in self.objects:
            raise ObjectNotFound(key)
        return self.objects[key].decode()


def _champion():
    mod = importlib.import_module("cascade.funding.champion")
    assert hasattr(mod.ChampionPublisher, "note_lane_b"), \
        "ChampionPublisher has no lane B path: a lane B slot winner is never published"
    assert hasattr(mod, "lane_b_wins") and hasattr(mod, "LANE_B_INDEX_KEY"), \
        "no lane B win extraction / index key in cascade.funding.champion"
    return mod


def _sync():
    path = REPO / "scripts" / "sync_lane_b_champions.py"
    assert path.is_file(), "scripts/sync_lane_b_champions.py missing: lane B winners never reach the repo"
    sys.path.insert(0, str(REPO / "scripts"))
    return importlib.import_module("sync_lane_b_champions")


def _vault(tmp_path, *, hotkey=HK1, files=GEN_FILES):
    from cascade.funding.store import SubmissionStore, vault_ref

    store = SubmissionStore(tmp_path / "vault")
    digest = store.put(_zip(files), hotkey)
    return store, vault_ref(digest), digest


def _merge(*, hotkey, gen_ref, targets, rid, block, rows, leg="ptr-leg", king="ptr-kingA",
           accepted=True):
    wins = [{"domain": t, "hotkey": hotkey, "uid": 7, "round_id": rid, "merge_block": 0,
             "gen_ref": gen_ref, "commit_block": block - 500, "targets": list(targets),
             "rebased_pointer": leg, "king_pointer": king, "selection": rows}
            for t in targets]
    m = {"rebased_pointer": leg, "king_pointer": king, "selection": rows, "round_id": rid,
         "qualified_wins": wins}
    if accepted:
        m["acceptance"] = {"block": block, "wins": wins, "targets": list(targets)}
    return m


def _inits(*eras):
    """era index -> (merged pointer, lane A king pointer, merges)."""
    return {str(i): {"pointer": p, "base": b, "merges": list(ms)} for i, p, b, ms in eras}


def _publisher(mod, store, public, tmp_path, policy="crown", delay=1):
    return mod.ChampionPublisher(store, public, policy=policy, delay_rounds=delay,
                                 state_path=tmp_path / "champion_publisher.json")


def _fetch_from(public):
    """``cascade fetch <vault ref>`` against the fake public bucket."""
    def fetch(ref, out_dir):
        digest = ref.split("@sha256:")[1]
        data = public.objects[f"champions/{digest}.zip"]
        assert hashlib.sha256(data).hexdigest() == digest
        out_dir.mkdir(parents=True)
        zipfile.ZipFile(io.BytesIO(data)).extractall(out_dir)
        return out_dir
    return fetch


ROWS_A = {"0": [1, 2, 3, 4], "1": [9, 10, 11, 12]}
ROWS_B = {"0": [20, 21, 22, 23], "1": [30, 31, 32, 33]}


def test_winning_slot_publishes_after_the_delay_with_files_and_index(tmp_path):
    with torch_blocked():
        mod = _champion()
        sync = _sync()
        store, ref, digest = _vault(tmp_path)
        public = FakePublic()
        pub = _publisher(mod, store, public, tmp_path, policy="delay", delay=2)
        inits = _inits((4, "hub/ckpt-era4-merged-init@sha256:" + "4" * 64,
                        "hub/king-era4@sha256:" + "a" * 64,
                        [_merge(hotkey=HK1, gen_ref=ref, targets=["energy"], rid="r100",
                                block=9_600_000, rows=ROWS_A)]))
        wins = mod.lane_b_wins(inits)
        assert pub.note_lane_b(wins, "r101") == []           # reign 0
        assert pub.note_lane_b(wins, "r101") == []           # same round: no advance
        assert pub.note_lane_b(wins, "r102") == []           # reign 1
        assert f"champions/{digest}.zip" not in public.objects
        assert pub.note_lane_b(wins, "r103") == ["energy|ptr-leg"]   # reign 2 = delay
        assert public.objects[f"champions/{digest}.zip"] == _zip(GEN_FILES)
        index = json.loads(public.objects[mod.LANE_B_INDEX_KEY])
        assert mod.verify_lane_b_index(index)
        e = index["slots"]["energy"]
        assert e["published"] is True and e["hotkey"] == HK1 and e["uid"] == 7
        assert e["gen_ref"] == ref and e["generator_digest"] == f"sha256:{digest}"
        assert e["zip_key"] == f"champions/{digest}.zip"
        assert e["rows"] == ROWS_A and e["slot"] == mod.rows_slot_id(ROWS_A)
        assert e["merged_into"] == "hub/ckpt-era4-merged-init@sha256:" + "4" * 64
        assert e["lane_a_king_pointer"] == "hub/king-era4@sha256:" + "a" * 64
        assert e["round_id"] == "r100" and e["merge_block"] == 9_600_000
        # lane A's index never learns about lane B (cascade fetch king reads it)
        assert mod.CHAMPION_INDEX_KEY not in public.objects

        touched, rc = sync.sync_tree(index, tmp_path / "repo", fetch=_fetch_from(public))
        assert rc == 0
        d = tmp_path / "repo" / "champions" / "lane_b" / "energy" / mod.rows_slot_id(ROWS_A)
        for name in ("generator.py", "config.json", "requirements.txt",
                     "PROVENANCE.json", "SCAN.json"):
            assert (d / name).is_file(), name
        prov = json.loads((d / "PROVENANCE.json").read_text())
        assert prov["gen_ref"] == ref and prov["merged_into"] == e["merged_into"]
        assert json.loads((tmp_path / "repo" / "champions" / "lane_b" / "index.json")
                          .read_text()) == index
        # a second sync of the same index touches nothing
        assert sync.sync_tree(index, tmp_path / "repo", fetch=_fetch_from(public)) == ([], 0)


def test_rewinning_a_slot_archives_the_previous_holder(tmp_path):
    with torch_blocked():
        mod = _champion()
        sync = _sync()
        store, ref1, d1 = _vault(tmp_path, hotkey=HK1)
        files2 = {**GEN_FILES, "generator.py": b"class Generator:\n    pass\n"}
        d2 = store.put(_zip(files2), HK2)
        ref2 = ref1.replace(d1, d2)
        public = FakePublic()
        pub = _publisher(mod, store, public, tmp_path, policy="crown")
        repo = tmp_path / "repo"
        m1 = _merge(hotkey=HK1, gen_ref=ref1, targets=["energy"], rid="r100",
                    block=9_600_000, rows=ROWS_A, leg="ptr-leg1")
        era1 = (4, "hub/merged4@sha256:" + "4" * 64, "hub/kingA@sha256:" + "a" * 64, [m1])
        pub.note_lane_b(mod.lane_b_wins(_inits(era1)), "r101")
        sync.sync_tree(json.loads(public.objects[mod.LANE_B_INDEX_KEY]), repo,
                       fetch=_fetch_from(public))
        m2 = _merge(hotkey=HK2, gen_ref=ref2, targets=["energy"], rid="r140",
                    block=9_610_000, rows=ROWS_B, leg="ptr-leg2", king="hub/merged4")
        era2 = (8, "hub/merged8@sha256:" + "8" * 64, "hub/kingB@sha256:" + "b" * 64, [m2])
        assert pub.note_lane_b(mod.lane_b_wins(_inits(era1, era2)), "r141") == ["energy|ptr-leg2"]
        index = json.loads(public.objects[mod.LANE_B_INDEX_KEY])
        assert index["slots"]["energy"]["hotkey"] == HK2
        assert [a["hotkey"] for a in index["archive"]] == [HK1]
        _touched, rc = sync.sync_tree(index, repo, fetch=_fetch_from(public))
        assert rc == 0
        lb = repo / "champions" / "lane_b"
        cur = lb / "energy" / mod.rows_slot_id(ROWS_B)
        old = lb / "archive" / "energy" / mod.rows_slot_id(ROWS_A) / "r100"
        assert (cur / "generator.py").read_bytes() == files2["generator.py"]
        assert (old / "generator.py").read_bytes() == GEN_FILES["generator.py"]
        assert json.loads((old / "PROVENANCE.json").read_text())["hotkey"] == HK1
        assert not (lb / "energy" / mod.rows_slot_id(ROWS_A)).exists()


def test_index_lists_every_generator_composed_into_the_king(tmp_path):
    with torch_blocked():
        mod = _champion()
        store, ref1, d1 = _vault(tmp_path, hotkey=HK1)
        d2 = store.put(_zip({**GEN_FILES, "config.json": b'{"x": 2}\n'}), HK2)
        d3 = store.put(_zip({**GEN_FILES, "config.json": b'{"x": 3}\n'}), HK3)
        ref2, ref3 = ref1.replace(d1, d2), ref1.replace(d1, d3)
        public = FakePublic()
        pub = _publisher(mod, store, public, tmp_path, policy="crown")
        inits = _inits(
            (4, "hub/merged4@sha256:" + "4" * 64, "hub/kingA@sha256:" + "a" * 64,
             [_merge(hotkey=HK1, gen_ref=ref1, targets=["energy", "horizon:720"],
                     rid="r100", block=9_600_000, rows=ROWS_A, leg="leg1")]),
            (8, "hub/merged8@sha256:" + "8" * 64, "hub/kingB@sha256:" + "b" * 64,
             [_merge(hotkey=HK2, gen_ref=ref2, targets=["sales"], rid="r140",
                     block=9_610_000, rows=ROWS_B, leg="leg2"),
              _merge(hotkey=HK3, gen_ref=ref3, targets=["energy"], rid="r140",
                     block=9_610_000, rows={"0": [40, 41, 42, 43]}, leg="leg3")]))
        pub.note_lane_b(mod.lane_b_wins(inits), "r141")
        index = json.loads(public.objects[mod.LANE_B_INDEX_KEY])
        assert set(index["slots"]) == {"energy", "horizon:720", "sales"}
        assert index["slots"]["energy"]["hotkey"] == HK3      # the newer accepted merge
        composed = {(c["target"], c["gen_ref"]) for c in index["composed"]}
        assert composed == {("energy", ref1), ("horizon:720", ref1), ("sales", ref2),
                            ("energy", ref3)}
        assert all(c["published"] for c in index["composed"])
        assert index["king_checkpoint"] == "hub/merged8@sha256:" + "8" * 64
        # hashed like the lane A record: digests per entry + one body hash
        tampered = json.loads(json.dumps(index))
        tampered["slots"]["sales"]["hotkey"] = HK1
        assert not mod.verify_lane_b_index(tampered)
        with pytest.raises(ValueError):
            _sync().sync_tree(tampered, tmp_path / "repo", fetch=_fetch_from(public))


def test_repo_copy_uses_the_lane_a_scan_and_provenance(tmp_path, monkeypatch):
    with torch_blocked():
        mod = _champion()
        sync = _sync()
        import sync_king

        assert hasattr(sync_king, "scan_gate") and hasattr(sync_king, "write_scanned_copy"), \
            "the lane A king sync's scan gate / staged copy are not shared"
        calls = []
        real_scan, real_write = sync_king.scan_gate, sync_king.write_scanned_copy
        monkeypatch.setattr(sync_king, "scan_gate",
                            lambda t: calls.append("scan") or real_scan(t))
        monkeypatch.setattr(sync_king, "write_scanned_copy",
                            lambda *a: calls.append("write") or real_write(*a))
        bad = {**GEN_FILES, "generator.py": b"import socket\nclass Generator: pass\n"}
        store, ref, digest = _vault(tmp_path, files=bad)
        public = FakePublic()
        _publisher(mod, store, public, tmp_path).note_lane_b(mod.lane_b_wins(_inits(
            (4, "hub/m@sha256:" + "4" * 64, "hub/k@sha256:" + "a" * 64,
             [_merge(hotkey=HK1, gen_ref=ref, targets=["energy"], rid="r1",
                     block=9_600_000, rows=ROWS_A)]))), "r2")
        index = json.loads(public.objects[mod.LANE_B_INDEX_KEY])
        repo = tmp_path / "repo"
        _t, rc = sync.sync_tree(index, repo, fetch=_fetch_from(public))
        assert rc == 1 and calls == ["scan"]                  # high finding: refused
        assert not (repo / "champions" / "lane_b" / "energy").exists()
        _t, rc = sync.sync_tree(index, repo, fetch=_fetch_from(public), allow_findings=True)
        d = repo / "champions" / "lane_b" / "energy" / mod.rows_slot_id(ROWS_A)
        scan = json.loads((d / "SCAN.json").read_text())
        king_scan = json.loads((REPO / "champions" / "king" / "SCAN.json").read_text())
        assert set(scan) == set(king_scan) and scan["high"] >= 1
        assert "write" in calls


def test_nothing_published_for_a_refused_or_unmerged_leg(tmp_path):
    with torch_blocked():
        mod = _champion()
        store, ref, digest = _vault(tmp_path)
        public = FakePublic()
        pub = _publisher(mod, store, public, tmp_path, policy="crown")
        # a deferred init (acceptance timed out): pointer == base, no merges;
        # a qualified-but-rejected leg never enters merged_inits at all
        inits = _inits((4, "hub/k@sha256:" + "a" * 64, "hub/k@sha256:" + "a" * 64, []))
        assert mod.lane_b_wins(inits) == []
        assert pub.note_lane_b(mod.lane_b_wins(inits), "r1") == []
        assert f"champions/{digest}.zip" not in public.objects
        index = json.loads(public.objects[mod.LANE_B_INDEX_KEY])
        assert index["slots"] == {} and index["composed"] == []
        # "off": nothing at all, not even state
        off = _publisher(mod, store, FakePublic(), tmp_path / "off", policy="off")
        win = mod.lane_b_wins(_inits((4, "hub/m@sha256:" + "4" * 64, "hub/k", [
            _merge(hotkey=HK1, gen_ref=ref, targets=["energy"], rid="r1",
                   block=9_600_000, rows=ROWS_A)])))
        assert off.note_lane_b(win, "r2") == [] and not off.public.objects


def test_lane_a_reign_change_keeps_lane_b_state_and_trainer_wires_it(tmp_path):
    with torch_blocked():
        mod = _champion()
        store, ref, digest = _vault(tmp_path)
        public = FakePublic()
        pub = _publisher(mod, store, public, tmp_path, policy="delay", delay=1)
        inits = _inits((4, "hub/m@sha256:" + "4" * 64, "hub/k@sha256:" + "a" * 64, [
            _merge(hotkey=HK1, gen_ref=ref, targets=["energy"], rid="r1",
                   block=9_600_000, rows=ROWS_A)]))
        pub.note_lane_b(mod.lane_b_wins(inits), "r2")
        pub.note_king(HK2, "kingns/gen@sha256:" + "e" * 64, "r3")   # lane A dethrone
        state = json.loads((tmp_path / "champion_publisher.json").read_text())
        assert "energy" in state["lane_b"]["holders"]

        from cascade.shared.config import RoundConfig
        from cascade.trainer.loop import TrainerRunner

        assert hasattr(TrainerRunner, "_maybe_publish_lane_b_champions")
        public2 = FakePublic()
        fake = SimpleNamespace(
            cfg=SimpleNamespace(round=RoundConfig(champion_publish="crown",
                                                  submission_vault_dir=str(tmp_path / "vault"))),
            work_root=tmp_path / "wr", manifest_store=lambda: public2,
            lane_b_driver=SimpleNamespace(st=SimpleNamespace(merged_inits=inits),
                                          _lock=None))
        for name in ("_submission_store", "_maybe_publish_champion",
                     "_maybe_publish_lane_b_champions"):
            setattr(fake, name, getattr(TrainerRunner, name).__get__(fake))
        fake._maybe_publish_champion(SimpleNamespace(hotkey=HK2, ref="ns/g@sha256:" + "e" * 64),
                                     "r9")
        assert f"champions/{digest}.zip" in public2.objects
        assert json.loads(public2.objects[mod.LANE_B_INDEX_KEY])["slots"]["energy"]["published"]
