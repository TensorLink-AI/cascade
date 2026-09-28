"""The per-round training summary (``training/round-<id>.json``) — what each
duel leg actually trained, republished public-read for the dashboard's
Training tab (:mod:`cascade.shared.training_summary`).

Covers the fold of a leg's log records into one row, the local-row-first /
log-read-back resolution per manifest entry, the ACL fallback, and the trainer
hook that publishes right after a manifest. No network, no torch."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cascade.shared.hippius import StorageError, log_key
from cascade.shared.manifest import TrainedEntry, TrainingManifest, format_trained_pointer
from cascade.shared.training_summary import (
    TRAINING_SUMMARY_PREFIX,
    build_training_summary,
    collect_round_legs,
    contract_block,
    dump_training_summary,
    leg_from_records,
    leg_log_role,
    parse_log_records,
    publish_training_summary,
    training_summary_key,
)
from cascade.trainer.loop import TrainerRunner

KING_PTR = format_trained_pointer("cascade/ckpt-king@sha256:" + "c" * 64)
CHAL_PTR = format_trained_pointer("cascade/ckpt-chal@sha256:" + "d" * 64)


def _entry(role, ptr, uid, size="22m"):
    return TrainedEntry(miner_hotkey=f"hk-{role}", miner_uid=uid, role=role,
                        gen_ref=f"{role}/gen@sha256:" + "a" * 64, trained_pointer=ptr,
                        corpus_digest="e" * 64, train_block=100, size=size)


def _manifest(entries, **kw):
    return TrainingManifest(round_id="777", created_block=1234, contract_digest="c" * 64,
                            base_arch_digest="b" * 64, eval_dataset="pool", entries=list(entries), **kw)


DONE = {"event": "done", "final_loss": 0.4, "steps": 152, "tokens_seen": 39_000,
        "channel_tokens": 78_000, "tokens_frac": 0.975, "deadline_hit": True,
        "gpu_name": "L40S", "channel_telemetry": {"max_channels_seen": 2, "n_multichannel_series": 5},
        "lr_schedule": "wsd", "optim_state_resumed": True}
SUMMARY = {"event": "summary", "role": "king-22m", "n_series": 40, "total_points": 39_000,
           "train_seconds": 10_800.5, **{k: v for k, v in DONE.items() if k != "event"}}


def test_key_lives_under_the_training_prefix():
    assert training_summary_key("777") == "training/round-777.json"
    assert training_summary_key("777").startswith(TRAINING_SUMMARY_PREFIX)
    assert leg_log_role("king", "22m") == "king-22m"


def test_parse_log_records_skips_junk():
    text = json.dumps({"event": "host"}) + "\nnot json\n\n" + json.dumps(DONE) + "\n[1,2]\n"
    recs = parse_log_records(text)
    assert [r.get("event") for r in recs] == ["host", "done"]


def test_leg_fold_takes_the_run_summary_and_the_channel_telemetry():
    leg = leg_from_records([{"event": "host", "gpu": "x"}, {"event": "step", "step": 50}, DONE, SUMMARY])
    assert leg["steps"] == 152 and leg["tokens_seen"] == 39_000 and leg["channel_tokens"] == 78_000
    assert leg["deadline_hit"] is True and leg["tokens_frac"] == 0.975
    assert leg["max_channels_seen"] == 2 and leg["n_multichannel_series"] == 5
    assert leg["train_seconds"] == 10_800.5 and leg["n_series"] == 40      # from the summary row
    assert "final_loss" not in leg and "role" not in leg                    # not a log mirror


def test_leg_fold_is_none_without_a_run_summary():
    assert leg_from_records([{"event": "host"}, {"event": "step", "step": 1}]) is None
    assert leg_from_records([]) is None


def test_collect_prefers_the_local_row_then_the_log_then_marks_unmeasured():
    reads = []

    def read_log(log_role):
        reads.append(log_role)
        if log_role == "challenger-22m":
            return json.dumps({"event": "host"}) + "\n" + json.dumps(DONE) + "\n"
        return None

    entries = [_entry("king", KING_PTR, 7), _entry("challenger", CHAL_PTR, 11),
               _entry("challenger", CHAL_PTR.replace("d" * 64, "f" * 64), 12, size="44m")]
    legs = collect_round_legs(entries, "22m", cached={"king-22m": SUMMARY}, read_log=read_log)
    assert [x["role"] for x in legs] == ["king", "challenger", "challenger"]
    k, c, u = legs
    assert k["measured"] and k["source"] == "trainer" and k["train_seconds"] == 10_800.5
    assert k["trained_pointer"] == KING_PTR and k["miner_uid"] == 7 and k["size"] == "22m"
    assert c["measured"] and c["source"] == "log" and c["steps"] == 152
    assert u["measured"] is False and "steps" not in u and u["size"] == "44m"
    assert reads == ["challenger-22m", "challenger-44m"]      # the cached king never hits the log


def test_collect_survives_a_raising_log_reader():
    def boom(_):
        raise RuntimeError("s3 down")

    legs = collect_round_legs([_entry("king", KING_PTR, 7)], "22m", read_log=boom)
    assert legs[0]["measured"] is False


def test_contract_block_reads_the_pricing_terms_leniently():
    c = SimpleNamespace(batch_size=64, context_length=4096, budget_denomination="series_points",
                        target_train_hours=3.0, ref_throughput_tokens_per_s=3_700_000,
                        train_tokens=39_960_000_000)
    b = contract_block(c)
    assert b["batch_size"] == 64 and b["context_length"] == 4096
    assert b["budget_denomination"] == "series_points" and b["token_budget"] == 39_960_000_000
    assert "max_train_seconds" not in b                       # absent on the fake ⇒ omitted


def test_build_and_dump_are_sorted_json():
    doc = build_training_summary("777", 1234, {"batch_size": 64}, [{"role": "king", "measured": False}])
    text = dump_training_summary(doc)
    back = json.loads(text)
    assert back["kind"] == "training_summary" and back["telemetry_only"] is True
    assert back["warm_start_ckpt"] == "" and back["warm_start_size"] == ""   # random init
    assert back["round_id"] == "777" and back["created_block"] == 1234
    assert back["legs"][0]["role"] == "king"
    assert text == json.dumps(back, indent=2, sort_keys=True)


class _Store:
    def __init__(self, acl_ok=True):
        self.texts, self.acls, self.acl_ok = {}, {}, acl_ok

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        if acl is not None and not self.acl_ok:
            raise StorageError("no canned ACLs here")
        self.texts[key] = text
        self.acls[key] = acl

    def get_text(self, key):
        if key not in self.texts:
            raise StorageError(f"missing {key}")
        return self.texts[key]


def test_publish_is_public_read_with_a_private_fallback():
    s = _Store()
    assert publish_training_summary(s, "{}", "777") == "training/round-777.json"
    assert s.acls["training/round-777.json"] == "public-read"
    s2 = _Store(acl_ok=False)
    publish_training_summary(s2, "{}", "777")
    assert s2.acls["training/round-777.json"] is None and s2.texts["training/round-777.json"] == "{}"


def _fake_runner(manifest_store, logs_store, leg_summaries):
    contract = SimpleNamespace(arch_preset="22m", batch_size=64, context_length=4096,
                               budget_denomination="series_points", batch_denomination="series",
                               target_train_hours=3.0, ref_throughput_tokens_per_s=3_700_000,
                               max_train_seconds=18000, train_tokens=39_960_000_000)
    cfg = SimpleNamespace(throne_contracts=lambda: [contract],
                          storage=SimpleNamespace(manifest_bucket="cascade-manifests"))
    return SimpleNamespace(cfg=cfg, _leg_summaries=leg_summaries,
                           logs_store=lambda: logs_store, manifest_store=lambda: manifest_store)


def test_trainer_hook_publishes_local_and_pod_legs():
    """The king trained in-process (row cached under its log role); the
    challenger trained on a pod (its worker flushed the log to the logs
    bucket). Both land in one public-read doc keyed by the round."""
    ms, ls = _Store(), _Store()
    ls.texts[log_key("777", "challenger-22m")] = json.dumps({"event": "host"}) + "\n" + json.dumps(DONE) + "\n"
    runner = _fake_runner(ms, ls, {"777": {"king-22m": SUMMARY}, "778": {"king-22m": SUMMARY}})
    m = _manifest([_entry("king", KING_PTR, 7), _entry("challenger", CHAL_PTR, 11)],
                  warm_start_ckpt=CHAL_PTR, warm_start_size="22m")

    key = TrainerRunner._publish_training_summary(runner, m)

    assert key == "training/round-777.json" and ms.acls[key] == "public-read"
    doc = json.loads(ms.texts[key])
    assert doc["round_id"] == "777" and doc["created_block"] == 1234
    # the init every leg continued — the dashboard's link back along the lineage
    assert doc["warm_start_ckpt"] == CHAL_PTR and doc["warm_start_size"] == "22m"
    assert doc["contract"]["batch_size"] == 64 and doc["contract"]["token_budget"] == 39_960_000_000
    king, chal = doc["legs"]
    assert king["source"] == "trainer" and king["steps"] == 152 and king["channel_tokens"] == 78_000
    assert chal["source"] == "log" and chal["measured"] and chal["miner_uid"] == 11
    assert "contract" not in king                             # primary size: top-level block applies
    # the published round's cached rows are released; another round's stay
    assert "777" not in runner._leg_summaries and "778" in runner._leg_summaries


def test_trainer_hook_never_raises():
    class _Broken:
        def put_text(self, *a, **k):
            raise RuntimeError("bucket offline")

        def get_text(self, key):
            raise RuntimeError("bucket offline")

    runner = _fake_runner(_Broken(), _Broken(), {})
    m = _manifest([_entry("king", KING_PTR, 7)])
    assert TrainerRunner._publish_training_summary(runner, m) is None


@pytest.mark.parametrize("denom", ["points", "series_points"])
def test_trainer_metrics_carry_exact_channel_tokens(tmp_path, denom):
    """``channel_tokens`` counts every value entry (B×C×L) whatever the billing
    denomination — equal to ``tokens_seen`` at C = 1, C× it on a wide corpus."""
    pytest.importorskip("torch")
    import numpy as np

    from cascade.trainer.toto2_trainer import Toto2Trainer

    contract = SimpleNamespace(
        context_length=16, horizon=8, patch_size=4, d_model=16, num_layers=1,
        num_heads=1, head_dim=16, mlp_expansion=2, num_quantiles=9,
        batch_size=2, max_train_seconds=30, base_lr=1e-3, weight_decay=0.0,
        optimizer="adamw", warmup_tokens=0, input_transform="arcsinh_causal",
        batch_denomination="series", budget_denomination=denom,
    )
    rng = np.random.default_rng(0)
    series = ([rng.normal(size=(2, 32)).cumsum(axis=-1) for _ in range(4)]
              + [rng.normal(size=32).cumsum() for _ in range(2)])
    result = Toto2Trainer(device="cpu", deterministic=False).train(
        iter(series), contract, training_seed=1, token_budget=10**6, out_dir=tmp_path / "ckpt")
    assert result.metrics["channel_tokens"] == 4 * 2 * 16 + 2 * 16
    if denom == "points":
        assert result.metrics["channel_tokens"] == result.metrics["tokens_seen"]
    else:
        assert result.metrics["channel_tokens"] > result.metrics["tokens_seen"]
