"""r47/r48 provisioner fixes (2026-08-28), all three pure-logic seams:

1. VM-boot env pin — the config-push hook writes the chain.toml
   ``train_image_digest`` into the pod's /etc/environment + .bashrc over SSH,
   idempotently, so shadeform VM-boot pods stop dispatching with the STALE
   digest baked into the image at CI time (final workers refused the round
   with TrainImageMismatch; a manual ssh patcher ran every round until now).
2. lium digest-ref degradation — lium's API 400-rejects ``repo@sha256:<64hex>``
   refs, so lium launches send the tag/repo form; the pin is still enforced by
   the launch env digest + the health gate's byte comparison.
3. Cohort-aware final sizing — the duel cohort can never exceed the revealed
   field, so a king-only round (0 fresh submissions) rents 1 final slot, not
   the full 1 + max_finalists fleet (2 pods / 4 slots, ~$10 wasted in r48).
"""

from __future__ import annotations

import types

from cascade.provision.core import (
    LiumProvider,
    PodAddress,
    image_digest_of,
    lium_image_ref,
)
from cascade.provision.main import digest_env_command, make_config_push
from cascade.provision.policy import ProvisionPolicy, StagePolicy, size_fleet
from tests.unit.test_provision_loop import (
    FakeProvider,
    cycle,
    make_loop,
)

DIGEST = "sha256:" + "a" * 64
IMG = f"reg.example/cascade-worker@{DIGEST}"
IMG_TAGGED = f"reg.example/cascade-worker:v0.7.0@{DIGEST}"


def _policy(*, heat_gpus=8, heat_max=4, final_gpus=2, final_max=4):
    return ProvisionPolicy(
        heat=StagePolicy(sku="NVIDIA RTX A6000", gpus_per_pod=heat_gpus,
                         max_pods=heat_max, providers=("lium",), max_price_hr=4.0),
        final=StagePolicy(sku="NVIDIA L40S", gpus_per_pod=final_gpus,
                          max_pods=final_max, providers=("lium",), max_price_hr=3.0),
        trigger_margin_blocks=25,
        max_spend_per_round=25.0,
    )


# ── FIX 2: lium digest-ref degradation ───────────────────────────────────────


def test_lium_image_ref_strips_digest_keeps_tag():
    # repo:tag@sha256 → repo:tag (lium's parser 400s on the full ref)
    assert lium_image_ref(IMG_TAGGED) == "reg.example/cascade-worker:v0.7.0"


def test_lium_image_ref_strips_digest_bare_repo():
    # digest-only pin, no tag → bare repo
    assert lium_image_ref(IMG) == "reg.example/cascade-worker"


def test_lium_image_ref_leaves_unpinned_refs_alone():
    assert lium_image_ref("reg.example/cascade-worker:v0.7.0") == \
        "reg.example/cascade-worker:v0.7.0"
    assert lium_image_ref("ubuntu:22.04") == "ubuntu:22.04"
    assert lium_image_ref("") == ""
    # a malformed @suffix that is not a sha256 digest is passed through
    assert lium_image_ref("repo@md5:abc") == "repo@md5:abc"


def test_lium_launch_degrades_ref_but_keeps_digest_env(caplog):
    """The launch call must never carry @sha256 (lium 400s: 'Digest must be
    sha256: followed by 64 hex characters'), while CASCADE_TRAIN_IMAGE_DIGEST
    still carries the FULL pin for the health gate / final workers."""
    spawned: list[list[str]] = []

    def _run(argv):
        out = '[{"id": "exec-1"}]' if "ls" in argv else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    prov = LiumProvider(bin="lium", _run=_run, _spawn=lambda argv: spawned.append(argv))
    from cascade.provision.core import LaunchSpec

    with caplog.at_level("WARNING", logger="cascade.provision.core"):
        prov.launch(LaunchSpec(sku="L40S", count=1, image=IMG_TAGGED,
                               ssh_pubkey="ssh-ed25519 AAAA orch"))
    up = spawned[0]
    assert up[up.index("--image") + 1] == "reg.example/cascade-worker:v0.7.0"
    assert not any("@sha256:" in a for a in up if a.startswith("reg.example")), up
    assert f"CASCADE_TRAIN_IMAGE_DIGEST={DIGEST}" in up          # pin rides the env
    assert any("degrading image ref" in r.message for r in caplog.records)


def test_lium_launch_unpinned_ref_logs_no_degradation(caplog):
    spawned: list[list[str]] = []

    def _run(argv):
        out = '[{"id": "exec-1"}]' if "ls" in argv else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    prov = LiumProvider(bin="lium", _run=_run, _spawn=lambda argv: spawned.append(argv))
    from cascade.provision.core import LaunchSpec

    with caplog.at_level("WARNING", logger="cascade.provision.core"):
        prov.launch(LaunchSpec(sku="L40S", count=1, image="ubuntu:22.04",
                               ssh_pubkey="ssh-ed25519 AAAA orch"))
    assert spawned[0][spawned[0].index("--image") + 1] == "ubuntu:22.04"
    assert not any("degrading image ref" in r.message for r in caplog.records)


# ── FIX 3: cohort-aware final sizing ─────────────────────────────────────────


def test_king_only_round_rents_one_final_slot():
    """r48: 0 fresh submissions, finalists=1 + max_finalists=3 in config →
    the old sizing rented 1 + 3 = 4 slots (2 pods). The field caps the
    cohort: zero challengers ⇒ king-only ⇒ exactly 1 slot / 1 pod."""
    plan = size_fleet(0, 1, 0.5, 24.0, 3.0, _policy(), max_finalists=3)
    assert plan.final.slots == 1
    assert plan.final.pods == 1                  # the king still trains
    assert plan.heat.pods == 0 and plan.heat.slots == 0


def test_small_field_caps_the_cohort():
    # 2 challengers can produce at most 2 finalists: king + 2 = 3 slots,
    # not 1 + max_finalists = 4.
    plan = size_fleet(2, 1, 0.5, 24.0, 3.0, _policy(), max_finalists=3)
    assert plan.final.slots == 3
    assert plan.heat.pods == 0                   # everyone advances, no screen


def test_large_field_sizing_is_unchanged():
    # The clamp is inert whenever the field covers the cohort cap.
    plan = size_fleet(20, 1, 0.5, 24.0, 3.0, _policy(), max_finalists=3)
    assert plan.final.slots == 4 and plan.final.pods == 2
    base = size_fleet(20, 2, 0.5, 24.0, 3.0, _policy())
    assert base.final.slots == 3


def test_final_slots_now_clamped_by_field(tmp_path):
    """The JIT/retry path's pre-marker prediction obeys the same field cap
    (the marker path already uses the ACTUAL finalist list)."""
    loop, _ = make_loop(tmp_path)
    loop._round_plan = {"finalists": 1, "max_finalists": 3,
                        "eligible_challengers": 0}
    assert loop._final_slots_now() == 1          # king-only round
    loop._round_plan = {"finalists": 1, "max_finalists": 3,
                        "eligible_challengers": 12, "screened_challengers": 2}
    assert loop._final_slots_now() == 3          # post-dedup field of 2
    loop._round_plan = {"finalists": 1, "max_finalists": 3,
                        "eligible_challengers": 12}
    assert loop._final_slots_now() == 4          # cap covered by the field
    loop._round_plan = None
    assert loop._final_slots_now() == 2          # restart fallback: minimal duel


def test_loop_king_only_round_rents_single_final_pod(tmp_path):
    """End-to-end through the service loop: an r48-shaped plan (zero eligible)
    rents exactly one final pod at the margin and no heat fleet."""
    prov = FakeProvider("lium")
    plan = {"block": 880, "epoch_blocks": 900, "next_boundary_block": 900,
            "blocks_to_boundary": 20, "king": "5King", "resolved": 1,
            "challengers": 0, "eligible_challengers": 0,
            "heat_train_hours": 0.5, "finalists": 1, "max_finalists": 3}
    loop, _ = make_loop(tmp_path, providers={"lium": prov}, plan=plan)
    cycle(loop)
    assert prov.launched == ["cascade-900-final-0"]          # one pod, no heat


# ── FIX 1: VM-boot CASCADE_TRAIN_IMAGE_DIGEST env pin ────────────────────────


def test_digest_env_command_root_is_idempotent_and_sudoless():
    cmd = digest_env_command(DIGEST, user="root")
    assert "sudo" not in cmd
    # idempotence: delete-then-append on both targets
    assert 'sed -i "/^CASCADE_TRAIN_IMAGE_DIGEST=/d" /etc/environment' in cmd
    assert f"echo CASCADE_TRAIN_IMAGE_DIGEST={DIGEST} >> /etc/environment" in cmd
    assert 'sed -i "/^export CASCADE_TRAIN_IMAGE_DIGEST=/d" "$HOME/.bashrc"' in cmd
    assert f'echo "export CASCADE_TRAIN_IMAGE_DIGEST={DIGEST}" >> "$HOME/.bashrc"' in cmd
    # /etc/environment may not exist on minimal images
    assert "touch /etc/environment" in cmd


def test_digest_env_command_non_root_uses_sudo_for_etc_only():
    cmd = digest_env_command(DIGEST, user="shadeform")
    assert cmd.startswith("sudo -n sh -c '")     # /etc/environment needs root
    # exactly one sudo, at the front: the .bashrc half runs as the login
    # user so it lands in THEIR $HOME, not root's
    assert cmd.count("sudo") == 1


class _RecordingRun:
    def __init__(self, rc=0):
        self.calls: list[list[str]] = []
        self.rc = rc

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        return types.SimpleNamespace(returncode=self.rc, stdout="", stderr="")


def _render(**profiles):
    from cascade.provision.loop import PodProfile, RenderSettings

    return RenderSettings(
        image=IMG, ssh_pubkey="ssh-ed25519 AAAA orch",
        key_path="~/.ssh/cascade_ed25519",
        profiles={k: PodProfile(**v) for k, v in profiles.items()},
    )


def test_config_push_pins_digest_env_over_same_channel(monkeypatch):
    """The hook scp's chain.toml AND ssh-writes the digest pin — the r47/r48
    permanent fix replacing the manual per-round ssh patcher."""
    rec = _RecordingRun()
    monkeypatch.setattr("cascade.provision.main.subprocess.run", rec)
    hook = make_config_push(_render(), box_chain_toml="/root/chain.toml",
                            pod_user="root", image_digest=IMG)
    hook(PodAddress("10.0.0.5", 2222), "final", "")
    assert len(rec.calls) == 2
    scp, ssh = rec.calls
    assert scp[0] == "scp" and "/root/chain.toml" in scp
    assert ssh[0] == "ssh" and "root@10.0.0.5" in ssh
    assert ssh[ssh.index("-p") + 1] == "2222"
    remote = ssh[-1]
    assert f"CASCADE_TRAIN_IMAGE_DIGEST={DIGEST}" in remote
    assert "/etc/environment" in remote and ".bashrc" in remote


def test_config_push_uses_provider_profile_user(monkeypatch):
    """shadeform VM-boot pods land as the ``shadeform`` user — the pin must
    target that user's home and sudo the /etc/environment write."""
    rec = _RecordingRun()
    monkeypatch.setattr("cascade.provision.main.subprocess.run", rec)
    hook = make_config_push(
        _render(shadeform={"user": "shadeform", "workdir": "/home/shadeform/cascade"}),
        box_chain_toml="/root/chain.toml", pod_user="root", image_digest=IMG)
    hook(PodAddress("10.0.0.6", 22), "final", "shadeform")
    ssh = rec.calls[1]
    assert "shadeform@10.0.0.6" in ssh
    assert "sudo -n" in ssh[-1]


def test_config_push_without_pin_skips_the_env_write(monkeypatch):
    # Unpinned deployment (bootstrap/testnet): behaviour is byte-identical to
    # the pre-fix hook — one scp, no ssh.
    rec = _RecordingRun()
    monkeypatch.setattr("cascade.provision.main.subprocess.run", rec)
    hook = make_config_push(_render(), box_chain_toml="/root/chain.toml",
                            pod_user="root", image_digest="")
    hook(PodAddress("10.0.0.7", 22), "heat", "")
    assert len(rec.calls) == 1 and rec.calls[0][0] == "scp"


def test_config_push_env_pin_survives_failed_scp(monkeypatch):
    """A failed chain.toml copy must not strand the digest fix — the two
    writes are independent."""
    calls: list[list[str]] = []

    def _run(argv, **kw):
        calls.append(list(argv))
        rc = 1 if argv[0] == "scp" else 0
        return types.SimpleNamespace(returncode=rc, stdout="", stderr="scp lost")

    monkeypatch.setattr("cascade.provision.main.subprocess.run", _run)
    hook = make_config_push(_render(), box_chain_toml="/root/chain.toml",
                            pod_user="root", image_digest=IMG)
    hook(PodAddress("10.0.0.8", 22), "final", "")
    assert [c[0] for c in calls] == ["scp", "ssh"]


def test_config_push_normalises_a_full_ref_pin():
    # The chain.toml pin may be a full repo@sha256 ref or a bare digest —
    # the env value written is always the bare sha256:<64hex>.
    assert image_digest_of(IMG) == DIGEST
    cmd_from_ref = digest_env_command(DIGEST)
    assert DIGEST in cmd_from_ref and "reg.example" not in cmd_from_ref
