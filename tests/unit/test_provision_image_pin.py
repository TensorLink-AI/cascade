"""Two pure-logic provisioner seams:

1. VM-boot env pin — the config-push hook writes the chain.toml
   ``train_image_digest`` into the pod's /etc/environment + .bashrc over SSH,
   idempotently, so shadeform VM-boot pods never dispatch with the STALE
   digest baked into the image at CI time (TrainImageMismatch).
2. lium digest-ref degradation — lium's API 400-rejects ``repo@sha256:<64hex>``
   refs, so lium launches send the tag/repo form; the pin is still enforced by
   the launch env digest + the health gate's byte comparison.
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

DIGEST = "sha256:" + "a" * 64
IMG = f"reg.example/cascade-worker@{DIGEST}"
IMG_TAGGED = f"reg.example/cascade-worker:v0.7.0@{DIGEST}"




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

    payloads = []
    prov = LiumProvider(bin="lium", _run=_run, _spawn=lambda argv: spawned.append(argv),
                        _template=lambda p: payloads.append(p) or "tmpl-t")
    from cascade.provision.core import LaunchSpec

    with caplog.at_level("WARNING", logger="cascade.provision.core"):
        prov.launch(LaunchSpec(sku="L40S", count=1, image=IMG_TAGGED,
                               ssh_pubkey="ssh-ed25519 AAAA orch"))
    up = spawned[0]
    assert up[up.index("--template_id") + 1] == "tmpl-t"
    # The template carries the repo/tag split (lium 400-rejects the digest
    # form) and the exact pin in its env.
    (p,) = payloads
    assert (p["docker_image"], p["docker_image_tag"]) == ("reg.example/cascade-worker", "v0.7.0")
    assert p["environment"]["CASCADE_TRAIN_IMAGE_DIGEST"] == DIGEST   # pin rides the env
    assert any("degrading image ref" in r.message for r in caplog.records)


def test_lium_launch_unpinned_ref_logs_no_degradation(caplog):
    spawned: list[list[str]] = []

    def _run(argv):
        out = '[{"id": "exec-1"}]' if "ls" in argv else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    payloads = []
    prov = LiumProvider(bin="lium", _run=_run, _spawn=lambda argv: spawned.append(argv),
                        _template=lambda p: payloads.append(p) or "tmpl-u")
    from cascade.provision.core import LaunchSpec

    with caplog.at_level("WARNING", logger="cascade.provision.core"):
        prov.launch(LaunchSpec(sku="L40S", count=1, image="ubuntu:22.04",
                               ssh_pubkey="ssh-ed25519 AAAA orch"))
    assert spawned[0][spawned[0].index("--template_id") + 1] == "tmpl-u"
    assert (payloads[0]["docker_image"], payloads[0]["docker_image_tag"]) == ("ubuntu", "22.04")
    assert "CASCADE_TRAIN_IMAGE_DIGEST" not in payloads[0]["environment"]   # unpinned ref
    assert not any("degrading image ref" in r.message for r in caplog.records)














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
