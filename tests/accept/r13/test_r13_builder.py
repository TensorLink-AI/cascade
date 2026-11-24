"""R13: the elastic pool_d builder on Shadeform, CPU only (README.md)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from .helpers import CONFIG, MARKET, T0, FakeClock, FakeShadeform, config, mod, shade_type


def _client(m, cfg, fake):
    return m.ShadeformCpuOnly(cfg.builder, inner=fake)


def test_shipped_config_parses_with_the_owner_defaults():
    m = mod()
    cfg = m.load_config(CONFIG)
    b, e = cfg.builder, cfg.eval_host
    assert b.provider == "shadeform" and e.provider == "lium"
    assert b.min_disk_gb >= 150 and b.max_price_per_hour > 0 and b.monthly_cap_usd > 0
    assert b.build_hours <= 3.0 and not b.keep_warm
    assert ("massedcompute", "cpu_medium") in b.types and ("verda", "cpu_16") in b.types
    assert e.min_vram_gb >= 96 and e.max_price_per_hour > 0 and e.monthly_cap_usd > 0
    assert e.per_host_entries >= 1 and e.max_hosts >= 1


@pytest.mark.parametrize("table,body,code", [
    ("pool_d_builder", 'provider = "lium"', "usage"),
    ("eval_host", 'provider = "shadeform"', "usage"),
    ("pool_d_builder", "max_price_per_hour = 0", "usage"),
    ("eval_host", "monthly_cap_usd = 0", "usage"),
    ("pool_d_builder", "min_disk_gb = 100", "usage"),
    ("pool_d_builder", 'api_key = "x"', "usage"),
    ("pool_d_builder", "bogus = 1", "usage"),
])
def test_config_refusals(tmp_path, table, body, code):
    m = mod()
    p = tmp_path / "e.toml"
    p.write_text(f'schema = "laned_elastic/1"\n[{table}]\n{body}\n')
    with pytest.raises(m.ElasticError) as ei:
        m.load_config(p)
    assert ei.value.code == code
    assert '"x"' not in str(ei.value)          # a secret-shaped value is never echoed


def test_cpu_offers_cheapest_available_and_never_a_gpu_type():
    m = mod()
    b = m.load_config(CONFIG).builder
    offers = m.shadeform_cpu_offers(MARKET, b)
    ids = [o.type_id for o in offers]
    assert "gpu_cheap" not in ids and "cpu_gone" not in ids
    assert all(o.num_gpus == 0 for o in offers)
    assert ids == ["cpu_small", "cpu_medium", "cpu_16"]          # cheapest first
    assert offers[0].price_per_hour == pytest.approx(0.22)      # cents → USD
    tight = replace(b, min_memory_gb=64, max_price_per_hour=0.45)
    assert [o.type_id for o in m.shadeform_cpu_offers(MARKET, tight)] == ["cpu_medium"]


def test_shadeform_client_refuses_gpu_types(tmp_path):
    m = mod()
    cfg = config(tmp_path)
    fake = FakeShadeform()
    cl = _client(m, cfg, fake)
    gpu = m.CpuOffer("massedcompute", "us-east", "gpu_cheap", 32, 128, 2000, 0.10, num_gpus=1)
    with pytest.raises(m.ElasticError) as ei:
        cl.create(gpu, name="x", ttl_seconds=3600, spend_threshold_usd=1)
    assert ei.value.code == "gpu_refused"
    # an offer claiming 0 GPUs whose listing has GPUs is refused too
    lying = replace(gpu, num_gpus=0)
    with pytest.raises(m.ElasticError) as ei:
        cl.create(lying, name="x", ttl_seconds=3600, spend_threshold_usd=1)
    assert ei.value.code == "gpu_refused"
    assert not [p for p in fake.posts if p[0] == "/instances/create"]


def test_dry_run_prints_choice_and_rents_nothing(tmp_path):
    m = mod()
    cfg = config(tmp_path)
    fake = FakeShadeform()
    out = m.builder_up(cfg, _client(m, cfg, fake), dry_run=True, clock=FakeClock())
    assert out["dry_run"] and out["type"] == "cpu_small" and out["price_per_hour"] == 0.22
    assert out["provider"] == "shadeform"
    assert fake.posts == []


def test_up_creates_vm_mode_with_backstops(tmp_path):
    m = mod()
    cfg = config(tmp_path)
    fake = FakeShadeform()
    out = m.builder_up(cfg, _client(m, cfg, fake), dry_run=False, clock=FakeClock())
    (path, body), = fake.posts
    assert path == "/instances/create"
    assert body["ssh_key_id"] == "key-123" and "launch_configuration" not in body
    assert body["auto_delete"]["date_threshold"]
    assert 0 < float(body["auto_delete"]["spend_threshold"]) <= 0.22 * cfg.builder.ttl_hours
    assert out["instance"]["ip"] == "203.0.113.7"
    rows = m.SpendLedger(tmp_path / "state" / "spend.jsonl").rows()
    assert rows[0]["event"] == "open" and rows[0]["price_per_hour"] == 0.22


def test_price_cap_and_monthly_cap_refuse_before_renting(tmp_path):
    m = mod()
    cfg = config(tmp_path)
    led = m.SpendLedger(tmp_path / "state" / "spend.jsonl")
    with pytest.raises(m.ElasticError) as ei:
        m.check_spend(price=0.60, cap_per_hour=0.50, ledger=led, role=m.BUILDER, hours=3,
                      monthly_cap=100, count=1, now=T0)
    assert ei.value.code == "price_cap"
    # $349 already billed this month: one more 3-h build at $0.44 breaks $350
    led.open(role=m.BUILDER, instance_id="old", provider="shadeform", type_id="t",
             price_per_hour=349.0, ts=T0 - 7200)
    led.close(role=m.BUILDER, instance_id="old", ts=T0 - 3600)
    assert led.spent(m.BUILDER, T0) == pytest.approx(349.0)
    fake = FakeShadeform(market={"instance_types": [
        shade_type("massedcompute", "cpu_medium", vcpus=16, mem=80, disk=1600, cents=44)]})
    with pytest.raises(m.ElasticError) as ei:
        m.builder_up(cfg, _client(m, cfg, fake), dry_run=False, clock=FakeClock())
    assert ei.value.code == "monthly_cap"
    assert fake.posts == []


def test_ledger_counts_only_this_month():
    m = mod()
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        led = m.SpendLedger(Path(d) / "s.jsonl")
        m0 = m.month_start(T0)
        led.open(role=m.BUILDER, instance_id="a", provider="shadeform", type_id="t",
                 price_per_hour=1.0, ts=m0 - 3600)
        led.close(role=m.BUILDER, instance_id="a", ts=m0 + 3600)
        led.open(role=m.BUILDER, instance_id="b", provider="shadeform", type_id="t",
                 price_per_hour=2.0, ts=T0 - 1800)            # still open: accrues
        assert led.spent(m.BUILDER, T0) == pytest.approx(1.0 + 1.0)
        assert led.spent(m.EVAL_HOST, T0) == 0


def test_failed_boot_deletes_and_verifies(tmp_path):
    m = mod()
    cfg = config(tmp_path)
    fake = FakeShadeform(ready=False)
    with pytest.raises(m.ElasticError):
        m.builder_up(cfg, _client(m, cfg, fake), dry_run=False, clock=FakeClock())
    assert fake.deleted == ["sf-1"] and not fake.live
    rows = m.SpendLedger(tmp_path / "state" / "spend.jsonl").rows()
    assert [r["event"] for r in rows] == ["open", "close"]


def test_unverified_delete_alerts(tmp_path):
    m = mod()
    cfg = config(tmp_path)
    fake = FakeShadeform(ready=False, delete_works=False)
    with pytest.raises(m.ElasticError):
        m.builder_up(cfg, _client(m, cfg, fake), dry_run=False, clock=FakeClock())
    alerts = (tmp_path / "state" / "alerts.jsonl").read_text()
    assert "teardown_unverified" in alerts


def test_run_script_is_the_w6_sequence():
    m = mod()
    b = m.load_config(CONFIG).builder
    s = m.builder_run_script(b, as_of_ts=T0 - 300, now_ts=T0, started_ts=T0)
    order = ["elastic lane-a-pool", "cascade.pool.first_seen record",
             "cascade.pool.first_seen commit", "cascade.pool.pool_d_build",
             "cascade.pool.pool_d_publish publish", "elastic state-push"]
    pos = [s.index(x) for x in order]
    assert pos == sorted(pos)
    assert f"--as-of-ts {T0 - 300}" in s and f"--started-ts {T0}" in s
    assert '--finished-ts "$finished"' in s and "--prior-dir" in s
    assert "set -euo pipefail" in s


def _runner(log, fail_on=None):
    def run(argv):
        log.append(argv)
        joined = " ".join(argv)
        return 1 if fail_on and fail_on in joined else 0
    return run


def test_cycle_builds_then_deletes(tmp_path):
    m = mod()
    cfg = config(tmp_path)
    fake = FakeShadeform()
    log = []
    out = m.cycle(cfg, _client(m, cfg, fake), dry_run=False, runner=_runner(log),
                  clock=FakeClock())
    assert out["down"]["deleted"] == ["sf-1"] and not fake.live
    joined = [" ".join(a) for a in log]
    assert any("pin-host-key" in j for j in joined)
    assert any("pod_bootstrap.sh" in j for j in joined)
    assert any("mirror-sync" in j for j in joined)
    assert any("pool_d_publish publish" in j for j in joined)
    # every ssh pins the host key
    for a in log:
        if a[0] in ("ssh", "scp"):
            assert "StrictHostKeyChecking=yes" in a
            assert not any("accept-new" in x or "/dev/null" in x for x in a)


def test_cycle_deletes_on_failure_even_with_keep_warm(tmp_path):
    m = mod()
    cfg = config(tmp_path)
    cfg = replace(cfg, builder=replace(cfg.builder, keep_warm=True))
    fake = FakeShadeform()
    with pytest.raises(m.ElasticError) as ei:
        m.cycle(cfg, _client(m, cfg, fake), dry_run=False,
                runner=_runner([], fail_on="pool_d_publish"), clock=FakeClock())
    assert ei.value.code == "step_failed"
    assert not fake.live


def test_keep_warm_keeps_the_vm_after_a_good_build(tmp_path):
    m = mod()
    cfg = config(tmp_path)
    cfg = replace(cfg, builder=replace(cfg.builder, keep_warm=True))
    fake = FakeShadeform()
    out = m.cycle(cfg, _client(m, cfg, fake), dry_run=False, runner=_runner([]),
                  clock=FakeClock())
    assert "down" not in out and fake.live == {"sf-1"}
    again = m.cycle(cfg, _client(m, cfg, fake), dry_run=False, runner=_runner([]),
                    clock=FakeClock())
    assert again["up"]["reused"] and len(fake.posts) == 1


def test_env_template_names_only_mode_600(tmp_path):
    m = mod()
    p = tmp_path / "pool-d.env"
    out = m.env_template(p)
    assert out["created"] and (p.stat().st_mode & 0o777) == 0o600
    lines = [ln for ln in p.read_text().splitlines() if not ln.startswith("#")]
    assert lines and all(ln.endswith("=") for ln in lines)
    p.write_text("HIPPIUS_S3_ACCESS_KEY=filled\n")
    assert not m.env_template(p)["created"]
    assert "filled" in p.read_text()              # never overwritten
    p.chmod(0o644)
    with pytest.raises(m.ElasticError):
        m.check_env_file(p)
