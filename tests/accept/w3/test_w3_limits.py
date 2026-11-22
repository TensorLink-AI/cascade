"""W3: rate limits per IP and per hotkey (deterministic test clock) and the
RAM budget (bounded bodies, bounded in-memory structures). Contract:
README.md "Rate limits" and "RAM budget"."""

from __future__ import annotations

import io

from .helpers import QUEUE, SUBMIT, Env, Miner, NoRead, call, checkpoint

# ── per IP ───────────────────────────────────────────────────────────────────


def test_rate_limit_constants_are_pinned():
    from cascade.lane_d import service

    for name in ("DEFAULT_PER_IP_RPM", "DEFAULT_PER_HOTKEY_RPM"):
        v = getattr(service, name)
        assert type(v) is int and 1 <= v <= 600, (name, v)
    assert type(service.RATE_LIMIT_MAX_KEYS) is int
    assert 1024 <= service.RATE_LIMIT_MAX_KEYS <= 16384
    assert type(service.MAX_BODY_BYTES) is int
    assert 4096 <= service.MAX_BODY_BYTES <= 65536


def test_per_ip_limit_is_a_sliding_minute(tmp_path):
    env = Env(tmp_path, per_ip_rpm=3)
    for _ in range(3):
        assert call(env.app, "GET", QUEUE, ip="198.51.100.9").status == 200
    r = call(env.app, "GET", QUEUE, ip="198.51.100.9")
    assert r.status == 429 and r.code == "rate_limited", r
    assert int(r.headers["retry-after"]) >= 1
    # another client is unaffected
    assert call(env.app, "GET", QUEUE, ip="198.51.100.10").status == 200
    env.clock.t += 61
    assert call(env.app, "GET", QUEUE, ip="198.51.100.9").status == 200


def test_per_ip_limit_counts_refused_and_unknown_requests(tmp_path):
    env = Env(tmp_path, per_ip_rpm=2)
    assert call(env.app, "POST", SUBMIT, ip="198.51.100.11").status == 400
    assert call(env.app, "GET", "/v1/lane-d/nope", ip="198.51.100.11").status == 404
    r = call(env.app, "GET", QUEUE, ip="198.51.100.11")
    assert r.status == 429 and r.code == "rate_limited", r


def test_client_ip_is_the_proxys_forwarded_for_only_from_loopback(tmp_path):
    env = Env(tmp_path, per_ip_rpm=1)
    # behind Caddy (REMOTE_ADDR loopback): X-Forwarded-For names the client
    assert call(env.app, "GET", QUEUE, ip="198.51.100.20").status == 200
    assert call(env.app, "GET", QUEUE, ip="198.51.100.21").status == 200
    # a direct (non-loopback) peer cannot pick its identity with the header
    hdr = {"X-Forwarded-For": "198.51.100.30"}
    assert call(env.app, "GET", QUEUE, remote="192.0.2.40", headers=hdr).status == 200
    hdr = {"X-Forwarded-For": "198.51.100.31"}
    r = call(env.app, "GET", QUEUE, remote="192.0.2.40", headers=hdr)
    assert r.status == 429 and r.code == "rate_limited", r


# ── per hotkey ───────────────────────────────────────────────────────────────


def test_per_hotkey_limit_spans_ips(tmp_path):
    env = Env(tmp_path, per_hotkey_rpm=2)
    m = Miner(env, "rl-a")
    files = checkpoint("a")
    m.ip = "198.51.100.40"
    assert m.submit(files).status == 201
    m.ip = "198.51.100.41"
    assert m.submit(files).status == 201
    m.ip = "198.51.100.42"
    r = m.submit(files)
    assert r.status == 429 and r.code == "rate_limited", r
    env.clock.t += 61
    m._ts = int(env.clock.t) - 250
    assert m.submit(files).status == 201


def test_unverified_requests_never_spend_a_hotkeys_budget(tmp_path):
    """A forger sending the victim's hotkey header with bad signatures from
    many IPs cannot rate-limit the victim."""
    env = Env(tmp_path, per_hotkey_rpm=2)
    victim = Miner(env, "rl-victim")
    Miner(env, "rl-forger", coldkey="ck-f")
    files = checkpoint("v")
    for i in range(6):
        victim.ip = f"198.51.100.{60 + i}"
        r = victim.submit(files, signer="rl-forger")
        assert r.status == 400 and r.code == "bad_signature", r
    victim.ip = "198.51.100.90"
    assert victim.submit(files).status == 201


# ── RAM budget ───────────────────────────────────────────────────────────────


def test_oversize_body_is_refused_unread(tmp_path):
    from cascade.lane_d import service

    env = Env(tmp_path)
    m = Miner(env, "ram-a")
    h = m.headers("submit", "ab" * 32, b"")
    r = call(env.app, "POST", SUBMIT, ip=m.ip, headers=h, stream=NoRead(),
             content_length=service.MAX_BODY_BYTES + 1)
    assert r.status == 413 and r.code == "body_too_large", r
    r = call(env.app, "POST", SUBMIT, ip=m.ip, headers=h, stream=NoRead(),
             content_length=2_500_000_000)
    assert r.status == 413 and r.code == "body_too_large", r


def test_bad_content_length_is_refused_unread(tmp_path):
    env = Env(tmp_path)
    for raw in (-1, "abc"):
        from wsgiref.util import setup_testing_defaults

        environ: dict = {}
        setup_testing_defaults(environ)
        environ.update({"REQUEST_METHOD": "POST", "PATH_INFO": SUBMIT, "REMOTE_ADDR": "127.0.0.1",
                        "HTTP_X_FORWARDED_FOR": "198.51.100.70", "CONTENT_LENGTH": str(raw),
                        "wsgi.input": NoRead()})
        got = {}

        def start_response(status, headers, exc_info=None, got=got):
            got["status"] = int(status.split()[0])
            return lambda _data: None

        b"".join(env.app(environ, start_response))
        assert 400 <= got["status"] < 500, (raw, got)


def test_body_reads_stop_at_content_length(tmp_path):
    """A lying client streams more than it declared: the app reads at most
    the declared length."""
    env = Env(tmp_path)
    m = Miner(env, "ram-b")

    class Counting(io.BytesIO):
        def __init__(self):
            super().__init__(b"{" * 1_000_000)
            self.taken = 0

        def read(self, n=-1):
            chunk = super().read(n)
            self.taken += len(chunk)
            return chunk

    s = Counting()
    h = m.headers("submit", "ab" * 32, b"{" * 100)
    call(env.app, "POST", SUBMIT, ip=m.ip, headers=h, stream=s, content_length=100)
    assert s.taken <= 100, s.taken


def test_rate_limiter_memory_is_bounded(tmp_path):
    from cascade.lane_d import service

    env = Env(tmp_path, per_ip_rpm=5)
    n = service.RATE_LIMIT_MAX_KEYS + 300
    for i in range(n):
        call(env.app, "POST", SUBMIT, ip=f"10.{(i >> 16) & 255}.{(i >> 8) & 255}.{i & 255}")
    assert env.app.rate_limit_keys() <= service.RATE_LIMIT_MAX_KEYS


def test_one_record_per_hotkey_however_often_it_resubmits(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "ram-c")
    for i in range(25):
        assert m.submit(checkpoint(f"r{i}")).status == 201
    assert len([e for e in env.queue() if e["hotkey"] == m.hotkey]) == 1
