"""cascade.funding.robots: per-pod Harbor robot mint/revoke."""

from __future__ import annotations

import re

import pytest

from cascade.funding.robots import HarborRobots, RobotError, robot_name


class _Http:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, body):
        self.calls.append((method, url, body))
        return self.responses.pop(0)


def test_create_is_project_scoped_push_only_with_expiry():
    http = _Http([(201, {"id": 9, "name": "robot$cascade+funded-n91-1-abc-x1",
                         "secret": "s3cr3t"})])
    robots = HarborRobots("https://registry.hippius.com/", "Basic b3A6cHc=", http=http)
    cred = robots.create("funded-n91-1-abc-x1", "cascade", duration_days=1)
    method, url, body = http.calls[0]
    assert (method, url) == ("POST", "https://registry.hippius.com/api/v2.0/robots")
    assert body["level"] == "project" and body["duration"] == 1 and body["disable"] is False
    (perm,), = [body["permissions"]]
    assert perm["kind"] == "project" and perm["namespace"] == "cascade"
    assert perm["access"] == [{"resource": "repository", "action": "push"}]
    assert (cred.id, cred.username, cred.secret, cred.project) == (
        9, "robot$cascade+funded-n91-1-abc-x1", "s3cr3t", "cascade")
    assert dict(cred.as_env()) == {"HIPPIUS_HUB_USERNAME": cred.username,
                                   "HIPPIUS_HUB_PASSWORD": "s3cr3t"}
    assert "s3cr3t" not in repr(cred)


def test_create_failures_raise_and_never_return_a_credential():
    for resp in [(403, {"errors": [{"message": "forbidden"}]}),
                 (201, {"id": 1, "name": "robot$x"}),          # no secret
                 (201, {"id": 1, "name": "robot$x", "secret": ""})]:
        robots = HarborRobots("https://r", "Basic x", http=_Http([resp]))
        with pytest.raises(RobotError):
            robots.create("n", "cascade")


def test_delete_is_idempotent_on_404_and_loud_otherwise():
    robots = HarborRobots("https://r", "Basic x", http=_Http([(200, {}), (404, {}), (500, {"raw": "boom"})]))
    assert robots.delete(5) and robots.delete(5)
    with pytest.raises(RobotError):
        robots.delete(5)


def test_minter_refuses_without_an_operator_credential():
    with pytest.raises(RobotError):
        HarborRobots("https://r", "", http=_Http([]))


def test_robot_name_is_harbor_legal_and_attributable():
    name = robot_name(91, "13164094732016089897", "5GjHagMkUTnjkzX31t7LX5gthEiJDagvafk3gJLL43goUnET")
    assert re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", name)
    assert name.startswith("funded-n91-13164094732016089897-5gjhagmkutnj-")
    assert robot_name(91, "1", "x") != robot_name(91, "1", "x")   # retry-safe suffix
