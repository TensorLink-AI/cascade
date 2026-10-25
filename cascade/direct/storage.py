"""Immutable content-addressed objects: shared filesystem, pod-local and Hippius S3."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


def sha256(body):
    return hashlib.sha256(body).hexdigest()


class FileObjects:
    """Durable shared filesystem for local/staging workers; not cross-cloud storage."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.prefix = self.root.as_uri() + "/"

    def _path(self, uri):
        if not uri.startswith(self.prefix):
            raise ValueError("object outside configured store")
        name = uri[len(self.prefix) :]
        if len(name) != 64 or any(c not in "0123456789abcdef" for c in name):
            raise ValueError("object key must be sha256")
        return self.root / name

    def put(self, uri, body):
        path = self._path(uri)
        if path.name != sha256(body):
            raise ValueError("object content digest mismatch")
        fd, temporary = tempfile.mkstemp(dir=self.root, prefix=".upload-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(temporary, path)  # atomic create-only publication
            except FileExistsError:
                self.verify(uri, path.name)
            directory = os.open(self.root, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.unlink(temporary)

    def get(self, uri):
        path = self._path(uri)
        body = path.read_bytes()
        if sha256(body) != path.name:
            raise ValueError("corrupt object")
        return body

    def verify(self, uri, expected):
        if sha256(self.get(uri)) != expected:
            raise ValueError("object checksum mismatch")


class PodObjects(FileObjects):
    """The credential-free worker store on a rented pod (``kind=pod``).

    Pods hold no storage credentials (DEC-CA-0036): the orchestrator stages
    every object a job reads into this directory over SSH and harvests what the
    worker writes. Reads resolve by content digest whatever store the manifest
    names (``hippius://…/<sha>`` resolves to ``<root>/<sha>``); writes publish
    ``file://`` URIs the orchestrator rewrites after upload.
    """

    def _path(self, uri):
        name = uri.rsplit("/", 1)[-1]
        if len(name) != 64 or any(c not in "0123456789abcdef" for c in name):
            raise ValueError("object key must be sha256")
        return self.root / name


class HippiusObjects:
    """Content-addressed objects in a Hippius S3 bucket (``kind=hippius``).

    Credentials come from the environment (``HIPPIUS_S3_ACCESS_KEY`` /
    ``HIPPIUS_S3_SECRET_KEY``), only on the orchestrator. Keys are the SHA-256
    of the body, so a write is idempotent and a read is verified end to end.
    ``public_read`` writes objects world-readable so anyone can fetch the
    checkpoints and reproduce the trainer's scores (``cascade-audit``).
    """

    def __init__(self, client, bucket, prefix, *, public_read=False):
        if not bucket or not prefix.strip("/"):
            raise ValueError("explicit bucket/prefix required")
        self.client, self.bucket = client, bucket
        self.acl = {"ACL": "public-read"} if public_read else {}
        self.key_prefix = prefix.strip("/") + "/"
        self.prefix = f"hippius://{bucket}/{self.key_prefix}"

    def _key(self, uri):
        if not uri.startswith(self.prefix):
            raise ValueError("object outside configured Hippius prefix")
        name = uri[len(self.prefix) :]
        if len(name) != 64 or any(c not in "0123456789abcdef" for c in name):
            raise ValueError("object key must be sha256")
        return self.key_prefix + name

    def exists(self, uri):
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(uri))
            return True
        except Exception as exc:
            from ..shared.hippius import _is_missing_object

            if _is_missing_object(exc):
                return False
            raise OSError(f"hippius head failed: {exc}") from None

    def put(self, uri, body):
        key = self._key(uri)
        checksum = sha256(body)
        if key.rsplit("/", 1)[-1] != checksum:
            raise ValueError("object content digest mismatch")
        if self.exists(uri):
            return
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body, **self.acl)

    def put_file(self, uri, path):
        """Stream a harvested file up without holding it in memory."""
        key = self._key(uri)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8 * 1024 * 1024):
                h.update(chunk)
        if key.rsplit("/", 1)[-1] != h.hexdigest():
            raise ValueError("object content digest mismatch")
        if not self.exists(uri):
            self.client.upload_file(str(path), self.bucket, key,
                                    **({"ExtraArgs": self.acl} if self.acl else {}))

    def get(self, uri):
        stream = self.client.get_object(Bucket=self.bucket, Key=self._key(uri))["Body"]
        try:
            body = stream.read()
        finally:
            stream.close()
        if sha256(body) != uri.rsplit("/", 1)[-1]:
            raise ValueError("Hippius object checksum mismatch")
        return body

    def download(self, uri, path):
        """Stream an object to ``path`` and verify it (staging to a pod)."""
        stream = self.client.get_object(Bucket=self.bucket, Key=self._key(uri))["Body"]
        h = hashlib.sha256()
        try:
            with open(path, "wb") as f:
                while chunk := stream.read(8 * 1024 * 1024):
                    h.update(chunk)
                    f.write(chunk)
        finally:
            stream.close()
        if h.hexdigest() != uri.rsplit("/", 1)[-1]:
            Path(path).unlink(missing_ok=True)
            raise ValueError("Hippius object checksum mismatch")

    def verify(self, uri, expected):
        # Stream verification avoids holding bulk tensors on the controller.
        stream = self.client.get_object(Bucket=self.bucket, Key=self._key(uri))["Body"]
        h = hashlib.sha256()
        try:
            while chunk := stream.read(8 * 1024 * 1024):
                h.update(chunk)
        finally:
            stream.close()
        if h.hexdigest() != expected:
            raise ValueError("Hippius object checksum mismatch")


def open_objects(config):
    if config["kind"] == "file":
        return FileObjects(config["root"])
    if config["kind"] == "pod":
        return PodObjects(config["root"])
    if config["kind"] != "hippius":
        raise ValueError("unsupported object storage")
    from ..shared.hippius import S3Config, _s3_client

    return HippiusObjects(
        _s3_client(S3Config(endpoint=config.get("endpoint", "https://s3.hippius.com"),
                            region=config.get("region", "decentralized"),
                            bucket=config["bucket"])),
        config["bucket"],
        config["prefix"],
        public_read=config.get("public_read", False),
    )


def _anonymous_client(endpoint, region):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", endpoint_url=endpoint, region_name=region,
                        config=Config(signature_version=UNSIGNED, s3={"addressing_style": "path"}))


def anonymous_objects(uri, *, endpoint="https://s3.hippius.com", region="decentralized"):
    """Read-only, credential-free access to the public objects under ``uri``'s prefix."""
    bucket, _, rest = uri.removeprefix("hippius://").partition("/")
    return HippiusObjects(_anonymous_client(endpoint, region), bucket, rest.rsplit("/", 1)[0])


def public_exists(bucket, key, *, endpoint="https://s3.hippius.com", region="decentralized",
                  client=None):
    """Whether ``key`` is readable WITHOUT credentials (a credential-free HEAD).

    Absent and private (403) both read as False; any other failure raises, so a
    network fault never passes for a missing object or the reverse.
    """
    from ..shared.hippius import _is_missing_object

    client = client or _anonymous_client(endpoint, region)
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        resp = getattr(exc, "response", None) or {}
        denied = (str(resp.get("Error", {}).get("Code", "")) in {"403", "AccessDenied", "Forbidden"}
                  or resp.get("ResponseMetadata", {}).get("HTTPStatusCode") == 403)
        if _is_missing_object(exc) or denied:
            return False
        raise OSError(f"hippius head failed: {exc}") from None


def publish(objects, body):
    checksum = sha256(body)
    uri = objects.prefix + checksum
    objects.put(uri, body)
    return {"uri": uri, "sha256": checksum}
