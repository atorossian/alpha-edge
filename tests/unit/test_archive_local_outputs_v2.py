"""Local data archives require remote byte verification before deletion."""

import io
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from alpha_edge.market.archive_local_outputs_v2 import (
    transfer,
    inventory,
    ArchiveError,
)


class FakeS3:
    def __init__(self):
        self.objects = {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kwargs):
        if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body

    def upload_file(self, Filename, Bucket, Key):
        self.objects[Key] = Path(Filename).read_bytes()


def test_archive_requires_verified_remote_copy_before_prune(tmp_path):
    root = tmp_path / "outputs"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "gold.parquet").write_bytes(b"gold v2")
    s3 = FakeS3()
    kwargs = {"root": root, "bucket": "test-bucket", "archive_id": "20261110-local", "s3": s3}
    assert transfer(**kwargs)["status"] == "dry_run"
    assert (root / "nested" / "gold.parquet").exists()
    with pytest.raises(ClientError):
        transfer(**kwargs, prune_verified=True)
    assert (root / "nested" / "gold.parquet").exists()
    receipt = transfer(**kwargs, upload=True)
    assert receipt["status"] == "verified_in_s3"
    assert (root / "nested" / "gold.parquet").exists()
    cleaned = transfer(**kwargs, prune_verified=True)
    assert cleaned["files_removed"] == 1
    assert not (root / "nested" / "gold.parquet").exists()


def test_archive_rejects_symlinks_and_secret_keys(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    (root / "private.key").write_text("sensitive")
    with pytest.raises(ArchiveError, match="Sensitive-looking"):
        inventory(root)
    (root / "private.key").unlink()
    (root / "real.csv").write_text("rows")
    try:
        (root / "link.csv").symlink_to(root / "real.csv")
    except OSError as exc:
        pytest.skip(f"Operating system does not permit symlinks: {exc}")
    with pytest.raises(ArchiveError, match="Non-regular"):
        inventory(root)


def test_changed_archive_must_not_prune(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    path = root / "gold.parquet"
    path.write_bytes(b"v1")
    s3 = FakeS3()
    kwargs = {"root": root, "bucket": "bucket", "archive_id": "archive1", "s3": s3}
    transfer(**kwargs, upload=True)
    path.write_bytes(b"v2")
    with pytest.raises(ArchiveError, match="inventory has changed"):
        transfer(**kwargs, prune_verified=True)
    assert path.read_bytes() == b"v2"
