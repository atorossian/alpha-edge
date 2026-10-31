"""AEA-87 staging resume and progress tests using in-memory S3 operations."""
import json
from io import BytesIO
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError

from alpha_edge.market import aea87_release_runner as runner
from alpha_edge.market.aea87_publication_transaction import checksum


class FakeS3:
    def __init__(self, data):
        self.data = dict(data)
    def get_object(self, Bucket, Key):
        if Key not in self.data:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": BytesIO(self.data[Key])}
    def head_object(self, Bucket, Key):
        if Key not in self.data:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ETag": '"source"', "Metadata": {}}
    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, **kw):
        if IfNoneMatch == "*" and Key in self.data:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.data[Key] = Body
    def copy_object(self, Bucket, CopySource, Key, **kw):
        self.data[Key] = self.data[CopySource["Key"]]


def fixture():
    old = b'{"parts":["old"]}'
    part = b"parquet-content"
    op = {"asset_id": "A", "manifest_key": "active", "before_sha": checksum(old),
          "backup_key": "backups/0", "staged_key": "staged", "staged_sha": checksum(part),
          "part_key": "dest"}
    manifest = {"release_id": "AEA87-TEST-RESUME", "plan_key": "plan", "operations": [op]}
    return FakeS3({"active": old, "staged": part}), manifest


def test_resumes_preparing_with_existing_objects(capsys):
    s3, manifest = fixture()
    with patch.object(runner, "entries", side_effect=[[], [{"step": 0, "state": "PREPARING"}]]), patch.object(runner, "write_event") as write:
        runner.stage(s3, "bucket", manifest, workers=2, progress_every=1)
        runner.stage(s3, "bucket", manifest, workers=2, progress_every=1)
    assert s3.data["backups/0"] == s3.data["active"]
    assert s3.data["dest"] == s3.data["staged"]
    assert "1/1" in capsys.readouterr().out
    assert write.call_count == 3


def test_refuses_corrupted_reused_partition():
    s3, manifest = fixture()
    s3.data["dest"] = b"wrong"
    with patch.object(runner, "entries", return_value=[]), patch.object(runner, "write_event"):
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            runner.stage(s3, "bucket", manifest, workers=1, progress_every=1)


def test_refuses_different_release_operations():
    s3, manifest = fixture()
    s3.data["market/repairs/aea87/publications/AEA87-TEST-RESUME/operations.json"] = b"{}"
    with pytest.raises(RuntimeError, match="operations differ"):
        runner.stage(s3, "bucket", manifest, workers=1, progress_every=1)
