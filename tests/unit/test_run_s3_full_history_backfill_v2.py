"""S3 research backfill verifies bytes and never changes production pointers."""
import hashlib
import io
import json

import pytest
from botocore.exceptions import ClientError

from alpha_edge.market.run_s3_full_history_backfill_v2 import (
    upload_verified, verify_commit, S3BackfillError,
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
        self.objects[Key] = open(Filename, "rb").read()


def test_s3_commit_contains_verified_objects_and_resumes(tmp_path):
    root = tmp_path / "ephemeral"
    root.mkdir()
    (root / "bronze").mkdir()
    (root / "bronze" / "bar.parquet").write_bytes(b"immutable source")
    (root / "audit.csv").write_text("asset,missing\nA,0\n")
    s3 = FakeS3()
    prefix = "dev/market/lake_v2/backfills/abc/end-2026-11-10/cohort-0001"
    result = upload_verified(
        s3, bucket="alpha-edge-algo", prefix=prefix,
        local_dir=root, plan_hash="a" * 64, end="2026-11-10",
    )
    assert len(result["files"]) == 2
    assert verify_commit(s3, "alpha-edge-algo", prefix, "a" * 64) == result
    assert all("/current.json" not in k and k.startswith("dev/market/") for k in s3.objects)
    with pytest.raises(S3BackfillError, match="already committed"):
        upload_verified(s3, bucket="alpha-edge-algo", prefix=prefix,
                        local_dir=root, plan_hash="a" * 64, end="2026-11-10")


def test_s3_tampering_detected_before_cohort_is_reused(tmp_path):
    root = tmp_path / "ephemeral"
    root.mkdir()
    (root / "bars").write_bytes(b"first")
    s3 = FakeS3()
    prefix = "dev/market/lake_v2/backfills/test/end-2026-11-10/cohort-0001"
    upload_verified(s3, bucket="alpha-edge-algo", prefix=prefix,
                    local_dir=root, plan_hash="b" * 64, end="2026-11-10")
    s3.objects[f"{prefix}/files/bars"] = b"tampered"
    with pytest.raises(S3BackfillError, match="checksum mismatch"):
        verify_commit(s3, "alpha-edge-algo", prefix, "b" * 64)
