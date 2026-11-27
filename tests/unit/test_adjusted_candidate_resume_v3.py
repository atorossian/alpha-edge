"""Interrupted downloads and immutable, input-bound candidate checkpoints."""

import io

import boto3
import pytest
from botocore.exceptions import ClientError, ResponseStreamingError
from moto import mock_aws

from alpha_edge.market import build_silver_adjusted_v3 as adjusted
from alpha_edge.market.lakehouse_v3 import storage


class BrokenBody(io.BytesIO):
    def read(self, *args):
        raise ResponseStreamingError(error="connection broken")


def test_stream_restarts_from_byte_zero(monkeypatch):
    first = BrokenBody(b"partial")
    second = io.BytesIO(b"complete")
    bodies = iter([first, second])

    class Client:
        def get_object(self, **kwargs):
            return {"Body": next(bodies), "ContentLength": 8}

    monkeypatch.setattr(storage.time, "sleep", lambda _: None)
    assert storage.read_bytes(Client(), "bucket", "key") == b"complete"
    assert first.closed and second.closed


def test_permission_failure_not_retried(monkeypatch):
    class Client:
        calls = 0

        def get_object(self, **kwargs):
            self.calls += 1
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")

    client = Client()
    with pytest.raises(ClientError):
        storage.read_object(client, "bucket", "key", optional=True)
    assert client.calls == 1


def test_short_body_is_retried_and_integrity_mismatch_is_not_waived(monkeypatch):
    class Client:
        calls = 0

        def get_object(self, **kwargs):
            self.calls += 1
            return {"Body": io.BytesIO(b"bad" if self.calls == 1 else b"complete"), "ContentLength": 8}

    monkeypatch.setattr(storage.time, "sleep", lambda _: None)
    client = Client()
    ref = {"key": f"{storage.ROOT}/bronze/object", "sha256": storage.digest(b"different")}
    with pytest.raises(storage.LakehouseError, match="Corrupt S3 artifact"):
        storage.verified_bytes(client, "bucket", ref)
    assert client.calls == 2


@mock_aws
def test_existing_asset_replayed_once_then_resumed_without_rebuild():
    s3 = boto3.client("s3", region_name="us-east-1")
    bucket = "test-bucket"
    s3.create_bucket(Bucket=bucket)
    root = adjusted.OUTPUT_ROOT + "/runs/test-run"
    aid, fingerprint, blob = "asset-a", "f" * 64, b"candidate-test-bytes"
    cohort = {(aid, "2025-01-01")}
    report = {"asset_id": aid, "cohort_cases": 1}
    key = f"{root}/assets/{storage.digest(aid.encode())}/prices-{storage.digest(blob)}.parquet"
    storage.put_immutable(s3, bucket, key, blob, content_type="application/vnd.apache.parquet")
    assert adjusted._resume_asset(s3, bucket, root, aid, fingerprint, cohort) is None
    assert adjusted._publish_asset(s3, bucket, root, aid, fingerprint, report, cohort, blob)
    assert adjusted._resume_asset(s3, bucket, root, aid, fingerprint, cohort) == (report, cohort)
    with pytest.raises(ValueError, match="identity mismatch"):
        adjusted._resume_asset(s3, bucket, root, aid, "other", cohort)
    # A corrupt existing output cannot pass resume even when the path is correct.
    s3.put_object(Bucket=bucket, Key=key, Body=b"corrupt", Metadata={"sha256": storage.digest(blob)})
    with pytest.raises(ValueError, match="checksum mismatch"):
        adjusted._resume_asset(s3, bucket, root, aid, fingerprint, cohort)
