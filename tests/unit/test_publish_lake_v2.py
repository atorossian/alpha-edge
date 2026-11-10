"""Read-only, isolated publisher and promotion-gate contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from alpha_edge.market.publish_lake_v2 import (
    PublicationError, _root, stage_candidate, validate_local_candidate,
    verify_staged_candidate, promote_candidate,
)


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.write_keys = []
    def put_object(self, *, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None,
                   ContentType=None):
        if IfNoneMatch == "*" and (Bucket, Key) in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        old = self.objects.get((Bucket, Key))
        if IfMatch is not None and (
            old is None or hashlib.md5(old).hexdigest() != IfMatch.strip('"')
        ):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[(Bucket, Key)] = Body
        self.write_keys.append(Key)
    def get_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        import io
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}
    def head_object(self, *, Bucket, Key):
        data = self.objects[(Bucket, Key)]
        return {"ETag": '"' + hashlib.md5(data).hexdigest() + '"'}


def candidate(tmp_path, *, acceptance="review_required", manual=1):
    folder = tmp_path / "candidate"
    folder.mkdir()
    names = [
        "silver.parquet", "gold.parquet", "bronze_validated.parquet",
        "fx_normalized.parquet", "fx_evidence.parquet",
        "session_issues.parquet", "asset_quality_summary.csv",
    ]
    for filename in names:
        (folder / filename).write_bytes(("candidate " + filename).encode())
    manifest = {
        "schema": "market_data_lake_v2_silver_gold_candidate_1",
        "publication_state": "candidate_only",
        "legacy_production_unchanged": True,
        "bronze_manifest_sha256": "a" * 64,
        "source_rows": 3,
        "candidate_input_rows": 3 + manual,
        "silver_rows": 3 + manual,
        "gold_rows": 3 + manual,
        "manual_correction_rows": manual,
        "missing_expected_sessions": 0,
        "unexpected_sessions": 0,
        "fx_rejected_rows": 0,
        "acceptance": acceptance,
        "assets": [{"quarantined_prices": int(manual > 0), "review_returns": 0}],
        "files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.iterdir()
        },
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def test_isolated_root_guards():
    for root in ["market", "prod/market", "dev", "dev/market/../market",
                 "dev/market//lake_v2", "dev/market/./lake_v2"]:
        with pytest.raises(PublicationError):
            _root(root)
    assert _root("dev/market") == "dev/market"


def test_local_validation_catches_tampering(tmp_path):
    local = candidate(tmp_path)
    manifest, paths = validate_local_candidate(local)
    assert len(paths) == 7
    (local / "silver.parquet").write_bytes(b"tampered")
    with pytest.raises(PublicationError, match="SHA-256 mismatch"):
        validate_local_candidate(local)


def test_stage_verify_immutable_receipt_written_last(tmp_path):
    local = candidate(tmp_path)
    s3 = FakeS3()
    receipt = stage_candidate(s3, "example-bucket", local, run_id="pilot-1")
    assert s3.write_keys[-1].endswith("/_committed.json")
    staged, manifest = verify_staged_candidate(
        s3, "example-bucket", run_id="pilot-1"
    )
    assert staged == receipt
    assert manifest["manual_correction_rows"] == 1
    assert all(key.startswith("dev/market/lake_v2/") for key in s3.write_keys)
    with pytest.raises(PublicationError, match="already committed"):
        stage_candidate(s3, "example-bucket", local, run_id="pilot-1")


def test_commit_marker_not_created_for_failed_upload(tmp_path):
    local = candidate(tmp_path)
    s3 = FakeS3()
    s3.objects[("example-bucket", "dev/market/lake_v2/candidates/pilot-1/silver.parquet")] = b"old"
    with pytest.raises(PublicationError, match="partially staged"):
        stage_candidate(s3, "example-bucket", local, run_id="pilot-1")
    assert not any(k.endswith("_committed.json") for k in s3.write_keys)


def test_manual_candidate_is_never_promoted(tmp_path):
    local = candidate(tmp_path)
    s3 = FakeS3()
    receipt = stage_candidate(s3, "example-bucket", local, run_id="pilot-1")
    with pytest.raises(PublicationError, match="not eligible"):
        promote_candidate(
            s3, "example-bucket", run_id="pilot-1",
            approval=receipt["candidate_manifest_sha256"],
        )
    assert "dev/market/lake_v2/current.json" not in s3.write_keys


def test_only_eligible_candidate_promotes_after_explicit_digest_approval(tmp_path):
    local = candidate(tmp_path, acceptance="initial_contract_pass", manual=0)
    s3 = FakeS3()
    receipt = stage_candidate(s3, "example-bucket", local, run_id="pilot-1")
    with pytest.raises(PublicationError, match="Approval digest"):
        promote_candidate(s3, "example-bucket", run_id="pilot-1",
                          approval="bad-approval")
    pointer = promote_candidate(s3, "example-bucket", run_id="pilot-1",
                                approval=receipt["candidate_manifest_sha256"])
    assert pointer["run_id"] == "pilot-1"
    assert s3.write_keys[-1] == "dev/market/lake_v2/current.json"
    with pytest.raises(PublicationError, match="already the current"):
        promote_candidate(s3, "example-bucket", run_id="pilot-1",
                          approval=receipt["candidate_manifest_sha256"])
