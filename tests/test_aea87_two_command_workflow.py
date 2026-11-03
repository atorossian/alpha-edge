import io
import json
import pytest
from alpha_edge.market.aea87_rebuild import selected_from_latest, AFFECTED
from alpha_edge.market.aea87_publication_transaction import safe_restore, checksum


class S3:
    def __init__(self, blobs):
        self.blobs = blobs
    def get_object(self, Bucket, Key):
        from botocore.exceptions import ClientError
        if Key not in self.blobs:
            raise ClientError({"Error":{"Code":"NoSuchKey"}}, "GetObject")
        return {"Body":io.BytesIO(self.blobs[Key])}
    def put_object(self, Bucket, Key, Body, **kwargs):
        self.blobs[Key] = Body
    def delete_object(self, Bucket, Key):
        self.blobs.pop(Key, None)


def test_validate_picks_all_implicated_and_not_healthy():
    report = {"run_id":"v1", "complete_universe":True, "universe_count":4,
              "scanned":4, "classification_counts":{},
              "assets":[{"asset_id":"healthy", "status":"PASSED_NUMERICAL_SCREEN_ONLY"},
                        {"asset_id":"bad", "status":"REBUILD_OR_DATA_REPAIR_REVIEW"},
                        {"asset_id":"unavailable", "status":"INSUFFICIENT_DATA"},
                        {"asset_id":"odd", "status":"INVESTIGATION_REQUIRED"}]}
    pointer = {"report_key":"report"}
    s3 = S3({"market/repairs/aea87/universe_screen/v1/latest.json":
              json.dumps(pointer).encode(), "report":json.dumps(report).encode()})
    ids, meta = selected_from_latest(s3,"bucket")
    assert ids == ["bad", "odd", "unavailable"]
    assert meta["complete_universe"] is True


def test_incomplete_pilot_cannot_drive_default_bulk_rebuild():
    report = {"run_id":"v1", "complete_universe":False,
              "universe_count":2084, "scanned":100, "assets":[]}
    s3 = S3({"market/repairs/aea87/universe_screen/v1/latest.json":
             b'{"report_key":"report"}', "report":json.dumps(report).encode()})
    with pytest.raises(RuntimeError, match="complete-universe"):
        selected_from_latest(s3, "bucket")


def test_missing_manifest_rollback_deletes_only_new_manifest():
    s3 = S3({"active_manifest":b"new"})
    op = {"manifest_key":"active_manifest", "before_sha":None,
          "after_sha":checksum(b"new"), "backup_key":None}
    safe_restore(s3,"bucket",op)
    assert "active_manifest" not in s3.blobs


def test_source_manifest_drift_prevents_rollback():
    s3 = S3({"active_manifest":b"third_party"})
    op = {"manifest_key":"active_manifest", "before_sha":None,
          "after_sha":checksum(b"new"), "backup_key":None}
    with pytest.raises(RuntimeError, match="Concurrent"):
        safe_restore(s3,"bucket",op)
