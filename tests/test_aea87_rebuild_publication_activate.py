import hashlib
import json
import pytest
from alpha_edge.market.aea87_rebuild_publication_activate import (
    require_ready_plan, convert_to_transaction_ops,
)


class InMemoryS3:
    def __init__(self, items):
        self.items = items

    def get_object(self, Bucket, Key):
        from io import BytesIO
        if Key not in self.items:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": BytesIO(self.items[Key])}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sample():
    old = b'{"parts":["old"]}'
    part = b"parquet"
    new = json.dumps({"parts": ["new"]})
    ops = []
    objects = {}
    for table in ("ohlcv_usd", "returns_usd"):
        key = "active/" + table
        backup = "backup/" + table
        dest = "new/" + table
        objects.update({key: old, backup: old, dest: part})
        ops.append({"asset_id": "HAE_ID", "table": table,
                    "year": 2025, "manifest_key": key, "before_missing": False,
                    "before_sha": sha(old), "backup_key": backup,
                    "part_key": dest, "part_sha": sha(part),
                    "after": new, "after_sha": sha(new.encode())})
    return InMemoryS3(objects), {"asset_id":"HAE_ID",
                                "active_histories_modified":False,
                                "status":"PREPARED_NOT_PUBLISHED","operations":ops}


def test_prepared_publication_requires_verified_backups():
    s3, plan = sample()
    assert len(require_ready_plan(s3, "test", plan)) == 2
    ops = convert_to_transaction_ops(plan["operations"])
    assert ops[0]["staged_key"] == ops[0]["part_key"]
    del s3.items[plan["operations"][0]["backup_key"]]
    with pytest.raises(ValueError, match="backup unavailable"):
        require_ready_plan(s3, "test", plan)


def test_manifest_drift_is_blocked():
    s3, plan = sample()
    s3.items[plan["operations"][0]["manifest_key"]] = b"changed"
    with pytest.raises(ValueError, match="drift"):
        require_ready_plan(s3, "test", plan)


def test_missing_original_manifest_requires_legacy_retirement():
    s3, plan = sample()
    plan["operations"][0]["before_missing"] = True
    with pytest.raises(ValueError, match="legacy-retirement"):
        require_ready_plan(s3, "test", plan)


def test_downstream_backup_and_restore_exact_bytes(monkeypatch):
    from alpha_edge.market import aea87_rebuild_publication_activate as module
    class MutableS3(InMemoryS3):
        def put_object(self, Bucket, Key, Body, **kwargs):
            if kwargs.get("IfNoneMatch") == "*" and Key in self.items:
                raise AssertionError("Overwrite during immutable staging")
            self.items[Key] = Body
        def delete_object(self, Bucket, Key):
            self.items.pop(Key, None)
    monkeypatch.setattr(module, "DOWNSTREAM",
                        ["market/snapshots/v1/latest_returns.parquet",
                         "market/cache/v1/returns_wide_min5y.parquet"])
    initial = b"original-cache"
    s3 = MutableS3({"market/cache/v1/returns_wide_min5y.parquet": initial})
    previous = module.backup_downstream(s3, "test", "release/transaction")
    s3.items["market/cache/v1/returns_wide_min5y.parquet"] = b"bad-cache"
    s3.items["market/snapshots/v1/latest_returns.parquet"] = b"new-file"
    module.restore_downstream(s3, "test", previous)
    assert s3.items["market/cache/v1/returns_wide_min5y.parquet"] == initial
    assert "market/snapshots/v1/latest_returns.parquet" not in s3.items
