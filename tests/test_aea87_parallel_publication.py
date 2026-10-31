"""Parallel release transaction tests; no AWS calls."""
from io import BytesIO
import json
from unittest.mock import patch

from botocore.exceptions import ClientError
import pytest

from alpha_edge.market import aea87_publication_transaction as tx


class FakeS3:
    def __init__(self, objects):
        self.objects = dict(objects)
    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": BytesIO(self.objects[Key])}
    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ETag": '"old"'}
    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body
    def delete_object(self, Bucket, Key, **kwargs):
        self.objects.pop(Key, None)


def fixture():
    old = b"old"
    new = b"new"
    ops = [dict(manifest_key=f"live/{i}", backup_key=f"backup/{i}",
                before_sha=tx.checksum(old), after_sha=tx.checksum(new),
                after=new.decode()) for i in range(5)]
    objects = {**{f"live/{i}": old for i in range(5)},
               **{f"backup/{i}": old for i in range(5)}}
    return FakeS3(objects), ops


def test_parallel_activate_and_rollback():
    s3, ops = fixture()
    with patch.object(tx, "entries", return_value=[{"step": 1, "state": "READY"}]), patch.object(tx, "write_event") as journal:
        tx.switch_manifests(s3, "b", "root", ops, workers=3, batch_size=2)
    assert all(s3.objects[x["manifest_key"]] == b"new" for x in ops)
    assert journal.call_args.args[4] == "CACHE_REBUILD_REQUIRED"
    with patch.object(tx, "entries", return_value=[{"step": 8, "state": "CACHE_REBUILD_REQUIRED"}]), patch.object(tx, "write_event"):
        tx.rollback(s3, "b", "root", ops, workers=3)
    assert all(s3.objects[x["manifest_key"]] == b"old" for x in ops)


def test_parallel_activation_refuses_manifest_drift():
    s3, ops = fixture()
    s3.objects["live/2"] = b"external"
    with patch.object(tx, "entries", return_value=[{"step": 1, "state": "READY"}]), patch.object(tx, "write_event") as journal:
        with pytest.raises(RuntimeError, match="activation-check failed"):
            tx.switch_manifests(s3, "b", "root", ops, workers=3, batch_size=2)
    journal.assert_not_called()
