"""Unit tests for AEA-87 journaled publication helpers."""
import hashlib
import json
import unittest
from unittest.mock import Mock

from alpha_edge.market.aea87_publication_transaction import checksum, safe_restore


class S3Fake:
    def __init__(self, objects):
        self.objects = dict(objects)

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        from io import BytesIO
        return {"Body": BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)


class TransactionTests(unittest.TestCase):
    def test_restore_own_switch(self):
        before, after = b'{"parts":["old"]}', b'{"parts":["new"]}'
        s3 = S3Fake({"active": after, "backup": before})
        safe_restore(s3, "bucket", dict(manifest_key="active", backup_key="backup",
                                        before_sha=checksum(before), after_sha=checksum(after)))
        self.assertEqual(s3.objects["active"], before)

    def test_refuse_to_override_concurrent_change(self):
        before, after = b"old", b"new"
        s3 = S3Fake({"active": b"third-party", "backup": before})
        with self.assertRaisesRegex(RuntimeError, "Concurrent change"):
            safe_restore(s3, "bucket", dict(manifest_key="active", backup_key="backup",
                                            before_sha=checksum(before), after_sha=checksum(after)))
        self.assertEqual(s3.objects["active"], b"third-party")

    def test_restore_missing_original(self):
        after = b"new"
        s3 = S3Fake({"active": after})
        safe_restore(s3, "bucket", dict(manifest_key="active", backup_key="backup",
                                        before_sha=None, after_sha=checksum(after)))
        self.assertNotIn("active", s3.objects)


if __name__ == "__main__":
    unittest.main()
