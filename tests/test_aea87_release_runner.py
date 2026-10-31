import pytest

from alpha_edge.market.aea87_release_runner import build
from alpha_edge.market.aea87_publication_transaction import checksum, safe_restore


def test_checksum_of_missing_manifest():
    assert checksum(None) is None


def test_rollback_rejects_third_party_version():
    class Store:
        def get_object(self, Bucket, Key):
            from io import BytesIO
            return {"Body": BytesIO(b"unexpected")}
    op={"manifest_key":"active","backup_key":"backup","before_sha":checksum(b"old"),
        "after_sha":checksum(b"new")}
    with pytest.raises(RuntimeError,match="Concurrent change"):
        safe_restore(Store(),"bucket",op)


def test_release_build_rejects_invalid_plan(monkeypatch):
    class Dummy: pass
    def invalid_plan(s3,bucket,key):
        if key == "plan":
            return {"reconciliation_id":"a","reconciliation_sha256":"bogus"}, "hash"
        return {"run_id":"a"}, "different-hash"
    monkeypatch.setattr("alpha_edge.market.aea87_release_runner.read_document",invalid_plan)
    with pytest.raises(ValueError,match="Reconciliation changed"):
        build(Dummy(),"bucket","plan","AEA87-TEST-01")
