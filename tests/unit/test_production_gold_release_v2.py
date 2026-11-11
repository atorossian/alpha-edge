"""Gold v2 release candidate and pointer activation must fail closed."""

import hashlib
import io
import json

import pytest
from botocore.exceptions import ClientError

from alpha_edge.market.production_gold_release_v2 import (
    CURRENT,
    ReleaseError,
    activate,
    encoded,
    sha,
    stage,
)


class S3:
    def __init__(self):
        self.objects = {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kw):
        if kw.get("IfNoneMatch") == "*" and Key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if kw.get("IfMatch") and Key in self.objects:
            current_etag = '"' + hashlib.md5(self.objects[Key]).hexdigest() + '"'
            if kw["IfMatch"] != current_etag:
                raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body

    def head_object(self, Bucket, Key):
        return {"ETag": '"' + hashlib.md5(self.objects[Key]).hexdigest() + '"'}


def candidate(s3):
    root = "market/lake_v2/releases/data/fixture"
    files = {}
    for field in ("bronze", "silver", "gold", "fx", "indicator_candidate"):
        key = f"{root}/{field}.parquet"
        blob = field.encode()
        s3.objects[key] = blob
        files[field] = {"key": key, "sha256": sha(blob)}
    report = {
        "internal_integrity_status": "passed",
        "acceptance": "review_required",
        "blocker_counts": {"quarantined_prices": 1},
        "open_external_evidence_gates": ["independent_source_observation_verification"],
    }
    key = f"{root}/acceptance.json"
    body = encoded(report)
    s3.objects[key] = body
    files["acceptance"] = {"key": key, "sha256": sha(body)}
    return {
        "schema": "alpha_edge_v2_release_candidate_1",
        "release_id": "test_release_1",
        "as_of": "2026-11-11",
        "expected_cohorts": 1,
        "expected_assets": 1,
        "cohorts": [{"cohort": 1, "assets": ["A"], "gold_rows": 1, **files}],
        "universe_reconciliation": "pending",
        "indicator_validation": "pending",
        "consumer_contract_validation": "pending",
    }


def test_stage_works_but_activation_blocks_quality_and_external_gates():
    s3 = S3()
    release = candidate(s3)
    staged = stage(s3, bucket="test", release=release, execute=True)
    assert staged["state"] == "staged"
    with pytest.raises(ReleaseError, match="acceptance blocked"):
        activate(
            s3, bucket="test", release_key=staged["release_key"], approved_sha256=staged["release_sha256"], execute=True
        )
    assert CURRENT not in s3.objects


def test_full_approval_requires_indicator_and_universe_validation():
    s3 = S3()
    release = candidate(s3)
    cohort = release["cohorts"][0]
    report = {
        "internal_integrity_status": "passed",
        "acceptance": "accepted",
        "blocker_counts": {},
        "open_external_evidence_gates": [],
    }
    key = cohort["acceptance"]["key"]
    s3.objects[key] = encoded(report)
    cohort["acceptance"]["sha256"] = sha(s3.objects[key])
    cohort["independent_approval"] = {
        "reviewer": "independent-reviewer",
        "acceptance_manifest_sha256": cohort["acceptance"]["sha256"],
    }
    staged = stage(s3, bucket="test", release=release, execute=True)
    with pytest.raises(ReleaseError, match="indicator validation"):
        activate(
            s3, bucket="test", release_key=staged["release_key"], approved_sha256=staged["release_sha256"], execute=True
        )
    assert CURRENT not in s3.objects


def test_complete_approved_release_pointer_switches_and_preserves_previous():
    s3 = S3()
    release = candidate(s3)
    cohort = release["cohorts"][0]
    report = {
        "internal_integrity_status": "passed",
        "acceptance": "accepted",
        "blocker_counts": {},
        "open_external_evidence_gates": [],
    }
    key = cohort["acceptance"]["key"]
    s3.objects[key] = encoded(report)
    cohort["acceptance"]["sha256"] = sha(s3.objects[key])
    cohort["independent_approval"] = {
        "reviewer": "independent-reviewer",
        "acceptance_manifest_sha256": cohort["acceptance"]["sha256"],
    }
    release.update(
        {
            "indicator_validation": "passed",
            "consumer_contract_validation": "passed",
            "universe_reconciliation": "passed",
            "independent_signoff": "external release approval 1",
        }
    )
    staged = stage(s3, bucket="test", release=release, execute=True)
    dry = activate(
        s3, bucket="test", release_key=staged["release_key"], approved_sha256=staged["release_sha256"], execute=False
    )
    assert dry["state"] == "dry_run" and CURRENT not in s3.objects
    actual = activate(
        s3, bucket="test", release_key=staged["release_key"], approved_sha256=staged["release_sha256"], execute=True
    )
    assert actual["state"] == "activated"
    assert json.loads(s3.objects[CURRENT])["release_id"] == "test_release_1"
    assert (
        activate(
            s3, bucket="test", release_key=staged["release_key"], approved_sha256=staged["release_sha256"], execute=True
        )["state"]
        == "already_current"
    )


def test_incomplete_cohort_inventory_cannot_be_staged():
    s3 = S3()
    release = candidate(s3)
    release["expected_cohorts"] = 2
    with pytest.raises(ReleaseError, match="coverage"):
        stage(s3, bucket="test", release=release, execute=True)
    assert CURRENT not in s3.objects
