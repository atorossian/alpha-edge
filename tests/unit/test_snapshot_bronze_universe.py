"""Tests for five-file immutable S3 snapshot publishing; no live AWS/Git."""

from __future__ import annotations

import io
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from alpha_edge.market import snapshot_bronze_universe as snapshot
from alpha_edge.market.ingest_bronze_full_history import (
    BronzeIngestError,
    SNAPSHOT_ROOT,
    SOURCE_NAMES,
    _digest,
    _json_bytes,
)
from tests.unit.test_ingest_bronze_full_history import _sources


BUCKET = "alpha-edge-algo"
COMMIT = "a" * 40


@pytest.fixture
def sources():
    return _sources()


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="eu-west-1")
        client.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
        )
        yield client


def _fingerprint(blobs):
    return _digest(_json_bytes({
        name: {"bytes": len(blobs[name]), "sha256": _digest(blobs[name])}
        for name in SOURCE_NAMES
    }))


def test_dry_run_has_no_s3_writes(s3, sources):
    report = snapshot.publish_snapshot(
        s3, BUCKET, blobs=sources, commit=COMMIT,
        fingerprint=_fingerprint(sources), execute=False,
    )
    assert report["file_count"] == 5
    assert not report["executed"]
    assert s3.list_objects_v2(Bucket=BUCKET, Prefix=SNAPSHOT_ROOT).get("KeyCount") == 0


def test_snapshot_is_immutable_and_readback_compatible_with_ingestion(s3, sources):
    fingerprint = _fingerprint(sources)
    first = snapshot.publish_snapshot(
        s3, BUCKET, blobs=sources, commit=COMMIT,
        fingerprint=fingerprint, execute=True,
    )
    assert first["readback_verified"]
    assert first["included"] == 4
    assert first["excluded"] == 1
    for name in SOURCE_NAMES:
        entry = s3.get_object(Bucket=BUCKET, Key=f"{SNAPSHOT_ROOT}/{fingerprint}/{name}")
        assert entry["Body"].read() == sources[name]
        assert entry["Metadata"]["source-git-commit"] == COMMIT
        assert entry["Metadata"]["sha256"] == _digest(sources[name])

    # A retried publish must never alter the canonical snapshot.
    again = snapshot.publish_snapshot(
        s3, BUCKET, blobs=sources, commit=COMMIT,
        fingerprint=fingerprint, execute=True,
    )
    assert again["readback_verified"]
    assert again["snapshot_fingerprint"] == first["snapshot_fingerprint"]


def test_corrupted_existing_object_is_never_overwritten(s3, sources):
    fingerprint = _fingerprint(sources)
    key = f"{SNAPSHOT_ROOT}/{fingerprint}/{SOURCE_NAMES[0]}"
    s3.put_object(
        Bucket=BUCKET, Key=key, Body=b"other version",
        Metadata={"source-git-commit": COMMIT, "sha256": "0" * 64},
    )
    with pytest.raises(BronzeIngestError, match="incompatible"):
        snapshot.publish_snapshot(
            s3, BUCKET, blobs=sources, commit=COMMIT,
            fingerprint=fingerprint, execute=True,
        )
    assert s3.get_object(Bucket=BUCKET, Key=key)["Body"].read() == b"other version"


def test_snapshot_hash_and_git_sha_are_mandatory(s3, sources):
    with pytest.raises(BronzeIngestError, match="fingerprint"):
        snapshot.publish_snapshot(
            s3, BUCKET, blobs=sources, commit=COMMIT,
            fingerprint="0" * 64, execute=True,
        )
    with pytest.raises(BronzeIngestError, match="Git SHA"):
        snapshot.publish_snapshot(
            s3, BUCKET, blobs=sources, commit="invalid",
            fingerprint=_fingerprint(sources), execute=True,
        )


def test_collect_sources_requires_clean_five_configs(monkeypatch, tmp_path, sources):
    (tmp_path / ".git").mkdir()
    root = tmp_path / "data" / "universe"
    root.mkdir(parents=True)
    for name, data in sources.items():
        (root / name).write_bytes(data)

    def clean_git(_root: Path, *args: str) -> str:
        if "status" in args:
            return ""
        if "rev-parse" in args:
            return COMMIT
        raise AssertionError(args)

    monkeypatch.setattr(snapshot, "_git", clean_git)
    blobs, commit, fingerprint = snapshot.collect_sources(tmp_path)
    assert blobs == sources
    assert commit == COMMIT
    assert fingerprint == _fingerprint(sources)

    def dirty_git(_root: Path, *args: str) -> str:
        if "status" in args:
            return " M data/universe/universe.csv"
        return COMMIT

    monkeypatch.setattr(snapshot, "_git", dirty_git)
    with pytest.raises(BronzeIngestError, match="uncommitted"):
        snapshot.collect_sources(tmp_path)


def test_collect_sources_does_not_create_local_artifacts(monkeypatch, tmp_path, sources):
    (tmp_path / ".git").mkdir()
    root = tmp_path / "data" / "universe"
    root.mkdir(parents=True)
    for name, data in sources.items():
        (root / name).write_bytes(data)
    monkeypatch.setattr(
        snapshot, "_git",
        lambda _root, *args: "" if "status" in args else COMMIT,
    )
    before = {str(x.relative_to(tmp_path)) for x in tmp_path.rglob("*")}
    snapshot.collect_sources(tmp_path)
    after = {str(x.relative_to(tmp_path)) for x in tmp_path.rglob("*")}
    assert before == after
