from __future__ import annotations


import boto3
import pytest
from moto import mock_aws

from alpha_edge.core.schemas import RuntimeConfig


@pytest.fixture(autouse=True)
def _safe_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests never inherit production AWS settings or contact EC2 metadata."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("ALPHA_EDGE_ENV", "dev")
    monkeypatch.setenv("ALPHA_EDGE_BUCKET", "alpha-edge-tests")
    monkeypatch.setenv("ALPHA_EDGE_REGION", "eu-west-1")


@pytest.fixture
def runtime_cfg() -> RuntimeConfig:
    return RuntimeConfig(
        env="dev",
        bucket="alpha-edge-tests",
        region="eu-west-1",
        engine_root="dev/engine/v1",
        market_root="dev/market",
        warehouse_root="dev/engine/v1/warehouse",
        is_prod=False,
    )


@pytest.fixture
def aws_mock():
    with mock_aws():
        yield


@pytest.fixture
def s3_bucket(runtime_cfg: RuntimeConfig, aws_mock):
    s3 = boto3.client("s3", region_name=runtime_cfg.region)
    s3.create_bucket(
        Bucket=runtime_cfg.bucket,
        CreateBucketConfiguration={"LocationConstraint": runtime_cfg.region},
    )
    return s3
