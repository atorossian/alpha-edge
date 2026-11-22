from copy import deepcopy
from decimal import Decimal

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from alpha_edge.market.dividend_currency_registry_v3 import (
    RATE_REFERENCE,
    REGISTRY_ROOT,
    SCHEMA,
    UNITS_PER_EUR,
    DividendCurrencyError,
    apply_currency_corrections,
    load_registry,
    publish_registry,
    registry_version,
)


SHA = "a" * 64


def registry(currency="ESP", source="45", declared="45", multiplier="1", status="proposed"):
    return {
        "schema": SCHEMA,
        "rate_reference": RATE_REFERENCE,
        "units_per_eur": UNITS_PER_EUR,
        "entries": [
            {
                "asset_id": "EQH-TEST",
                "provider_symbol": "TEST.MC",
                "effective_date": "2000-07-03",
                "bronze_sha256": SHA,
                "correction_id": "test-legacy-dividend",
                "source_value": source,
                "original_currency": currency,
                "price_currency": "EUR",
                "declared_amount": declared,
                "share_basis_multiplier": multiplier,
                "basis_evidence": "Test issuer evidence",
                "evidence_url": "https://example.com/issuer",
                "status": status,
                "approved_by": "reviewer",
            }
        ],
    }


def actions(value=45):
    return pd.DataFrame(
        {
            "asset_id": ["EQH-TEST"] * 3,
            "provider_symbol": ["TEST.MC"] * 3,
            "effective_date": ["2000-07-03", "2000-07-03", "2001-07-03"],
            "action_type": ["DIVIDEND", "SPLIT", "DIVIDEND"],
            "source_value": [value, 2.0, 0.25],
            "source_currency": ["EUR", None, "EUR"],
        }
    )


def apply(a, r, sha=SHA):
    return apply_currency_corrections(a, r, asset_id="EQH-TEST", provider_symbol="TEST.MC", bronze_sha256=sha)


@pytest.mark.parametrize(
    "currency,source,declared,multiplier",
    [("ESP", "45", "45", "1"), ("ITL", "300", "300", "1"), ("FRF", "11", "22", "0.5")],
)
def test_approved_conversion_preserves_original_and_same_date_split(currency, source, declared, multiplier):
    original = actions(float(source))
    pristine = original.copy(deep=True)
    result, stats = apply(original, registry(currency, source, declared, multiplier, "approved"))
    assert result.iloc[0].normalized_value == pytest.approx(float(Decimal(source) / Decimal(UNITS_PER_EUR[currency])))
    assert result.iloc[0].source_value == float(source)
    assert result.iloc[1].normalized_value == 2
    assert result.iloc[2].normalized_value == 0.25
    assert stats["approved"] == 1 and not stats["production_eligible"]
    pd.testing.assert_frame_equal(original, pristine)


def test_proposed_preview_does_not_change_normalized_value():
    result, stats = apply(actions(), registry())
    assert result.iloc[0].normalized_value == 45
    assert result.iloc[0].proposed_value == pytest.approx(45 / 166.386)
    assert stats["proposed"] == 1 and stats["approved"] == 0


@pytest.mark.parametrize(
    "case", ["stale_hash", "changed_value", "duplicate", "basis", "missing", "post_euro", "unapproved_identity"]
)
def test_fail_closed(case):
    r, a, sha = registry(), actions(), SHA
    e = r["entries"][0]
    if case == "stale_hash":
        sha = "b" * 64
    elif case == "changed_value":
        a.loc[0, "source_value"] = 44
    elif case == "duplicate":
        r["entries"].append(deepcopy(e))
    elif case == "basis":
        e["declared_amount"] = "44"
    elif case == "missing":
        a = a.iloc[1:]
    elif case == "post_euro":
        e["effective_date"] = "2003-07-03"
    elif case == "unapproved_identity":
        e["status"] = "approved"
        e["approved_by"] = ""
    with pytest.raises(DividendCurrencyError):
        apply(a, r, sha)


def test_s3_pinned_registry_read():
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-1")
        s3.create_bucket(Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "eu-west-1"})
        r = registry()
        version = registry_version(r)
        dry = publish_registry(s3, "test-bucket", r)
        assert dry["version"] == version
        assert s3.list_objects_v2(Bucket="test-bucket", Prefix=REGISTRY_ROOT)["KeyCount"] == 0
        publish_registry(s3, "test-bucket", r, execute=True)
        assert load_registry(s3, "test-bucket", version) == r
