"""Athena/Glue publication of a reviewed Gold release as an Iceberg table.

Only S3 persists query results and data; no local files, no legacy writes.
A release becomes visible only when its vetted, immutable Iceberg table is
selected by the stable user-facing view. No silent automatic approval.
"""

from __future__ import annotations

import argparse
import json
import re
import time

from alpha_edge.market.lakehouse_v3.storage import (
    ROOT, LakehouseError, check_id, digest, put_json, read_bytes, read_json, verified_frame,
    verified_bytes,
)

DATABASE = "alpha_edge_market"
VIEW = "market_asset_daily"
# Only explicit, stable fields become the warehouse consumption contract.
COLUMNS = {
    "asset_id": "string",
    "session_date": "string",
    "provider_symbol": "string",
    "currency_original": "string",
    "close_raw_usd": "double",
    "close_adjusted_usd": "double",
    "close_price_usd": "double",
    "close_total_return_index_usd": "double",
    "fx_to_usd": "double",
    "dividend_per_share_local": "double",
    "split_new_shares_per_old_share": "double",
    "ret_price_usd": "double",
    "ret_total_usd": "double",
    "ret_log_total_usd": "double",
    "sma_20": "double",
    "sma_50": "double",
    "sma_200": "double",
    "rsi_14": "double",
    "atr_14": "double",
    "macd": "double",
    "macd_signal": "double",
    "macd_hist": "double",
    "annualized_vol_20": "double",
    "annualized_vol_60": "double",
    "drawdown_20": "double",
    "drawdown_60": "double",
    "drawdown_252": "double",
    "quality_status": "string",
    "return_quality_status": "string",
    "price_basis": "string",
    "total_return_basis": "string",
}
_APPROVED_GATES = (
    "identity", "exchange_calendar", "source_prices",
    "corporate_actions", "fx_timing", "provider_revisions",
    "warehouse_contract",
)


def _assert_safe_sql_id(value: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,100}", value):
        raise LakehouseError(f"Unsafe Athena SQL identifier {value!r}")
    return value


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def stage_table_name(run_id: str) -> str:
    return _assert_safe_sql_id("market_stage_" + check_id(run_id).replace("-", "_"))


def release_table_name(run_id: str) -> str:
    return _assert_safe_sql_id("market_gold_" + check_id(run_id).replace("-", "_"))


def warehouse_sql(bucket: str, run_id: str) -> dict:
    """Pure DDL generator: inspect SQL before authorizing AWS execution."""
    check_id(run_id)
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{2,62}", bucket):
        raise LakehouseError("Unsafe S3 bucket")
    base = f"s3://{bucket}/{ROOT}"
    stage = stage_table_name(run_id)
    release = release_table_name(run_id)
    columns = ",\n  ".join(f"{name} {type_name}" for name, type_name in COLUMNS.items())
    select_columns = ",\n  ".join(COLUMNS)
    return {
        "database": f"CREATE DATABASE IF NOT EXISTS {DATABASE}",
        "stage": (
            f"CREATE EXTERNAL TABLE {DATABASE}.{stage} (\n  {columns}\n)"
            " STORED AS PARQUET "
            f"LOCATION {_quote(f'{base}/candidates/{run_id}/gold/')}"
            " TBLPROPERTIES ('classification'='parquet')"
        ),
        "iceberg": (
            f"CREATE TABLE {DATABASE}.{release} WITH ("
            "table_type = 'ICEBERG', is_external = false, format = 'PARQUET', "
            f"location = {_quote(f'{base}/warehouse/releases/{run_id}/')}"
            ") AS SELECT\n  "
            f"{select_columns}\nFROM {DATABASE}.{stage}"
            " WHERE quality_status = 'accepted'"
        ),
        "count": f"SELECT count(*) FROM {DATABASE}.{release}",
        "distinct": (
            f"SELECT count(*) FROM (SELECT asset_id, session_date, count(*) AS n "
            f"FROM {DATABASE}.{release} GROUP BY 1, 2 HAVING count(*) > 1)"
        ),
        "view": (
            f"CREATE OR REPLACE VIEW {DATABASE}.{VIEW} AS "
            f"SELECT {select_columns} FROM {DATABASE}.{release}"
        ),
    }


def approved_candidate(s3, bucket: str, run_id: str, approval_key: str) -> tuple[dict, dict]:
    """Require an external, pinned reviewer decision; never self-approve."""
    check_id(run_id)
    prefix = f"{ROOT}/candidates/{run_id}/"
    manifest_bytes = read_bytes(s3, bucket, prefix + "manifest.json")
    candidate = json.loads(manifest_bytes)
    if candidate.get("schema") != "alpha_edge_gold_candidate_v3_1":
        raise LakehouseError("Invalid warehouse candidate")
    if candidate.get("run_id") != run_id or candidate.get("status") != "review_required":
        raise LakehouseError("Candidate identity/state invalid")
    if not approval_key.startswith(f"{ROOT}/approvals/{run_id}/"):
        raise LakehouseError("Approval must be a dedicated immutable S3 record")
    approval = read_json(s3, bucket, approval_key)
    if (approval.get("schema") != "alpha_edge_gold_approval_v3_1"
            or approval.get("run_id") != run_id
            or approval.get("candidate_sha256") != digest(manifest_bytes)
            or not approval.get("reviewer")
            or not approval.get("approved_at_utc")
            or approval.get("decision") != "approved"):
        raise LakehouseError("Missing independent, candidate-pinned approval")
    if any(approval.get("gates", {}).get(name) != "passed" for name in _APPROVED_GATES):
        raise LakehouseError("One or more required independent review gates remain open")
    # Actual data content and keys are verified, not trusted based on approval alone.
    quality_blob = verified_bytes(s3, bucket, candidate["artifacts"]["quality"])
    if json.loads(quality_blob) != candidate["quality"]:
        raise LakehouseError("Candidate quality evidence mismatch")
    gold = verified_frame(s3, bucket, candidate["artifacts"]["gold"])
    if set(COLUMNS) - set(gold):
        raise LakehouseError("Gold lacks required warehouse columns")
    if gold.duplicated(["asset_id", "session_date"]).any():
        raise LakehouseError("Duplicate Gold asset/session keys")
    if len(gold) != candidate["quality"]["gold_rows"]:
        raise LakehouseError("Gold row count mismatch")
    if not gold["quality_status"].eq("accepted").all():
        raise LakehouseError("Warehouse publication requires complete accepted Gold price coverage")
    if gold["return_quality_status"].eq("review").any():
        raise LakehouseError("Unresolved Gold return reviews")
    if gold["close_price_usd"].isna().any() or gold["close_total_return_index_usd"].isna().any():
        raise LakehouseError("Missing certified Gold prices")
    return candidate, approval


def _query(athena, sql: str, *, bucket: str, workgroup: str, wait_seconds: int = 600) -> str:
    """Submit SQL and fail closed on any Athena query failure or timeout."""
    query_id = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": DATABASE, "Catalog": "AwsDataCatalog"},
        WorkGroup=workgroup,
        ResultConfiguration={"OutputLocation": f"s3://{bucket}/{ROOT}/athena-results/"},
    )["QueryExecutionId"]
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        state = athena.get_query_execution(QueryExecutionId=query_id)["QueryExecution"]
        status = state["Status"]["State"]
        if status == "SUCCEEDED":
            return query_id
        if status in {"FAILED", "CANCELLED"}:
            raise LakehouseError(f"Athena {status}: {query_id} {state['Status'].get('StateChangeReason', '')}")
        time.sleep(2)
    raise LakehouseError(f"Athena timed out: {query_id}")


def _scalar(athena, query_id: str) -> int:
    rows = athena.get_query_results(QueryExecutionId=query_id)["ResultSet"]["Rows"]
    if len(rows) < 2:
        raise LakehouseError(f"Missing Athena scalar result: {query_id}")
    return int(rows[1]["Data"][0]["VarCharValue"])


def publish(
    s3, athena, *, bucket: str, run_id: str, approval_key: str,
    workgroup: str, execute: bool = False,
) -> dict:
    candidate, approval = approved_candidate(s3, bucket, run_id, approval_key)
    sql = warehouse_sql(bucket, run_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", workgroup):
        raise LakehouseError("Unsafe Athena workgroup")
    expected = candidate["quality"]["gold_rows"]
    if not execute:
        return {"state": "dry_run", "release_id": run_id, "expected_rows": expected, "queries": sql, "writes": 0}
    receipt_key = f"{ROOT}/warehouse/publications/{run_id}.json"
    # Prevent repeat publication: caller must inspect existing release and view.
    try:
        read_json(s3, bucket, receipt_key)
    except LakehouseError as exc:
        if "missing" not in str(exc):
            raise
    else:
        raise LakehouseError("Release already published; verify, never repeat side effects")
    ids = {}
    for step in ("database", "stage", "iceberg", "count", "distinct"):
        ids[step] = _query(athena, sql[step], bucket=bucket, workgroup=workgroup)
    actual = _scalar(athena, ids["count"])
    duplicate_count = _scalar(athena, ids["distinct"])
    if actual != expected or duplicate_count:
        raise LakehouseError(f"Warehouse release rejected: {actual}/{expected} rows, {duplicate_count} duplicates")
    # Only now expose the immutable release through one consumer-facing view.
    ids["view"] = _query(athena, sql["view"], bucket=bucket, workgroup=workgroup)
    receipt = {
        "schema": "alpha_edge_gold_warehouse_release_v3_1",
        "run_id": run_id, "candidate_manifest_sha256": approval["candidate_sha256"],
        "approval_key": approval_key,
        "iceberg_table": f"{DATABASE}.{release_table_name(run_id)}",
        "consumer_view": f"{DATABASE}.{VIEW}",
        "accepted_rows": actual, "athena_query_ids": ids, "status": "published",
    }
    put_json(s3, bucket, receipt_key, receipt)
    return receipt


def main() -> None:
    p = argparse.ArgumentParser(description="Publish independently approved Gold to Athena Iceberg")
    p.add_argument("--bucket", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--approval-key", required=True)
    p.add_argument("--workgroup", required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    import boto3
    print(json.dumps(publish(
        boto3.client("s3", region_name="eu-west-1"),
        boto3.client("athena", region_name="eu-west-1"),
        bucket=a.bucket, run_id=a.run_id, approval_key=a.approval_key,
        workgroup=a.workgroup, execute=a.execute,
    ), indent=2))


if __name__ == "__main__":
    main()
