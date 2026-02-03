# data_loader.py  (S3-only I/O helpers + parsing utilities)
from __future__ import annotations

import io
import json
import re
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
from collections.abc import KeysView, ValuesView, ItemsView

from alpha_edge.core.schemas import ScoreConfig, Position, PortfolioHealth


# ----------------------------
# JSON serialization helpers
# ----------------------------

def _json_default(o: Any):
    # dataclasses
    if is_dataclass(o):
        return asdict(o)

    # pandas timestamps
    if isinstance(o, pd.Timestamp):
        return o.isoformat()

    # numpy scalars
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)

    # dict views (dict_keys / dict_values / dict_items)
    if isinstance(o, (KeysView, ValuesView, ItemsView)):
        return list(o)

    # sets
    if isinstance(o, (set, frozenset)):
        return list(o)

    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


# ----------------------------
# S3 generic read helpers
# ----------------------------

def s3_read_json(s3, *, bucket: str, key: str) -> dict:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def s3_read_df(
    s3,
    *,
    bucket: str,
    key: str,
    index_col: Optional[str] = None,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Reads S3 object into a DataFrame.
    Supports:
      - CSV (plain)
      - CSV (gzip)  [by magic header 1f 8b]
      - Parquet      [by magic header PAR1]
    """
    obj = s3.get_object(Bucket=bucket, Key=key)
    body: bytes = obj["Body"].read()

    # ---- sniff format ----
    is_gzip = len(body) >= 2 and body[0] == 0x1F and body[1] == 0x8B
    is_parquet = len(body) >= 4 and body[:4] == b"PAR1"

    if is_parquet:
        df = pd.read_parquet(io.BytesIO(body))
    else:
        bio = io.BytesIO(body)
        if is_gzip:
            df = pd.read_csv(bio, compression="gzip")
        else:
            df = pd.read_csv(bio)

    # ---- index handling ----
    if index_col is not None and index_col in df.columns:
        if parse_dates:
            df[index_col] = pd.to_datetime(df[index_col], errors="coerce")
        df = df.set_index(index_col).sort_index()

    return df


# ----------------------------
# S3 dt-partitioned "engine" store helpers
# ----------------------------

_DT_RE = re.compile(r"/dt=(\d{4}-\d{2}-\d{2})/")

def s3_init(region: str = "eu-west-1"):
    return boto3.client("s3", region_name=region)


def s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    body = json.dumps(payload, indent=2, default=_json_default).encode("utf-8")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def s3_put_parquet(s3, *, bucket: str, key: str, df: pd.DataFrame, index: bool = False) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=index)
    buf.seek(0)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.read(),
        ContentType="application/octet-stream",
    )


def s3_list_keys(s3, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kwargs = dict(Bucket=bucket, Prefix=prefix)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []):
            keys.append(it["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def s3_latest_dt(
    s3,
    *,
    bucket: str,
    table_prefix: str,
) -> Optional[str]:
    """
    Finds latest dt under: {table_prefix}/dt=YYYY-MM-DD/
    table_prefix example: "engine/v1/daily_reports"
    Returns "YYYY-MM-DD" or None.
    """
    prefix = table_prefix.rstrip("/") + "/"
    keys = s3_list_keys(s3, bucket=bucket, prefix=prefix)
    latest = None
    for k in keys:
        m = _DT_RE.search(k)
        if not m:
            continue
        dt_str = m.group(1)
        if latest is None or dt_str > latest:
            latest = dt_str
    return latest


def s3_dt_key(*, root_prefix: str, table: str, dt: pd.Timestamp, filename: str) -> str:
    dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
    return f"{root_prefix.rstrip('/')}/{table}/dt={dt_str}/{filename}"


def s3_latest_key(*, root_prefix: str, table: str, filename: str = "latest.json") -> str:
    return f"{root_prefix.rstrip('/')}/{table}/{filename}"


def s3_write_json_event(
    s3,
    *,
    bucket: str,
    root_prefix: str,
    table: str,
    dt: pd.Timestamp,
    filename: str,
    payload: Dict[str, Any],
    update_latest: bool = False,
) -> str:
    """
    Writes:
      s3://bucket/{root_prefix}/{table}/dt=YYYY-MM-DD/{filename}
    Optionally writes:
      s3://bucket/{root_prefix}/{table}/latest.json
    Returns the dt-key.
    """
    key = s3_dt_key(root_prefix=root_prefix, table=table, dt=dt, filename=filename)
    s3_put_json(s3, bucket=bucket, key=key, payload=payload)

    if update_latest:
        latest = s3_latest_key(root_prefix=root_prefix, table=table, filename="latest.json")
        s3_put_json(s3, bucket=bucket, key=latest, payload=payload)

    return key


def s3_write_parquet_partition(
    s3,
    *,
    bucket: str,
    root_prefix: str,
    table: str,
    dt: pd.Timestamp,
    filename: str,
    df: pd.DataFrame,
) -> str:
    """
    Writes Parquet to:
      s3://bucket/{root_prefix}/{table}/dt=YYYY-MM-DD/{filename}
    Returns the dt-key.
    """
    key = s3_dt_key(root_prefix=root_prefix, table=table, dt=dt, filename=filename)
    s3_put_parquet(s3, bucket=bucket, key=key, df=df, index=False)
    return key


def s3_load_latest_json(
    s3,
    *,
    bucket: str,
    root_prefix: str,
    table: str,
) -> Optional[dict]:
    """
    Loads s3://bucket/{root_prefix}/{table}/latest.json if it exists.
    """
    key = s3_latest_key(root_prefix=root_prefix, table=table, filename="latest.json")
    try:
        return s3_read_json(s3, bucket=bucket, key=key)
    except Exception:
        return None


def s3_load_latest_dt_json(
    s3,
    *,
    bucket: str,
    root_prefix: str,
    table: str,
    filename: str,
) -> Optional[dict]:
    """
    Loads JSON from latest dt partition:
      s3://bucket/{root_prefix}/{table}/dt=LATEST/{filename}
    """
    table_prefix = f"{root_prefix.rstrip('/')}/{table}"
    dt_str = s3_latest_dt(s3, bucket=bucket, table_prefix=table_prefix)
    if not dt_str:
        return None
    key = f"{table_prefix}/dt={dt_str}/{filename}"
    try:
        return s3_read_json(s3, bucket=bucket, key=key)
    except Exception:
        return None


def s3_load_latest_report_score(
    s3,
    *,
    bucket: str,
    root_prefix: str,
    table: str = "daily_reports",
    filename: str = "report.json",
) -> Optional[float]:
    payload = s3_load_latest_dt_json(
        s3,
        bucket=bucket,
        root_prefix=root_prefix,
        table=table,
        filename=filename,
    )
    if not payload:
        return None
    try:
        return float(payload["report"]["eval"]["score"])
    except Exception:
        return None


# ----------------------------
# Data cleaning helpers (still used in engine)
# ----------------------------

def clean_returns_matrix(
    returns: pd.DataFrame,
    *,
    min_history_days: int = 252 * 2,
    max_nan_frac: float = 0.25,
    min_vol: float = 1e-6,
) -> tuple[pd.DataFrame, dict]:
    diag: dict = {}
    if returns.empty:
        return returns, {"final_n_assets": 0, "final_n_days": 0}

    r = returns.copy()
    r = r.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    nan_frac = r.isna().mean()
    keep = nan_frac <= max_nan_frac
    diag["dropped_too_many_nans"] = nan_frac[~keep].sort_values(ascending=False).to_dict()
    r = r.loc[:, keep]

    non_nan = r.notna().sum()
    keep = non_nan >= min_history_days
    diag["dropped_too_short"] = non_nan[~keep].sort_values().to_dict()
    r = r.loc[:, keep]

    r = r.sort_index().ffill()

    vols = r.std(axis=0, ddof=1)
    keep = vols > min_vol
    diag["dropped_flat"] = vols[~keep].sort_values().to_dict()
    r = r.loc[:, keep]

    diag["final_n_assets"] = int(r.shape[1])
    diag["final_n_days"] = int(r.shape[0])
    return r, diag


# ----------------------------
# Engine parsing utilities
# ----------------------------

def new_run_id(dt_: Optional[pd.Timestamp] = None) -> str:
    dt_ = pd.Timestamp(dt_ or pd.Timestamp.utcnow())
    return f"{dt_.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def parse_positions_obj(obj: dict) -> dict[str, Position]:
    """
    Accepts either:
      - { "VNQ": {"ticker":"VNQ", ...}, ... }  (stored JSON)
      - or already Position-like dicts
    Returns dict[str, Position].
    """
    if obj is None:
        return {}

    out: dict[str, Position] = {}
    for t, v in obj.items():
        if isinstance(v, Position):
            out[t] = v
        elif isinstance(v, dict):
            out[t] = Position(**v)
        else:
            raise TypeError(f"Invalid position payload for {t}: {type(v)}")
    return out

def parse_ledger_positions_obj(obj: dict) -> tuple[list[dict], list[dict]]:
    """
    Expects ledger positions payload from rebuild_ledger.py:
      {
        "spot_positions": [ {...}, ... ],
        "derivatives_positions": [ {...}, ... ],
        ...
      }
    Returns (spot_positions, derivatives_positions) as lists of dict.
    """
    if obj is None or not isinstance(obj, dict):
        return ([], [])

    spot = obj.get("spot_positions") or []
    deriv = obj.get("derivatives_positions") or []

    if not isinstance(spot, list) or not isinstance(deriv, list):
        raise TypeError("Invalid ledger positions payload: expected lists for spot_positions/derivatives_positions")

    return spot, deriv



def parse_portfolio_health(obj) -> PortfolioHealth:
    if isinstance(obj, PortfolioHealth):
        return obj
    if not isinstance(obj, dict):
        raise TypeError(f"Invalid baseline type: {type(obj)}")

    d = dict(obj)
    d["date"] = pd.to_datetime(d["date"])
    return PortfolioHealth(**d)

def s3_get_json(s3, *, bucket: str, key: str) -> dict:
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return json.loads(body.decode("utf-8"))

def s3_load_latest_json_asof(
    s3,
    *,
    bucket: str,
    root_prefix: str,
    table: str,
    as_of: str,
) -> dict | None:
    """
    Loads the most recent dt-partitioned JSON <= as_of (YYYY-MM-DD),
    falling back to latest.json if nothing found.
    """
    as_of_str = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    prefix = f"{root_prefix.rstrip('/')}/{table}/"
    keys = s3_list_keys(s3, bucket=bucket, prefix=prefix)

    best_dt = None
    best_key = None
    for k in keys:
        m = _DT_RE.search(k)
        if not m:
            continue
        dt_str = m.group(1)
        if dt_str <= as_of_str and (best_dt is None or dt_str > best_dt):
            # prefer "latest.json" within dt partition if it exists, else any json
            if k.endswith(".json"):
                best_dt = dt_str
                best_key = k

    if best_key:
        return s3_get_json(s3, bucket=bucket, key=best_key)

    # fallback
    return s3_load_latest_json(s3, bucket=bucket, root_prefix=root_prefix, table=table)
