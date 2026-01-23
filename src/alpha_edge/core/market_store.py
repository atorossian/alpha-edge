# market_store.py
from __future__ import annotations

import io
import json
import uuid
from dataclasses import dataclass
from typing import Iterable, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import threading
from botocore.config import Config

import pandas as pd


@dataclass
class MarketStore:
    bucket: str
    base_prefix: str = "market"
    version: str = "v1"
    region: str = "eu-west-1"

    def __post_init__(self) -> None:
        # Thread-local client so botocore HTTP sessions don't fight each other
        self._tls = threading.local()

        # Bigger pool + adaptive retries helps a lot under concurrency
        self._boto_cfg = Config(
            region_name=self.region,
            max_pool_connections=64,
            retries={"max_attempts": 10, "mode": "adaptive"},
        )

        self._session = boto3.session.Session(region_name=self.region)

    def _client(self):
        c = getattr(self._tls, "s3", None)
        if c is None:
            c = self._session.client("s3", config=self._boto_cfg)
            self._tls.s3 = c
        return c

    # -------------------------
    # Low-level S3 helpers
    # -------------------------
    def _put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self._client().put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def _get_bytes(self, key: str) -> bytes:
        obj = self._client().get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def _key_exists(self, key: str) -> bool:
        try:
            self._client().head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def _list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self._client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for it in (page.get("Contents") or []):
                k = it.get("Key")
                if k:
                    keys.append(k)
        return keys


    # -------------------------
    # Prefixes (S3 KEYS, not s3://)
    # -------------------------
    @property
    def ohlcv_prefix(self) -> str:
        return f"{self.base_prefix}/ohlcv_usd/{self.version}"

    @property
    def returns_prefix(self) -> str:
        return f"{self.base_prefix}/returns_usd/{self.version}"

    @property
    def snapshots_prefix(self) -> str:
        return f"{self.base_prefix}/snapshots/{self.version}"

    @property
    def state_prefix(self) -> str:
        return f"{self.base_prefix}/state/{self.version}"

    @property
    def manifests_prefix(self) -> str:
        return f"{self.base_prefix}/manifests/{self.version}"



    # -------------------------
    # Partition path builders
    # -------------------------
    def _part_key(self, table_prefix: str, asset_id: str, year: int) -> str:
        part = uuid.uuid4().hex[:12]
        asset_id = str(asset_id).strip()
        return f"{table_prefix}/asset_id={asset_id}/year={int(year)}/part-{part}.parquet"

    # -------------------------
    # MANIFESTS (asset_id/year)
    # -------------------------
    def _manifest_key(self, *, table: str, asset_id: str, year: int) -> str:
        asset_id = str(asset_id).strip()
        return f"{self.manifests_prefix}/{table}/asset_id={asset_id}/year={int(year)}/manifest.json"

    def read_asset_year_manifest(self, *, table: str, asset_id: str, year: int) -> dict:
        key = self._manifest_key(table=table, asset_id=asset_id, year=year)
        try:
            return json.loads(self._get_bytes(key).decode("utf-8"))
        except Exception:
            return {}


    def write_asset_year_manifest(self, *, table: str, asset_id: str, year: int, dates: list[str], parts: list[str] | None = None) -> None:
        key = self._manifest_key(table=table, asset_id=asset_id, year=year)
        payload = {
            "asset_id": str(asset_id).strip(),
            "year": int(year),
            "dates": sorted(set(str(d) for d in (dates or []))),
            "parts": sorted(set(str(p) for p in (parts or []))),
            "as_of_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self._put_bytes(key, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"), content_type="application/json")


    # -------------------------
    # RETURNS FULL DONE CHECKPOINTS (per-asset)
    # -------------------------
    def _returns_full_done_key(self, asset_id: str) -> str:
        asset_id = str(asset_id).strip()
        return f"{self.state_prefix}/returns_full_done/asset_id={asset_id}.json"

    def read_returns_full_done(self, asset_id: str) -> dict:
        key = self._returns_full_done_key(asset_id)
        try:
            return json.loads(self._get_bytes(key).decode("utf-8"))
        except Exception:
            return {}

    def write_returns_full_done(self, asset_id: str, payload: dict) -> None:
        key = self._returns_full_done_key(asset_id)
        self._put_bytes(key, json.dumps(payload, indent=2, default=str).encode("utf-8"), content_type="application/json")

    # -------------------------
    # WRITE (append-only parquet)
    # -------------------------
    def write_ohlcv_usd_partitioned(self, df: pd.DataFrame) -> list[str]:
        return self._write_partitioned(df, self.ohlcv_prefix)

    def write_returns_usd_partitioned(self, df: pd.DataFrame) -> list[str]:
        return self._write_partitioned(df, self.returns_prefix)

    def _write_partitioned(self, df: pd.DataFrame, table_prefix: str) -> list[str]:
        if df is None or df.empty:
            return []

        df = df.copy()

        if "asset_id" not in df.columns:
            raise RuntimeError("Expected 'asset_id' column for partitioning by asset_id/year.")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "asset_id"])
        df["date"] = df["date"].dt.tz_localize(None).dt.normalize()

        df["asset_id"] = df["asset_id"].astype(str).str.strip()
        for c in df.columns:
            if pd.api.types.is_categorical_dtype(df[c]):
                df[c] = df[c].astype(str)

        df["year"] = df["date"].dt.year.astype(int)

        written: list[str] = []

        for (asset_id, year), g in df.groupby(["asset_id", "year"], sort=False):
            out = g.drop(columns=["year"]).sort_values("date")

            bio = io.BytesIO()
            out.to_parquet(bio, index=False)
            bio.seek(0)

            key = self._part_key(table_prefix, str(asset_id), int(year))
            self._put_bytes(key, bio.read(), content_type="application/octet-stream")
            written.append(key)

        return written

    # -------------------------
    # SNAPSHOTS
    # -------------------------
    def write_latest_prices_snapshot(self, latest_prices: pd.DataFrame) -> None:
        key = f"{self.snapshots_prefix}/latest_prices.parquet"
        bio = io.BytesIO()
        latest_prices.to_parquet(bio, index=False)
        bio.seek(0)
        self._put_bytes(key, bio.read())

    def write_latest_returns_snapshot(self, latest_returns: pd.DataFrame) -> None:
        key = f"{self.snapshots_prefix}/latest_returns.parquet"
        bio = io.BytesIO()
        latest_returns.to_parquet(bio, index=False)
        bio.seek(0)
        self._put_bytes(key, bio.read())

    # -------------------------
    # STATE (json)
    # -------------------------
    def write_last_date_state(self, last_date_by_asset: dict[str, str]) -> None:
        """
        Canonical state for the new asset_id pipeline.
        """
        key = f"{self.state_prefix}/last_date_by_asset_id.json"
        self._put_bytes(key, json.dumps(last_date_by_asset, indent=2).encode("utf-8"), content_type="application/json")

    def read_last_date_state(self) -> dict[str, str]:
        """
        Reads canonical asset_id state. (Older code that wrote last_date_by_ticker.json
        should be migrated; we keep a best-effort fallback read for it.)
        """
        key_asset = f"{self.state_prefix}/last_date_by_asset_id.json"
        try:
            return json.loads(self._get_bytes(key_asset).decode("utf-8"))
        except Exception:
            # fallback: legacy file name (if exists)
            key_legacy = f"{self.state_prefix}/last_date_by_ticker.json"
            try:
                return json.loads(self._get_bytes(key_legacy).decode("utf-8"))
            except Exception:
                return {}

    def _provider_symbol_state_key(self) -> str:
        return f"{self.state_prefix}/provider_symbol_by_asset_id.json"

    def write_provider_symbol_state(self, mapping: dict[str, str]) -> None:
        key = self._provider_symbol_state_key()
        self._put_bytes(key, json.dumps(mapping, indent=2, sort_keys=True).encode("utf-8"), content_type="application/json")

    def read_provider_symbol_state(self) -> dict[str, str]:
        key = self._provider_symbol_state_key()
        try:
            return json.loads(self._get_bytes(key).decode("utf-8"))
        except Exception:
            return {}

    def write_ingest_failures(self, df: pd.DataFrame) -> None:
        # latest
        key_latest = f"{self.base_prefix}/ingest_failures/{self.version}/latest.parquet"
        bio = io.BytesIO()
        df.to_parquet(bio, index=False)
        bio.seek(0)
        self._put_bytes(key_latest, bio.read())

        # history
        dt_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        key_hist = f"{self.base_prefix}/ingest_failures/{self.version}/dt={dt_str}/failures.parquet"
        bio = io.BytesIO()
        df.to_parquet(bio, index=False)
        bio.seek(0)
        self._put_bytes(key_hist, bio.read())

    def write_returns_latest_state(self, payload: dict) -> None:
        key = f"{self.state_prefix}/returns_latest.json"
        self._put_bytes(key, json.dumps(payload, indent=2, default=str).encode("utf-8"), content_type="application/json")

    def read_returns_latest_state(self) -> dict:
        key = f"{self.state_prefix}/returns_latest.json"
        try:
            return json.loads(self._get_bytes(key).decode("utf-8"))
        except Exception:
            return {}

    def _regime_filter_state_key(self) -> str:
        return f"{self.state_prefix}/regime_filter_state.json"

    def write_regime_filter_state(self, payload: dict) -> None:
        key = self._regime_filter_state_key()
        self._put_bytes(key, json.dumps(payload, indent=2, default=str).encode("utf-8"), content_type="application/json")

    def read_regime_filter_state(self) -> dict:
        key = self._regime_filter_state_key()
        try:
            return json.loads(self._get_bytes(key).decode("utf-8"))
        except Exception:
            return {}

    # -------------------------
    # Universe triage outputs (unchanged API)
    # -------------------------
    def write_universe_triage_outputs(
        self,
        *,
        as_of: str,
        triage_report: pd.DataFrame,
        suggested_overrides: pd.DataFrame,
        suggested_exclusions: pd.DataFrame,
        mapping_changes: pd.DataFrame | None = None,
        mapping_validation: pd.DataFrame | None = None,
    ) -> None:
        # These are big-ish; keep pandas direct-to-s3 behavior if you want,
        # but to stay consistent with "boto3 only", we write via bytes too.
        base = f"{self.base_prefix}/universe_triage/{self.version}/dt={as_of}"

        # triage_report parquet
        bio = io.BytesIO()
        triage_report.to_parquet(bio, index=False)
        bio.seek(0)
        self._put_bytes(f"{base}/triage_report.parquet", bio.read())

        # CSVs
        self._put_bytes(f"{base}/suggested_overrides.csv", suggested_overrides.to_csv(index=False).encode("utf-8"), content_type="text/csv")
        self._put_bytes(f"{base}/suggested_exclusions.csv", suggested_exclusions.to_csv(index=False).encode("utf-8"), content_type="text/csv")

        if mapping_changes is not None:
            self._put_bytes(f"{base}/mapping_changes.csv", mapping_changes.to_csv(index=False).encode("utf-8"), content_type="text/csv")
        if mapping_validation is not None:
            self._put_bytes(f"{base}/mapping_validation.csv", mapping_validation.to_csv(index=False).encode("utf-8"), content_type="text/csv")

    # -------------------------
    # READ (boto3 + BytesIO)
    # -------------------------
    def read_latest_prices_snapshot(self) -> pd.DataFrame:
        key = f"{self.snapshots_prefix}/latest_prices.parquet"
        try:
            raw = self._get_bytes(key)
            return pd.read_parquet(io.BytesIO(raw))
        except Exception:
            return pd.DataFrame()

    def read_latest_returns_snapshot(self) -> pd.DataFrame:
        key = f"{self.snapshots_prefix}/latest_returns.parquet"
        try:
            raw = self._get_bytes(key)
            return pd.read_parquet(io.BytesIO(raw))
        except Exception:
            return pd.DataFrame()

    def read_ohlcv_usd(
        self,
        asset_ids: Iterable[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        return self._read_partitioned(self.ohlcv_prefix, asset_ids, start, end, columns)

    def read_returns_usd(
        self,
        asset_ids: Iterable[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        return self._read_partitioned(self.returns_prefix, asset_ids, start, end, columns)


    def _read_partitioned(
        self,
        table_prefix: str,
        asset_ids: Iterable[str],
        start: Optional[str],
        end: Optional[str],
        columns: Optional[list[str]],
    ) -> pd.DataFrame:
        def _load_one(key: str) -> pd.DataFrame | None:
            raw = self._get_bytes(key)
            df = pd.read_parquet(io.BytesIO(raw), columns=columns)
            if df is None or df.empty:
                return None
            if "asset_id" in df.columns:
                df["asset_id"] = df["asset_id"].astype(str).str.strip()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
            return df

        ids = [str(a).strip() for a in asset_ids if str(a).strip()]
        if not ids:
            return pd.DataFrame()

        start_ts = pd.to_datetime(start).normalize() if start else None
        end_ts = pd.to_datetime(end).normalize() if end else None

        # years range (tight)
        if start_ts is None and end_ts is None:
            years = [pd.Timestamp.utcnow().year]
        else:
            y0 = int(start_ts.year) if start_ts is not None else 1900
            y1 = int(end_ts.year) if end_ts is not None else 2100
            years = list(range(y0, y1 + 1))

        table_name = "ohlcv_usd" if table_prefix == self.ohlcv_prefix else "returns_usd"

        # Collect keys once (manifest parts preferred, fallback to list)
        all_keys: list[str] = []

        for asset_id in ids:
            for y in years:
                man = self.read_asset_year_manifest(table=table_name, asset_id=asset_id, year=int(y)) or {}
                parts = [k for k in (man.get("parts") or []) if isinstance(k, str) and k.endswith(".parquet")]

                if parts:
                    all_keys.extend(parts)
                    continue

                # Legacy fallback (only when no parts)
                prefix = f"{table_prefix}/asset_id={asset_id}/year={int(y)}/"
                keys = [k for k in self._list_keys(prefix) if k.endswith(".parquet")]
                all_keys.extend(keys)

        # De-dupe (manifests can accumulate duplicates if you ever union twice)
        if not all_keys:
            return pd.DataFrame()
        all_keys = list(dict.fromkeys(all_keys))  # stable unique

        dfs: list[pd.DataFrame] = []
        max_workers = 16  # safe default; your botocore pool is 64

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_load_one, k) for k in all_keys]
            for fut in as_completed(futs):
                try:
                    df = fut.result()
                    if df is not None and not df.empty:
                        dfs.append(df)
                except Exception:
                    pass

        if not dfs:
            return pd.DataFrame()

        out = pd.concat(dfs, ignore_index=True)

        if "date" in out.columns:
            out = out.dropna(subset=["date"])
            if start_ts is not None:
                out = out[out["date"] >= start_ts]
            if end_ts is not None:
                out = out[out["date"] <= end_ts]

        if "asset_id" in out.columns and "date" in out.columns:
            out = out.sort_values(["asset_id", "date"])
        elif "date" in out.columns:
            out = out.sort_values(["date"])

        return out.reset_index(drop=True)
