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


    def _read_parquet_tolerant(
        self,
        key: str,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Read one parquet file while tolerating schema evolution.

        If requested columns are missing in older parquet files, read the full
        file, add missing requested columns as NA, and return columns in the
        requested order.

        This is important for append-only partitioned datasets where old parts
        may not contain newer compatibility columns.
        """
        raw = self._get_bytes(key)

        if columns is None:
            return pd.read_parquet(io.BytesIO(raw))

        try:
            return pd.read_parquet(io.BytesIO(raw), columns=columns)
        except Exception:
            df = pd.read_parquet(io.BytesIO(raw))

            if df is None or df.empty:
                return pd.DataFrame(columns=columns)

            out = df.copy()

            for col in columns:
                if col not in out.columns:
                    out[col] = pd.NA

            return out[columns].copy()
        

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

    def write_asset_year_manifest(
        self,
        *,
        table: str,
        asset_id: str,
        year: int,
        dates: list[str],
        parts: list[str] | None = None,
    ) -> None:
        key = self._manifest_key(table=table, asset_id=asset_id, year=year)
        payload = {
            "asset_id": str(asset_id).strip(),
            "year": int(year),
            "dates": sorted(set(str(d) for d in (dates or []))),
            "parts": sorted(set(str(p) for p in (parts or []))),
            "as_of_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self._put_bytes(
            key,
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
            content_type="application/json",
        )

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
        self._put_bytes(
            key,
            json.dumps(payload, indent=2, default=str).encode("utf-8"),
            content_type="application/json",
        )

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
            if isinstance(df[c].dtype, pd.CategoricalDtype):
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
    def _snapshot_key(self, name: str) -> str:
        return f"{self.snapshots_prefix}/{name}.parquet"

    def _write_snapshot(self, name: str, df: pd.DataFrame) -> None:
        bio = io.BytesIO()
        df.to_parquet(bio, index=False)
        bio.seek(0)
        self._put_bytes(self._snapshot_key(name), bio.read())

    def _read_snapshot(self, name: str) -> pd.DataFrame:
        key = self._snapshot_key(name)
        try:
            raw = self._get_bytes(key)
            return pd.read_parquet(io.BytesIO(raw))
        except Exception:
            return pd.DataFrame()

    # Backward-compatible combined snapshot
    def write_latest_prices_snapshot(self, latest_prices: pd.DataFrame) -> None:
        self._write_snapshot("latest_prices", latest_prices)

    def read_latest_prices_snapshot(self) -> pd.DataFrame:
        return self._read_snapshot("latest_prices")

    # New Phase 0 raw/adjusted snapshot APIs
    def write_latest_prices_raw_snapshot(self, latest_prices_raw: pd.DataFrame) -> None:
        self._write_snapshot("latest_prices_raw", latest_prices_raw)

    def read_latest_prices_raw_snapshot(self) -> pd.DataFrame:
        return self._read_snapshot("latest_prices_raw")

    def write_latest_prices_adjusted_snapshot(self, latest_prices_adjusted: pd.DataFrame) -> None:
        self._write_snapshot("latest_prices_adjusted", latest_prices_adjusted)

    def read_latest_prices_adjusted_snapshot(self) -> pd.DataFrame:
        return self._read_snapshot("latest_prices_adjusted")

    # Returns snapshot remains analytics-oriented
    def write_latest_returns_snapshot(self, latest_returns: pd.DataFrame) -> None:
        self._write_snapshot("latest_returns", latest_returns)

    def read_latest_returns_snapshot(self) -> pd.DataFrame:
        return self._read_snapshot("latest_returns")

    # -------------------------
    # STATE (json)
    # -------------------------
    def write_last_date_state(self, last_date_by_asset: dict[str, str]) -> None:
        """
        Canonical state for the new asset_id pipeline.
        """
        key = f"{self.state_prefix}/last_date_by_asset_id.json"
        self._put_bytes(
            key,
            json.dumps(last_date_by_asset, indent=2).encode("utf-8"),
            content_type="application/json",
        )

    def read_last_date_state(self) -> dict[str, str]:
        """
        Reads canonical asset_id state. (Older code that wrote last_date_by_ticker.json
        should be migrated; we keep a best-effort fallback read for it.)
        """
        key_asset = f"{self.state_prefix}/last_date_by_asset_id.json"
        try:
            return json.loads(self._get_bytes(key_asset).decode("utf-8"))
        except Exception:
            key_legacy = f"{self.state_prefix}/last_date_by_ticker.json"
            try:
                return json.loads(self._get_bytes(key_legacy).decode("utf-8"))
            except Exception:
                return {}

    def _provider_symbol_state_key(self) -> str:
        return f"{self.state_prefix}/provider_symbol_by_asset_id.json"

    def write_provider_symbol_state(self, mapping: dict[str, str]) -> None:
        key = self._provider_symbol_state_key()
        self._put_bytes(
            key,
            json.dumps(mapping, indent=2, sort_keys=True).encode("utf-8"),
            content_type="application/json",
        )

    def read_provider_symbol_state(self) -> dict[str, str]:
        key = self._provider_symbol_state_key()
        try:
            return json.loads(self._get_bytes(key).decode("utf-8"))
        except Exception:
            return {}

    def write_ingest_failures(self, df: pd.DataFrame) -> None:
        key_latest = f"{self.base_prefix}/ingest_failures/{self.version}/latest.parquet"
        bio = io.BytesIO()
        df.to_parquet(bio, index=False)
        bio.seek(0)
        self._put_bytes(key_latest, bio.read())

        dt_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        key_hist = f"{self.base_prefix}/ingest_failures/{self.version}/dt={dt_str}/failures.parquet"
        bio = io.BytesIO()
        df.to_parquet(bio, index=False)
        bio.seek(0)
        self._put_bytes(key_hist, bio.read())

    def write_returns_latest_state(self, payload: dict) -> None:
        key = f"{self.state_prefix}/returns_latest.json"
        self._put_bytes(
            key,
            json.dumps(payload, indent=2, default=str).encode("utf-8"),
            content_type="application/json",
        )

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
        self._put_bytes(
            key,
            json.dumps(payload, indent=2, default=str).encode("utf-8"),
            content_type="application/json",
        )

    def read_regime_filter_state(self) -> dict:
        key = self._regime_filter_state_key()
        try:
            return json.loads(self._get_bytes(key).decode("utf-8"))
        except Exception:
            return {}



    # -------------------------
    # MARKET REGIME HMM OUTPUTS
    # -------------------------
    @property
    def market_hmm_prefix(self) -> str:
        """
        Regime HMM output prefix.

        Use a MarketStore whose base_prefix is the engine root, for example:
            MarketStore(bucket=bucket, region=region, base_prefix="engine/v1")

        Result:
            engine/v1/regimes/market_hmm
        """
        return f"{self.base_prefix}/regimes/market_hmm"

    def _market_hmm_regime_key(self, as_of: str) -> str:
        as_of = str(as_of).strip()
        if not as_of:
            raise ValueError("as_of is required for market HMM regime key")
        return f"{self.market_hmm_prefix}/dt={as_of}/regime.json"

    def _market_hmm_latest_key(self) -> str:
        return f"{self.market_hmm_prefix}/latest.json"

    def write_market_hmm_regime(
        self,
        *,
        as_of: str,
        payload: dict,
        update_latest: bool = True,
    ) -> list[str]:
        """
        Write market HMM regime output to the existing regime path.

        Writes:
            <base_prefix>/regimes/market_hmm/dt=YYYY-MM-DD/regime.json
            <base_prefix>/regimes/market_hmm/latest.json, if update_latest=True
        """
        key = self._market_hmm_regime_key(as_of)

        self._put_bytes(
            key,
            json.dumps(payload, indent=2, default=str, sort_keys=True).encode("utf-8"),
            content_type="application/json",
        )

        written = [key]

        if update_latest:
            latest_key = self._market_hmm_latest_key()
            self._put_bytes(
                latest_key,
                json.dumps(payload, indent=2, default=str, sort_keys=True).encode("utf-8"),
                content_type="application/json",
            )
            written.append(latest_key)

        return written

    def read_market_hmm_regime(self, as_of: str | None = None) -> dict:
        """
        Read market HMM regime output.

        If as_of is None, reads latest.json.
        Otherwise reads dt=YYYY-MM-DD/regime.json.
        """
        key = self._market_hmm_latest_key() if as_of is None else self._market_hmm_regime_key(as_of)

        try:
            return json.loads(self._get_bytes(key).decode("utf-8"))
        except Exception:
            return {}

    def list_market_hmm_regime_dates(self) -> list[str]:
        """
        Return available dt=YYYY-MM-DD partitions under the existing regime path.
        """
        prefix = f"{self.market_hmm_prefix}/dt="
        keys = self._list_keys(prefix)

        dates: set[str] = set()

        for key in keys:
            marker = "/dt="
            if marker not in key:
                continue

            tail = key.split(marker, 1)[1]
            dt_part = tail.split("/", 1)[0]

            if len(dt_part) == 10:
                dates.add(dt_part)

        return sorted(dates)

    def read_market_hmm_regime_history(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        include_mixed: bool = True,
        require_point_in_time: bool = False,
    ) -> pd.DataFrame:
        """
        Read point-in-time market HMM regime history from the existing path.

        Existing path:
            <base_prefix>/regimes/market_hmm/dt=YYYY-MM-DD/regime.json

        This method does not create a new table. It reconstructs regime history
        from the daily JSON files that already exist under regimes/market_hmm.
        """
        start_ts = pd.to_datetime(start).normalize() if start else None
        end_ts = pd.to_datetime(end).normalize() if end else None

        rows: list[dict[str, Any]] = []

        for d in self.list_market_hmm_regime_dates():
            d_ts = pd.to_datetime(d, errors="coerce")
            if pd.isna(d_ts):
                continue

            d_ts = pd.Timestamp(d_ts).normalize()

            if start_ts is not None and d_ts < start_ts:
                continue
            if end_ts is not None and d_ts > end_ts:
                continue

            key = self._market_hmm_regime_key(d)

            try:
                payload = json.loads(self._get_bytes(key).decode("utf-8"))
            except Exception:
                continue

            if not isinstance(payload, dict):
                continue

            hmm = payload.get("hmm") or {}
            if not isinstance(hmm, dict):
                hmm = {}

            p_label = hmm.get("p_label_today") or {}
            if not isinstance(p_label, dict):
                p_label = {}

            label = hmm.get("label_commit")
            label_or_mixed = str(label) if label else "MIXED"

            if not include_mixed and label_or_mixed == "MIXED":
                continue

            meta = payload.get("meta") or {}
            hmm_meta = hmm.get("meta") or {}
            point_in_time = bool(meta.get("point_in_time", False) or hmm_meta.get("point_in_time", False))
            lookahead_safe = bool(meta.get("lookahead_safe", False) or hmm_meta.get("lookahead_safe", False))

            if require_point_in_time and not (point_in_time and lookahead_safe):
                continue

            lev = payload.get("leverage_recommendation") or {}
            if not isinstance(lev, dict):
                lev = {}

            try:
                confidence = max(float(v) for v in p_label.values()) if p_label else None
            except Exception:
                confidence = None

            rows.append(
                {
                    "date": d_ts,
                    "as_of": str(payload.get("as_of") or d),
                    "label": label,
                    "label_or_mixed": label_or_mixed,
                    "regime": label_or_mixed,
                    "confidence": confidence,
                    "p_CALM_BULL": float(p_label.get("CALM_BULL", 0.0) or 0.0),
                    "p_CHOPPY_BULL": float(p_label.get("CHOPPY_BULL", 0.0) or 0.0),
                    "p_CHOPPY_BEAR": float(p_label.get("CHOPPY_BEAR", 0.0) or 0.0),
                    "p_STRESS_BEAR": float(p_label.get("STRESS_BEAR", 0.0) or 0.0),
                    "target_leverage": (
                        None if lev.get("leverage") is None else float(lev.get("leverage"))
                    ),
                    "source_key": key,
                    "point_in_time": point_in_time,
                    "lookahead_safe": lookahead_safe,
                }
            )

        columns = [
            "date",
            "as_of",
            "label",
            "label_or_mixed",
            "regime",
            "confidence",
            "p_CALM_BULL",
            "p_CHOPPY_BULL",
            "p_CHOPPY_BEAR",
            "p_STRESS_BEAR",
            "target_leverage",
            "source_key",
            "point_in_time",
            "lookahead_safe",
        ]

        if not rows:
            return pd.DataFrame(columns=columns)

        out = pd.DataFrame(rows)
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        out = out.dropna(subset=["date"])
        out = out.sort_values("date", kind="stable")
        out = out.drop_duplicates(subset=["date"], keep="last")
        out = out.reset_index(drop=True)

        for col in columns:
            if col not in out.columns:
                out[col] = pd.NA

        return out[columns].copy()

    # -------------------------
    # Universe triage outputs
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
        base = f"{self.base_prefix}/universe_triage/{self.version}/dt={as_of}"

        bio = io.BytesIO()
        triage_report.to_parquet(bio, index=False)
        bio.seek(0)
        self._put_bytes(f"{base}/triage_report.parquet", bio.read())

        self._put_bytes(
            f"{base}/suggested_overrides.csv",
            suggested_overrides.to_csv(index=False).encode("utf-8"),
            content_type="text/csv",
        )
        self._put_bytes(
            f"{base}/suggested_exclusions.csv",
            suggested_exclusions.to_csv(index=False).encode("utf-8"),
            content_type="text/csv",
        )

        if mapping_changes is not None:
            self._put_bytes(
                f"{base}/mapping_changes.csv",
                mapping_changes.to_csv(index=False).encode("utf-8"),
                content_type="text/csv",
            )
        if mapping_validation is not None:
            self._put_bytes(
                f"{base}/mapping_validation.csv",
                mapping_validation.to_csv(index=False).encode("utf-8"),
                content_type="text/csv",
            )
    # -------------------------
    # CORPORATE ACTIONS
    # -------------------------
    @property
    def corporate_actions_prefix(self) -> str:
        # market/reference/v1/corporate_actions
        return f"{self.base_prefix}/reference/{self.version}/corporate_actions"


    def _corporate_actions_part_key(self, asset_id: str) -> str:
        asset_id = str(asset_id).strip()
        return f"{self.corporate_actions_prefix}/asset_id={asset_id}/part-00000.parquet"


    def write_corporate_actions_partitioned(self, df: pd.DataFrame) -> list[str]:
        """
        Overwrite one parquet part per asset_id:
        market/reference/v1/corporate_actions/asset_id=<asset_id>/part-00000.parquet
        """
        if df is None or df.empty:
            return []

        df = df.copy()
        if "asset_id" not in df.columns:
            raise RuntimeError("Expected 'asset_id' column for corporate actions partitioning.")

        df["asset_id"] = df["asset_id"].astype(str).str.strip()
        if "effective_date" in df.columns:
            df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce").dt.date

        written: list[str] = []

        for asset_id, g in df.groupby("asset_id", sort=False):
            out = g.sort_values(["effective_date", "ticker"], kind="stable").reset_index(drop=True)

            bio = io.BytesIO()
            out.to_parquet(bio, index=False)
            bio.seek(0)

            key = self._corporate_actions_part_key(str(asset_id))
            self._put_bytes(key, bio.read(), content_type="application/octet-stream")
            written.append(key)

        return written


    # -------------------------
    # READ (boto3 + BytesIO)
    # -------------------------
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

    def read_corporate_actions(
        self,
        asset_ids: Iterable[str] | None = None,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Read partitioned corporate actions. If asset_ids is None, read all partitions.
        """
        keys: list[str] = []

        if asset_ids is None:
            keys = [k for k in self._list_keys(self.corporate_actions_prefix + "/") if k.endswith(".parquet")]
        else:
            ids = [str(a).strip() for a in asset_ids if str(a).strip()]
            for asset_id in ids:
                key = self._corporate_actions_part_key(asset_id)
                if self._key_exists(key):
                    keys.append(key)

        if not keys:
            return pd.DataFrame()

        dfs: list[pd.DataFrame] = []
        for key in keys:
            try:
                raw = self._get_bytes(key)
                df = pd.read_parquet(io.BytesIO(raw), columns=columns)
                if df is not None and not df.empty:
                    if "asset_id" in df.columns:
                        df["asset_id"] = df["asset_id"].astype(str).str.strip()
                    if "effective_date" in df.columns:
                        df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce").dt.date
                    dfs.append(df)
            except Exception:
                continue

        if not dfs:
            return pd.DataFrame()

        out = pd.concat(dfs, ignore_index=True)
        if "asset_id" in out.columns and "effective_date" in out.columns:
            out = out.sort_values(["asset_id", "effective_date"], kind="stable")
        return out.reset_index(drop=True)

    def _read_partitioned(
        self,
        table_prefix: str,
        asset_ids: Iterable[str],
        start: Optional[str],
        end: Optional[str],
        columns: Optional[list[str]],
    ) -> pd.DataFrame:
        def _load_one(key: str) -> pd.DataFrame | None:
            df = self._read_parquet_tolerant(key, columns=columns)

            if df is None or df.empty:
                return None

            if "asset_id" in df.columns:
                df["asset_id"] = df["asset_id"].astype(str).str.strip()

            if "date" in df.columns:
                df["date"] = (
                    pd.to_datetime(df["date"], errors="coerce", utc=True)
                    .dt.tz_convert(None)
                    .dt.normalize()
                )

            return df

        ids = [str(a).strip() for a in asset_ids if str(a).strip()]
        if not ids:
            return pd.DataFrame()

        start_ts = pd.to_datetime(start).normalize() if start else None
        end_ts = pd.to_datetime(end).normalize() if end else None

        if start_ts is None and end_ts is None:
            years = [pd.Timestamp.utcnow().year]
        else:
            y0 = int(start_ts.year) if start_ts is not None else 1900
            y1 = int(end_ts.year) if end_ts is not None else 2100
            years = list(range(y0, y1 + 1))

        table_name = "ohlcv_usd" if table_prefix == self.ohlcv_prefix else "returns_usd"

        all_keys: list[str] = []

        for asset_id in ids:
            for y in years:
                man = self.read_asset_year_manifest(table=table_name, asset_id=asset_id, year=int(y)) or {}
                parts = [k for k in (man.get("parts") or []) if isinstance(k, str) and k.endswith(".parquet")]

                if parts:
                    all_keys.extend(parts)
                    continue

                prefix = f"{table_prefix}/asset_id={asset_id}/year={int(y)}/"
                keys = [k for k in self._list_keys(prefix) if k.endswith(".parquet")]
                all_keys.extend(keys)

        if not all_keys:
            return pd.DataFrame()
        all_keys = list(dict.fromkeys(all_keys))

        dfs: list[pd.DataFrame] = []
        max_workers = 16

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

        clean_dfs: list[pd.DataFrame] = []

        for df in dfs:
            if df is None or df.empty:
                continue

            # Drop columns that are entirely NA in this individual part.
            # This avoids pandas FutureWarning when concatenating mixed old/new schemas.
            df2 = df.dropna(axis=1, how="all").copy()

            if df2.empty:
                continue

            clean_dfs.append(df2)

        if not clean_dfs:
            return pd.DataFrame(columns=columns or None)

        out = pd.concat(clean_dfs, ignore_index=True, sort=False)

        if columns is not None:
            for col in columns:
                if col not in out.columns:
                    out[col] = pd.NA
            out = out[columns].copy()

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