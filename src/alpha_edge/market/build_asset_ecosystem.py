# src/alpha_edge/market/build_asset_ecosystem.py
from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

import boto3
import pandas as pd

from alpha_edge.core.schemas import RuntimeConfig
from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import (
    load_runtime_config,
    require_prod_confirmation,
    runtime_engine_key,
    runtime_warehouse_key,
)


WAREHOUSE_ROOT = "warehouse"
WAREHOUSE_VERSION = "v=1"

TABLE_NAME_MAP = {
    "regions": "dim_regions",
    "countries": "dim_countries",
    "markets": "dim_markets",
    "calendar_definitions": "dim_calendar_definitions",
    "calendar_dates": "dim_calendar_dates",
    "assets": "dim_assets",
}


# =============================================================================
# S3 helpers
# =============================================================================
def s3_client(cfg: RuntimeConfig):
    return boto3.client("s3", region_name=cfg.region)


def s3_put_bytes(
    s3,
    *,
    bucket: str,
    key: str,
    body: bytes,
    content_type: str,
) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
    )


def df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()

def write_parquet_table(
    *,
    s3,
    cfg: RuntimeConfig,
    table_name: str,
    df: pd.DataFrame,
    as_of: str,
    dry_run: bool,
) -> list[str]:
    warehouse_table = TABLE_NAME_MAP.get(table_name, table_name)

    current_key = runtime_warehouse_key(
        cfg,
        warehouse_table,
        "current",
        f"{warehouse_table}.parquet",
    )

    snapshot_key = runtime_warehouse_key(
        cfg,
        warehouse_table,
        f"dt={as_of}",
        f"{warehouse_table}.parquet",
    )

    print(f"[write] {warehouse_table}: rows={len(df)}")
    print(f"        current:  s3://{cfg.bucket}/{current_key}")
    print(f"        snapshot: s3://{cfg.bucket}/{snapshot_key}")

    if dry_run:
        return [current_key, snapshot_key]

    body = df_to_parquet_bytes(df)

    s3_put_bytes(
        s3,
        bucket=cfg.bucket,
        key=current_key,
        body=body,
        content_type="application/octet-stream",
    )
    s3_put_bytes(
        s3,
        bucket=cfg.bucket,
        key=snapshot_key,
        body=body,
        content_type="application/octet-stream",
    )

    return [current_key, snapshot_key]


def write_json(
    *,
    s3,
    cfg: RuntimeConfig,
    key: str,
    payload: dict[str, Any],
    dry_run: bool,
) -> None:
    print(f"[write] json: s3://{cfg.bucket}/{key}")

    if dry_run:
        return

    s3_put_bytes(
        s3,
        bucket=cfg.bucket,
        key=key,
        body=json.dumps(payload, indent=2, default=str).encode("utf-8"),
        content_type="application/json",
    )


# =============================================================================
# Seed tables
# =============================================================================
def utc_now() -> str:
    return pd.Timestamp.utcnow().isoformat()


def build_regions(now: str) -> pd.DataFrame:
    rows = [
        ("NORTH_AMERICA", "North America", "GEOGRAPHIC"),
        ("LATIN_AMERICA", "Latin America", "GEOGRAPHIC"),
        ("EUROPE", "Europe", "GEOGRAPHIC"),
        ("UNITED_KINGDOM", "United Kingdom", "GEOGRAPHIC"),
        ("ASIA_PACIFIC", "Asia Pacific", "GEOGRAPHIC"),
        ("JAPAN", "Japan", "GEOGRAPHIC"),
        ("CHINA_HK", "China / Hong Kong", "GEOGRAPHIC"),
        ("GLOBAL", "Global", "SYNTHETIC"),
        ("CRYPTO", "Crypto", "ASSET_SYSTEM"),
        ("FX", "Foreign Exchange", "ASSET_SYSTEM"),
        ("COMMODITIES", "Commodities", "ASSET_SYSTEM"),
        ("UNKNOWN", "Unknown", "FALLBACK"),
    ]

    return pd.DataFrame(
        [
            {
                "region_id": region_id,
                "region_name": region_name,
                "region_group": region_group,
                "is_active": True,
                "created_at_utc": now,
                "updated_at_utc": now,
            }
            for region_id, region_name, region_group in rows
        ]
    )


def build_countries(now: str) -> pd.DataFrame:
    rows = [
        ("US", "United States", "US", "USA", "NORTH_AMERICA", "USD", "America/New_York"),
        ("CA", "Canada", "CA", "CAN", "NORTH_AMERICA", "CAD", "America/Toronto"),
        ("GB", "United Kingdom", "GB", "GBR", "UNITED_KINGDOM", "GBP", "Europe/London"),
        ("DE", "Germany", "DE", "DEU", "EUROPE", "EUR", "Europe/Berlin"),
        ("IT", "Italy", "IT", "ITA", "EUROPE", "EUR", "Europe/Rome"),
        ("ES", "Spain", "ES", "ESP", "EUROPE", "EUR", "Europe/Madrid"),
        ("FR", "France", "FR", "FRA", "EUROPE", "EUR", "Europe/Paris"),
        ("NL", "Netherlands", "NL", "NLD", "EUROPE", "EUR", "Europe/Amsterdam"),
        ("CH", "Switzerland", "CH", "CHE", "EUROPE", "CHF", "Europe/Zurich"),
        ("JP", "Japan", "JP", "JPN", "JAPAN", "JPY", "Asia/Tokyo"),
        ("HK", "Hong Kong", "HK", "HKG", "CHINA_HK", "HKD", "Asia/Hong_Kong"),
        ("GLOBAL", "Global", "GLOBAL", "GLOBAL", "GLOBAL", "USD", "UTC"),
        ("UNKNOWN", "Unknown", "UNKNOWN", "UNKNOWN", "UNKNOWN", "USD", "UTC"),
    ]

    return pd.DataFrame(
        [
            {
                "country_id": country_id,
                "country_name": country_name,
                "iso2": iso2,
                "iso3": iso3,
                "region_id": region_id,
                "currency": currency,
                "timezone_default": timezone_default,
                "is_active": True,
                "created_at_utc": now,
                "updated_at_utc": now,
            }
            for (
                country_id,
                country_name,
                iso2,
                iso3,
                region_id,
                currency,
                timezone_default,
            ) in rows
        ]
    )


def build_calendar_definitions(now: str) -> pd.DataFrame:
    rows = [
        # calendar_id, name, provider, provider_code, timezone, weekend_rule, market_type
        ("US_MARKET", "US Market", "pandas_market_calendars", "NYSE", "America/New_York", "SAT_SUN", "EXCHANGE"),
        ("UK_MARKET", "UK Market", "pandas_market_calendars", "LSE", "Europe/London", "SAT_SUN", "EXCHANGE"),
        ("GERMANY_MARKET", "Germany Market", "pandas_market_calendars", "XETR", "Europe/Berlin", "SAT_SUN", "EXCHANGE"),
        ("ITALY_MARKET", "Italy Market", "pandas_market_calendars", "XMIL", "Europe/Rome", "SAT_SUN", "EXCHANGE"),
        ("SPAIN_MARKET", "Spain Market", "pandas_market_calendars", "XMAD", "Europe/Madrid", "SAT_SUN", "EXCHANGE"),
        ("FRANCE_MARKET", "France Market", "pandas_market_calendars", "XPAR", "Europe/Paris", "SAT_SUN", "EXCHANGE"),
        ("NETHERLANDS_MARKET", "Netherlands Market", "pandas_market_calendars", "XAMS", "Europe/Amsterdam", "SAT_SUN", "EXCHANGE"),
        ("SWITZERLAND_MARKET", "Switzerland Market", "pandas_market_calendars", "SIX", "Europe/Zurich", "SAT_SUN", "EXCHANGE"),
        ("JAPAN_MARKET", "Japan Market", "pandas_market_calendars", "JPX", "Asia/Tokyo", "SAT_SUN", "EXCHANGE"),
        ("HONG_KONG_MARKET", "Hong Kong Market", "pandas_market_calendars", "HKEX", "Asia/Hong_Kong", "SAT_SUN", "EXCHANGE"),
        ("CRYPTO_24_7", "Crypto 24/7", "internal", "CRYPTO_24_7", "UTC", "NONE", "SYNTHETIC"),
        ("FX_WEEKDAYS", "FX Weekdays", "internal", "FX_WEEKDAYS", "UTC", "SAT_SUN", "SYNTHETIC"),
        ("COMMODITIES_US", "US Commodities", "pandas_market_calendars", "NYSE", "America/New_York", "SAT_SUN", "EXCHANGE"),
        ("UNKNOWN", "Unknown Calendar", "internal", "UNKNOWN", "UTC", "SAT_SUN", "FALLBACK"),
    ]

    return pd.DataFrame(
        [
            {
                "calendar_id": calendar_id,
                "calendar_name": calendar_name,
                "calendar_provider": provider,
                "provider_calendar_code": provider_code,
                "timezone": timezone,
                "weekend_rule": weekend_rule,
                "market_type": market_type,
                "is_active": True,
                "created_at_utc": now,
                "updated_at_utc": now,
            }
            for (
                calendar_id,
                calendar_name,
                provider,
                provider_code,
                timezone,
                weekend_rule,
                market_type,
            ) in rows
        ]
    )


def build_markets(now: str) -> pd.DataFrame:
    rows = [
        # market_id, market_name, country_id, region_id, mic, exchange_code, timezone, calendar_id, market_type
        ("US_NYSE", "New York Stock Exchange", "US", "NORTH_AMERICA", "XNYS", "NYSE", "America/New_York", "US_MARKET", "EXCHANGE"),
        ("US_NASDAQ", "Nasdaq", "US", "NORTH_AMERICA", "XNAS", "NASDAQ", "America/New_York", "US_MARKET", "EXCHANGE"),
        ("US_NYSE_ARCA", "NYSE Arca", "US", "NORTH_AMERICA", "ARCX", "NYSE_ARCA", "America/New_York", "US_MARKET", "EXCHANGE"),
        ("UK_LSE", "London Stock Exchange", "GB", "UNITED_KINGDOM", "XLON", "LSE", "Europe/London", "UK_MARKET", "EXCHANGE"),
        ("DE_XETRA", "Xetra", "DE", "EUROPE", "XETR", "XETRA", "Europe/Berlin", "GERMANY_MARKET", "EXCHANGE"),
        ("IT_BORSA_ITALIANA", "Borsa Italiana", "IT", "EUROPE", "XMIL", "BORSA_ITALIANA", "Europe/Rome", "ITALY_MARKET", "EXCHANGE"),
        ("ES_BME", "BME Spanish Exchanges", "ES", "EUROPE", "XMAD", "BME", "Europe/Madrid", "SPAIN_MARKET", "EXCHANGE"),
        ("FR_EURONEXT_PARIS", "Euronext Paris", "FR", "EUROPE", "XPAR", "EURONEXT_PARIS", "Europe/Paris", "FRANCE_MARKET", "EXCHANGE"),
        ("NL_EURONEXT_AMSTERDAM", "Euronext Amsterdam", "NL", "EUROPE", "XAMS", "EURONEXT_AMSTERDAM", "Europe/Amsterdam", "NETHERLANDS_MARKET", "EXCHANGE"),
        ("CH_SIX", "SIX Swiss Exchange", "CH", "EUROPE", "XSWX", "SIX", "Europe/Zurich", "SWITZERLAND_MARKET", "EXCHANGE"),
        ("JP_TSE", "Tokyo Stock Exchange", "JP", "JAPAN", "XTKS", "TSE", "Asia/Tokyo", "JAPAN_MARKET", "EXCHANGE"),
        ("HK_HKEX", "Hong Kong Exchange", "HK", "CHINA_HK", "XHKG", "HKEX", "Asia/Hong_Kong", "HONG_KONG_MARKET", "EXCHANGE"),
        ("CRYPTO_GLOBAL", "Crypto Global", "GLOBAL", "CRYPTO", None, "CRYPTO", "UTC", "CRYPTO_24_7", "SYNTHETIC"),
        ("FX_GLOBAL", "Foreign Exchange Global", "GLOBAL", "FX", None, "FX", "UTC", "FX_WEEKDAYS", "SYNTHETIC"),
        ("UNKNOWN", "Unknown Market", "UNKNOWN", "UNKNOWN", None, "UNKNOWN", "UTC", "UNKNOWN", "FALLBACK"),
    ]

    return pd.DataFrame(
        [
            {
                "market_id": market_id,
                "market_name": market_name,
                "country_id": country_id,
                "region_id": region_id,
                "mic": mic,
                "exchange_code": exchange_code,
                "timezone": timezone,
                "calendar_id": calendar_id,
                "market_type": market_type,
                "is_active": True,
                "created_at_utc": now,
                "updated_at_utc": now,
            }
            for (
                market_id,
                market_name,
                country_id,
                region_id,
                mic,
                exchange_code,
                timezone,
                calendar_id,
                market_type,
            ) in rows
        ]
    )


# =============================================================================
# Calendar dates
# =============================================================================
def _date_range(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start).date(), pd.Timestamp(end).date(), freq="D")


def _build_internal_calendar_dates(
    *,
    calendar_id: str,
    start: str,
    end: str,
    timezone: str,
    weekend_rule: str,
    now: str,
) -> pd.DataFrame:
    dates = _date_range(start, end)
    rows: list[dict[str, Any]] = []

    for d in dates:
        weekday = int(d.weekday())

        if weekend_rule == "NONE":
            is_trading_day = True
            reason = None
        elif weekend_rule == "SAT_SUN":
            is_trading_day = weekday not in (5, 6)
            reason = None if is_trading_day else "weekend"
        else:
            is_trading_day = weekday not in (5, 6)
            reason = None if is_trading_day else "weekend"

        rows.append(
            {
                "calendar_id": calendar_id,
                "date": d.date().isoformat(),
                "is_trading_day": bool(is_trading_day),
                "session_type": "regular" if is_trading_day else "closed",
                "open_time": "00:00" if is_trading_day else None,
                "close_time": "23:59" if is_trading_day else None,
                "timezone": timezone,
                "reason": reason,
                "source": "internal",
                "created_at_utc": now,
            }
        )

    return pd.DataFrame(rows)


def _build_pmc_calendar_dates(
    *,
    calendar_id: str,
    provider_code: str,
    start: str,
    end: str,
    timezone: str,
    now: str,
) -> pd.DataFrame:
    try:
        import pandas_market_calendars as mcal
    except ImportError as exc:
        raise RuntimeError(
            "pandas_market_calendars is required to generate exchange calendars. "
            "Install with: poetry add pandas-market-calendars"
        ) from exc

    dates = _date_range(start, end)
    all_days = pd.DataFrame({"date": [d.date().isoformat() for d in dates]})

    try:
        cal = mcal.get_calendar(provider_code)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load pandas_market_calendars calendar provider_code={provider_code!r} "
            f"for calendar_id={calendar_id!r}"
        ) from exc

    schedule = cal.schedule(start_date=start, end_date=end).reset_index()

    # pandas_market_calendars usually returns 'market_open' and 'market_close',
    # with index/calendar date in column 'index' or 'date' depending on version.
    date_col = "index" if "index" in schedule.columns else schedule.columns[0]

    schedule["date"] = pd.to_datetime(schedule[date_col]).dt.date.astype(str)

    open_map = {}
    close_map = {}

    if "market_open" in schedule.columns:
        open_map = dict(
            zip(
                schedule["date"],
                pd.to_datetime(schedule["market_open"]).dt.strftime("%H:%M"),
            )
        )

    if "market_close" in schedule.columns:
        close_map = dict(
            zip(
                schedule["date"],
                pd.to_datetime(schedule["market_close"]).dt.strftime("%H:%M"),
            )
        )

    trading_days = set(schedule["date"].astype(str).tolist())

    rows = []
    for d in all_days["date"].astype(str):
        is_trading_day = d in trading_days
        weekday = pd.Timestamp(d).weekday()

        if is_trading_day:
            reason = None
        elif weekday in (5, 6):
            reason = "weekend"
        else:
            reason = "holiday_or_closed"

        rows.append(
            {
                "calendar_id": calendar_id,
                "date": d,
                "is_trading_day": bool(is_trading_day),
                "session_type": "regular" if is_trading_day else "closed",
                "open_time": open_map.get(d),
                "close_time": close_map.get(d),
                "timezone": timezone,
                "reason": reason,
                "source": f"pandas_market_calendars:{provider_code}",
                "created_at_utc": now,
            }
        )

    return pd.DataFrame(rows)


def build_calendar_dates(
    *,
    calendar_definitions: pd.DataFrame,
    start: str,
    end: str,
    now: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for _, row in calendar_definitions.iterrows():
        calendar_id = str(row["calendar_id"])
        provider = str(row["calendar_provider"])
        provider_code = str(row["provider_calendar_code"])
        timezone = str(row["timezone"])
        weekend_rule = str(row["weekend_rule"])

        print(f"[calendar] generating {calendar_id} provider={provider}:{provider_code}")

        if provider == "internal":
            df = _build_internal_calendar_dates(
                calendar_id=calendar_id,
                start=start,
                end=end,
                timezone=timezone,
                weekend_rule=weekend_rule,
                now=now,
            )
        else:
            df = _build_pmc_calendar_dates(
                calendar_id=calendar_id,
                provider_code=provider_code,
                start=start,
                end=end,
                timezone=timezone,
                now=now,
            )

        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["calendar_id", "date"]).reset_index(drop=True)
    return out


# =============================================================================
# Asset enrichment
# =============================================================================
def normalize_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


def upper_or_none(x: Any) -> Optional[str]:
    s = normalize_str(x)
    return s.upper() if s else None


def infer_market_from_ticker(
    *,
    ticker: str,
    asset_class: Optional[str],
) -> str:
    t = ticker.upper().strip()
    ac = (asset_class or "").upper().strip()

    if ac in {"CRYPTO", "CRYPTOCURRENCY"} or "-USD" in t and t.split("-")[0] in {"BTC", "ETH", "SOL", "ADA", "SUI"}:
        return "CRYPTO_GLOBAL"

    if ac in {"FX", "FOREX"} or t.endswith("=X"):
        return "FX_GLOBAL"

    if t.endswith(".MI"):
        return "IT_BORSA_ITALIANA"

    if t.endswith(".DE"):
        return "DE_XETRA"

    if t.endswith(".MC"):
        return "ES_BME"

    if t.endswith(".PA"):
        return "FR_EURONEXT_PARIS"

    if t.endswith(".AS"):
        return "NL_EURONEXT_AMSTERDAM"

    if t.endswith(".SW"):
        return "CH_SIX"

    if t.endswith(".L"):
        return "UK_LSE"

    if t.endswith(".T"):
        return "JP_TSE"

    if t.endswith(".HK"):
        return "HK_HKEX"

    # Default for US tickers / ETFs when no suffix exists.
    return "US_NYSE_ARCA"


def pick_col(row: pd.Series, *cols: str) -> Optional[str]:
    for c in cols:
        if c in row.index:
            v = normalize_str(row.get(c))
            if v:
                return v
    return None


def build_assets(
    *,
    universe_path: str,
    markets: pd.DataFrame,
    now: str,
    manual_additions_path: Optional[str] = None,
    overrides_path: Optional[str] = None,
    excluded_path: Optional[str] = None,
    force_refresh_path: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = pd.read_csv(universe_path)

    market_meta = markets.set_index("market_id").to_dict("index")

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for i, row in universe.iterrows():
        asset_id = pick_col(row, "asset_id")
        ticker = pick_col(row, "ticker", "broker_ticker", "yahoo_ticker")
        broker_ticker = pick_col(row, "broker_ticker", "ticker", "yahoo_ticker")
        yahoo_ticker = pick_col(row, "yahoo_ticker", "ticker", "broker_ticker")
        asset_name = pick_col(row, "asset_name", "name", "description")
        asset_class = upper_or_none(pick_col(row, "asset_class", "class", "type"))
        currency = upper_or_none(pick_col(row, "currency", "quote_currency")) or "USD"

        is_active_raw = pick_col(row, "is_active", "active")
        if is_active_raw is None:
            is_active = True
        else:
            is_active = str(is_active_raw).strip().lower() not in {"false", "0", "no", "n"}

        if not asset_id:
            asset_id = f"UNKNOWN:{i}"

        if not ticker:
            issues.append(
                {
                    "row_idx": i,
                    "asset_id": asset_id,
                    "issue": "missing_ticker",
                }
            )
            ticker = str(asset_id)

        explicit_market_id = pick_col(row, "market_id")
        market_id = explicit_market_id or infer_market_from_ticker(
            ticker=str(yahoo_ticker or broker_ticker or ticker),
            asset_class=asset_class,
        )

        if market_id not in market_meta:
            issues.append(
                {
                    "row_idx": i,
                    "asset_id": asset_id,
                    "ticker": ticker,
                    "issue": "unknown_market_id",
                    "market_id": market_id,
                }
            )
            market_id = "UNKNOWN"

        mm = market_meta[market_id]

        out = {
            "asset_id": str(asset_id).strip(),
            "ticker": str(ticker).upper().strip(),
            "broker_ticker": str(broker_ticker or ticker).upper().strip(),
            "yahoo_ticker": str(yahoo_ticker or ticker).upper().strip(),
            "asset_name": asset_name,
            "asset_class": asset_class or "UNKNOWN",
            "market_id": market_id,
            "calendar_id": mm["calendar_id"],
            "country_id": mm["country_id"],
            "region_id": mm["region_id"],
            "currency": currency,
            "data_source": pick_col(row, "data_source", "source") or "yahoo",
            "is_active": bool(is_active),
            "start_date": pick_col(row, "start_date"),
            "end_date": pick_col(row, "end_date"),
            "created_at_utc": now,
            "updated_at_utc": now,
        }

        rows.append(out)

    assets = pd.DataFrame(rows)
    issues_df = pd.DataFrame(issues)

    return assets, issues_df


# =============================================================================
# Validation
# =============================================================================
@dataclass
class ValidationSummary:
    regions_rows: int
    countries_rows: int
    markets_rows: int
    calendar_definitions_rows: int
    calendar_dates_rows: int
    assets_rows: int
    assets_active_rows: int
    asset_issues_rows: int
    missing_calendar_id: int
    missing_market_id: int
    missing_country_id: int
    missing_region_id: int
    unknown_calendar_id: int
    unknown_market_id: int
    calendar_start: str
    calendar_end: str


def build_validation_report(
    *,
    regions: pd.DataFrame,
    countries: pd.DataFrame,
    markets: pd.DataFrame,
    calendar_definitions: pd.DataFrame,
    calendar_dates: pd.DataFrame,
    assets: pd.DataFrame,
    asset_issues: pd.DataFrame,
    start: str,
    end: str,
) -> dict[str, Any]:
    calendar_ids = set(calendar_definitions["calendar_id"].astype(str))
    market_ids = set(markets["market_id"].astype(str))

    summary = ValidationSummary(
        regions_rows=len(regions),
        countries_rows=len(countries),
        markets_rows=len(markets),
        calendar_definitions_rows=len(calendar_definitions),
        calendar_dates_rows=len(calendar_dates),
        assets_rows=len(assets),
        assets_active_rows=int(assets["is_active"].sum()) if "is_active" in assets.columns else 0,
        asset_issues_rows=len(asset_issues),
        missing_calendar_id=int(assets["calendar_id"].isna().sum()) if "calendar_id" in assets.columns else len(assets),
        missing_market_id=int(assets["market_id"].isna().sum()) if "market_id" in assets.columns else len(assets),
        missing_country_id=int(assets["country_id"].isna().sum()) if "country_id" in assets.columns else len(assets),
        missing_region_id=int(assets["region_id"].isna().sum()) if "region_id" in assets.columns else len(assets),
        unknown_calendar_id=int((~assets["calendar_id"].astype(str).isin(calendar_ids)).sum()) if "calendar_id" in assets.columns else len(assets),
        unknown_market_id=int((~assets["market_id"].astype(str).isin(market_ids)).sum()) if "market_id" in assets.columns else len(assets),
        calendar_start=start,
        calendar_end=end,
    )

    return {
        "summary": asdict(summary),
        "asset_issues": asset_issues.to_dict("records"),
    }


# =============================================================================
# Main build
# =============================================================================
def build_asset_ecosystem(
    *,
    cfg: RuntimeConfig,
    universe_path: str,
    start: str,
    end: str,
    as_of: Optional[str],
    dry_run: bool,
    local_out_dir: Optional[str],
    run_id: str,
    input_args: dict[str, Any],
) -> None:
    now = utc_now()
    as_of_norm = pd.Timestamp(as_of).date().isoformat() if as_of else pd.Timestamp.utcnow().date().isoformat()

    print("\n=== BUILD ASSET ECOSYSTEM ===")
    print(f"env:           {cfg.env}")
    print(f"bucket:        {cfg.bucket}")
    print(f"root:          {cfg.engine_root}")
    print(f"universe_path: {universe_path}")
    print(f"calendar:      {start} -> {end}")
    print(f"as_of:         {as_of_norm}")
    print(f"dry_run:       {dry_run}")
    print(f"run_id:        {run_id}")
    print("")

    regions = build_regions(now)
    countries = build_countries(now)
    calendar_definitions = build_calendar_definitions(now)
    markets = build_markets(now)
    calendar_dates = build_calendar_dates(
        calendar_definitions=calendar_definitions,
        start=start,
        end=end,
        now=now,
    )
    assets, asset_issues = build_assets(
        universe_path=universe_path,
        markets=markets,
        now=now,
    )

    report = build_validation_report(
        regions=regions,
        countries=countries,
        markets=markets,
        calendar_definitions=calendar_definitions,
        calendar_dates=calendar_dates,
        assets=assets,
        asset_issues=asset_issues,
        start=start,
        end=end,
    )

    print("\n=== VALIDATION SUMMARY ===")
    for k, v in report["summary"].items():
        print(f"{k}: {v}")

    if len(asset_issues) > 0:
        print("\n=== ASSET ISSUES SAMPLE ===")
        print(asset_issues.head(25).to_string(index=False))

    tables = {
        "regions": regions,
        "countries": countries,
        "markets": markets,
        "calendar_definitions": calendar_definitions,
        "calendar_dates": calendar_dates,
        "assets": assets,
    }

    if local_out_dir:
        out = Path(local_out_dir)
        out.mkdir(parents=True, exist_ok=True)

        print(f"\n[local] writing tables to {out}")

        for name, df in tables.items():
            path = out / f"{name}.parquet"
            df.to_parquet(path, index=False)
            print(f"[local] {path}")

        report_path = out / "validation_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"[local] {report_path}")

    s3 = s3_client(cfg)

    print("\n=== S3 OUTPUTS ===")
    written_keys: list[str] = []

    for table_name, df in tables.items():
        written_keys.extend(
            write_parquet_table(
                s3=s3,
                cfg=cfg,
                table_name=table_name,
                df=df,
                as_of=as_of_norm,
                dry_run=dry_run,
            )
        )

    validation_key = runtime_warehouse_key(
        cfg,
        "validation",
        "asset_ecosystem",
        f"dt={as_of_norm}",
        "validation_report.json",
    )

    write_json(
        s3=s3,
        cfg=cfg,
        key=validation_key,
        payload=report,
        dry_run=dry_run,
    )

    run_summary_key = runtime_warehouse_key(
        cfg,
        "runs",
        "build_asset_ecosystem",
        f"dt={as_of_norm}",
        "summary.json",
    )

    run_summary = {
        "run_id": run_id,
        "status": "dry_run" if dry_run else "success",
        "as_of": as_of_norm,
        "created_at_utc": now,
        "universe_path": universe_path,
        "calendar_start": start,
        "calendar_end": end,
        "tables": {name: {"rows": len(df)} for name, df in tables.items()},
        "validation_report_key": validation_key,
        "written_keys": written_keys,
    }

    write_json(
        s3=s3,
        cfg=cfg,
        key=run_summary_key,
        payload=run_summary,
        dry_run=dry_run,
    )


    audit_event = build_audit_event(
        cfg=cfg,
        event_type="create_or_update",
        entity_type="warehouse_asset_ecosystem",
        entity_id=f"asset_ecosystem:{as_of_norm}",
        as_of=as_of_norm,
        source_script="build_asset_ecosystem.py",
        source_mode="dry_run" if dry_run else "prod_write",
        status="dry_run" if dry_run else "success",
        run_id=run_id,
        reason="Build canonical asset ecosystem warehouse dimension tables for market calendar eligibility.",
        input_args=input_args,
        output_keys=[*written_keys, validation_key, run_summary_key],
        metadata={
            "tables": {name: {"rows": len(df)} for name, df in tables.items()},
            "validation_summary": report["summary"],
            "calendar_start": start,
            "calendar_end": end,
            "warehouse_version": WAREHOUSE_VERSION,
        },
    )

    audit_key = write_audit_event(
        cfg=cfg,
        event=audit_event,
        dry_run=dry_run,
    )

    print(f"[audit] {audit_key}")

    print("\n[done] build_asset_ecosystem completed.")
    print("")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build Alpha Edge asset ecosystem reference tables."
    )

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument(
        "--universe-path",
        required=True,
        help="Path to local universe.csv used to seed/enrich the assets reference table.",
    )
    ap.add_argument(
        "--start",
        default="2010-01-01",
        help="Calendar start date YYYY-MM-DD.",
    )
    ap.add_argument(
        "--end",
        default="2035-12-31",
        help="Calendar end date YYYY-MM-DD.",
    )
    ap.add_argument(
        "--as-of",
        default=None,
        help="Snapshot date YYYY-MM-DD. Defaults to current UTC date.",
    )
    ap.add_argument(
        "--local-out-dir",
        default=None,
        help="Optional local output directory to write parquet/json files for inspection.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_runtime_config(args.env)

    if not bool(args.dry_run):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    input_args = vars(args)

    with capture_script_run(
        cfg=cfg,
        script_name="build_asset_ecosystem.py",
        input_args=input_args,
        dry_run=bool(args.dry_run),
    ) as run_id:
        build_asset_ecosystem(
            cfg=cfg,
            universe_path=str(args.universe_path),
            start=str(args.start),
            end=str(args.end),
            as_of=(str(args.as_of) if args.as_of else None),
            dry_run=bool(args.dry_run),
            local_out_dir=(str(args.local_out_dir) if args.local_out_dir else None),
            run_id=run_id,
            input_args=input_args,
        )


if __name__ == "__main__":
    main()