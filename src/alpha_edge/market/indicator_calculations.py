# indicator_calculations.py
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _first_existing(columns: set[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None


def _to_day_naive(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce", utc=True)
    return d.dt.tz_convert(None).dt.normalize()


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _make_adjusted_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build analytics-consistent OHLC prices.

    Ingest writes raw/execution and adjusted/analytics fields. For indicators, we prefer
    adjusted close. For OHLC indicators like ATR, we adjust raw open/high/low using:

        adjustment_factor = close_adjusted_usd / close_raw_usd

    This avoids split distortions in ATR when adjusted high/low fields are not explicitly stored.
    """
    cols = set(df.columns)

    close_adj_col = _first_existing(
        cols,
        [
            "close_adjusted_usd",
            "adj_close_usd",
            "adjusted_close_usd",
            "adj_close",
            "close",
        ],
    )
    close_raw_col = _first_existing(
        cols,
        [
            "close_raw_usd",
            "close_usd",
            "raw_close_usd",
            "close",
        ],
    )

    open_raw_col = _first_existing(cols, ["open_raw_usd", "open_usd", "open"])
    high_raw_col = _first_existing(cols, ["high_raw_usd", "high_usd", "high"])
    low_raw_col = _first_existing(cols, ["low_raw_usd", "low_usd", "low"])
    volume_col = _first_existing(cols, ["volume", "volume_raw", "volume_usd"])

    if close_adj_col is None:
        raise ValueError("Could not find adjusted/analytics close column.")
    if close_raw_col is None:
        close_raw_col = close_adj_col

    out = pd.DataFrame(index=df.index)
    out["close"] = _safe_num(df[close_adj_col])

    close_raw = _safe_num(df[close_raw_col])
    close_adj = _safe_num(df[close_adj_col])

    factor = close_adj / close_raw.replace(0.0, np.nan)
    factor = factor.replace([np.inf, -np.inf], np.nan)

    if open_raw_col is not None:
        out["open"] = _safe_num(df[open_raw_col]) * factor
    else:
        out["open"] = np.nan

    if high_raw_col is not None:
        out["high"] = _safe_num(df[high_raw_col]) * factor
    else:
        out["high"] = np.nan

    if low_raw_col is not None:
        out["low"] = _safe_num(df[low_raw_col]) * factor
    else:
        out["low"] = np.nan

    if volume_col is not None:
        out["volume"] = _safe_num(df[volume_col])
    else:
        out["volume"] = np.nan

    # Fallbacks when OHLC adjustment is not possible.
    out["open"] = out["open"].fillna(out["close"])
    out["high"] = out["high"].fillna(out[["open", "close"]].max(axis=1))
    out["low"] = out["low"].fillna(out[["open", "close"]].min(axis=1))

    return out


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=int(span), adjust=False, min_periods=int(span)).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def compute_market_indicators_for_asset(
    ohlcv: pd.DataFrame,
    returns: pd.DataFrame | None = None,
    *,
    annualization_days: int = 252,
) -> pd.DataFrame:
    """
    Compute historical point-in-time indicators for one asset.

    Required input:
      - date
      - asset_id
      - ticker if available
      - adjusted/analytics close column, preferably close_adjusted_usd

    Output:
      one row per asset/date with historical indicators.
    """
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame()

    if "date" not in ohlcv.columns:
        raise ValueError("Input OHLCV dataframe must contain 'date'.")

    if "asset_id" not in ohlcv.columns:
        raise ValueError("Input OHLCV dataframe must contain 'asset_id'.")

    raw = ohlcv.copy()

    raw["date"] = _to_day_naive(raw["date"])
    raw = raw.dropna(subset=["date", "asset_id"])
    raw["asset_id"] = raw["asset_id"].astype(str).str.strip()

    if "ticker" in raw.columns:
        raw["ticker"] = raw["ticker"].astype(str).str.upper().str.strip()
    else:
        raw["ticker"] = None

    raw = raw.sort_values(["asset_id", "date"], kind="stable")
    raw = raw.drop_duplicates(subset=["asset_id", "date"], keep="last")

    outputs: list[pd.DataFrame] = []

    for asset_id, g in raw.groupby("asset_id", sort=False):
        g = g.sort_values("date", kind="stable").reset_index(drop=True)

        px = _make_adjusted_ohlc(g)

        out = pd.DataFrame()
        out["date"] = g["date"]
        out["asset_id"] = str(asset_id)
        out["ticker"] = g["ticker"] if "ticker" in g.columns else None

        out["open"] = px["open"]
        out["high"] = px["high"]
        out["low"] = px["low"]
        out["close"] = px["close"]
        out["volume"] = px["volume"]

        close = out["close"]
        high = out["high"]
        low = out["low"]

        # Returns / momentum.
        # Canonical log returns from market/returns_usd/v1.
        # Do not recalculate 1d returns here.
        out["ret_log_1d"] = np.nan

        if returns is not None and not returns.empty:
            rr = returns.copy()
            rr["asset_id"] = rr["asset_id"].astype(str).str.strip()
            rr["date"] = pd.to_datetime(rr["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()

            if "ret_log_close_adjusted_usd" in rr.columns:
                ret_col = "ret_log_close_adjusted_usd"
            elif "ret_close_adjusted_usd" in rr.columns:
                ret_col = "ret_close_adjusted_usd"
            elif "ret_adj_close_usd" in rr.columns:
                ret_col = "ret_adj_close_usd"
            else:
                raise ValueError("Returns dataframe has no recognized return column.")

            rr[ret_col] = pd.to_numeric(rr[ret_col], errors="coerce")
            rr = rr[rr["asset_id"] == str(asset_id)].copy()
            rr = rr.dropna(subset=["date", ret_col])
            rr = rr.sort_values("date", kind="stable").drop_duplicates(subset=["date"], keep="last")

            out = out.merge(
                rr[["date", ret_col]].rename(columns={ret_col: "ret_log_1d"}),
                on="date",
                how="left",
                suffixes=("", "_from_returns"),
            )

            if "ret_log_1d_from_returns" in out.columns:
                out["ret_log_1d"] = out["ret_log_1d_from_returns"].combine_first(out["ret_log_1d"])
                out = out.drop(columns=["ret_log_1d_from_returns"])

        # Multi-day log returns from price path.
        out["ret_log_5d"] = np.log(close / close.shift(5).replace(0.0, np.nan))
        out["ret_log_20d"] = np.log(close / close.shift(20).replace(0.0, np.nan))
        out["ret_log_60d"] = np.log(close / close.shift(60).replace(0.0, np.nan))

        # True range / ATR.
        prev_close = close.shift(1)
        tr_components = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        out["true_range"] = tr_components.max(axis=1)

        out["atr_14"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["atr_20"] = out["true_range"].rolling(20, min_periods=20).mean()

        out["atr_pct_14"] = out["atr_14"] / close.replace(0.0, np.nan)
        out["atr_pct_20"] = out["atr_20"] / close.replace(0.0, np.nan)

        # Realized volatility.
        ret = out["ret_log_1d"]
        out["daily_vol_20"] = ret.rolling(20, min_periods=20).std()
        out["daily_vol_60"] = ret.rolling(60, min_periods=60).std()

        out["annualized_vol_20"] = out["daily_vol_20"] * np.sqrt(float(annualization_days))
        out["annualized_vol_60"] = out["daily_vol_60"] * np.sqrt(float(annualization_days))

        # Moving averages.
        out["sma_20"] = close.rolling(20, min_periods=20).mean()
        out["sma_50"] = close.rolling(50, min_periods=50).mean()
        out["sma_200"] = close.rolling(200, min_periods=200).mean()

        out["dist_sma_20"] = close / out["sma_20"].replace(0.0, np.nan) - 1.0
        out["dist_sma_50"] = close / out["sma_50"].replace(0.0, np.nan) - 1.0
        out["dist_sma_200"] = close / out["sma_200"].replace(0.0, np.nan) - 1.0

        # Rolling highs/lows and drawdowns.
        out["rolling_high_20"] = close.rolling(20, min_periods=20).max()
        out["rolling_high_60"] = close.rolling(60, min_periods=60).max()
        out["rolling_high_252"] = close.rolling(252, min_periods=252).max()

        out["rolling_low_20"] = close.rolling(20, min_periods=20).min()
        out["rolling_low_60"] = close.rolling(60, min_periods=60).min()
        out["rolling_low_252"] = close.rolling(252, min_periods=252).min()

        out["drawdown_20"] = close / out["rolling_high_20"].replace(0.0, np.nan) - 1.0
        out["drawdown_60"] = close / out["rolling_high_60"].replace(0.0, np.nan) - 1.0
        out["drawdown_252"] = close / out["rolling_high_252"].replace(0.0, np.nan) - 1.0

        # RSI.
        out["rsi_14"] = _rsi(close, window=14)

        # Bollinger Bands.
        bb_mid = close.rolling(20, min_periods=20).mean()
        bb_std = close.rolling(20, min_periods=20).std()
        out["bb_mid_20"] = bb_mid
        out["bb_upper_20"] = bb_mid + 2.0 * bb_std
        out["bb_lower_20"] = bb_mid - 2.0 * bb_std
        out["bb_width_20"] = (out["bb_upper_20"] - out["bb_lower_20"]) / bb_mid.replace(0.0, np.nan)
        out["bb_pct_b_20"] = (close - out["bb_lower_20"]) / (
            out["bb_upper_20"] - out["bb_lower_20"]
        ).replace(0.0, np.nan)

        # MACD.
        ema_12 = _ema(close, 12)
        ema_26 = _ema(close, 26)
        out["macd"] = ema_12 - ema_26
        out["macd_signal"] = _ema(out["macd"], 9)
        out["macd_hist"] = out["macd"] - out["macd_signal"]

        out["n_obs_asset"] = np.arange(1, len(out) + 1)
        out["updated_at_utc"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        out = out.replace([np.inf, -np.inf], np.nan)
        outputs.append(out)

    if not outputs:
        return pd.DataFrame()

    final = pd.concat(outputs, ignore_index=True)
    final = final.sort_values(["asset_id", "date"], kind="stable").reset_index(drop=True)

    return final


def latest_indicator_rows(indicators: pd.DataFrame) -> pd.DataFrame:
    """
    Latest valid indicator row per asset_id.

    This intentionally requires close to be present, but does not require all indicators
    to be populated because young assets may not have SMA_200 or vol_60 yet.
    """
    if indicators is None or indicators.empty:
        return pd.DataFrame()

    df = indicators.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "asset_id", "close"])
    df = df.sort_values(["asset_id", "date"], kind="stable")
    return df.groupby("asset_id", as_index=False, sort=False).tail(1).reset_index(drop=True)