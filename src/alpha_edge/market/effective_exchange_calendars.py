"""Effective-dated primary listing evidence and bounded session schedules."""

from __future__ import annotations

import pandas as pd
import pandas_market_calendars as mcal

TRANSFERS = {
    "EQH272fff4855629062": (
        "DDS",
        "NYSE",
        "2026-10-05",
        "https://www.sec.gov/Archives/edgar/data/28917/000110465926108945/tm2625597d1_ex99-1.htm",
    ),
    "EQH3787181fc4e164f6": (
        "IBOC",
        "NASDAQ",
        "2026-10-19",
        "https://www.sec.gov/Archives/edgar/data/315709/000110465926113899/tm2627039d1_8k.htm",
    ),
    "EQH6ebde0c7cae5efdb": (
        "CFR",
        "NYSE",
        "2026-11-12",
        "https://www.txse.com/press/texas-stock-exchange-celebrates-transfer-of-cullen-frost-s-primary-listing",
    ),
    "EQH7dbb1bdf9c987e2d": (
        "ET",
        "NYSE",
        "2026-10-05",
        "https://www.sec.gov/Archives/edgar/data/1276187/000127618726000044/et-20260903.htm",
    ),
}
TXSE_POLICY = {
    "calendar_id": "TXSE",
    "valid_from": "2026-10-05",
    "valid_through": "2026-11-27",
    "timezone": "America/New_York",
    "regular_close": "16:00",
    "closed": ["2026-11-26"],
    "early_closes": {"2026-11-27": "13:00"},
    "source_references": [
        "https://www.txse.com/alerts/txse-production-launch-and-market-activation",
        "https://www.txse.com/alerts/004f30f7-f4f4-4020-9918-c24e8d15b3ce",
    ],
}


def transfer_history(asset_id: str, symbol: str) -> list[dict] | None:
    entry = TRANSFERS.get(asset_id)
    if entry is None or entry[0] != symbol:
        return None
    _, previous, effective, reference = entry
    return [
        {"effective_from": None, "calendar_id": previous, "source_reference": reference},
        {"effective_from": effective, "calendar_id": "TXSE", "source_reference": reference},
    ]


def calendar_on(history: list[dict], day: str) -> str:
    return next(
        p["calendar_id"] for p in reversed(history) if p["effective_from"] is None or p["effective_from"] <= day
    )


def effective_schedule(history: list[dict], start, end, txse_policy: dict) -> pd.DataFrame:
    """Join non-overlapping schedules across transfer boundaries, not price series."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    for i, period in enumerate(history):
        first = max(start, pd.Timestamp(period["effective_from"])) if period["effective_from"] else start
        last = (
            min(end, pd.Timestamp(history[i + 1]["effective_from"]) - pd.Timedelta(days=1))
            if i + 1 < len(history)
            else end
        )
        if first > last:
            continue
        cal = period["calendar_id"]
        if cal != "TXSE":
            frames.append(mcal.get_calendar(cal).schedule(start_date=first, end_date=last)[["market_close"]])
            continue
        if first < pd.Timestamp(txse_policy["valid_from"]) or last > pd.Timestamp(txse_policy["valid_through"]):
            raise ValueError("TXSE schedule outside bounded evidence; extend official calendar policy")
        days = pd.date_range(first, last, freq="B")
        days = days[~days.strftime("%Y-%m-%d").isin(txse_policy["closed"])]
        closes = [
            pd.Timestamp(
                f"{d.date()} {txse_policy['early_closes'].get(str(d.date()), txse_policy['regular_close'])}",
                tz=txse_policy["timezone"],
            ).tz_convert("UTC")
            for d in days
        ]
        frames.append(pd.DataFrame({"market_close": pd.DatetimeIndex(closes, tz="UTC")}, index=days))
    return pd.concat(frames) if frames else pd.DataFrame(columns=["market_close"])
