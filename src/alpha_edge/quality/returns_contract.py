"""AEA-87: explicit, audited return conventions and hard quality checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

LOG = 'ret_log_close_adjusted_usd'
SIMPLE = 'ret_close_adjusted_usd'
SIMPLE_ALIAS = 'ret_adj_close_usd'


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate both conventions per asset; do not bridge invalid observations."""
    required = {'asset_id', 'date', 'close_adjusted_usd'}
    if required - set(prices):
        raise ValueError(f'Missing: {sorted(required - set(prices))}')
    df = prices.copy()
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_convert(None).dt.normalize()
    df['close_adjusted_usd'] = pd.to_numeric(df['close_adjusted_usd'], errors='coerce')
    if df[['asset_id','date']].isna().any().any():
        raise ValueError('Null asset/date')
    if df.duplicated(['asset_id','date']).any():
        raise ValueError('Duplicate asset/date: resolve at ingest with source provenance')
    df = df.sort_values(['asset_id','date'], kind='stable')
    price = df['close_adjusted_usd'].where(lambda x: np.isfinite(x) & (x > 0))
    previous = price.groupby(df['asset_id']).shift(1)
    # Invalid rows stay missing; next valid row cannot silently bridge missing periods.
    valid = price.notna() & previous.notna()
    ratio = price.div(previous).where(valid)
    df[LOG] = np.log(ratio)
    df[SIMPLE] = ratio - 1.0
    df[SIMPLE_ALIAS] = df[SIMPLE]
    return df


def validate_simple_returns(wide: pd.DataFrame, *, max_abs_review: float = 0.50) -> dict:
    """Hard-fail mathematical impossibilities; quarantine extreme movements for review."""
    values = wide.to_numpy(dtype='float64', copy=False)
    finite = np.isfinite(values)
    invalid = finite & (values <= -1.0)
    extreme = finite & (np.abs(values) > max_abs_review)
    summary = {
        'return_type': 'simple',
        'n_rows': len(wide),
        'n_assets': len(wide.columns),
        'n_missing': int((~finite).sum()),
        'n_invalid_le_minus_one': int(invalid.sum()),
        'n_extreme_review': int(extreme.sum()),
        'critical': bool(invalid.any() or extreme.any()),
    }
    return summary


def assert_no_critical(wide: pd.DataFrame) -> dict:
    report = validate_simple_returns(wide)
    if report['critical']:
        raise ValueError('AEA-87 returns quality gate failed: ' + str(report))
    return report
