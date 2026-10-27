"""Read-only AEA-87 price/return diagnosis on extracted OHLCV history.

Inputs: consolidated price history CSV/Parquet + returns_quality_cells.csv.
Outputs: anomaly classification candidates and per-date summaries.
No repairs or production S3 writes.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.lower().endswith(('.parquet','.pq')) else pd.read_csv(path)


def diagnose(prices: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    p=prices.copy()
    p['date']=pd.to_datetime(p['date'],errors='coerce',utc=True).dt.tz_convert(None).dt.normalize()
    p['asset_id']=p['asset_id'].astype(str).str.strip()
    if p.duplicated(['asset_id','date']).any():
        raise ValueError('duplicate asset/date prices: resolve source provenance before diagnosis')
    p=p.sort_values(['asset_id','date']).reset_index(drop=True)
    value_col=next((c for c in ['close_adjusted_usd','adj_close_usd'] if c in p), None)
    if not value_col:
        raise ValueError('No close_adjusted_usd or adj_close_usd found')
    price=pd.to_numeric(p[value_col],errors='coerce')
    price=price.where(np.isfinite(price)&(price>0))
    p['price']=price
    prev=price.groupby(p['asset_id']).shift(1)
    next_=price.groupby(p['asset_id']).shift(-1)
    p['prev_price']=prev
    p['next_price']=next_
    p['price_simple_return']=price.div(prev)-1
    p['price_log_return']=np.log(price.div(prev))
    p['reversal_simple_return']=next_.div(price)-1
    if {'close','adj_close'}.issubset(p.columns):
        cl=pd.to_numeric(p['close'],errors='coerce'); adj=pd.to_numeric(p['adj_close'],errors='coerce')
        factor=adj.div(cl.where(cl>0))
        p['adjustment_factor']=factor
        p['factor_change']=factor.div(factor.groupby(p['asset_id']).shift(1))
    else:
        p['factor_change']=np.nan
    if 'fx_to_usd' in p:
        fx=pd.to_numeric(p['fx_to_usd'],errors='coerce')
        p['fx_ratio']=fx.div(fx.groupby(p['asset_id']).shift(1))
    else:
        p['fx_ratio']=np.nan
    a=anomalies.copy()
    a['date']=pd.to_datetime(a['date'],errors='coerce',utc=True).dt.tz_convert(None).dt.normalize()
    a['asset_id']=a['asset_id'].astype(str).str.strip()
    # Identify assets included in the historical price extraction.
    extracted_assets = set(p['asset_id'].unique())

    a['asset_extracted'] = a['asset_id'].isin(extracted_assets)
    retcol='return' if 'return' in a else 'return_value'
    if retcol not in a: raise ValueError('No return column in bad-cells report')
    cols=['asset_id','date','price','prev_price','next_price','price_simple_return','price_log_return','reversal_simple_return','factor_change','fx_ratio']
    for c in ['ticker','yahoo_ticker','close','adj_close','currency','source_s3_key','volume']:
        if c in p: cols.append(c)
    result=a.merge(p[cols],on=['asset_id','date'],how='left',validate='many_to_one')
    value=pd.to_numeric(result[retcol],errors='coerce')
    result['matches_simple']=np.isclose(value,result['price_simple_return'],atol=1e-6,rtol=1e-5)
    result['matches_log']=np.isclose(value,result['price_log_return'],atol=1e-6,rtol=1e-5)
    result['near_split_factor']=result['factor_change'].apply(lambda x: any(np.isfinite(x) and np.isclose(x,f,rtol=.02) for f in [2,3,4,5,10,.5,1/3,.25,.2,.1]))
    result['large_fx_change']=result['fx_ratio'].apply(lambda x: bool(np.isfinite(x) and (x>=1.5 or x<=2/3)))
    r1=result['price_simple_return'];r2=result['reversal_simple_return']
    result['spike_reversal']=((r1>0.5)&(r2<-.33))|((r1<-.33)&(r2>0.5))
    # AEA-87: distinguish unextracted assets from genuinely missing prices.

    result['diagnosis_candidate'] = np.select(
        [
            ~result['asset_extracted'],
            result['asset_extracted'] & result['price'].isna(),
            result['matches_log'] & ~result['matches_simple'],
            result['near_split_factor'],
            result['large_fx_change'],
            result['spike_reversal'],
            result['matches_simple'],
        ],
        [
            'ASSET_NOT_EXTRACTED',
            'SOURCE_PRICE_MISSING_FOR_EXTRACTED_ASSET',
            'RETURN_CONVENTION_LOG_IN_SIMPLE_CACHE',
            'CHECK_SPLIT_OR_ADJUSTMENT_FACTOR',
            'CHECK_FX_CONVERSION',
            'PRICE_SPIKE_AND_REVERSAL_CHECK_YAHOO',
            'SOURCE_PRICE_MOVE_MATCHES_RETURN',
        ],
        default='RETURNS_PIPELINE_MISMATCH_OR_INCOMPLETE_HISTORY'
    )
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prices',required=True)
    ap.add_argument('--bad-cells',required=True)
    ap.add_argument('--out-dir',default='data/quality/returns/triage')
    args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    result=diagnose(load(args.prices),load(args.bad_cells))
    result.to_csv(out/'anomaly_price_join.csv',index=False)
    result.groupby(['date','diagnosis_candidate']).size().rename('n').reset_index().to_csv(out/'diagnosis_by_date.csv',index=False)
    result['diagnosis_candidate'].value_counts().rename_axis('diagnosis_candidate').reset_index(name='n').to_csv(out/'diagnosis_summary.csv',index=False)
    print(result['diagnosis_candidate'].value_counts().to_string())

if __name__=='__main__':main()
