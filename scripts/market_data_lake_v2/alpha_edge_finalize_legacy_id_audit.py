#!/usr/bin/env python3
"""Alpha Edge legacy/current ID audit. Read-only S3; local evidence only.

Inputs: prior 879-pair JSONL audit. Current asset IDs remain canonical.
For the seven non-nested date histories, compare active-manifest Parquet observations.
Never import, overwrite, delete, or repoint production objects.

Usage:
  poetry run python alpha_edge_finalize_legacy_id_audit.py \
    --pairs legacy_coverage_pairs.jsonl --bucket alpha-edge-algo \
    --end 2026-11-10 --output-dir legacy_identity_final_audit

Requires pandas, pyarrow, boto3 (already part of the Alpha Edge project stack).
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config

MANIFEST_ROOT = 'market/manifests/v1/ohlcv_usd'
PRICE_GROUPS = {
    'raw_close_usd': ('close_raw_usd', 'close_usd'),
    'adjusted_close_usd': ('close_adjusted_usd', 'adj_close_usd'),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')


def get_json(s3: Any, bucket: str, key: str) -> dict:
    return json.loads(s3.get_object(Bucket=bucket, Key=key)['Body'].read())


def active_history(s3: Any, bucket: str, asset_id: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Read only the Parquet parts referenced by the active annual OHLCV manifest.

    Important: listing raw asset/year prefixes includes superseded data; do not do that.
    """
    prefix = f'{MANIFEST_ROOT}/asset_id={asset_id}/'
    keys: list[str] = []
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(x['Key'] for x in page.get('Contents', []) if x['Key'].endswith('/manifest.json'))
    if not keys:
        raise ValueError(f'No OHLCV manifests: {asset_id}')
    dates: set[str] = set()
    parts: list[str] = []
    for key in sorted(keys):
        m = get_json(s3, bucket, key)
        if m.get('asset_id') != asset_id:
            raise ValueError(f'Manifest ID mismatch: {key}')
        for d in m.get('dates', []):
            if not isinstance(d, str) or len(d) != 10:
                raise ValueError(f'Invalid manifest date: {key} {d}')
            date.fromisoformat(d)
            if d < end:
                dates.add(d)
        for p in m.get('parts', []):
            if not p.startswith('market/') or p in parts:
                if p in parts:
                    continue
                raise ValueError(f'Unexpected part path: {p}')
            parts.append(p)
    if not parts:
        raise ValueError(f'Manifest has no parts: {asset_id}')
    chunks = []
    for part in parts:
        raw = s3.get_object(Bucket=bucket, Key=part)['Body'].read()
        chunks.append(pd.read_parquet(io.BytesIO(raw), engine='pyarrow'))
    frame = pd.concat(chunks, ignore_index=True, sort=False)
    if 'date' not in frame or 'asset_id' not in frame:
        raise ValueError(f'OHLCV date/asset_id columns missing: {asset_id}')
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce', utc=True).dt.strftime('%Y-%m-%d')
    frame = frame.loc[frame['date'].lt(end)].copy()
    if frame['date'].isna().any():
        raise ValueError(f'Invalid observation dates: {asset_id}')
    if frame['asset_id'].astype(str).ne(asset_id).any():
        raise ValueError(f'Unexpected asset ID in Parquet: {asset_id}')
    duplicates = int(frame.duplicated(subset=['date']).sum())
    observation_dates = set(frame['date'])
    meta = {
        'asset_id': asset_id,
        'manifest_years': len(keys), 'active_parts': len(parts),
        'manifest_dates': len(dates), 'observed_rows': len(frame),
        'duplicate_observation_dates': duplicates,
        'manifest_dates_without_rows': sorted(dates - observation_dates),
        'rows_without_manifest_dates': sorted(observation_dates - dates),
        'manifest_first_date': min(dates) if dates else None,
        'manifest_last_date': max(dates) if dates else None,
        'observed_first_date': min(observation_dates) if observation_dates else None,
        'observed_last_date': max(observation_dates) if observation_dates else None,
    }
    return frame, meta


def numerical_column(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    for n in names:
        if n in frame.columns:
            val = pd.to_numeric(frame[n], errors='coerce')
            if val.notna().any():
                return val
    return None


def compare_prices(old: pd.DataFrame, current: pd.DataFrame, tolerance: float) -> dict:
    """Compare raw/adjusted USD prices on same dates, and log returns on the common date grid."""
    a = old.set_index('date').sort_index()
    b = current.set_index('date').sort_index()
    dates = a.index.intersection(b.index)
    report: dict[str, Any] = {'overlap_observation_dates': len(dates), 'price_fields': {}}
    for label, names in PRICE_GROUPS.items():
        xa = numerical_column(a, names)
        xb = numerical_column(b, names)
        if xa is None or xb is None:
            report['price_fields'][label] = {'status': 'column_missing'}
            continue
        x = xa.reindex(dates).to_numpy(dtype='float64')
        y = xb.reindex(dates).to_numpy(dtype='float64')
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        n = int(valid.sum())
        if n < 2:
            report['price_fields'][label] = {'status': 'insufficient_valid_overlap', 'valid_overlap_rows': n}
            continue
        xx = x[valid]; yy = y[valid]
        rel = np.abs(xx - yy) / np.maximum(np.maximum(np.abs(xx), np.abs(yy)), 1e-12)
        # Important: compare aligned log returns on consecutive shared sessions, not
        # separately computed returns that jump over different missing dates.
        lr_x = np.diff(np.log(xx)); lr_y = np.diff(np.log(yy))
        diff = lr_x - lr_y
        report['price_fields'][label] = {
            'status': 'compared', 'valid_overlap_rows': n,
            'invalid_or_missing_overlap_rows': int(len(dates) - n),
            'matching_rows_within_tolerance': int(np.sum(rel <= tolerance)),
            'matching_fraction': float(np.mean(rel <= tolerance)),
            'median_relative_price_difference': float(np.median(rel)),
            'p99_relative_price_difference': float(np.quantile(rel, 0.99)),
            'max_relative_price_difference': float(np.max(rel)),
            'log_return_rmse': float(np.sqrt(np.mean(diff * diff))),
            'sum_log_return_difference': float(np.sum(diff)),
            'max_abs_log_return_difference': float(np.max(np.abs(diff))),
            'log_return_correlation': float(np.corrcoef(lr_x, lr_y)[0, 1])
                if np.std(lr_x) > 0 and np.std(lr_y) > 0 else None,
        }
    return report


def audit_pair(s3: Any, bucket: str, pair: dict, end: str, tolerance: float, out: Path) -> dict:
    symbol = pair['provider_symbol']
    old_id = pair['legacy_asset_id']; current_id = pair['current_asset_id']
    old, old_meta = active_history(s3, bucket, old_id, end)
    cur, cur_meta = active_history(s3, bucket, current_id, end)
    old_dates = set(old['date']); cur_dates = set(cur['date'])
    old_only = sorted(old_dates - cur_dates)
    current_only = sorted(cur_dates - old_dates)
    overlap = len(old_dates & cur_dates)
    price = {'status': 'blocked_by_duplicate_or_manifest_mismatch'}
    structural_ok = all(
        x['duplicate_observation_dates'] == 0
        and not x['manifest_dates_without_rows']
        and not x['rows_without_manifest_dates']
        for x in (old_meta, cur_meta)
    )
    if structural_ok:
        price = compare_prices(old, cur, tolerance)
    raw = price.get('price_fields', {}).get('raw_close_usd', {})
    adj = price.get('price_fields', {}).get('adjusted_close_usd', {})
    def passes(m: dict) -> bool:
        return (m.get('status') == 'compared'
                and m.get('invalid_or_missing_overlap_rows') == 0
                and m.get('matching_fraction') == 1.0)
    consistent = structural_ok and passes(raw) and passes(adj)
    # Data is only a proposal. Even if prices match, unique dates need exchange
    # calendar, corporate-action, issuer identity and FX evidence before promotion.
    status = 'overlap_consistent_unique_dates_need_external_review' if consistent else 'hold_price_or_structure_discrepancies'
    result = {
        'symbol': symbol, 'legacy_asset_id': old_id, 'canonical_asset_id': current_id,
        'canonical_decision': 'retain_current_asset_id',
        'legacy': old_meta, 'current': cur_meta,
        'overlap_observation_dates': overlap,
        'legacy_only_observation_dates': len(old_only),
        'current_only_observation_dates': len(current_only),
        'legacy_only_dates': old_only,
        'legacy_only_date_range': {'first': old_only[0] if old_only else None, 'last': old_only[-1] if old_only else None},
        'price_comparison': price,
        'status': status,
        'exchange_calendar_verified': False,
        'corporate_action_and_fx_independently_verified': False,
        'same_instrument_independently_verified': False,
        'legacy_rows_automatically_merged': False,
        'production_eligible': False,
    }
    # Separate evidence file for missing observations, never write merged OHLCV.
    # An additional quality-gated promotion must recompute returns, check session
    # continuity and build immutable manifests; this script does not do it.
    if old_only:
        out.mkdir(parents=True, exist_ok=True)
        unique = old.loc[old['date'].isin(old_only)].sort_values('date').copy()
        unique['candidate_current_asset_id'] = current_id
        unique['legacy_source_asset_id'] = old_id
        unique['proposal_only'] = True
        unique.to_parquet(out / f'{symbol.replace("/", "_")}_legacy_only_evidence.parquet', index=False)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs', type=Path, default=Path('legacy_coverage_pairs.jsonl'))
    ap.add_argument('--bucket', default='alpha-edge-algo')
    ap.add_argument('--region', default='eu-west-1')
    ap.add_argument('--end', default='2026-11-10')
    ap.add_argument('--output-dir', type=Path, default=Path('legacy_identity_final_audit'))
    ap.add_argument('--price-relative-tolerance', type=float, default=0.0001)
    args = ap.parse_args()
    date.fromisoformat(args.end)
    if not (0 < args.price_relative_tolerance < 0.01):
        ap.error('--price-relative-tolerance must be positive and < 1%')
    rows = read_jsonl(args.pairs)
    if len(rows) != 879 or len({(r['legacy_asset_id'], r['current_asset_id']) for r in rows}) != 879:
        ap.error('Expected the exact 879 distinct historical/current pairs; aborting')
    statuses = collections.Counter(x['coverage_status'] for x in rows)
    if statuses != {'current_strict_superset': 872, 'non_nested_coverage_review': 7}:
        ap.error(f'Unexpected input classification: {dict(statuses)}')
    exceptions = sorted((r for r in rows if r['coverage_status'] == 'non_nested_coverage_review'), key=lambda x: x['provider_symbol'])
    s3 = boto3.client('s3', region_name=args.region, config=Config(retries={'max_attempts': 8, 'mode': 'standard'}))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crosswalk = [
        {'legacy_asset_id': r['legacy_asset_id'], 'canonical_asset_id': r['current_asset_id'],
         'provider_symbol': r['provider_symbol'], 'canonical_id_decision': 'retain_current_asset_id',
         'coverage_status': r['coverage_status'],
         'legacy_unique_manifest_dates': r['legacy_unique_date_count'],
         'overlap_prices_verified': False,
         'identity_relationship_status': 'candidate_same_symbol_not_independently_approved',
         'source_history_deletion_authorized': False}
        for r in rows
    ]
    # Always preserve immutable local report of all 879 mappings, even if one
    # exceptional pair raises due to invalid data.
    with (args.output_dir / 'canonical_crosswalk_879.jsonl').open('w', encoding='utf-8') as f:
        for x in crosswalk:
            f.write(json.dumps(x, sort_keys=True) + '\n')
    results = []; failures = {}
    for i, pair in enumerate(exceptions, 1):
        sym = pair['provider_symbol']
        print(f'[{i}/7] Comparing {sym} active manifest Parquet rows...', flush=True)
        try:
            result = audit_pair(s3, args.bucket, pair, args.end, args.price_relative_tolerance, args.output_dir / 'legacy_only_evidence')
            results.append(result)
            write_json(args.output_dir / f'{sym}_comparison.json', result)
            print('  ' + result['status'], flush=True)
        except Exception as exc:
            failures[sym] = f'{type(exc).__name__}: {exc}'
            print('  ERROR ' + failures[sym], flush=True)
    summary = {
        'bucket': args.bucket, 'end_exclusive': args.end,
        'canonical_ids_retained': 879,
        'initial_coverage_classification': dict(statuses),
        'exception_assets_processed': len(results), 'exception_assets_failed': failures,
        'exception_outcomes': dict(collections.Counter(x['status'] for x in results)),
        'overall_status': 'read_only_evidence; no merges or production mutations',
        'note': 'Only 7 exceptional histories undergo price comparison; 872 supersets remain unverified at observation level.',
        'remaining_gates': ['independent instrument identity', 'exchange calendar / valid unique sessions',
                            'corporate-action and FX verification', 'accepted source values',
                            'returns recalculation for accepted candidates', 'immutable S3 candidate publication'],
    }
    write_json(args.output_dir / 'audit_summary.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()