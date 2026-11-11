"""Checksum-pinned read-only production Market Lake v2 consumer adapter.

Reads only an explicitly approved active release pointer; never falls back
to unapproved research, dev candidates or legacy MarketStore v1 data.
Asset lookups load one cohort rather than every dataset in the universe.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import pandas as pd

from alpha_edge.market.production_gold_release_v2 import CURRENT, ROOT, get, sha, verify_object, ReleaseError


class ActiveMarketDataError(RuntimeError):
    pass


@dataclass
class ActiveLakeV2:
    s3: object
    bucket: str
    release: dict
    pointer: dict

    @classmethod
    def open(cls, s3, bucket: str):
        raw = get(s3, bucket, CURRENT)
        if raw is None:
            raise ActiveMarketDataError("No active approved Market Lake v2 release")
        pointer = json.loads(raw)
        if pointer.get("schema") != "alpha_edge_v2_release_pointer_1":
            raise ActiveMarketDataError("Invalid v2 active pointer schema")
        key = pointer.get("release_key")
        if not isinstance(key, str) or not key.startswith(ROOT + "/releases/candidates/"):
            raise ActiveMarketDataError("Pointer targets untrusted release key")
        version = get(s3, bucket, key)
        if version is None or sha(version) != pointer.get("release_sha256"):
            raise ActiveMarketDataError("Active release manifest SHA mismatch")
        release = json.loads(version)
        if release.get("schema") != "alpha_edge_v2_release_candidate_1" or release.get("release_id") != pointer.get(
            "release_id"
        ):
            raise ActiveMarketDataError("Active release identity mismatch")
        if len(release.get("cohorts", [])) != release.get("expected_cohorts"):
            raise ActiveMarketDataError("Active release cohort population incomplete")
        return cls(s3=s3, bucket=bucket, release=release, pointer=pointer)

    def _cohort(self, asset_id: str) -> dict:
        matches = [c for c in self.release["cohorts"] if asset_id in c.get("assets", [])]
        if len(matches) != 1:
            raise ActiveMarketDataError(f"Asset missing or ambiguous in active release: {asset_id}")
        return matches[0]

    def read_asset(
        self, asset_id: str, *, layer: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        if layer not in {"silver", "gold", "indicator_candidate"}:
            raise ActiveMarketDataError("Only Silver, Gold or indicator data can be read")
        cohort = self._cohort(asset_id)
        reference = cohort.get(layer)
        if not isinstance(reference, dict):
            raise ActiveMarketDataError(f"No pinned {layer} artifact in release")
        try:
            blob = verify_object(self.s3, self.bucket, reference)
        except ReleaseError as exc:
            raise ActiveMarketDataError(f"Corrupt active release artifact: {exc}") from exc
        frame = pd.read_parquet(io.BytesIO(blob))
        if "asset_id" not in frame:
            raise ActiveMarketDataError("Release artifact missing asset_id")
        if frame.duplicated(["asset_id", "session_date" if layer != "indicator_candidate" else "date"]).any():
            raise ActiveMarketDataError("Duplicate asset/session in active release")
        selected = frame.loc[frame["asset_id"].astype(str).eq(asset_id)].copy()
        date_column = "date" if layer == "indicator_candidate" else "session_date"
        if date_column not in selected:
            raise ActiveMarketDataError(f"Release artifact lacks {date_column}")
        dates = pd.to_datetime(selected[date_column].astype(str).str[:10], errors="coerce")
        if dates.isna().any():
            raise ActiveMarketDataError("Malformed session labels in approved release")
        if start:
            selected = selected.loc[dates >= pd.Timestamp(start)]
            dates = dates.loc[dates >= pd.Timestamp(start)]
        if end:
            selected = selected.loc[dates <= pd.Timestamp(end)]
        return selected.reset_index(drop=True)

    def prices(self, asset_id: str, **kwargs) -> pd.DataFrame:
        data = self.read_asset(asset_id, layer="gold", **kwargs)
        columns = ["asset_id", "session_date", "close_raw_usd", "close_adjusted_usd", "quality_status"]
        if missing := set(columns) - set(data):
            raise ActiveMarketDataError(f"Gold price contract missing: {missing}")
        return data[columns].rename(columns={"session_date": "date"})

    def returns(self, asset_id: str, **kwargs) -> pd.DataFrame:
        data = self.read_asset(asset_id, layer="gold", **kwargs)
        columns = [
            "asset_id",
            "session_date",
            "ret_simple_close_adjusted_usd",
            "ret_log_close_adjusted_usd",
            "return_quality_status",
        ]
        if missing := set(columns) - set(data):
            raise ActiveMarketDataError(f"Gold return contract missing: {missing}")
        return data[columns].rename(columns={"session_date": "date"})

    def indicators(self, asset_id: str, **kwargs) -> pd.DataFrame:
        return self.read_asset(asset_id, layer="indicator_candidate", **kwargs)
