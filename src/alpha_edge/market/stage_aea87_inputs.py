"""Publish immutable AEA-87 input CSVs to S3; no production market datasets changed.

Run this ONE TIME from the usual Alpha Edge Poetry environment.
The source files may be local temporarily; persistent authoritative copies are S3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anomalies", type=Path, required=True)
    ap.add_argument("--universe", type=Path, required=True)
    ap.add_argument("--bucket", default="alpha-edge-algo")
    ap.add_argument("--region", default="eu-west-1")
    ap.add_argument("--run-id", default="aea87_inputs_20261028")
    args = ap.parse_args()

    if not args.run_id.startswith("aea87_inputs_") or not all(
        ch.isalnum() or ch in "_-" for ch in args.run_id
    ):
        raise ValueError("Invalid immutable run ID")
    s3 = boto3.client("s3", region_name=args.region)
    prefix = f"market/quality/aea87/inputs/{args.run_id}"
    specs = (
        ("returns_quality_cells.csv", args.anomalies, ("asset_id", "date")),
        ("universe.csv", args.universe, ("asset_id",)),
    )
    manifest = {
        "run_id": args.run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": args.bucket, "files": {},
    }
    prepared = []
    for name, file, required in specs:
        raw = file.read_bytes()
        frame = pd.read_csv(file)
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{file}: missing {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{file}: input is empty")
        if name == "returns_quality_cells.csv":
            frame["asset_id"] = frame["asset_id"].astype(str).str.strip()
            if frame["asset_id"].isna().any() or frame["asset_id"].eq("").any():
                raise ValueError("Anomaly CSV contains empty asset IDs")
        digest = hashlib.sha256(raw).hexdigest()
        key = f"{prefix}/{name}"
        manifest["files"][name] = {"key": key, "sha256": digest, "rows": len(frame)}
        prepared.append((key, raw, digest))

    for key, raw, digest in prepared:
        try:
            s3.head_object(Bucket=args.bucket, Key=key)
        except s3.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
                raise
        else:
            existing = s3.get_object(Bucket=args.bucket, Key=key)["Body"].read()
            if hashlib.sha256(existing).hexdigest() != digest:
                raise FileExistsError(f"Immutable input differs from existing object: {key}")
            print(f"ALREADY_VERIFIED s3://{args.bucket}/{key}")
            continue
        s3.put_object(Bucket=args.bucket, Key=key, Body=raw, ContentType="text/csv")
        observed = s3.get_object(Bucket=args.bucket, Key=key)["Body"].read()
        if hashlib.sha256(observed).hexdigest() != digest:
            raise IOError(f"Uploaded object checksum mismatch: {key}")
        print(f"UPLOADED_AND_VERIFIED s3://{args.bucket}/{key}")

    manifest_key = f"{prefix}/manifest.json"
    try:
        prior = json.loads(s3.get_object(Bucket=args.bucket, Key=manifest_key)["Body"].read())
        comparable = {k: v for k, v in prior.items() if k != "created_utc"}
        if comparable != {k: v for k, v in manifest.items() if k != "created_utc"}:
            raise FileExistsError("Existing input manifest disagrees with source files")
    except s3.exceptions.NoSuchKey:
        s3.put_object(Bucket=args.bucket, Key=manifest_key,
                      Body=json.dumps(manifest, indent=2).encode(), ContentType="application/json")
    print(f"INPUT_MANIFEST s3://{args.bucket}/{manifest_key}")


if __name__ == "__main__":
    main()
