"""Immutable S3 objects, checksum-pinned manifests and in-memory Parquet I/O."""

from __future__ import annotations

import hashlib
import io
import json
import re

import pandas as pd
from botocore.exceptions import ClientError

ROOT = "market/lake_v3"
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


class LakehouseError(RuntimeError):
    """Invalid state or violated market-data publication contract."""


def check_id(value: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise LakehouseError(f"Unsafe identifier: {value!r}")
    return value


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(obj: dict) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def read_bytes(s3, bucket: str, key: str) -> bytes:
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
            raise LakehouseError(f"Required S3 object missing: {key}") from exc
        raise


def read_json(s3, bucket: str, key: str) -> dict:
    payload = json.loads(read_bytes(s3, bucket, key))
    if not isinstance(payload, dict):
        raise LakehouseError(f"Expected JSON object: {key}")
    return payload


def put_immutable(s3, bucket: str, key: str, data: bytes, *, content_type: str) -> dict:
    """A retry is safe only if the existing object's bytes match exactly."""
    if not key.startswith(ROOT + "/") or "/../" in key or "/./" in key:
        raise LakehouseError(f"Refusing unsafe S3 write path: {key}")
    sha = digest(data)
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata={"sha256": sha},
            IfNoneMatch="*",
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {
            "PreconditionFailed",
            "412",
            "ConditionalRequestConflict",
            "409",
        }:
            raise
        if digest(read_bytes(s3, bucket, key)) != sha:
            raise LakehouseError(f"Immutable S3 collision: {key}") from exc
    if digest(read_bytes(s3, bucket, key)) != sha:
        raise LakehouseError(f"S3 read-after-write checksum mismatch: {key}")
    return {"key": key, "sha256": sha, "bytes": len(data)}


def put_json(s3, bucket: str, key: str, document: dict) -> dict:
    return put_immutable(s3, bucket, key, json_bytes(document), content_type="application/json")


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.BytesIO()
    frame.to_parquet(stream, index=False, engine="pyarrow", compression="snappy")
    return stream.getvalue()


def put_parquet(s3, bucket: str, key: str, frame: pd.DataFrame) -> dict:
    return put_immutable(s3, bucket, key, parquet_bytes(frame), content_type="application/vnd.apache.parquet")


def verified_bytes(s3, bucket: str, ref: dict) -> bytes:
    key, sha = ref.get("key"), ref.get("sha256")
    if (
        not isinstance(key, str)
        or not key.startswith(ROOT + "/")
        or "/../" in key
        or "/./" in key
        or not isinstance(sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", sha)
    ):
        raise LakehouseError("Malformed S3 artifact reference")
    blob = read_bytes(s3, bucket, key)
    if digest(blob) != sha:
        raise LakehouseError(f"Corrupt S3 artifact: {key}")
    return blob


def verified_frame(s3, bucket: str, ref: dict) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(verified_bytes(s3, bucket, ref)), engine="pyarrow")
