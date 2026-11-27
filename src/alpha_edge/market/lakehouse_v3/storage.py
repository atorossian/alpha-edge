"""Immutable S3 objects, checksum-pinned manifests and in-memory Parquet I/O."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import time

import pandas as pd
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    IncompleteReadError,
    ReadTimeoutError,
    ResponseStreamingError,
)

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


def read_object(s3, bucket: str, key: str, *, optional=False):
    """Retry interrupted response bodies from byte zero, never partial bytes."""
    for attempt in range(5):
        body = None
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            payload = body.read()
            if "ContentLength" in response and len(payload) != response["ContentLength"]:
                raise IncompleteReadError(actual_bytes=len(payload), expected_bytes=response["ContentLength"])
            return payload, response.get("Metadata", {})
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
                if optional:
                    return None
                raise LakehouseError(f"Required S3 object missing: {key}") from exc
            raise
        except (ResponseStreamingError, IncompleteReadError, ReadTimeoutError, ConnectionClosedError):
            if attempt == 4:
                raise
            logging.getLogger(__name__).warning("Retrying interrupted S3 read (%d/5): %s", attempt + 2, key)
            time.sleep(min(2**attempt, 8))
        finally:
            if body is not None:
                body.close()


def read_bytes(s3, bucket: str, key: str) -> bytes:
    return read_object(s3, bucket, key)[0]


def read_json(s3, bucket: str, key: str) -> dict:
    payload = json.loads(read_bytes(s3, bucket, key))
    if not isinstance(payload, dict):
        raise LakehouseError(f"Expected JSON object: {key}")
    return payload


def put_immutable(
    s3,
    bucket: str,
    key: str,
    data: bytes,
    *,
    content_type: str,
    scope: str = ROOT,
    metadata: dict[str, str] | None = None,
) -> dict:
    """Create a scoped object and verify its bytes and required metadata."""
    if (
        (scope != ROOT and not scope.startswith(ROOT + "/"))
        or not key.startswith(scope + "/")
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise LakehouseError(f"Refusing unsafe S3 write path: {key}")
    sha = digest(data)
    expected_metadata = {**(metadata or {}), "sha256": sha}
    conflict = None
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata=expected_metadata,
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
        conflict = exc
    payload, actual_metadata = read_object(s3, bucket, key)
    if digest(payload) != sha or any(actual_metadata.get(name) != value for name, value in expected_metadata.items()):
        if conflict is not None:
            raise LakehouseError(f"Immutable S3 collision: {key}") from conflict
        raise LakehouseError(f"S3 read-after-write checksum or metadata mismatch: {key}")
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
