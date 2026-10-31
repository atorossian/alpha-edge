"""AEA-87 publication transaction primitives. Never invoke without a maintenance window.

S3 has no multi-object transaction. The journal makes writes recoverable, but
readers MUST be stopped throughout activation, cache rebuild, and verification.
"""
from __future__ import annotations
import hashlib
import json
from botocore.exceptions import ClientError


def checksum(value):
    return hashlib.sha256(value).hexdigest() if value is not None else None


def fetch(s3, bucket, key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise


def put_once(s3, bucket, key, value):
    s3.put_object(Bucket=bucket, Key=key, Body=value, IfNoneMatch="*")


def write_event(s3, bucket, root, step, state, **extra):
    payload = json.dumps({"step":step, "state":state, **extra}, sort_keys=True).encode()
    put_once(s3, bucket, f"{root}/events/{step:08d}.json", payload)


def entries(s3, bucket, root):
    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=root+"/events/")
    keys = sorted(obj["Key"] for page in pages for obj in page.get("Contents", []))
    return [json.loads(fetch(s3,bucket,k)) for k in keys]


def safe_restore(s3, bucket, op):
    current = fetch(s3,bucket,op["manifest_key"])
    actual = checksum(current)
    if actual == op["before_sha"]:
        return
    if actual != op["after_sha"]:
        raise RuntimeError("Concurrent change prevents rollback: "+op["manifest_key"])
    original = fetch(s3,bucket,op["backup_key"])
    if checksum(original) != op["before_sha"]:
        raise RuntimeError("Missing/corrupt backup: "+op["manifest_key"])
    if original is None:
        s3.delete_object(Bucket=bucket,Key=op["manifest_key"])
    else:
        s3.put_object(Bucket=bucket,Key=op["manifest_key"],Body=original,ContentType="application/json")


def _parallel(items, fn, workers, label, progress_every=200):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    done, errors = 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append((futures[future]["manifest_key"], str(exc)))
            done += 1
            if done % progress_every == 0 or done == len(items):
                print(f"[{label}] {done:,}/{len(items):,} ({done/len(items):.1%})",
                      flush=True)
    if errors:
        raise RuntimeError(f"{label} failed: {len(errors)} objects; first={errors[0]}")


def rollback(s3, bucket, root, ops, workers=16):
    history = entries(s3, bucket, root)
    if not history:
        raise RuntimeError("Missing publication journal")
    if history[-1]["state"] == "ROLLED_BACK":
        return
    if history[-1]["state"] == "FINALIZED":
        raise RuntimeError("Finalized release requires separate reversal")
    step = history[-1]["step"] + 1
    write_event(s3, bucket, root, step, "ROLLING_BACK")
    _parallel(ops, lambda op: safe_restore(s3, bucket, op), workers, "rollback")
    write_event(s3, bucket, root, step + 1, "ROLLED_BACK")


def _activate_one(s3, bucket, op):
    key = op["manifest_key"]
    original = fetch(s3, bucket, key)
    if checksum(original) != op["before_sha"]:
        raise RuntimeError("Concurrent writer: " + key)
    if op["after"] is None:
        # Retirement requires external exclusive maintenance, same as writes.
        if original is not None:
            s3.delete_object(Bucket=bucket, Key=key)
    else:
        params = {"Bucket": bucket, "Key": key,
                  "Body": op["after"].encode(), "ContentType": "application/json"}
        if original is None:
            params["IfNoneMatch"] = "*"
        else:
            # ETag conditional prevents a writer changing this manifest
            # between verification and replacement.
            params["IfMatch"] = s3.head_object(Bucket=bucket, Key=key)["ETag"]
        s3.put_object(**params)


def switch_manifests(s3, bucket, root, ops, workers=16, batch_size=200):
    """Parallel manifest batches with durable checkpoints, not atomic visibility."""
    history = entries(s3, bucket, root)
    if not history or history[-1]["state"] != "READY":
        raise RuntimeError("Publication journal is not READY")
    for op in ops:
        if checksum(fetch(s3, bucket, op["manifest_key"])) != op["before_sha"]:
            raise RuntimeError("Manifest drift: " + op["manifest_key"])
        if checksum(fetch(s3, bucket, op["backup_key"])) != op["before_sha"]:
            raise RuntimeError("Missing backup: " + op["manifest_key"])
    step = history[-1]["step"] + 1
    write_event(s3, bucket, root, step, "SWITCHING")
    try:
        for i in range(0, len(ops), batch_size):
            batch = ops[i:i + batch_size]
            _parallel(batch, lambda op: _activate_one(s3, bucket, op), workers, "activate")
            step += 1
            write_event(s3, bucket, root, step, "SWITCHING",
                        completed=min(i + batch_size, len(ops)), total=len(ops))
            print(f"[activate] checkpoint {min(i+batch_size,len(ops)):,}/{len(ops):,}", flush=True)
        write_event(s3, bucket, root, step + 1, "CACHE_REBUILD_REQUIRED")
    except Exception:
        rollback(s3, bucket, root, ops, workers=workers)
        raise
