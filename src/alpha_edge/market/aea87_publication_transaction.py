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


def rollback(s3,bucket,root,ops):
    history=entries(s3,bucket,root)
    if not history:
        raise RuntimeError("Missing publication journal")
    if history[-1]["state"]=="ROLLED_BACK":
        return
    if history[-1]["state"]=="FINALIZED":
        raise RuntimeError("Finalized release requires a separate reversal campaign")
    step=history[-1]["step"]+1
    write_event(s3,bucket,root,step,"ROLLING_BACK")
    for op in reversed(ops):
        safe_restore(s3,bucket,op)
    write_event(s3,bucket,root,step+1,"ROLLED_BACK")


def switch_manifests(s3,bucket,root,ops):
    """Requires immutable backed-up operation plan and external reader shutdown."""
    history=entries(s3,bucket,root)
    if not history or history[-1]["state"]!="READY":
        raise RuntimeError("Publication journal is not READY")
    for op in ops:
        current=fetch(s3,bucket,op["manifest_key"])
        if checksum(current)!=op["before_sha"]:
            raise RuntimeError("Manifest drift: "+op["manifest_key"])
        if checksum(fetch(s3,bucket,op["backup_key"]))!=op["before_sha"]:
            raise RuntimeError("Missing backup: "+op["manifest_key"])
    step=history[-1]["step"]+1
    write_event(s3,bucket,root,step,"SWITCHING")
    try:
        for op in ops:
            if checksum(fetch(s3,bucket,op["manifest_key"]))!=op["before_sha"]:
                raise RuntimeError("Concurrent writer: "+op["manifest_key"])
            if op["after"] is None:
                s3.delete_object(Bucket=bucket,Key=op["manifest_key"])
            else:
                s3.put_object(Bucket=bucket,Key=op["manifest_key"],Body=op["after"].encode(),ContentType="application/json")
            step+=1
            write_event(s3,bucket,root,step,"SWITCHING",last_key=op["manifest_key"])
        write_event(s3,bucket,root,step+1,"CACHE_REBUILD_REQUIRED")
    except Exception:
        rollback(s3,bucket,root,ops)
        raise
