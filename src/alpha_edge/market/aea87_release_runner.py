"""AEA87 release workflow. Requires an exclusive maintenance window."""
import json
from botocore.exceptions import ClientError
from alpha_edge.market.aea87_publication_transaction import checksum, fetch, put_once, write_event, switch_manifests, rollback, entries
from alpha_edge.market.preview_aea87_publication import read_document, select_candidates

ROOT="market/repairs/aea87/publications"

def build(s3,bucket,plan_key,release_id):
    plan,plan_sha=read_document(s3,bucket,plan_key)
    recon,recon_sha=read_document(s3,bucket,"market/repairs/aea87/staging/"+plan["reconciliation_id"]+"/reconciliation.json")
    if recon_sha!=plan["reconciliation_sha256"]:
        raise ValueError("Reconciliation changed")
    selected,excluded=select_candidates(recon,plan)
    plans={x["asset_id"]:x for x in plan["assets"]}
    sources={}
    for src in recon["sources"].values():
        campaign,h=read_document(s3,bucket,src["manifest_key"])
        checks,vh=read_document(s3,bucket,src["validation_key"])
        if h!=src["manifest_sha256"] or vh!=src["validation_sha256"]:
            raise ValueError("Source validation changed")
        sources[src["run_id"]]=({a["asset_id"]:a for a in campaign["prepared"]},{a["asset_id"]:a for a in checks["assets"]})
    operations=[]
    for asset in recon["assets"]:
        aid=asset["asset_id"]
        if aid not in selected:
            continue
        prepared,validated=sources[asset["campaign"]]
        if validated[aid]["status"]!="VALID" or validated[aid].get("errors"):
            raise ValueError("Validation failed "+aid)
        records=prepared[aid]["records"]
        if len(records)!=plans[aid]["record_count"]:
            raise ValueError("Record count mismatch")
        by_key={r["manifest_key"]:r for r in records}
        if len(by_key)!=len(records):
            raise ValueError("Duplicate target "+aid)
        retire=set(plans[aid]["retire_manifest_keys"])
        if retire and aid!="CRYPTO:HYPE-USD":
            raise ValueError("Unexpected retirement")
        for key in sorted(set(by_key)|retire):
            before=fetch(s3,bucket,key)
            expected=plans[aid]["expected_manifest_hashes"][key]
            if checksum(before)!=expected:
                raise ValueError("Active manifest changed "+key)
            record=by_key.get(key)
            after=None
            if record:
                after=json.dumps({"asset_id":aid,"year":record["year"],"dates":record["dates"],"parts":[record["new_part"]],"repair_run_id":release_id},sort_keys=True)
            index=len(operations)
            operations.append(dict(manifest_key=key,asset_id=aid,before_sha=expected,
                after=after,after_sha=checksum(after.encode()) if after else None,
                staged_key=record["staged_part"] if record else None,
                staged_sha=record["sha256"] if record else None,
                part_key=record["new_part"] if record else None,
                backup_key=f"{ROOT}/{release_id}/backups/{index:06d}.json"))
    if len(selected)!=842 or len(excluded)!=11 or len([x for x in operations if x["asset_id"]=="CRYPTO:HYPE-USD" and x["after"] is None])==0:
        raise ValueError("Unapproved selection or missing HYPE retirement")
    return dict(kind="AEA87_PUBLICATION",release_id=release_id,plan_key=plan_key,plan_sha=plan_sha,
                candidates=selected,excluded=excluded,operations=operations)

def _stage_one(s3, bucket, op):
    """Idempotent operation: existing objects must match approved SHA-256."""
    original = fetch(s3, bucket, op["manifest_key"])
    if checksum(original) != op["before_sha"]:
        raise RuntimeError("Active manifest changed: " + op["manifest_key"])
    reused = 0
    if original is not None:
        backup = fetch(s3, bucket, op["backup_key"])
        if backup is None:
            try:
                put_once(s3, bucket, op["backup_key"], original)
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") not in ("PreconditionFailed", "412"):
                    raise
            backup = fetch(s3, bucket, op["backup_key"])
        else:
            reused += 1
        if checksum(backup) != op["before_sha"]:
            raise RuntimeError("Backup differs from approved original: " + op["backup_key"])
    if op["staged_key"]:
        # Never replace an existing destination: it may belong to a previous
        # attempt and must be validated instead.
        try:
            head = s3.head_object(Bucket=bucket, Key=op["part_key"])
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
                raise
            head = None
        if head is None:
            # S3-side copy avoids streaming Parquet bytes through the PC.
            # Verify staged content before copying; the existing campaign validator
            # already recorded its expected SHA-256.
            source = fetch(s3, bucket, op["staged_key"])
            if checksum(source) != op["staged_sha"]:
                raise RuntimeError("Staged partition changed: " + op["staged_key"])
            try:
                s3.copy_object(Bucket=bucket,
                    CopySource={"Bucket": bucket, "Key": op["staged_key"]},
                    Key=op["part_key"], MetadataDirective="REPLACE",
                    Metadata={"aea87-sha256": op["staged_sha"]},
                    CopySourceIfMatch=s3.head_object(
                        Bucket=bucket, Key=op["staged_key"])["ETag"].strip('"'))
            except ClientError as exc:
                # A second process may have created the destination. We still
                # require exact contents, rather than overwriting them.
                raise RuntimeError("S3 copy failed; inspect destination " + op["part_key"]) from exc
        else:
            reused += 1
        part_head = s3.head_object(Bucket=bucket, Key=op["part_key"])
        advertised = part_head.get("Metadata", {}).get("aea87-sha256")
        if advertised == op["staged_sha"] and head is None:
            pass  # just copied from checked source
        elif checksum(fetch(s3, bucket, op["part_key"])) != op["staged_sha"]:
            raise RuntimeError("Destination partition checksum mismatch: " + op["part_key"])
    return reused


def stage(s3, bucket, manifest, workers=16, progress_every=200):
    """Prepare or resume the SAME release ID. No manifest switching here."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    release_id = manifest["release_id"]
    root = f"{ROOT}/{release_id}"
    payload = json.dumps(manifest, sort_keys=True).encode()
    saved = fetch(s3, bucket, root + "/operations.json")
    if saved is None:
        put_once(s3, bucket, root + "/operations.json", payload)
    elif checksum(saved) != checksum(payload):
        raise RuntimeError("Existing release operations differ; do not reuse release ID")
    history = entries(s3, bucket, root)
    if not history:
        write_event(s3, bucket, root, 0, "PREPARING")
    elif history[-1]["state"] == "READY":
        print("[stage] release already READY; no writes required", flush=True)
        return
    elif history[-1]["state"] != "PREPARING":
        raise RuntimeError("Cannot stage in state " + history[-1]["state"])
    total = len(manifest["operations"])
    done = reused = 0
    started = time.monotonic()
    print(f"[stage] {total:,} operations; {workers} workers; resuming verified S3 objects", flush=True)
    # Bounded in-flight futures avoid queues of 34k jobs and allow Ctrl-C
    # to stop after running work has drained.
    from concurrent.futures import wait, FIRST_COMPLETED
    with ThreadPoolExecutor(max_workers=workers) as executor:
        it = iter(manifest["operations"])
        inflight = {}
        def submit():
            try:
                op = next(it)
            except StopIteration:
                return False
            inflight[executor.submit(_stage_one, s3, bucket, op)] = op
            return True
        for _ in range(workers * 2):
            if not submit():
                break
        while inflight:
            finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in finished:
                op = inflight.pop(future)
                reused += future.result()
                done += 1
                if done % progress_every == 0 or done == total:
                    elapsed = max(time.monotonic() - started, 0.001)
                    print(f"[stage] {done:,}/{total:,} ({done/total:.1%}); "
                          f"verified existing objects={reused:,}; rate={done/elapsed:.1f} ops/s",
                          flush=True)
                submit()
    # READY means all backup and partition operations were verified.
    write_event(s3, bucket, root, 1, "READY")
    print("[stage] READY; activate only under exclusive maintenance", flush=True)

def resume(s3,bucket,release_id,action):
    root=f"{ROOT}/{release_id}"
    raw=fetch(s3,bucket,root+"/operations.json")
    if raw is None:
        raise ValueError("Unknown release ID")
    doc=json.loads(raw)
    if action=="rollback":
        rollback(s3,bucket,root,doc["operations"])
    elif action=="activate":
        switch_manifests(s3,bucket,root,doc["operations"])
    elif action=="finalize":
        from alpha_edge.market.aea87_release_finalize import finalize
        return finalize(s3,bucket,root,doc)
    elif action=="verify":
        bad=[x["manifest_key"] for x in doc["operations"] if checksum(fetch(s3,bucket,x["manifest_key"]))!=x["after_sha"]]
        return {"mismatches":len(bad),"samples":bad[:10],"journal_state":entries(s3,bucket,root)[-1]["state"]}
    else:
        raise ValueError(action)
    return entries(s3,bucket,root)[-1]

def main():
    import argparse
    import boto3
    from alpha_edge.market.preview_aea87_publication import APPROVED_PLAN
    p=argparse.ArgumentParser(description="AEA87 maintenance-window release runner")
    p.add_argument("--action",choices=("inspect","stage","activate","verify","rollback","finalize"),default="inspect")
    p.add_argument("--release-id",required=True)
    p.add_argument("--plan-key",default=APPROVED_PLAN)
    p.add_argument("--bucket",default="alpha-edge-algo")
    p.add_argument("--region",default="eu-west-1")
    p.add_argument("--env",choices=("dev","prod"),default="dev")
    p.add_argument("--confirm-prod-write",action="store_true")
    p.add_argument("--maintenance-ack",action="store_true")
    p.add_argument("--workers",type=int,default=16)
    p.add_argument("--progress-every",type=int,default=200)
    a=p.parse_args()
    if a.action in ("stage","activate","rollback","finalize") and not (
        a.env=="prod" and a.confirm_prod_write and a.maintenance_ack
    ):
        p.error("Writes require --env prod --confirm-prod-write --maintenance-ack")
    if not 1 <= a.workers <= 32 or a.progress_every < 1:
        p.error("--workers must be 1..32 and --progress-every must be positive")
    from botocore.config import Config
    s3=boto3.client("s3",region_name=a.region,config=Config(max_pool_connections=max(32,a.workers*2)))
    if a.action=="inspect":
        doc=build(s3,a.bucket,a.plan_key,a.release_id)
        result={"release_id":a.release_id,"candidates":len(doc["candidates"]),
                "excluded":len(doc["excluded"]),"manifest_operations":len(doc["operations"]),
                "production_modified":False}
    elif a.action=="stage":
        old=fetch(s3,a.bucket,f"{ROOT}/{a.release_id}/operations.json")
        doc=json.loads(old) if old is not None else build(s3,a.bucket,a.plan_key,a.release_id)
        if doc["plan_key"]!=a.plan_key or doc["release_id"]!=a.release_id:
            raise ValueError("Existing release plan identity differs")
        stage(s3,a.bucket,doc,workers=a.workers,progress_every=a.progress_every)
        result={"release_id":a.release_id,"state":"READY","operations":len(doc["operations"])}
    else:
        result=resume(s3,a.bucket,a.release_id,a.action)
    print(json.dumps(result,indent=2))
    if a.action=="activate":
        print("READERS MUST REMAIN STOPPED: caches and snapshots have not been rebuilt.")

if __name__=="__main__":
    main()
