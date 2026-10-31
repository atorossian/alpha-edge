"""AEA87 release workflow. Requires an exclusive maintenance window."""
import json
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

def stage(s3,bucket,manifest):
    release_id=manifest["release_id"]
    root=f"{ROOT}/{release_id}"
    put_once(s3,bucket,root+"/operations.json",json.dumps(manifest,sort_keys=True).encode())
    write_event(s3,bucket,root,0,"PREPARING")
    for op in manifest["operations"]:
        original=fetch(s3,bucket,op["manifest_key"])
        if checksum(original)!=op["before_sha"]:
            raise RuntimeError("Active manifest changed "+op["manifest_key"])
        if original is not None:
            put_once(s3,bucket,op["backup_key"],original)
        if op["staged_key"]:
            data=fetch(s3,bucket,op["staged_key"])
            if checksum(data)!=op["staged_sha"]:
                raise RuntimeError("Stage checksum mismatch "+op["staged_key"])
            put_once(s3,bucket,op["part_key"],data)
    for op in manifest["operations"]:
        if checksum(fetch(s3,bucket,op["manifest_key"]))!=op["before_sha"]:
            raise RuntimeError("Active manifest changed during stage")
        if op["before_sha"] is not None and checksum(fetch(s3,bucket,op["backup_key"]))!=op["before_sha"]:
            raise RuntimeError("Backup corrupted")
        if op["staged_key"] and checksum(fetch(s3,bucket,op["part_key"]))!=op["staged_sha"]:
            raise RuntimeError("New partition corrupted")
    write_event(s3,bucket,root,1,"READY")

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
    elif action=="verify":
        bad=[x["manifest_key"] for x in doc["operations"] if checksum(fetch(s3,bucket,x["manifest_key"]))!=x["after_sha"]]
        return {"mismatches":len(bad),"samples":bad[:10],"journal_state":entries(s3,bucket,root)[-1]["state"]}
    else:
        raise ValueError(action)
    return entries(s3,bucket,root)[-1]
