"""Read-only self-contained HTML explorer for a verified lake-v2 candidate.

No S3, network calls, server, write-back, or external JavaScript dependencies.
Input files are verified against the immutable candidate manifest first.
The browser reads only the snapshot embedded during report generation.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_edge.market.publish_lake_v2 import validate_local_candidate


DETAIL_FIELDS = [
    "session_date", "asset_id", "provider_symbol", "currency_original",
    "observation_origin", "provider", "correction_id", "verification_status",
    "source_reference", "source_timestamp", "source_timezone",
    "open", "high", "low", "close", "adj_close", "volume",
    "fx_date", "fx_to_usd", "fx_age_days", "fx_quote_symbol",
    "fx_quote_direction", "quote_close", "close_raw_usd",
    "close_adjusted_usd", "quality_status", "quality_reason",
    "ret_simple_close_adjusted_usd", "ret_log_close_adjusted_usd",
    "return_quality_status",
]
MAX_EMBEDDED_ROWS = 100_000


def _safe_json(data: object) -> str:
    """JSON inside a script element must not contain literal closing tags."""
    return json.dumps(data, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")


def _serializable(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()[:10]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [
        {str(key): _serializable(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def build_explorer_model(candidate_dir: Path) -> dict:
    """Validate files, then produce a bounded and provenance-preserving model."""
    manifest, verified_files = validate_local_candidate(candidate_dir)
    gold = pd.read_parquet(candidate_dir / "gold.parquet")
    sessions = pd.read_parquet(candidate_dir / "session_issues.parquet")
    fx = pd.read_parquet(candidate_dir / "fx_evidence.parquet")
    if gold.empty:
        raise ValueError("Gold history is empty")
    if len(gold) > MAX_EMBEDDED_ROWS:
        raise ValueError("Candidate too large for static explorer; filter before export")
    required = {"asset_id", "provider_symbol", "session_date", "quality_status"}
    if missing := required - set(gold.columns):
        raise ValueError(f"Gold missing required explorer columns: {sorted(missing)}")
    if len(gold) != manifest.get("gold_rows"):
        raise ValueError("Manifest vs Gold row-count mismatch")
    if gold.duplicated(["asset_id", "session_date"]).any():
        raise ValueError("Gold has duplicate asset/session keys; refusing to hide conflicts")
    if not {"asset_id", "provider_symbol", "session_date", "issue"}.issubset(sessions.columns):
        raise ValueError("Session issue evidence lacks required columns")
    if "fx_validation_status" not in fx:
        raise ValueError("FX diagnostics lack validation status")

    subset = gold[[field for field in DETAIL_FIELDS if field in gold.columns]].copy()
    subset["session_date"] = pd.to_datetime(subset["session_date"], errors="coerce")
    if subset["session_date"].isna().any():
        raise ValueError("Gold includes invalid session labels")
    subset = subset.sort_values(["asset_id", "session_date"], kind="stable")
    sessions = sessions.copy()
    sessions["session_date"] = pd.to_datetime(sessions["session_date"], errors="coerce")
    assets = []
    for aid, group in subset.groupby("asset_id", sort=True):
        asset_issues = sessions.loc[sessions["asset_id"].astype(str).eq(str(aid))]
        assets.append({
            "asset_id": str(aid),
            "symbol": str(group.iloc[0]["provider_symbol"]),
            "currency": str(group.iloc[0].get("currency_original", "")),
            "observations": len(group),
            "quarantined": int(group["quality_status"].eq("quarantined").sum()),
            "manual": int(group.get("observation_origin",
                pd.Series("", index=group.index)).eq("user_supplied_manual").sum()),
            "session_issues": len(asset_issues),
            "first": group["session_date"].iloc[0].date().isoformat(),
            "last": group["session_date"].iloc[-1].date().isoformat(),
        })

    return {
        "schema": "alpha_edge_lake_v2_quality_explorer_1",
        "source_candidate_manifest_sha256": hashlib.sha256(
            (candidate_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "candidate_acceptance": manifest.get("acceptance"),
        "publication_state": manifest.get("publication_state"),
        "files_verified": len(verified_files),
        "manual_correction_approval": manifest.get("manual_correction_approval"),
        "assets": assets,
        "observations": _records(subset),
        "session_issues": _records(sessions),
        "fx_issues": _records(fx.loc[fx["fx_validation_status"].ne("accepted")]),
        "limitations": (
            "Read-only local candidate snapshot. A manual close does not establish "
            "an adjusted close or adjusted return. FX alignment is provisional. "
            "Exchange calendar audit only covers first-to-last observed sessions."
        ),
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alpha Edge — Market Data Quality Explorer</title>
<style>
:root{color-scheme:dark;--bg:#0b111b;--panel:#141e2c;--line:#2c384a;--text:#ebf1f9;
--muted:#a5b3c4;--blue:#72b9ff;--amber:#ffd083;--red:#ff8c8c;--green:#84e0b0}
*{box-sizing:border-box}body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;
background:var(--bg);color:var(--text);margin:0;padding:24px;line-height:1.45}
main{max-width:1460px;margin:auto}h1{font-size:1.6rem;margin:0}h2{font-size:1.1rem}
small,.sub{color:var(--muted)}.sub{margin:7px 0 19px}
header{display:flex;gap:16px;align-items:center;justify-content:space-between;flex-wrap:wrap}
.pill{padding:5px 11px;border:1px solid var(--line);border-radius:40px;color:var(--amber)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin:14px 0}
.controls{display:flex;flex-wrap:wrap;gap:13px;align-items:end}
label{display:flex;flex-direction:column;gap:5px;color:var(--muted);font-size:.85rem}
select,input,button{background:#0f1722;color:var(--text);border:1px solid #4a5a6e;
border-radius:6px;padding:9px;font-size:.93rem}button{cursor:pointer}button:hover{border-color:var(--blue)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.metric{background:#0d1520;border:1px solid var(--line);padding:13px;border-radius:8px}
.metric strong{display:block;font-size:1.4rem}.metric small{display:block}
.chart{width:100%;height:300px;background:#0d1520;border-radius:8px}
.chart path{fill:none;stroke:var(--blue);stroke-width:2}
.chart .axis{stroke:#526074;stroke-width:1}
.chart text{fill:var(--muted);font-size:12px}
.scroll{max-height:370px;overflow:auto}table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{border-bottom:1px solid var(--line);text-align:left;padding:8px 10px;white-space:nowrap}
th{background:var(--panel);position:sticky;top:0}tr[data-date]{cursor:pointer}
tr[data-date]:hover{background:#263349}.bad{color:var(--red)}.warn{color:var(--amber)}
.good{color:var(--green)}.split{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(300px,.7fr);gap:15px}
.detail{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:5px 12px}
.detail div{border-bottom:1px solid var(--line);padding:6px 0;overflow-wrap:anywhere}
.detail b{display:block;color:var(--muted);font-size:.76rem}
@media(max-width:850px){.split{grid-template-columns:1fr}body{padding:12px}}
</style></head><body><main>
<header><div><h1>Alpha Edge · Data Quality Explorer</h1>
<p class="sub">Read-only Bronze / Silver / Gold lineage · Self-contained snapshot</p></div>
<div class="pill" id="acceptance"></div></header>
<div class="panel controls">
<label>Asset <select id="asset"></select></label>
<label>From <input id="from" type="date"></label>
<label>To <input id="to" type="date"></label>
<label>Series <select id="series">
<option value="close">Bronze original close</option>
<option value="adj_close">Bronze adjusted close</option>
<option value="close_raw_usd">Silver USD raw close</option>
<option value="close_adjusted_usd" selected>Silver USD adjusted close</option>
<option value="ret_simple_close_adjusted_usd">Gold simple return</option>
<option value="ret_log_close_adjusted_usd">Gold log return</option>
</select></label>
<label>Observations <select id="filter">
<option value="all">All</option><option value="issues">Quality and return issues</option>
<option value="manual">Manual corrections</option></select></label>
<button id="reset" type="button">Reset dates</button>
<button id="csv" type="button">Export visible rows CSV</button>
</div>
<div class="grid" id="metrics"></div>
<div class="split"><section class="panel"><h2>Price / return series</h2>
<svg id="chart" class="chart" viewBox="0 0 900 300" role="img"
aria-label="Selected asset line chart"><text x="30" y="140">Select an asset</text></svg>
<small>Null or quarantined adjusted prices are displayed as gaps, never interpolated.</small>
</section><section class="panel"><h2>Observation drill-down</h2>
<div id="detail" class="detail"><p>Select a row below.</p></div></section></div>
<section class="panel"><h2>Observed sessions and anomalies</h2>
<div class="scroll"><table><thead><tr>
<th>Date</th><th>Origin</th><th>Original close</th><th>Adjusted close</th>
<th>USD adjusted</th><th>FX rate</th><th>Simple return</th><th>Price status</th>
<th>Return status</th><th>Reason</th></tr></thead><tbody id="rows"></tbody></table></div>
<small id="count"></small></section>
<section class="panel"><h2>Calendar gaps and unexpected sessions</h2>
<div id="calendar"></div></section>
<section class="panel"><h2>FX diagnostics</h2><div id="fx"></div></section>
<section class="panel"><h2>Source / audit metadata</h2><div id="metadata"></div>
<p class="sub" id="limitations"></p></section>
</main>
<script id="snapshot" type="application/json">__DATA__</script>
<script>
"use strict";
const model=JSON.parse(document.getElementById("snapshot").textContent);
const $=id=>document.getElementById(id);
const cells=(tr,values)=>values.forEach(v=>{const td=document.createElement("td");
 td.textContent=(v===null||v===undefined||v==="")?"—":String(v);tr.appendChild(td)});
const fmt=v=>v===null||v===undefined||v===""?"—":
 typeof v==="number"?Number(v.toPrecision(7)).toLocaleString("en-US"):String(v);
const assetMap=new Map(model.assets.map(a=>[a.asset_id,a]));
const chosen=()=>assetMap.get($("asset").value);
function dateText(value){return value ? String(value).slice(0,10) : "";}
function selectedRows(){
 const aid=$("asset").value,start=$("from").value,end=$("to").value,filter=$("filter").value;
 return model.observations.filter(o=>o.asset_id===aid&&
 (!start||dateText(o.session_date)>=start)&&(!end||dateText(o.session_date)<=end)&&
 (filter==="all"||(filter==="manual"?o.observation_origin==="user_supplied_manual":
 o.quality_status!=="accepted"||o.return_quality_status==="review"||
 o.observation_origin==="user_supplied_manual")));
}
function setDates(){
 const a=chosen();$("from").value=a.first;$("to").value=a.last;
}
function metrics(rows){
 const a=chosen(),stats=[
 ["Visible observations",rows.length],["Asset observations",a.observations],
 ["Quarantined",rows.filter(r=>r.quality_status!=="accepted").length],
 ["Manual corrections",rows.filter(r=>r.observation_origin==="user_supplied_manual").length],
 ["Unavailable returns",rows.filter(r=>r.return_quality_status==="unavailable").length],
 ["Calendar issues",model.session_issues.filter(x=>x.asset_id===a.asset_id).length]];
 const parent=$("metrics");parent.replaceChildren();
 for(const [name,value] of stats){const box=document.createElement("div");
 box.className="metric";const strong=document.createElement("strong");
 strong.textContent=value;const small=document.createElement("small");
 small.textContent=name;box.append(strong,small);parent.appendChild(box);}
}
function chart(rows){
 const svg=$("chart");while(svg.firstChild)svg.removeChild(svg.firstChild);
 const key=$("series").value,values=rows.map(r=>r[key]);
 const valid=values.filter(v=>typeof v==="number"&&Number.isFinite(v));
 const ns="http://www.w3.org/2000/svg";
 function node(tag,attrs,text){const el=document.createElementNS(ns,tag);
 for(const [k,v] of Object.entries(attrs||{}))el.setAttribute(k,String(v));
 if(text!==undefined)el.textContent=text;svg.appendChild(el);return el;}
 if(!valid.length){node("text",{x:30,y:140},"No valid values in this view");return;}
 const min=Math.min(...valid),max=Math.max(...valid),span=Math.max(1e-9,max-min);
 const x=i=>45+(i/Math.max(1,rows.length-1))*830;
 const y=v=>260-((v-min)/span)*220;
 node("line",{x1:45,y1:260,x2:880,y2:260,class:"axis"});
 node("text",{x:45,y:20},fmt(max));node("text",{x:45,y:280},fmt(min));
 let part=[];
 function flush(){if(part.length>1)node("path",{d:part.join(" ")});
 else if(part.length===1){const xy=part[0].slice(1).split(" ");
 node("circle",{cx:xy[0],cy:xy[1],r:2.5,fill:"var(--blue)"});}
 part=[];}
 rows.forEach((r,i)=>{const v=values[i];const good=typeof v==="number"&&Number.isFinite(v);
 if(!good){flush();return;}
 part.push((part.length?"L":"M")+x(i).toFixed(2)+" "+y(v).toFixed(2));});
 flush();
 node("text",{x:45,y:297},rows.length?dateText(rows[0].session_date):"");
 node("text",{x:790,y:297},rows.length?dateText(rows[rows.length-1].session_date):"");
}
function detail(r){
 const root=$("detail");root.replaceChildren();
 for(const [key,value] of Object.entries(r)){
 const box=document.createElement("div"),label=document.createElement("b");
 label.textContent=key;box.append(label,document.createTextNode(fmt(value)));
 root.appendChild(box);}
}
function table(rows){
 const root=$("rows");root.replaceChildren();
 for(const r of rows){
 const tr=document.createElement("tr");tr.dataset.date=dateText(r.session_date);
 cells(tr,[dateText(r.session_date),r.observation_origin||"Yahoo",
 fmt(r.close),fmt(r.adj_close),fmt(r.close_adjusted_usd),
 fmt(r.fx_to_usd),fmt(r.ret_simple_close_adjusted_usd),
 r.quality_status,r.return_quality_status,r.quality_reason]);
 tr.addEventListener("click",()=>detail(r));root.appendChild(tr);
 }
 $("count").textContent=rows.length+" displayed observations";
 if(rows.length)detail(rows[0]);else $("detail").textContent="No rows match this filter.";
}
function simpleEvidence(target,items,keys){
 const root=$(target);root.replaceChildren();
 if(!items.length){root.textContent="No flagged rows.";return;}
 const scroll=document.createElement("div");scroll.className="scroll";
 const tbl=document.createElement("table");const header=document.createElement("tr");
 keys.forEach(k=>{const th=document.createElement("th");th.textContent=k;header.appendChild(th)});
 const head=document.createElement("thead");head.appendChild(header);tbl.appendChild(head);
 const tbody=document.createElement("tbody");
 items.forEach(o=>{const tr=document.createElement("tr");cells(tr,keys.map(k=>fmt(o[k])));
 tbody.appendChild(tr)});tbl.appendChild(tbody);scroll.appendChild(tbl);root.appendChild(scroll);
}
function csv(){
 const columns=["session_date",...Object.keys(selectedRows()[0]||{}).filter(k=>k!=="session_date")];
 const rows=selectedRows();const esc=v=>'"'+String(v===null||v===undefined?"":v).replaceAll('"','""')+'"';
 const text=[columns.map(esc).join(","),...rows.map(r=>columns.map(k=>esc(r[k])).join(","))].join("\n");
 const a=document.createElement("a");const url=URL.createObjectURL(new Blob([text],{type:"text/csv;charset=utf-8"}));
 a.href=url;a.download="alpha_edge_"+chosen().symbol.replace(/[^A-Za-z0-9_-]/g,"_")+"_trace.csv";
 a.click();URL.revokeObjectURL(url);
}
function render(){
 const a=chosen(),rows=selectedRows();metrics(rows);chart(rows);table(rows);
 simpleEvidence("calendar",model.session_issues.filter(x=>x.asset_id===a.asset_id),
 ["session_date","issue","calendar_id"]);
 simpleEvidence("fx",model.fx_issues.filter(x=>x.currency_original===a.currency),
 ["fx_date","fx_quote_symbol","quote_close","fx_validation_status"]);
}
model.assets.forEach(a=>{const option=document.createElement("option");
 option.value=a.asset_id;option.textContent=a.symbol+" · "+a.currency;$("asset").appendChild(option)});
$("acceptance").textContent=model.candidate_acceptance||"unknown";
$("metadata").textContent="Manifest SHA-256: "+model.source_candidate_manifest_sha256+
 " · Verified files: "+model.files_verified+" · Candidate: "+model.publication_state+
 " · Manual correction approval: "+model.manual_correction_approval;
$("limitations").textContent=model.limitations;
$("asset").addEventListener("change",()=>{setDates();render()});
for(const id of ["from","to","series","filter"])$(id).addEventListener("change",render);
$("reset").addEventListener("click",()=>{setDates();render()});
$("csv").addEventListener("click",csv);
if(model.assets.length){setDates();render();}
</script></body></html>
"""


def generate_explorer(candidate_dir: Path, output: Path) -> dict:
    model = build_explorer_model(candidate_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite explorer report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    page = HTML_TEMPLATE.replace("__DATA__", _safe_json(model))
    output.write_text(page, encoding="utf-8")
    return {
        "path": str(output), "assets": len(model["assets"]),
        "observations": len(model["observations"]),
        "candidate_acceptance": model["candidate_acceptance"],
        "source_manifest_sha256": model["source_candidate_manifest_sha256"],
        "html_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "mode": "read_only_local_snapshot",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(generate_explorer(args.candidate_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
