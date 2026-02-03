import pandas as pd

prev = pd.read_csv("data/universe/.snapshots/universe.prev.csv")
curr = pd.read_csv("data/universe/universe.csv")

for df in (prev, curr):
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["yahoo_ticker"] = df.get("yahoo_ticker", df["ticker"]).astype(str).str.strip().str.upper()
    df["asset_id"] = df.get("asset_id", "").astype(str).str.strip()

p = prev.set_index("ticker")[["yahoo_ticker","asset_id"]]
c = curr.set_index("ticker")[["yahoo_ticker","asset_id"]]

joined = p.join(c, how="outer", lsuffix="_prev", rsuffix="_curr")
changed = joined[
    (joined["yahoo_ticker_prev"] != joined["yahoo_ticker_curr"]) |
    (joined["asset_id_prev"] != joined["asset_id_curr"])
].reset_index()

changed[["ticker","yahoo_ticker_prev","yahoo_ticker_curr","asset_id_prev","asset_id_curr"]].to_csv(
    "data/universe/ingest_force_refresh.csv", index=False
)

print("Wrote force refresh list:", len(changed))