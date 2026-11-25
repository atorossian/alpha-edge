import pandas as pd
import pytest

from alpha_edge.market.audit_unmatched_split_cases_v3 import classify_cases


def test_classify_counts_only_yahoo_population():
    days = ["2020-01-0" + str(i) for i in range(1, 8)]
    screens = pd.DataFrame([
        {"asset_id": "a" if i < 6 else "historical", "date": d, "ratio": 2.,
         "bronze_sha256": "hash", "group": "integer_forward", "same_day_dividend": False}
        for i, d in enumerate(days)
    ])
    scenarios = [("exact_date", "ratio_match"), ("near_date_candidate", "ratio_match"),
                 ("exact_date", "ratio_mismatch"), ("yahoo_only", "unpaired"),
                 ("outside_overlap", "unpaired"), ("ambiguous_near_date", "unpaired")]
    events = pd.DataFrame([
        {"kind": "split", "asset_id": "a", "yahoo_date": d,
         "eodhd_date": d if i in (0, 2) else "2020-01-03" if i == 1 else None,
         "yahoo_event_rows": 1, "yahoo_sha256": "hash", "yahoo_value": 2.,
         "date_status": ds, "value_status": vs}
        for i, (d, (ds, vs)) in enumerate(zip(days, scenarios))
    ] + [{"kind": "split", "asset_id": "a", "yahoo_event_rows": 0,
          "eodhd_date": days[-1], "date_status": "eodhd_only", "value_status": "unpaired"}])
    rows, matched = classify_cases(screens, events, {"a"}, {"historical"})
    assert matched == 1
    assert len(rows) == 6
    assert [r["reason"] for r in rows] == [
        "near_date_same_ratio", "exact_date_ratio_mismatch", "yahoo_only_within_overlap",
        "outside_price_history_overlap", "ambiguous_near_date", "historical_asset_without_eodhd_coverage"]
    assert rows[0]["date_offset_days"] == 1
    assert all(not r["production_eligible"] for r in rows)
    with pytest.raises(ValueError, match="Missing reconciliation evidence"):
        classify_cases(screens, events, {"a", "historical"}, set())
    events.loc[0, "yahoo_sha256"] = "different"
    with pytest.raises(ValueError, match="Source digest mismatch"):
        classify_cases(screens, events, {"a"}, {"historical"})
