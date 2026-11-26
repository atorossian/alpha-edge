import pandas as pd
import pytest

from alpha_edge.market.split_basis_eligibility_v3 import annotate_candidate


def fixture():
    days=pd.date_range("2020-01-01",periods=7).strftime("%Y-%m-%d")
    raw=pd.DataFrame({"stock_splits":[0,0,0,2,0,0,0]},index=days)
    frame=pd.DataFrame({"candidate_expected_session":True,"candidate_total_return":[None,.01,.01,.01,.01,.01,.01],
                        "historical_quarantine":days<="2020-01-04","production_eligible":False},index=days)
    findings=pd.DataFrame([
        {"effective_date":"2020-01-02","reason":"dividend_factor_mismatch","scope":"prefix"},
        {"effective_date":"2020-01-04","reason":"split_price_basis_unverified","scope":"prefix"},
        {"effective_date":"2020-01-05","reason":"invalid_raw_close","scope":"session"}])
    approvals={"2020-01-04":{"ratio":2.,"bronze_sha256":"hash","symbol":"ABC","approval_version":"v",
                            "large_return_review_dates":["2020-01-04"]}}
    return frame,raw,findings,approvals


def test_only_split_blocker_is_removed_in_counterfactual_view():
    frame,raw,findings,approvals=fixture()
    original=frame.historical_quarantine.copy()
    assert annotate_candidate(frame,raw,findings,approvals,asset_id="a",symbol="ABC",sha="hash")==1
    assert frame.historical_quarantine.equals(original)
    assert frame.candidate_split_basis_remaining_prefix_quarantine.sum()==2
    assert frame.candidate_return_potential_before_split_basis_review.sum()==1
    assert frame.candidate_return_potential_after_split_basis_review.sum()==2
    assert frame.loc["2020-01-04","candidate_split_basis_large_return_review"]
    assert not frame.production_eligible.any()


def test_source_hash_and_exact_ratio_are_required():
    frame,raw,findings,approvals=fixture()
    with pytest.raises(ValueError,match="source mismatch"):
        annotate_candidate(frame,raw,findings,approvals,asset_id="a",symbol="ABC",sha="different")
    raw.loc["2020-01-04","stock_splits"]=3
    with pytest.raises(ValueError,match="source mismatch"):
        annotate_candidate(frame,raw,findings,approvals,asset_id="a",symbol="ABC",sha="hash")
