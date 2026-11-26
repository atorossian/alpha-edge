import json

from alpha_edge.market.audit_v2_reconciliation_review_v3 import candidate_flags, residual_flags


def test_candidate_flags_distinguish_target_from_other_sessions():
    choice = {"status":"advanced_session_factor_candidate","large_return_review_required":True,
              "cash_raw_basis_review_required":True,"daily_checks":[
                  {"date":"2020-01-01","large_return_review_required":True,"cash_evidence":"reported_adjusted"},
                  {"date":"2020-01-02","large_return_review_required":False,"cash_evidence":"none"}]}
    row={"date":"2020-01-02","chosen_hypothesis":json.dumps(choice),
         "attempts":json.dumps([choice,choice]),"prior_factors":"[]"}
    flags=candidate_flags(row)
    assert flags["large_return_any_window"] and not flags["large_return_target"]
    assert flags["raw_cash_basis_any_window"] and not flags["raw_cash_basis_target"]
    assert flags["passing_windows"] == 2


def test_residual_field_and_cash_evidence_are_not_approvals():
    attempt={"method":"daily_factor_chain","status":"advanced_field_scale_disagrees",
             "field_max_relative_error":{"open":.001,"high":.02,"low":.001,"close":.001},
             "four_decimal_rounding_compatible":False}
    row={"attempts":json.dumps([attempt,attempt,{"status":"cash_amount_or_share_basis_disagreement",
                                              "yahoo_cash":1.,"eodhd_adjusted_cash":1.}])}
    flags=residual_flags(row)
    assert flags["failed_fields"] == ["high"]
    assert flags["isolated_field_hypotheses"] == ["high"]
    assert flags["adjusted_cash_matches_yahoo"]
    assert not flags["rounding_compatible_window"]
