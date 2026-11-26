import json

from alpha_edge.market.investigate_corporate_actions_v3 import parse_response, payload, select_cases


def fixture():
    url="https://issuer.example/split"
    proposal={"finding":"corroborated","event_type":"split","effective_date":"2020-01-01",
              "new_shares":2,"old_shares":1,"cash_amount":None,"cash_currency":None,
              "recommendation":"retain_reported_ratio","explanation":"Issuer announced two for one.",
              "sources":[{"url":url,"title":"Notice","source_type":"issuer","support":"Effective date and ratio."}]}
    response={"id":"resp_test","model":"model","status":"completed","usage":{"input_tokens":20,"output_tokens":10},
              "output":[{"type":"web_search_call","action":{"sources":[{"url":url}]}},
                        {"type":"message","content":[{"type":"output_text","text":json.dumps(proposal)}]}]}
    case={"symbol":"XYZ","reported_event_date":"2020-01-01","reported_share_ratio":2.0,"exception_category":"missing_independent_event_evidence"}
    return response,case,proposal


def test_retrieved_sources_and_ratio_are_review_only():
    response,case,proposal=fixture()
    result=parse_response(response,case)
    assert result["status"]=="source_linked_ratio_date_match_review"
    assert result["approval_required"] and not result["production_eligible"]
    proposal["new_shares"]=3
    response["output"][1]["content"][0]["text"]=json.dumps(proposal)
    assert parse_response(response,case)["status"]=="source_linked_conflict_review"
    proposal["sources"][0]["url"]="https://not-retrieved.example"
    response["output"][1]["content"][0]["text"]=json.dumps(proposal)
    assert parse_response(response,case)["status"]=="source_provenance_not_confirmed"


def test_pilot_round_robin_and_request_bounds():
    rows=[{"asset_id":a,"date":d,"status":"evidence_exhausted","root_category":"missing_independent_event_evidence"}
          for a,d in [("a","2020-01-01"),("a","2020-01-02"),("b","2020-01-01")]]
    chosen,total=select_cases(rows,"missing_independent_event_evidence",0,2)
    assert total==3
    assert [r["asset_id"] for r in chosen]==["a","b"]
    _,case,_=fixture()
    request=payload(case,"model")
    assert request["max_tool_calls"]==4
    assert request["max_output_tokens"]==3000
    assert request["store"] is False
    assert request["text"]["format"]["strict"]
