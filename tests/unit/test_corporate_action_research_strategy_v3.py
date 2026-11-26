import json

from alpha_edge.market import corporate_action_research_strategy_v3 as strategy
from alpha_edge.market.investigate_corporate_actions_v3 import https_url, select_cases


def fixture():
    uri="https://issuer.example/history"
    proposal={"finding":"corroborated","issuer_name":"Issuer","historical_names":["Old name"],
              "historical_symbols":[],"event_type":"split","effective_date":"2020-01-01",
              "date_kind":"trading_ex_date","share_ratio":2.,"distribution_security":None,
              "distributed_shares_per_parent_share":None,"price_adjustment_coefficient":None,
              "cash_amount":None,"cash_currency":None,"explanation":"Historical terms.",
              "sources":[{"url":uri,"title":"Notice","source_type":"issuer",
                          "supports":["identity","event_type","effective_date","share_ratio"],"support":"Terms."}]}
    response={"id":"resp_test","status":"completed","output":[
        {"type":"web_search_call","action":{"sources":[{"url":uri}]}},
        {"type":"message","content":[{"type":"output_text","text":json.dumps(proposal)}]}]}
    case={"reported_event_date":"2020-01-01","reported_share_ratio":2.}
    return response,case,proposal


def parse(response,case,proposal):
    response["output"][1]["content"][0]["text"]=json.dumps(proposal)
    return strategy.parse_response(response,case,https_url)


def test_fact_level_provenance_retains_valid_sources_without_claiming_unverified_ones():
    response,case,p=fixture()
    p["sources"].append({"url":"https://www.issuer.example/history","title":"Unconfirmed",
                         "source_type":"secondary","supports":["share_ratio"],"support":"Terms."})
    result=parse(response,case,p)
    assert result["status"]=="source_linked_ratio_date_match_review"
    assert len(result["unconfirmed_sources"])==1
    assert not result["production_eligible"]
    p["sources"][0]["supports"].remove("effective_date")
    assert parse(response,case,p)["status"]=="source_linked_partial_terms_review"


def test_complex_entitlement_is_not_parent_share_ratio():
    response,case,p=fixture()
    p.update(event_type="spin_off",finding="related_action",share_ratio=None,
             distributed_shares_per_parent_share=.4,distribution_security="Child")
    result=parse(response,case,p)
    assert result["status"]=="source_linked_complex_action_review"
    assert "reported_ratio_matches" not in result


def test_prior_context_and_extended_batch_limit():
    response,case,p=fixture()
    context=strategy.context_for_asset([{"case":{"asset_id":"a","date":"2020-01-01"},
        "outcome":{"proposal":p,"retrieved_urls":[p["sources"][0]["url"]]}}],"a")
    assert context[0]["historical_names"]==["Old name"]
    assert strategy.payload(case,"model",context)["max_tool_calls"]==6
    rows=[{"asset_id":str(i),"date":"2020-01-01","status":"evidence_exhausted",
           "root_category":"missing_independent_event_evidence"} for i in range(137)]
    assert len(select_cases(rows,"all",0,137,enhanced=True)[0])==137
