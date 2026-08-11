from __future__ import annotations

import pandas as pd

from alpha_edge.jobs import run_quarantine_analysis as q


def test_quarantine_discovery_skips_rejected_final_executable(monkeypatch):
    key = "engine/v1/portfolio_search/runs/dt=2026-06-22/run_rejected.json"
    payload = {
        "run_id": "run_rejected",
        "as_of": "2026-06-22",
        "inputs": {"equity0": 32000, "target_notional": 100000, "goals": [35000, 37000, 40000], "main_goal": 35000},
        "outputs": {
            "candidate_context": {
                "equity0": 32000,
                "target_notional": 100000,
                "goals": [35000, 37000, 40000],
                "main_goal": 35000,
            },
            "final_executable": {"status": "rejected", "metrics": {"score": -0.1}},
            "discrete_allocation": {"shares": {"AAA": 10, "BBB": -5}},
        },
    }
    monkeypatch.setattr(q, "_s3_list_keys", lambda *args, **kwargs: [key])
    monkeypatch.setattr(q, "s3_get_json", lambda *args, **kwargs: payload)

    out = q._discover_candidates_from_portfolio_runs(
        s3=object(),
        bucket="bucket",
        root_prefix="engine/v1",
        as_of_ts=pd.Timestamp("2026-06-22"),
        lookback_days=0,
    )
    assert out == []


def test_quarantine_discovery_uses_accepted_final_executable(monkeypatch):
    key = "engine/v1/portfolio_search/runs/dt=2026-06-22/run_accepted.json"
    payload = {
        "run_id": "run_accepted",
        "as_of": "2026-06-22",
        "inputs": {"equity0": 32000, "target_notional": 100000, "goals": [35000, 37000, 40000], "main_goal": 35000},
        "outputs": {
            "candidate_context": {
                "equity0": 32000,
                "target_notional": 100000,
                "goals": [35000, 37000, 40000],
                "main_goal": 35000,
            },
            "final_executable": {
                "status": "accepted",
                "selected_candidate_label": "ga_best",
                "metrics": {"score": 0.58, "ruin_prob_1y": 0.0},
            },
            "discrete_allocation": {"shares": {"AAA": 10, "BBB": -5}},
        },
    }
    monkeypatch.setattr(q, "_s3_list_keys", lambda *args, **kwargs: [key])
    monkeypatch.setattr(q, "s3_get_json", lambda *args, **kwargs: payload)

    out = q._discover_candidates_from_portfolio_runs(
        s3=object(),
        bucket="bucket",
        root_prefix="engine/v1",
        as_of_ts=pd.Timestamp("2026-06-22"),
        lookback_days=0,
    )
    assert len(out) == 1
    assert out[0]["baseline_search_eval"]["score"] == 0.58
    assert out[0]["source"]["portfolio_output"] == "final_executable"
    assert out[0]["source"]["final_executable_status"] == "accepted"
