from __future__ import annotations

import json

from alpha_edge.core.market_store import MarketStore


def test_read_market_hmm_regime_history(monkeypatch) -> None:
    store = MarketStore(bucket="test-bucket", base_prefix="engine/v1", region="eu-west-1")

    keys = [
        "engine/v1/regimes/market_hmm/dt=2024-01-02/regime.json",
        "engine/v1/regimes/market_hmm/dt=2024-01-03/regime.json",
    ]

    payloads = {
        keys[0]: {
            "as_of": "2024-01-02",
            "hmm": {
                "label_commit": "CALM_BULL",
                "p_label_today": {
                    "CALM_BULL": 0.80,
                    "CHOPPY_BULL": 0.10,
                    "CHOPPY_BEAR": 0.05,
                    "STRESS_BEAR": 0.05,
                },
                "meta": {"point_in_time": True, "lookahead_safe": True},
            },
            "leverage_recommendation": {"leverage": 2.0},
            "meta": {"point_in_time": True, "lookahead_safe": True},
        },
        keys[1]: {
            "as_of": "2024-01-03",
            "hmm": {
                "label_commit": None,
                "p_label_today": {
                    "CALM_BULL": 0.30,
                    "CHOPPY_BULL": 0.30,
                    "CHOPPY_BEAR": 0.25,
                    "STRESS_BEAR": 0.15,
                },
                "meta": {"point_in_time": True, "lookahead_safe": True},
            },
            "leverage_recommendation": {"leverage": 1.0},
            "meta": {"point_in_time": True, "lookahead_safe": True},
        },
    }

    monkeypatch.setattr(store, "_list_keys", lambda prefix: keys)
    monkeypatch.setattr(store, "_get_bytes", lambda key: json.dumps(payloads[key]).encode("utf-8"))

    out = store.read_market_hmm_regime_history(start="2024-01-01", end="2024-01-31")

    assert out.shape[0] == 2

    first = out.iloc[0]
    second = out.iloc[1]

    assert first["as_of"] == "2024-01-02"
    assert first["label"] == "CALM_BULL"
    assert first["label_or_mixed"] == "CALM_BULL"
    assert first["regime"] == "CALM_BULL"
    assert first["confidence"] == 0.80
    assert first["target_leverage"] == 2.0
    assert bool(first["point_in_time"]) is True
    assert bool(first["lookahead_safe"]) is True

    assert second["as_of"] == "2024-01-03"
    assert second["label"] is None
    assert second["label_or_mixed"] == "MIXED"
    assert second["regime"] == "MIXED"
    assert second["confidence"] == 0.30
    assert second["target_leverage"] == 1.0


def test_read_market_hmm_regime_history_can_exclude_mixed(monkeypatch) -> None:
    store = MarketStore(bucket="test-bucket", base_prefix="engine/v1", region="eu-west-1")

    keys = [
        "engine/v1/regimes/market_hmm/dt=2024-01-02/regime.json",
        "engine/v1/regimes/market_hmm/dt=2024-01-03/regime.json",
    ]

    payloads = {
        keys[0]: {
            "as_of": "2024-01-02",
            "hmm": {
                "label_commit": "CALM_BULL",
                "p_label_today": {"CALM_BULL": 0.80},
            },
            "leverage_recommendation": {"leverage": 2.0},
            "meta": {"point_in_time": True, "lookahead_safe": True},
        },
        keys[1]: {
            "as_of": "2024-01-03",
            "hmm": {
                "label_commit": None,
                "p_label_today": {"CALM_BULL": 0.40, "CHOPPY_BULL": 0.35},
            },
            "leverage_recommendation": {"leverage": 1.0},
            "meta": {"point_in_time": True, "lookahead_safe": True},
        },
    }

    monkeypatch.setattr(store, "_list_keys", lambda prefix: keys)
    monkeypatch.setattr(store, "_get_bytes", lambda key: json.dumps(payloads[key]).encode("utf-8"))

    out = store.read_market_hmm_regime_history(include_mixed=False)

    assert out.shape[0] == 1
    assert out.iloc[0]["regime"] == "CALM_BULL"
