import pandas as pd

from alpha_edge.portfolio.open_position_barriers import (
    BarrierMonitorConfig,
    OpenLotInput,
    run_barrier_monitor,
)


def _ohlcv():
    return pd.DataFrame(
        [
            {
                "date": "2026-09-01",
                "asset_id": "ETF:ARKK",
                "ticker": "ARKK",
                "high_raw_usd": 100.0,
                "low_raw_usd": 98.0,
                "adj_close_usd": 99.0,
            },
            {
                "date": "2026-09-02",
                "asset_id": "ETF:ARKK",
                "ticker": "ARKK",
                "high_raw_usd": 120.0,
                "low_raw_usd": 110.0,
                "adj_close_usd": 118.0,
            },
            {
                "date": "2026-09-03",
                "asset_id": "ETF:ARKK",
                "ticker": "ARKK",
                "high_raw_usd": 116.0,
                "low_raw_usd": 108.0,
                "adj_close_usd": 109.0,
            },
        ]
    )


def test_profit_lock_can_trigger_exit_for_long():
    lot = OpenLotInput(
        trade_id="t1",
        asset_id="ETF:ARKK",
        ticker="ARKK",
        side="LONG",
        entry_date="2026-09-01",
        entry_price=100.0,
        quantity_remaining=1.0,
        trade_contract={"stop_loss_price": 92.0, "target_profit_price": 115.0},
    )
    latest = pd.DataFrame([{"date": "2026-09-03", "asset_id": "ETF:ARKK", "adj_close_usd": 109.0}])
    indicators = pd.DataFrame([{"date": "2026-09-03", "asset_id": "ETF:ARKK", "atr_14": 2.0, "atr_pct_14": 0.018}])

    result = run_barrier_monitor(
        as_of="2026-09-03",
        open_lots=[lot],
        ohlcv_history=_ohlcv(),
        latest_prices=latest,
        indicators=indicators,
        config=BarrierMonitorConfig(
            profit_lock_activation_return=0.10,
            profit_lock_ratio=0.50,
            chandelier_atr_multiplier=3.0,
            conservative_advisory_mode=False,
            apply_liquidation_throttle=False,
        ),
    )

    row = result.lot_results[0]
    assert round(row.peak_return, 6) == 0.2
    assert row.profit_lock_activated is True
    assert round(row.profit_lock_stop_price, 6) == 110.0
    assert row.stop_triggered is True
    assert row.recommended_action == "EXIT"
    assert result.summary["position_actions"]["EXIT"] == 1


def test_short_uses_low_since_entry_for_profit_lock():
    ohlcv = pd.DataFrame(
        [
            {
                "date": "2026-09-01",
                "asset_id": "CRYPTO:BTC",
                "ticker": "BTC",
                "high_raw_usd": 100.0,
                "low_raw_usd": 98.0,
                "adj_close_usd": 99.0,
            },
            {
                "date": "2026-09-02",
                "asset_id": "CRYPTO:BTC",
                "ticker": "BTC",
                "high_raw_usd": 94.0,
                "low_raw_usd": 80.0,
                "adj_close_usd": 82.0,
            },
            {
                "date": "2026-09-03",
                "asset_id": "CRYPTO:BTC",
                "ticker": "BTC",
                "high_raw_usd": 94.0,
                "low_raw_usd": 89.0,
                "adj_close_usd": 91.0,
            },
        ]
    )
    lot = OpenLotInput(
        trade_id="t2",
        asset_id="CRYPTO:BTC",
        ticker="BTC",
        side="SHORT",
        entry_date="2026-09-01",
        entry_price=100.0,
        value_remaining=1000.0,
        trade_contract={"stop_loss_price": 112.0, "target_profit_price": 85.0},
    )
    latest = pd.DataFrame([{"date": "2026-09-03", "asset_id": "CRYPTO:BTC", "adj_close_usd": 91.0}])
    indicators = pd.DataFrame([{"date": "2026-09-03", "asset_id": "CRYPTO:BTC", "atr_14": 3.0}])

    result = run_barrier_monitor(
        as_of="2026-09-03",
        open_lots=[lot],
        ohlcv_history=ohlcv,
        latest_prices=latest,
        indicators=indicators,
        config=BarrierMonitorConfig(
            profit_lock_activation_return=0.10,
            profit_lock_ratio=0.50,
            conservative_advisory_mode=False,
            apply_liquidation_throttle=False,
            confirmed_break_breach_count=1,
            confirmed_break_exit_fraction=1.0,
        ),
    )

    row = result.lot_results[0]
    assert round(row.peak_return, 6) == 0.25
    assert row.profit_lock_activated is True
    assert round(row.profit_lock_stop_price, 6) == 87.5
    assert row.stop_triggered is True
    assert row.recommended_action == "EXIT"


def test_missing_atr_degrades_but_fixed_barrier_still_works():
    lot = OpenLotInput(
        trade_id="t3",
        asset_id="ETF:IEF",
        ticker="IEF",
        side="LONG",
        entry_date="2026-09-01",
        entry_price=100.0,
        quantity_remaining=2.0,
        trade_contract={"stop_loss_price": 95.0, "target_profit_price": 106.0},
    )
    latest = pd.DataFrame([{"date": "2026-09-03", "asset_id": "ETF:IEF", "adj_close_usd": 104.0}])
    ohlcv = pd.DataFrame(
        [
            {"date": "2026-09-01", "asset_id": "ETF:IEF", "ticker": "IEF", "adj_close_usd": 100.0},
            {"date": "2026-09-03", "asset_id": "ETF:IEF", "ticker": "IEF", "adj_close_usd": 104.0},
        ]
    )

    result = run_barrier_monitor(
        as_of="2026-09-03",
        open_lots=[lot],
        ohlcv_history=ohlcv,
        latest_prices=latest,
        indicators=pd.DataFrame(),
    )
    row = result.lot_results[0]
    assert row.barrier_quality == "degraded"
    assert "atr_14_missing" in row.warnings
    assert row.raw_active_stop_price == 95.0
    assert row.active_stop_price < row.raw_active_stop_price
    assert row.recommended_action == "WATCH"


def test_liquidity_aware_wick_breach_recovers_to_watch_not_exit():
    lot = OpenLotInput(
        trade_id="t4",
        asset_id="ETF:SMH",
        ticker="SMH",
        side="LONG",
        entry_date="2026-09-01",
        entry_price=100.0,
        trade_contract={"stop_loss_price": 95.0, "target_profit_price": 120.0},
    )
    ohlcv = pd.DataFrame(
        [
            {
                "date": "2026-09-01",
                "asset_id": "ETF:SMH",
                "ticker": "SMH",
                "open_raw_usd": 100.0,
                "high_raw_usd": 105.0,
                "low_raw_usd": 99.0,
                "adj_close_usd": 104.0,
            },
            {
                "date": "2026-09-02",
                "asset_id": "ETF:SMH",
                "ticker": "SMH",
                "open_raw_usd": 100.0,
                "high_raw_usd": 102.0,
                "low_raw_usd": 94.5,
                "adj_close_usd": 98.0,
            },
        ]
    )
    latest = pd.DataFrame([{"date": "2026-09-02", "asset_id": "ETF:SMH", "adj_close_usd": 98.0}])
    indicators = pd.DataFrame([{"date": "2026-09-02", "asset_id": "ETF:SMH", "atr_14": 2.0}])
    result = run_barrier_monitor(
        as_of="2026-09-02",
        open_lots=[lot],
        ohlcv_history=ohlcv,
        latest_prices=latest,
        indicators=indicators,
        config=BarrierMonitorConfig(stop_buffer_pct=0.0, stop_buffer_atr_multiplier=0.25, use_chandelier_stop=False),
    )
    row = result.lot_results[0]
    assert row.breach_type == "WICK_BREACH_RECOVERED"
    assert row.liquidity_sweep_recovered is True
    assert row.recommended_action == "WATCH"
    assert row.stop_triggered is False
    assert row.breach_count == 0


def test_second_close_breach_becomes_confirmed_exit_with_state():
    lot = OpenLotInput(
        trade_id="t5",
        asset_id="ETF:ARKK",
        ticker="ARKK",
        side="LONG",
        entry_date="2026-09-01",
        entry_price=100.0,
        trade_contract={"stop_loss_price": 95.0, "target_profit_price": 120.0},
    )
    ohlcv = pd.DataFrame(
        [
            {
                "date": "2026-09-01",
                "asset_id": "ETF:ARKK",
                "open_raw_usd": 100.0,
                "high_raw_usd": 101.0,
                "low_raw_usd": 99.0,
                "adj_close_usd": 100.0,
            },
            {
                "date": "2026-09-03",
                "asset_id": "ETF:ARKK",
                "open_raw_usd": 94.8,
                "high_raw_usd": 95.2,
                "low_raw_usd": 94.0,
                "adj_close_usd": 94.9,
            },
        ]
    )
    latest = pd.DataFrame([{"date": "2026-09-03", "asset_id": "ETF:ARKK", "adj_close_usd": 94.9}])
    indicators = pd.DataFrame([{"date": "2026-09-03", "asset_id": "ETF:ARKK", "atr_14": 2.0}])
    prev = {
        "lot_state": {
            "t5|ETF:ARKK|LONG": {
                "state_key": "t5|ETF:ARKK|LONG",
                "trade_id": "t5",
                "asset_id": "ETF:ARKK",
                "side": "LONG",
                "breach_count": 1,
            }
        }
    }
    result = run_barrier_monitor(
        as_of="2026-09-03",
        open_lots=[lot],
        ohlcv_history=ohlcv,
        latest_prices=latest,
        indicators=indicators,
        previous_state=prev,
        config=BarrierMonitorConfig(
            stop_buffer_pct=0.0,
            stop_buffer_atr_multiplier=0.25,
            confirmed_break_breach_count=2,
            conservative_advisory_mode=False,
            apply_liquidation_throttle=False,
            confirmed_break_exit_fraction=1.0,
        ),
    )
    row = result.lot_results[0]
    assert row.breach_type == "CONFIRMED_BREAK"
    assert row.confirmation_status == "CONFIRMED"
    assert row.recommended_action == "EXIT"
    assert row.recommended_exit_fraction == 1.0
    assert row.breach_count == 2


def test_asset_class_override_tightens_crypto_buffer():
    lot = OpenLotInput(
        trade_id="t6",
        asset_id="CRYPTO:BTC-USD",
        ticker="BTC-USD",
        side="LONG",
        entry_date="2026-09-01",
        entry_price=100.0,
        trade_contract={"stop_loss_price": 90.0, "target_profit_price": 130.0},
    )
    ohlcv = pd.DataFrame(
        [
            {
                "date": "2026-09-02",
                "asset_id": "CRYPTO:BTC-USD",
                "open_raw_usd": 100.0,
                "high_raw_usd": 102.0,
                "low_raw_usd": 98.0,
                "adj_close_usd": 100.0,
            }
        ]
    )
    latest = pd.DataFrame([{"date": "2026-09-02", "asset_id": "CRYPTO:BTC-USD", "adj_close_usd": 100.0}])
    result = run_barrier_monitor(
        as_of="2026-09-02",
        open_lots=[lot],
        ohlcv_history=ohlcv,
        latest_prices=latest,
        indicators=pd.DataFrame(),
        config=BarrierMonitorConfig(stop_buffer_pct=0.003, asset_class_overrides={"crypto": {"stop_buffer_pct": 0.02}}),
    )
    row = result.lot_results[0]
    assert round(row.stop_buffer_amount, 6) == 2.0
    assert round(row.execution_trigger_price, 6) == 88.0


def test_v1_conservative_time_exit_defaults_to_watch():
    lot = OpenLotInput(
        trade_id="t7",
        asset_id="ETF:VT",
        ticker="VT",
        side="LONG",
        entry_date="2026-01-01",
        entry_price=100.0,
        quantity_remaining=10.0,
        trade_contract={"stop_loss_price": 90.0, "target_profit_price": 130.0, "expiry_date": "2026-02-01"},
    )
    ohlcv = pd.DataFrame(
        [
            {
                "date": "2026-09-03",
                "asset_id": "ETF:VT",
                "open_raw_usd": 110.0,
                "high_raw_usd": 112.0,
                "low_raw_usd": 109.0,
                "adj_close_usd": 111.0,
            }
        ]
    )
    latest = pd.DataFrame([{"date": "2026-09-03", "asset_id": "ETF:VT", "adj_close_usd": 111.0}])
    result = run_barrier_monitor(
        as_of="2026-09-03",
        open_lots=[lot],
        ohlcv_history=ohlcv,
        latest_prices=latest,
        indicators=pd.DataFrame([{"date": "2026-09-03", "asset_id": "ETF:VT", "atr_14": 2.0}]),
        config=BarrierMonitorConfig(apply_liquidation_throttle=False),
    )
    row = result.position_results[0]
    assert row.recommended_action == "WATCH"
    assert row.recommended_exit_fraction == 0.0
    assert row.trigger_reason == "time_exit_due"


def test_liquidation_throttle_defers_lower_priority_actions():
    lots = [
        OpenLotInput(
            trade_id="ta",
            asset_id="ETF:A",
            ticker="A",
            side="LONG",
            entry_date="2026-09-01",
            entry_price=100.0,
            quantity_remaining=10.0,
            trade_contract={"stop_loss_price": 95.0, "target_profit_price": 120.0},
        ),
        OpenLotInput(
            trade_id="tb",
            asset_id="ETF:B",
            ticker="B",
            side="LONG",
            entry_date="2026-09-01",
            entry_price=100.0,
            quantity_remaining=10.0,
            trade_contract={"stop_loss_price": 95.0, "target_profit_price": 120.0},
        ),
    ]
    ohlcv = pd.DataFrame(
        [
            {
                "date": "2026-09-03",
                "asset_id": "ETF:A",
                "open_raw_usd": 90.0,
                "high_raw_usd": 91.0,
                "low_raw_usd": 89.0,
                "adj_close_usd": 90.0,
            },
            {
                "date": "2026-09-03",
                "asset_id": "ETF:B",
                "open_raw_usd": 90.0,
                "high_raw_usd": 91.0,
                "low_raw_usd": 89.0,
                "adj_close_usd": 90.0,
            },
        ]
    )
    latest = pd.DataFrame(
        [
            {"date": "2026-09-03", "asset_id": "ETF:A", "adj_close_usd": 90.0},
            {"date": "2026-09-03", "asset_id": "ETF:B", "adj_close_usd": 90.0},
        ]
    )
    ind = pd.DataFrame(
        [
            {"date": "2026-09-03", "asset_id": "ETF:A", "atr_14": 1.0},
            {"date": "2026-09-03", "asset_id": "ETF:B", "atr_14": 1.0},
        ]
    )
    result = run_barrier_monitor(
        as_of="2026-09-03",
        open_lots=lots,
        ohlcv_history=ohlcv,
        latest_prices=latest,
        indicators=ind,
        config=BarrierMonitorConfig(
            max_full_exits_per_day=1, max_daily_exit_fraction_of_portfolio=1.0, conservative_advisory_mode=False
        ),
    )
    actions = [r.recommended_action for r in result.position_results]
    assert actions.count("EXIT") == 1
    assert actions.count("WATCH") == 1
    assert result.summary["liquidation_throttle"]["deferred_actions"] == 1
