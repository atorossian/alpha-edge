from dataclasses import asdict
from report_engine.report_engine import (
    Position,
    build_portfolio_metrics,
    summarize_metrics
)
from storage.data_storage import load_closes_from_folder

prices_folder = "data/prices"  # folder with *_ohlcv.csv
closes = load_closes_from_folder(prices_folder)

# Either build positions manually…
positions = {
    "REGN": Position("REGN", 1, 634.63),
    "GLD":  Position("GLD",  2, 379.32),
    "TLT":  Position("TLT",  3, 89.18),
    "ARKK": Position("ARKK", 4, 76.27),
    "NVO":  Position("NVO",  7, 48.64),
    "QQQ":  Position("QQQ",  1, 604.00),
    "NVDA": Position("NVDA", 1, 187.61),
    "MSFT": Position("MSFT", 1, 510.36),
    "V":    Position("V",    1, 348.41),
    "SMH":  Position("SMH",  3, 343.61),
}

# …or load from a CSV
# positions = load_positions_from_csv("data/positions_today.csv")

equity = 650.79  # you can pass this as a CLI arg or env var in your real flow

metrics = build_portfolio_metrics(closes, positions, equity=equity)
metrics_dict = asdict(metrics)

# text_report = render_daily_report(metrics_dict, style="daily")
# print(text_report)

text_summary = summarize_metrics(metrics)
print(text_summary)