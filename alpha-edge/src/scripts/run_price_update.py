from universe import load_universe
from price_update import update_daily_partition
import datetime as dt

universe = load_universe("data/universe.csv")
tickers = list(universe.keys())

today = dt.date.today()
update_daily_partition(tickers, dest_root="data/prices", date=today)