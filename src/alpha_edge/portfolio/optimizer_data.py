from alpha_edge.universe.universe import load_universe
from alpha_edge.core.data_loader import load_closes_from_folder
from alpha_edge import paths

def main():
    universe = load_universe(paths.universe_dir() / "universe.csv")
    universe_tickers = list(universe.keys())

    closes = load_closes_from_folder(paths.prices_dir(), universe_tickers)

    print("Universe tickers:", universe_tickers)
    print("Tickers with data:", list(closes.columns))
    print(closes.tail())

if __name__ == "__main__":
    main()
