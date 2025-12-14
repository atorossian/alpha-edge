import requests
from bs4 import BeautifulSoup
import pandas as pd

root_url = 'https://help.quantfury.com/en/articles/5448760-etfs'
response = requests.get(root_url)
html = response.text
soup = BeautifulSoup(html, 'html.parser')

# Get the first (and in this case only) tbody
tbody = soup.find("tbody")

rows = []
for tr in tbody.find_all("tr"):
    cols = [td.get_text(strip=True) for td in tr.find_all("td")]
    if not cols:
        continue
    rows.append(cols)

# First row is the header: ['ETF', 'Exchange', 'Ticker']
header = rows[0]
data_rows = rows[1:]

df = pd.DataFrame(data_rows, columns=header)

universe_df = pd.DataFrame({
    "ticker": df["Ticker"],
    "name": df["ETF"],
    "asset_class": "equity",
    "role": "etf",
    "region": df["Exchange"],   # or map to US/Global/etc.
    "max_weight": 0.25,
    "min_weight": 0.0,
    "include": 1,
})
universe = pd.read_csv('data/universe/universe.csv')
universe = pd.concat([universe, universe_df], ignore_index=True)
universe.drop_duplicates(subset=['ticker'], keep='last', inplace=True)
universe.to_csv('data/universe/universe.csv', index=False)
