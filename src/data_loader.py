"""data_loader.py
Utilities to fetch and prepare OHLC/AdjClose data. Uses yfinance for convenience.
"""
import pandas as pd
import yfinance as yf

def get_data(tickers, start='2018-01-01', end=None, interval='1d'):
"""Download adjusted close prices for a list of tickers.


Returns a DataFrame of shape (dates, tickers).
"""
if isinstance(tickers, str):
tickers = [tickers]
data = yf.download(tickers, start=start, end=end, interval=interval, progress=False)
if 'Adj Close' in data:
df = data['Adj Close'].copy()
else:
# fallback to Close
df = data['Close'].copy()
df.dropna(axis=0, how='any', inplace=True)
return df

def load_local_csv(path):
df = pd.read_csv(path, parse_dates=True, index_col=0)
return df
