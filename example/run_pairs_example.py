"""Run a toy pairs example end-to-end."""
from src.data_loader import get_data
from src.cointegration import find_cointegrated_pairs
from src.kalman import kalman_regression
from src.backtester import generate_pair_signals, backtest_pair

import pandas as pd

def main():
  tickers = ['KO', 'PEP', 'PG', 'CL', 'XOM']
  prices = get_data(tickers, start='2020-01-01')
  # pick two for example
  a, b = 'KO', 'PEP'
  df = prices[[a, b]].dropna()

  # quick cointegration check
  pairs = find_cointegrated_pairs(df)
  print('Cointegrated pairs (p<0.05):', pairs)

  # estimate dynamic hedge ratio
  betas, residuals = kalman_regression(df[a], df[b])
  spread = residuals

  # signals & backtest
  signals = generate_pair_signals(spread)
  cum_rets, metrics = backtest_pair(df[[a, b]], signals)
  print('Metrics:', metrics)
  # plot if wanted

if __name__ == '__main__':
main()
