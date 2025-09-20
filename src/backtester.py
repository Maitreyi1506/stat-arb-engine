"""backtester.py
A small backtesting engine to evaluate simple stat-arb signals.
"""
import numpy as np
import pandas as pd

def zscore(series):
  return (series - series.rolling(60).mean()) / series.rolling(60).std()

def generate_pair_signals(spread, entry_z=2.0, exit_z=0.5):
  """Simple mean-reversion signals based on z-score of spread.
  
  
  Returns a DataFrame signals: long(short) positions on first(second) asset.
  +1 means long first asset, short second asset.
  """
  zs = zscore(spread)
  positions = pd.Series(0, index=spread.index)
  long_cond = zs > entry_z
  short_cond = zs < -entry_z
  exit_cond = abs(zs) < exit_z
  pos = 0
  for t in range(len(zs)):
    if long_cond.iloc[t] and pos == 0:
      pos = -1 # short spread => short first, long second (depends on spread def)
    elif short_cond.iloc[t] and pos == 0:
      pos = 1
    elif exit_cond.iloc[t]:
      pos = 0
    positions.iloc[t] = pos
    
  # signals for both legs (assume dollar-neutral equal weight)
  signals = pd.DataFrame(index=spread.index, columns=['leg1', 'leg2'])
  signals['leg1'] = positions
  signals['leg2'] = -positions
  return signals

def backtest_pair(prices, signals, cost=0.0005):
  """Backtest for two-leg strategy.
  prices: DataFrame with two columns [asset1, asset2]
  signals: DataFrame with columns ['leg1','leg2'] representing position sizes
  cost: proportional transaction cost per trade
  Returns cumulative returns series and metrics.
  """
  rets = prices.pct_change().fillna(0)
  # portfolio returns = sum(position * returns)
  port_rets = (signals.shift(1) * rets).sum(axis=1)
  # simple transaction cost when position changes
  trades = signals.diff().abs().sum(axis=1)
  tc = trades * cost
  port_rets_net = port_rets - tc
  cum_rets = (1 + port_rets_net).cumprod()
  ann_ret = (cum_rets.iloc[-1]) ** (252 / len(cum_rets)) - 1
  ann_vol = port_rets_net.std() * np.sqrt(252)
  sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
  dd = (cum_rets.cummax() - cum_rets).max()
  metrics = {'ann_return': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe, 'max_drawdown': dd}
  return cum_rets, metrics
