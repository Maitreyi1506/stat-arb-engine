# Statistical Arbitrage Engine (Pairs & Triplets Trading)

This repository implements a statistical-arbitrage research pipeline: data ingestion, cointegration identification (pairs & triplets), dynamic hedge ratios (Kalman Filter), signal generation and a simple backtester.


## Quickstart
1. `git clone` this repo
2. `python -m venv venv && source venv/bin/activate` (or use your env)
3. `pip install -r requirements.txt`
4. Run the example: `python examples/run_pairs_example.py`


## What you get
- `src/data_loader.py` : fetch data via `yfinance` (or load local CSVs)
- `src/cointegration.py` : Engle-Granger + Johansen tests
- `src/kalman.py` : Kalman filter for dynamic hedge ratio estimation
- `src/backtester.py` : simple backtesting engine with transaction costs
- `notebooks/` : exploratory notebooks to reproduce analysis and plots


## Data
By default the code pulls daily adjusted close prices via `yfinance`. For production/backtests, use cleaned vendor data.


## Future work
- add walk-forward parameter tuning, transaction cost model, slippage simulation
- scale to intraday LOB data
