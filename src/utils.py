import numpy as np
import pandas as pd

def rolling_zscore(x, window=60):
  return (x - x.rolling(window).mean()) / x.rolling(window).std()
