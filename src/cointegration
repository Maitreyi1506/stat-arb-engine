"""cointegration.py
Engle-Granger pair test and Johansen multivariate test wrapper.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def engle_granger_pvalue(y, x):
  """Return p-value for Engle-Granger cointegration test between y and x."""
  score, pvalue, _ = coint(y, x)
  return pvalue

def find_cointegrated_pairs(price_df, pval_threshold=0.05):
  """Scan all pairs in a price DataFrame and return list of (i,j,pval) below threshold."""
  symbols = price_df.columns
  n = len(symbols)
  pairs = []
  for i in range(n):
    for j in range(i+1, n):
      y = price_df.iloc[:, i]
      x = price_df.iloc[:, j]
  try:
    pval = engle_granger_pvalue(y, x)
  except Exception:
    pval = 1.0
  if pval < pval_threshold:
    pairs.append((symbols[i], symbols[j], pval))
  return pairs

def johansen_test(price_df, det_order=0, k_ar_diff=1):
  """Run Johansen cointegration test; returns result object."""
  # price_df columns = assets; use log prices or levels depending on usage
  result = coint_johansen(price_df, det_order, k_ar_diff)
  return result
