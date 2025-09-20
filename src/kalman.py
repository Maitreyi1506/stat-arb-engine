"""kalman.py
Simple Kalman Filter to estimate dynamic hedge ratio between two series.
Based on standard linear regression-with-time-varying-coefs Kalman implementation.
"""
import numpy as np
import pandas as pd

def kalman_regression(y, x, delta=1e-5, Ve=1.0):
"""Estimate time-varying beta_0 and beta_1 for y = beta0 + beta1 * x + eps.


Returns betas_df (index aligned with x), residuals series.
delta controls state covariance (smaller -> slower adaptation).
Ve is observation noise variance.
"""
n = len(x)
# design matrix: [1, x_t]
X = np.vstack([np.ones(n), x.values]).T

# state: beta (2x1)
beta = np.zeros(2)
P = np.eye(2) * 1.0
# state covariance
W = np.eye(2) * delta

betas = np.zeros((n, 2))
residuals = np.zeros(n)

for t in range(n):
Xt = X[t:t+1]
# prediction step (beta_t|t-1 = beta_{t-1})
beta_pred = beta
P_pred = P + W

# observation variance
yt = y.values[t]
Ft = Xt @ P_pred @ Xt.T + Ve
Kt = (P_pred @ Xt.T) / Ft # Kalman gain (2x1)

# update
e_t = yt - Xt @ beta_pred
beta = beta_pred + (Kt.flatten() * e_t).flatten()
P = P_pred - Kt @ Xt @ P_pred

betas[t, :] = beta
residuals[t] = e_t

betas_df = pd.DataFrame(betas, index=x.index, columns=['beta0', 'beta1'])
residuals = pd.Series(residuals, index=x.index)
return betas_df, residuals
