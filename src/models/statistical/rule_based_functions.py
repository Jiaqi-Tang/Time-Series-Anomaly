import numpy as np
import pandas as pd


# Outliers based on standardized residuals
def get_residual_outliers(residuals, threshold=3):
    z_scores = (residuals - residuals.mean()) / residuals.std()
    return np.abs(z_scores) > threshold


# Outliers based on rolling standardized residuals
def get_rolling_residual_outliers(residuals, threshold=3, window=10):
    rolling_mean = residuals.rolling(window).mean()
    rolling_std = residuals.rolling(window).std()
    z_scores = (residuals - rolling_mean) / rolling_std
    return np.abs(z_scores) > threshold


# Outliers based on confidence intervals
def get_prediction_outliers(ts, model_fit, alpha=0.05):
    pred = model_fit.get_prediction(alpha=alpha)
    conf_int = pred.conf_int()
    lower = conf_int.iloc[:, 0]
    upper = conf_int.iloc[:, 1]

    return (ts < lower) | (ts > upper)


# Cumulative Sum
def cusum(ts, mean=0, threshold=5, drift=0.02):
    pos_sum = 0
    neg_sum = 0

    anomalies = pd.Series(False, index=ts.index)

    for i, x in enumerate(ts):
        pos_sum = max(0, pos_sum + x - mean - drift)
        neg_sum = min(0, neg_sum + x - mean + drift)
        if pos_sum > threshold or neg_sum < -threshold:
            anomalies.iloc[i] = True
            pos_sum = 0
            neg_sum = 0
    return anomalies
