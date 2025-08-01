import numpy as np
import pandas as pd

from src.visualization.plots import plot_cusum


# Outliers based on standardized residuals
def get_residual_outliers(residuals, threshold=3):
    z_scores = (residuals - residuals.mean()) / residuals.std()
    return np.abs(z_scores) > threshold


# Outliers based on rolling standardized residuals
def rolling_residuals(residuals, window=10):
    rolling_mean = residuals.rolling(window).mean()
    rolling_std = residuals.rolling(window).std()
    z_scores = (residuals - rolling_mean) / rolling_std
    return z_scores


def rolling_variance(ts: pd.Series, window=10):
    rolling_std = ts.rolling(window=window).std()
    return rolling_std


# Outliers based on confidence intervals
def get_prediction_outliers(ts, model_fit, alpha=0.05):
    pred = model_fit.get_prediction(alpha=alpha)
    conf_int = pred.conf_int()
    lower = conf_int.iloc[:, 0]
    upper = conf_int.iloc[:, 1]

    return (ts < lower) | (ts > upper)


# Cumulative Sum
def cusum(ts: pd.Series, mean=None, threshold=None, drift=None, plot=False):
    if not mean:
        mean = ts.mean()
    if not threshold:
        threshold = 4 * ts.std()
    if not drift:
        drift = 0.1 * ts.std()

    pos_sum = 0
    neg_sum = 0
    pos_cusum = []
    neg_cusum = []
    anomalies = pd.Series(False, index=ts.index)

    for i, x in enumerate(ts):
        pos_sum = max(0, pos_sum + x - mean - drift)
        neg_sum = min(0, neg_sum + x - mean + drift)
        pos_cusum.append(pos_sum)
        neg_cusum.append(neg_sum)

        if pos_sum > threshold or neg_sum < -threshold:
            anomalies.iloc[i] = True
            pos_sum = 0
            neg_sum = 0

    if plot:
        cusum_df = pd.DataFrame({
            'positive_cusum': pos_cusum,
            'negative_cusum': neg_cusum
        }, index=ts.index)
        plot_cusum(cusum_df, threshold)

    return anomalies
