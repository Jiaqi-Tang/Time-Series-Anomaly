import numpy as np

from data.synthetic.tools.residuals import *
from data.synthetic.tools.seasonality import *
from data.synthetic.tools.trend import *


def get_ts_1():  # Monthly data
    n = 80

    t = np.arange(n)
    dates = pd.date_range("2015-01-01", periods=n, freq='ME')

    trend = create_trend_poly(t, np.array([200, 0.1]))  # Linear
    seasonality = create_seasonality_fourier(t, 12)  # Yearly seasonality
    noise, _, _ = random_causal_arma(0, 1)  # MA(1)
    noise = noise.generate_sample(nsample=n)

    ts = pd.Series(trend + seasonality + noise, index=dates)
    df = pd.DataFrame({'value': ts, 'labels': np.zeros(len(ts), dtype=int)})

    inject_anomaly(df, 8, 3)
    inject_anomaly(df, 52, 9, 'noise')
    return df


def get_ts_2():  # Yearly
    n = 40

    t = np.arange(n)
    dates = pd.date_range("1980-01-01", periods=n, freq='YE')

    trend = create_trend_poly(t, np.array([0, -0.05, 0.003]))  # Quadratic
    noise, _, _ = random_causal_arma(2, 2)
    noise = noise.generate_sample(nsample=n)

    ts = pd.Series(trend + noise, index=dates)
    return ts


def get_ts_3():  # Hourly
    n = 2000

    t = np.arange(n)
    dates = pd.date_range("2023-05-23 00:00:00", periods=n, freq='h')

    seasonality = create_seasonality_fourier(t, 12, 20)
    noise, _, _ = random_causal_arma(1, 1)
    noise = noise.generate_sample(nsample=n)

    ts = pd.Series(seasonality + noise, index=dates)
    return ts
