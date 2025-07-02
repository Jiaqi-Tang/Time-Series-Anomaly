import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess


def get_ARMA(n=100, ar=np.array([1]), ma=np.array([1])):
    return ArmaProcess(ar, ma).generate_sample(nsample=n)


def get_seasonality(t, period=50, n_harmonics=5):
    result = np.zeros_like(t, dtype=float)
    for k in range(1, n_harmonics + 1):
        result += (
            np.sin(2 * np.pi * k * t / period) +
            np.cos(2 * np.pi * k * t / period)
        )
    return result


def get_poly_trend(t, coef=np.array([])):
    trend = np.zeros_like(t, dtype=float)
    for power, coeff in enumerate(coef):
        trend += coeff * t**power
    return trend


def inject_anomaly(df, start, length, mode='shift', **kwargs):
    end = start + length
    df.loc[df.index[start:end], 'labels'] = 1

    if mode == 'shift':
        df.loc[df.index[start:end], 'value'] += kwargs.get('shift', 5)
    elif mode == 'noise':
        df.loc[df.index[start:end], 'value'] = np.random.normal(
            loc=df['value'].mean(), scale=kwargs.get('scale', 1), size=length)
    elif mode == 'flat':
        df.loc[df.index[start:end], 'value'] = df.iloc[start, 'value']


def get_ts_1():  # Monthly data
    n = 80

    t = np.arange(n)
    dates = pd.date_range("2015-01-01", periods=n, freq='ME')

    trend = get_poly_trend(t, np.array([200, 0.1]))  # Linear
    seasonality = get_seasonality(t, 12)  # Yearly seasonality
    noise = get_ARMA(n, ma=np.array([1, 0.1]))  # MA(1)

    ts = pd.Series(trend + seasonality + noise, index=dates)
    df = pd.DataFrame({'value': ts, 'labels': np.zeros(len(ts), dtype=int)})

    inject_anomaly(df, 8, 3)
    inject_anomaly(df, 52, 9, 'noise')
    return df


def get_ts_2():  # Yearly
    n = 40

    t = np.arange(n)
    dates = pd.date_range("1980-01-01", periods=n, freq='YE')

    trend = get_poly_trend(t, np.array([0, -0.05, 0.003]))  # Quadratic
    noise = get_ARMA(n, ar=np.array(
        [1, -0.59, 0.17]), ma=np.array([1, 0.2, -0.7]))  # MA(1)

    ts = pd.Series(trend + noise, index=dates)
    return ts


def get_ts_3():  # Hourly
    n = 2000

    t = np.arange(n)
    dates = pd.date_range("2023-05-23 00:00:00", periods=n, freq='h')

    seasonality = get_seasonality(t, 12, 20)
    noise = get_ARMA(n, ar=np.array([1, 0.4]), ma=np.array([1, -0.9]))

    ts = pd.Series(seasonality + noise, index=dates)
    return ts
