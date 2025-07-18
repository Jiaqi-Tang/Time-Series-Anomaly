import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess


def check_causal(ar: np.array) -> bool:
    ar_poly = ar[::-1]
    return not np.any(np.abs(np.roots(ar_poly)) <= 1)


def random_causal_arma(p, q, min_ar=1.2, max_ar=2.0, max_ma=0.5):
    # AR generation
    if p > 0:
        roots = []
        # Creates conjugate roots to ensure real coefficients
        for _ in range(p // 2):
            r = np.random.uniform(min_ar, max_ar)
            theta = np.random.uniform(0, np.pi)
            root = r * np.exp(1j * theta)
            roots.extend([root, np.conj(root)])

        # If p is odd, add one real root
        if p % 2 == 1:
            r = np.random.uniform(min_ar, max_ar)
            roots.append(r)

        ar_poly = np.poly(roots).real  # Ensure it's real-valued
        ar_poly = ar_poly / ar_poly[p]
        ar = ar_poly[::-1]
    else:
        ar = np.array([1])

    # MA generation
    if q > 0:
        ma_coeffs = np.random.uniform(-max_ma, max_ma, size=q)
        ma = np.concatenate(([1.0], ma_coeffs))
    else:
        ma = np.array([1.0])

    arma_process = ArmaProcess(ar, ma)
    return arma_process, ar, ma


def create_seasonality_fourier(t, period=50, n_harmonics=5):
    result = np.zeros_like(t, dtype=float)
    for k in range(1, n_harmonics + 1):
        result += (
            np.sin(2 * np.pi * k * t / period) +
            np.cos(2 * np.pi * k * t / period)
        )
    return result


def create_trend_poly(t, coef=np.array([])):
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


# def get_ts_1():  # Monthly data
#     n = 80

#     t = np.arange(n)
#     dates = pd.date_range("2015-01-01", periods=n, freq='ME')

#     trend = create_trend_poly(t, np.array([200, 0.1]))  # Linear
#     seasonality = create_seasonality_fourier(t, 12)  # Yearly seasonality
#     noise = create_residual_ARMA(n, ma=np.array([1, 0.1]))  # MA(1)

#     ts = pd.Series(trend + seasonality + noise, index=dates)
#     df = pd.DataFrame({'value': ts, 'labels': np.zeros(len(ts), dtype=int)})

#     inject_anomaly(df, 8, 3)
#     inject_anomaly(df, 52, 9, 'noise')
#     return df


# def get_ts_2():  # Yearly
#     n = 40

#     t = np.arange(n)
#     dates = pd.date_range("1980-01-01", periods=n, freq='YE')

#     trend = create_trend_poly(t, np.array([0, -0.05, 0.003]))  # Quadratic
#     noise = create_residual_ARMA(n, ar=np.array(
#         [1, -0.59, 0.17]), ma=np.array([1, 0.2, -0.7]))  # MA(1)

#     ts = pd.Series(trend + noise, index=dates)
#     return ts


# def get_ts_3():  # Hourly
#     n = 2000

#     t = np.arange(n)
#     dates = pd.date_range("2023-05-23 00:00:00", periods=n, freq='h')

#     seasonality = create_seasonality_fourier(t, 12, 20)
#     noise = create_residual_ARMA(
#         n, ar=np.array([1, 0.4]), ma=np.array([1, -0.9]))

#     ts = pd.Series(seasonality + noise, index=dates)
#     return ts
