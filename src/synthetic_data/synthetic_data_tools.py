import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess


def seasonality_fourier(t, period=50, n_harmonics=5):
    result = np.zeros_like(t, dtype=float)
    for k in range(1, n_harmonics + 1):
        result += (
            np.sin(2 * np.pi * k * t / period) +
            np.cos(2 * np.pi * k * t / period)
        )
    return result


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


def inject_shift(df, start, end, shift=5):
    assert (end >= start)
    df.loc[df.index[start:end], 'labels'] = True
    df.loc[df.index[start:end], 'value'] += shift


def inject_noise(df, start, end, mean=0, scale=5):
    assert (end >= start)
    df.loc[df.index[start:end], 'labels'] = True
    df.loc[df.index[start:end], 'value'] = np.random.normal(
        loc=mean, scale=scale, size=(end-start))


def inject_flat(df, start, end):
    assert (end >= start)
    df.loc[df.index[start:end], 'labels'] = True
    df.loc[df.index[start:end], 'value'] = df.iloc[start, 'value']


def load_synthetic_data(csv_file):
    df = pd.read_csv('../data/short_seasonal.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    ts = df.set_index("timestamp")["value"]
    ts = ts.asfreq('ME')

    lables = df.set_index("timestamp")["labels"]
    lables = lables.asfreq('ME')
    return df, ts, lables
