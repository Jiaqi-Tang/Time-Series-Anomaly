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
