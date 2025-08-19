import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess


# Generates seasonality using fourier
def seasonality_fourier(t, period=50, n_harmonics=5):
    result = np.zeros_like(t, dtype=float)
    for k in range(1, n_harmonics + 1):
        result += (
            np.sin(2 * np.pi * k * t / period) +
            np.cos(2 * np.pi * k * t / period)
        )
    return result


# Generates causal arma process
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

        ar_poly = np.poly(roots).real
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
    df.loc[df.index[start:end], 'value'] = df.loc[df.index[start], 'value']


def load_synthetic_data(csv_file, as_freq=None):
    df = pd.read_csv(csv_file)
    if as_freq:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    ts = df.set_index("timestamp")["value"]
    labels = df.set_index("timestamp")["labels"]

    if as_freq:
        ts = ts.asfreq(as_freq)
        labels = labels.asfreq(as_freq)

    return df, ts, labels


# Generator for damped oscillation series
def gen_damped_oscillator(
    A=1.0, zeta=0.1, omega=2*np.pi, phi=0.0,
    dt=0.01, T=10.0, sigma_meas=0.05,
    start_time="2025-01-01 00:00:00"
):
    rng = np.random.default_rng()
    t = np.arange(0, T, dt)
    omega_d = omega * np.sqrt(1 - zeta**2)

    x_true = A * np.exp(-zeta * omega * t) * np.cos(omega_d * t + phi)
    v_true = A * np.exp(-zeta * omega * t) * (
        -zeta * omega * np.cos(omega_d * t + phi)
        - omega_d * np.sin(omega_d * t + phi)
    )

    y_meas = x_true + rng.normal(0.0, sigma_meas, size=t.size)

    is_anom = np.zeros_like(t, dtype=bool)

    start_dt = pd.Timestamp(start_time)
    timestamps = start_dt + pd.to_timedelta(t, unit="s")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "x_true": x_true,
        "v_true": v_true,
        "value": y_meas,
        "labels": is_anom
    })
    return df
