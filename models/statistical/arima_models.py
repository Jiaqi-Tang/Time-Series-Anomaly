from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf, pacf

import numpy as np

from visualization.line_plot import *


def get_candidates(acf_vals, conf_interval):
    candidates = []
    last = 0

    for lag in range(1, len(acf_vals)):
        if abs(acf_vals[lag]) > conf_interval:
            candidates.append(lag)
            last = lag
        elif (lag - last) >= 3:  # if 3 consecutive 0 lags, ignore the rest
            break
    return candidates


def ARMA_pq_selection(ts, max_lag=10, CI_cutoff=0.05, output=False):
    n = len(ts)

    conf_interval = 1.96 / np.sqrt(n)  # Approx 95% confidence for acf and pacf

    acf_vals = acf(ts, nlags=max_lag, fft=True)
    pacf_vals = pacf(ts, nlags=max_lag)

    plot_lag_with_ci(acf_vals, conf_interval, title='ACF')
    plot_lag_with_ci(pacf_vals, conf_interval, title='PACF')

    p_candidates = get_candidates(pacf_vals, conf_interval)
    q_candidates = get_candidates(acf_vals, conf_interval)

    return p_candidates, q_candidates


def ARIMA_pdq_selection(ts, period=1, cutoff=0.05, output=False):
    ts_diff = ts
    d = 0

    # while adfuller(ts_diff)[1] >= cutoff:
    #     ts_diff = ts_diff.diff()
    #     d += 1

    p_candidates, q_candidates = ARMA_pq_selection(ts_diff)

    best_aic = float('inf')
    best_order = None
    best_model = None

    for p in p_candidates:
        for q in q_candidates:
            try:
                model = ARIMA(ts, order=(p, d, q))
                model_fit = model.fit()
                aic = model_fit.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_model = model_fit

                if output:
                    plot_arima_fit(ts, model_fit)
                    print(f'Model: ARIMA ({p}, {d}, {q})')
                    print(f'AIC: {aic}')

            except Exception as e:
                print(f'Cannot fit ARIMA: {e}')
                continue

    return best_order, best_model


def SARIMA_residual_model(ts, p, d, q, P, D, Q, s):
    model = SARIMAX(ts, order=(p, d, q), seasonal_order=(P, D, Q, s))
    results = model.fit()
    fitted_vals = results.fittedvalues
    residuals = ts - fitted_vals
    z_scores = (residuals - residuals.mean()) / residuals.std()
    anomalies = np.abs(z_scores) > 3

    pred = results.get_prediction()
    conf_int = pred.conf_int()
    lower = conf_int.iloc[:, 0]
    upper = conf_int.iloc[:, 1]

    anomalies = (ts < lower) | (ts > upper)
