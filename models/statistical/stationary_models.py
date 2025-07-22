from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf

from scipy.stats import norm

import numpy as np
import math

from visualization.line_plot import *


# Gets (p, q) candidates for ARMA model
def get_candidates(acf_pacf_vals: list, conf_interval: float) -> list:
    candidates = []
    last = 0  # tracks last statistically significant values

    for lag in range(0, len(acf_pacf_vals)):
        if abs(acf_pacf_vals[lag]) > conf_interval:
            candidates.append(lag)
            last = lag
        elif (lag - last) >= 3:  # if 3 consecutive 0 lags, ignore the rest
            break
    return candidates


# Takes in stationary ts and returns (p, q)(P, Q) candidates
def get_acf_pacf(ts: pd.Series, max_lag=10, period=1, alpha=0.05, output=False):
    n = len(ts)

    # Bounded above by n / 3 to ensure there is sufficient data to calculate the acf
    nlags = (period * max_lag) if (period *
                                   max_lag < n / 3) else math.floor(n/3)

    # (1-alpha/2)% confidence for acf and pacf
    conf_interval = norm.ppf(1 - alpha / 2) / np.sqrt(n)

    acf_vals = acf(ts, nlags=nlags, fft=True)
    pacf_vals = pacf(ts, nlags=nlags)

    # Indecies for seasonal lags
    seasonal_lags = [lag for lag in range(
        0, nlags + 1) if lag % period == 0]
    # Seasonal ACF/PACF values based on the indicies
    seasonal_acf_vals = [acf_vals[lag] for lag in seasonal_lags]
    seasonal_pacf_vals = [pacf_vals[lag] for lag in seasonal_lags]

    if output:  # Plot lags
        plot_lag_with_ci(pacf_vals[:max_lag], conf_interval, title='PACF')
        plot_lag_with_ci(acf_vals[:max_lag], conf_interval, title='ACF')
        if period > 1:
            plot_lag_with_ci(seasonal_pacf_vals, conf_interval,
                             title='Seasonal PACF')
            plot_lag_with_ci(seasonal_acf_vals, conf_interval,
                             title='Seasonal ACF')

    # Finds candidates based on the confidence interval
    p_candidates = get_candidates(pacf_vals[:max_lag], conf_interval)
    q_candidates = get_candidates(acf_vals[:max_lag], conf_interval)
    P_candidates = get_candidates(seasonal_pacf_vals, conf_interval)
    Q_candidates = get_candidates(seasonal_acf_vals, conf_interval)

    return p_candidates, q_candidates, P_candidates, Q_candidates


# Gets the best ARIMA / SARIMA model based on the ts
def SARIMA_selection(ts: pd.Series, period=1, cutoff=0.05, output=False):
    # OR has strong seasonal patter (try to detect, to be implemented)
    if period > 1:
        D = 1
        ts_diff = ts.diff(period).dropna()
    else:
        D = 0
        ts_diff = ts

    d = 0
    while adfuller(ts_diff)[1] < cutoff:  # Keep differencing until stationary
        ts_diff = ts_diff.diff().dropna()
        d += 1

    p_candidates, q_candidates, P_candidates, Q_candidates = get_acf_pacf(
        ts_diff, period=(period if D == 1 else 1), output=output)

    best_aic = float('inf')  # Uses AIC for model selection
    best_order = None
    best_model = None

    if D == 0:  # If no seasonal component, use ARIMA model
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
                        plot_fit(ts, model_fit)
                        print(f'Model: ARIMA ({p}, {d}, {q})')
                        print(f'AIC: {aic}')

                except Exception as e:
                    print(f'Cannot fit ARIMA: {e}')
                    continue
    else:  # If there is a seasonal component, use SARIMA model
        for p in p_candidates:
            for q in q_candidates:
                for P in P_candidates:
                    for Q in Q_candidates:
                        try:
                            model = SARIMAX(ts,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, period),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                            model_fit = model.fit()

                            diff = D * period + d
                            fitted_vals = model_fit.fittedvalues[diff:]

                            aic = model_fit.aic
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q, P, D, Q)
                                best_model = model_fit

                            if output:
                                plot_fit(ts, fitted_vals)
                                print(
                                    f'Model: SARIMA ({p}, {d}, {q})({P}, {D}, {Q})')
                                print(f'AIC: {aic}')

                        except Exception as e:
                            print(f'Cannot fit SARIMA: {e}')
                            continue

    return best_model, fitted_vals, best_order
