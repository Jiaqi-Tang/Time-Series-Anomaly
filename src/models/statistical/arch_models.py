from arch import arch_model
import numpy as np
import pandas as pd
import warnings


# Uses GARCH Z-scores for anomaly detection
def garch_anomalies(resid: pd.Series, threshold: float = 3.0):
    model = arch_model(resid, vol='Garch', p=1, q=1, rescale=False)
    result = model.fit(disp="off")

    cond_vol = result.conditional_volatility

    z = resid / cond_vol
    anomalies = np.abs(z) > threshold

    return anomalies.astype(bool)


# GARCH grid search for model selection
def garch_grid_search(ts: pd.Series, max_p=5, max_q=5):
    results_table = []

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                with warnings.catch_warnings(record=True) as w:
                    model = arch_model(ts, vol='Garch', p=p,
                                       q=q, rescale=False)
                    model_fit = model.fit(disp="off")
                    aic = model_fit.aic
                    bic = model_fit.bic

                    cond_vol = model_fit.conditional_volatility
                    z = ts / cond_vol
                    anom = np.abs(z) > 3

                    results_table.append({
                        'order': (p, q),
                        'AIC': aic,
                        'BIC': bic,
                        'model': model_fit,
                        'anom': anom,
                        'warnings': w,
                        'error': None
                    })
            except Exception as e:
                results_table.append({
                    'order': (p, q),
                    'AIC': None,
                    'BIC': None,
                    'model': None,
                    'anom': None,
                    'warnings': None,
                    'error': str(e)
                })
    return results_table
