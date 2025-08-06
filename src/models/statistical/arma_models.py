from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import warnings

from src.visualization.plots import *


def count_anoms(anoms: pd.DataFrame):
    anom_count = None
    for i, row in anoms.iterrows():
        if anom_count is None:
            anom_count = pd.Series(
                np.zeros(len(row['anom'])), row['anom'].index)

        anom_count += row['anom'].values
    return anom_count


def SARIMA_grid_search(ts: pd.Series, period: int, d: int, D: int, max_p=5, max_q=5, max_P=2, max_Q=2):
    results_table = []

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for P in range(max_P + 1):
                for Q in range(max_Q + 1):
                    try:
                        with warnings.catch_warnings(record=True) as w:
                            model = SARIMAX(ts,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, period), freq='ME')
                            model_fit = model.fit(disp=False)
                            aic = model_fit.aic
                            bic = model_fit.bic

                            pred = model_fit.get_prediction(
                                start=13, alpha=0.05)
                            conf_int = pred.conf_int()

                            offset = D * period + d
                            anom = (ts[offset:] < conf_int.iloc[:, 0]) | (
                                ts[offset:] > conf_int.iloc[:, 1])

                            results_table.append({
                                'order': (p, d, q),
                                'seasonal_order': (P, D, Q, 12),
                                'AIC': aic,
                                'BIC': bic,
                                'model': model_fit,
                                'pred': pred,
                                'conf_int': conf_int,
                                'anom': anom,
                                'warnings': w,
                                'error': None
                            })

                    except Exception as e:
                        results_table.append({
                            'order': (p, d, q),
                            'seasonal_order': (P, D, Q, period),
                            'AIC': None,
                            'BIC': None,
                            'model': None,
                            'pred': None,
                            'conf_int': None,
                            'anom': None,
                            'warnings': None,
                            'error': str(e)
                        })
    return results_table
