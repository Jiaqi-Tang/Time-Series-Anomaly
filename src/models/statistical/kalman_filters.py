from statsmodels.tsa.statespace.structural import UnobservedComponents
import pandas as pd


def kalman_local_level(ts: pd.Series):
    ts = ts.dropna()
    ts.index = pd.to_datetime(ts.index)

    model = UnobservedComponents(ts, level='local linear trend')

    result = model.fit(disp=False)

    # Extract smoothed level (latent trend)
    level = pd.Series(result.smoothed_state[0], index=ts.index, name='level')
    slope = pd.Series(result.smoothed_state[1], index=ts.index, name='slope')

    resid = ts - level

    return {
        'level': level,
        'slope': slope,
        'resid': resid,
        'model': result
    }
