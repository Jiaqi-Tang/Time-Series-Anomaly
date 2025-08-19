from statsmodels.tsa.statespace.structural import UnobservedComponents
import pandas as pd


# Wrapper function for local level kalman filters
def kalman_local_level(ts: pd.Series):
    ts = ts.dropna()
    ts.index = pd.to_datetime(ts.index)

    model = UnobservedComponents(ts, level='local linear trend')

    result = model.fit(disp=False)

    level = pd.Series(result.smoothed_state[0], index=ts.index, name='level')
    slope = pd.Series(result.smoothed_state[1], index=ts.index, name='slope')

    resid = ts - level

    return {
        'level': level,
        'slope': slope,
        'resid': resid,
        'model': result
    }


# Wrapper function for Kalman Filter cycle for damped ocsillation modeling
def kalman_cycle(ts: pd.Series, period: float, allow_drift=True):
    ts = ts.dropna()
    ts.index = pd.to_datetime(ts.index)

    lb, ub = period * 0.8, period * 1.2

    model = UnobservedComponents(
        ts,
        level=False,
        cycle=True,
        stochastic_cycle=False,
        damped_cycle=True,
        cycle_period_bounds=(lb, ub)
    )

    result = model.fit(disp=False)

    try:
        cycle = result.states.smoothed['cycle']
    except Exception:
        cycle = pd.Series(
            result.smoothed_state[0], index=ts.index, name='cycle')

    resid = ts - cycle

    return {
        'cycle': cycle.rename('cycle'),
        'resid': resid.rename('resid'),
        'model': result
    }
