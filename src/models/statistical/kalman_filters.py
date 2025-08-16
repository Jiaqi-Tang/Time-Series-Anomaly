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


def kalman_cycle(ts: pd.Series, period: float, allow_drift=True):
    """
    Kalman filter for an oscillator using a stochastic cycle.
    period: oscillation period in *time steps* (not seconds; convert using your sampling dt).
    allow_drift: if True, cycle is stochastic (amplitude/phase may drift); if False, nearly deterministic.
    """
    ts = ts.dropna()
    ts.index = pd.to_datetime(ts.index)

    lb, ub = period * 0.8, period * 1.2

    model = UnobservedComponents(
        ts,
        level=False,                 # <-- no local level
        cycle=True,                  # <-- include cycle component
        stochastic_cycle=False,
        damped_cycle=True,          # set True if you want a built-in damping factor
        # fixes the period exactly; widen for estimation
        cycle_period_bounds=(lb, ub)
    )

    result = model.fit(disp=False)

    # Get the estimated cycle component and residuals
    # (API varies slightly across statsmodels versions; try the named states first)
    try:
        cycle = result.states.smoothed['cycle']
    except Exception:
        # Fall back to the first state if names aren't available
        cycle = pd.Series(
            result.smoothed_state[0], index=ts.index, name='cycle')

    resid = ts - cycle

    return {
        'cycle': cycle.rename('cycle'),
        'resid': resid.rename('resid'),
        'model': result
    }
