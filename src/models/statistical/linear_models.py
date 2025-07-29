import pandas as pd
import numpy as np

import statsmodels.api as sm


def linear_model(ts: pd.Series):
    x = np.arange(len(ts))
    x = sm.add_constant(x)
    model = sm.OLS(ts.values, x)

    return model, x
