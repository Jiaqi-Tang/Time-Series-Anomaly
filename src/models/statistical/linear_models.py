import pandas as pd
import numpy as np

import statsmodels.api as sm
from scipy.stats import multivariate_normal


# Wrapper function for linear model
def linear_model(ts: pd.Series):
    x = np.arange(len(ts))
    x = sm.add_constant(x)
    model = sm.OLS(ts.values, x)

    return model, x


# Wrapper function for MVN model
def mvn_model(df: pd.DataFrame):
    mu = df.mean().values
    cov = np.cov(df.values.T)

    return multivariate_normal(mean=mu, cov=cov)
