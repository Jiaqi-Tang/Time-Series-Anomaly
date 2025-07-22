import numpy as np


def create_trend_poly(t, coef=np.array([])):
    trend = np.zeros_like(t, dtype=float)
    for power, coeff in enumerate(coef):
        trend += coeff * t**power
    return trend
