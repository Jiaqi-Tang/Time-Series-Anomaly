import numpy as np


def create_seasonality_fourier(t, period=50, n_harmonics=5):
    result = np.zeros_like(t, dtype=float)
    for k in range(1, n_harmonics + 1):
        result += (
            np.sin(2 * np.pi * k * t / period) +
            np.cos(2 * np.pi * k * t / period)
        )
    return result
