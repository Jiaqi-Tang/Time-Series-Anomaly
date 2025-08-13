import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


def standardize_residuals(resid: pd.Series) -> pd.Series:
    mean = resid.mean()
    std = resid.std()
    if std == 0:
        raise ValueError(
            "Standard deviation is zero. Cannot standardize residuals.")
    return (resid - mean) / std


def standardize_ts(ts):
    scaler = StandardScaler()

    if isinstance(ts, pd.Series):
        ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1)).flatten()
        return pd.Series(ts_scaled, index=ts.index, name=ts.name)

    elif isinstance(ts, pd.DataFrame):
        ts_scaled = scaler.fit_transform(ts)
        return pd.DataFrame(ts_scaled, index=ts.index, columns=ts.columns)

    else:
        raise TypeError("Input must be a pandas Series or DataFrame")


def create_sliding_windows(ts, window_size=10):
    ts = np.array(ts)

    if ts.ndim == 1:
        ts = ts[:, None]

    X = []
    for i in range(len(ts) - window_size):
        X.append(ts[i:i + window_size])
    return np.array(X)


def np_to_dataloader(np_arr: np.array, batch_size=16, generator=None):
    tensor_data = torch.tensor(np_arr, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def ts_to_dataloader(ts: pd.Series, window_size=10, batch_size=16):
    return np_to_dataloader(create_sliding_windows(ts, window_size=window_size), batch_size=batch_size)
