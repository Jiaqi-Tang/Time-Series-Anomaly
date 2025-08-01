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


def standardize_ts(ts: pd.Series):
    scaler = StandardScaler()
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))
    return pd.Series(ts_scaled.flatten(), index=ts.index)


def create_sliding_windows(ts, window_size=10):
    X = []
    for i in range(len(ts) - window_size):
        X.append(ts[i:i + window_size])
    return np.array(X)


def np_to_dataloader(np_arr: np.array, batch_size=16, generator=None):
    tensor_data = torch.tensor(np_arr, dtype=torch.float32).unsqueeze(-1)
    dataset = TensorDataset(tensor_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def ts_to_dataloader(ts: pd.Series, window_size=10, batch_size=16):
    return np_to_dataloader(create_sliding_windows(ts, window_size=window_size), batch_size=batch_size)
