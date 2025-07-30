import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def create_sliding_windows(ts: pd.Series, window_size=10):
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
