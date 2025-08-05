import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Callable

from src.models.preprocessing import standardize_ts, create_sliding_windows


def detect_anomalies(features_df: pd.DataFrame, contamination: float = 0.05):
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(features_df)
    scores = model.decision_function(features_df)
    labels = model.predict(features_df)  # -1 = anomaly, 1 = normal
    return scores, labels


def model_isolation_forest(ts: pd.Series, extract_features: Callable[[list[pd.Series]], pd.DataFrame], window_size: int = 10, contamination: float = 0.05):
    windows = create_sliding_windows(
        standardize_ts(ts), window_size=window_size)
    features_df = extract_features(windows)
    scores, labels = detect_anomalies(features_df, contamination)

    # Assemble results
    results = pd.DataFrame({
        'timestamp': ts.index[window_size:],
        'anomaly': (labels == -1),
        'score': scores
    }).set_index('timestamp')

    return results
