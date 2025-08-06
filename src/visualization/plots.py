import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ts(ts: pd.Series, fit=None, CI=None, anom=None, title='Time Series'):
    plt.figure(figsize=(12, 6))
    plt.plot(ts, label='Original', linewidth=2)

    if fit is not None:
        plt.plot(fit, label='Fitted', linestyle='--')

    if CI is not None:
        ci_lower = CI.iloc[:, 0]
        ci_upper = CI.iloc[:, 1]
        plt.fill_between(CI.index, ci_lower, ci_upper,
                         color='lightgreen', alpha=0.4, label='Confidence Interval')
    if anom is not None:
        anom = anom.reindex(ts.index, fill_value=False)
        plt.scatter(ts[anom].index, ts[anom].values,
                    color='red', label='Anomalies', zorder=5)

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_data_with_anom(df: pd.DataFrame, title="Time Series with Anomalies"):
    plt.plot(df['timestamp'], df['value'], label='Time Series')

    anomalies = df[df['labels'] == 1]
    plt.scatter(anomalies['timestamp'],
                anomalies['value'], color='red', label='Anomalies', zorder=5)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_scatter(resid: pd.Series, hlines=None, title="Residual plot"):
    plt.scatter(resid.index, resid.values)
    plt.hlines(0, xmin=min(resid.index), xmax=max(resid.index), colors='black')

    if (hlines):
        plt.hlines(hlines, xmin=min(resid.index),
                   xmax=max(resid.index), colors='red')

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_lag(corelation_vals, conf_interval, title='ACF / PACF Lag'):
    n = np.arange(len(corelation_vals))

    plt.stem(n, corelation_vals)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axhline(y=conf_interval, color='red', linestyle='--', label='CI')
    plt.axhline(y=-conf_interval, color='red', linestyle='--')

    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cusum(cusum_df: pd.DataFrame, threshold):
    plt.figure(figsize=(12, 5))
    plt.plot(cusum_df.index,
             cusum_df['positive_cusum'], label='Positive CUSUM')
    plt.plot(cusum_df.index,
             cusum_df['negative_cusum'], label='Negative CUSUM')
    plt.axhline(threshold, color='r',
                linestyle='--', label='Threshold')
    plt.axhline(-threshold, color='r', linestyle='--')
    plt.title('CUSUM over Time')
    plt.legend()
    plt.show()


def plot_training_accuracy(losses, title="Training Loss"):
    plt.figure(figsize=(12, 5))
    plt.plot(losses)

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (MSE)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
