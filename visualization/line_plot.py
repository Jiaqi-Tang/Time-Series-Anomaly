import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ts(ts):
    plt.plot(ts, label='Original', linewidth=2)
    plt.title('Times series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_data_with_anom(df):
    plt.plot(df.index, df['value'], label='Time Series', color='blue')

    # Overlay anomalies as red dots
    anomalies = df[df['labels'] == 1]
    plt.scatter(anomalies.index,
                anomalies['value'], color='red', label='Anomalies', zorder=5)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Time Series with Anomaly Highlights")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_lag_with_ci(corelation_vals, conf_interval, title='ACF / PACF Lag', ylabel='Value', xlabel='Lag'):
    n = np.arange(len(corelation_vals))

    plt.stem(n, corelation_vals)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axhline(y=conf_interval, color='red', linestyle='--', label='CI')
    plt.axhline(y=-conf_interval, color='red', linestyle='--')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_fit(ts, fitted, start=0, title='ARIMA Model Fit'):

    plt.figure(figsize=(12, 6))
    plt.plot(ts, label='Original', linewidth=2)
    plt.plot(fitted, label='Fitted', linestyle='--')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
