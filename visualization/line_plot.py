import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ts(ts, title='Times series'):
    plt.plot(ts, linewidth=2)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_ts_with_anom(ts, anom, title="Time Series with Anomalies"):
    plt.plot(ts, label='Time Series', color='blue')
    plt.scatter(ts[anom].index, ts[anom].values,
                color='red', label='Anomalies', zorder=5)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_data_with_anom(df, title="Time Series with Anomalies"):
    plt.plot(df.index, df['value'], label='Time Series', color='blue')

    anomalies = df[df['labels'] == 1]
    plt.scatter(anomalies.index,
                anomalies['value'], color='red', label='Anomalies', zorder=5)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_lag_with_ci(corelation_vals, conf_interval, title='ACF / PACF Lag'):
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


def plot_fit(ts, fitted_vals, CI=None, title='ARIMA Model Fit'):

    plt.figure(figsize=(12, 6))
    plt.plot(ts, label='Original', linewidth=2)
    plt.plot(fitted_vals, label='Fitted', linestyle='--')
    if CI is not None:
        ci_lower = CI.iloc[:, 0]
        ci_upper = CI.iloc[:, 1]
        plt.fill_between(CI.index, ci_lower, ci_upper,
                         color='lightgreen', alpha=0.4, label='Confidence Interval')

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
