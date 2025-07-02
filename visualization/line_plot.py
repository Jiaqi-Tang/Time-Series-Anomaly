import matplotlib.pyplot as plt


def plot_with_anom(df):
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
