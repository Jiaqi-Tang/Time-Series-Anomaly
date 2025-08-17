# Time Series Anomaly

A comparative, end-to-end study on time series anomaly detection for the purpose of developing a deeper, hands-on understanding of the methods available and how they behave across diverse contexts. Using classical statistical models, self-trained ML methods, and ready-to-use pretrained frameworks, the goal is to understand the strengths, weaknesses, and deployment considerations of each approach.

## Navigation

See `main_report.ipynb` for more details of this project.

See `/notebooks` for detailed reports on each dataset.

---

## Tech Stack

**Languages & Tools**: Python, Jupyter, pandas, NumPy, matplotlib, seaborn

**Libraries**: statsmodels, scipy, arch, scikit-learn, PyTorch, Prophet, yfinance

**Models**:

- Classical/statistical: STL, SARIMA, Kalman filters, GARCH, BOCPD, CUSUM
- Machine learning: LSTM Autoencoder, Isolation Forest
- Pre-trained: Prophet
  Documented strengths, weaknesses, and assumptions of each model when applied to different datasets.

## Technical Highlights

**Data Pipeline**

- Designed and followed an anomaly detection pipeline, including data preprocessing, synthetic data generation, model training, anomaly scoring, and visualization.
- Gained hands-on experience with both statistical and machine learning methods, applying them in diverse contexts.

**Breadth of Models**

Implemented and compared a wide range of models:

- Classical/statistical: STL, SARIMA, Kalman Filters, GARCH, BOCPD
- Machine learning: LSTM Autoencoder
- Pre-trained/package-based: Isolation Forest, Prophet, AWS Lookout for Metrics

**Dataset Diversity**

- Worked with 4 datasets spanning synthetic, financial, sensor, and physics domains.
- Explored challenges across univariate vs. multivariate data, binary vs. continuous features, short vs. long series, labeled vs. unlabeled datasets.

**Code & Modularity**

- Structured project to support scalability and modularity, maximizing reusability of all code blocks and functions
- Encapsulated model specific logic within individual files for better abstraction, allowing code to remain clean and declarative.

## References & Credits

BOCPD library from [Johannes Kulick on GitHub](https://github.com/hildensia/bayesian_changepoint_detection)

NASA Anomaly Detection Dataset from [astro\_\_pat on kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl?resource=download)

---

## Learnings and Initiative

I am particularly drawn to time series because it feels like solving puzzles hidden in plain sight. With only one timeline to study, every dataset becomes an opportunity to uncover the underlying processes that shape what we observe. That combination of constraint and discovery is what excites me most, and why I see time series as an area full of potential for deeper exploration.
