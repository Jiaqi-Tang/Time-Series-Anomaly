from scipy.stats import chi2
from pykalman import KalmanFilter
import numpy as np
import pandas as pd


from bayesian_changepoint_detection.priors import const_prior
from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
import bayesian_changepoint_detection.offline_likelihoods as offline_ll
from functools import partial

prior_function = partial(const_prior, p=1/(len(resid) + 1))

Q, P, Pcp = offline_changepoint_detection(
    resid, prior_function, offline_ll.StudentT(), truncate=-10)
values = np.insert(np.exp(Pcp).sum(0), 0, 0)
values = pd.Series(values.flatten(), index=resid.index)
# Pcp =

fig, ax = plt.subplots(2, figsize=[18, 16], sharex=True)
ax[0].plot(resid[:])
ax[1].plot(values)


kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
kf = kf.em(ts.values, n_iter=10)

filtered_state_means, filtered_state_covariances = kf.filter(ts.values)

print(filtered_state_means)
print(filtered_state_covariances)

predicted_means, predicted_covs = kf.filter_update(
    filtered_state_means[:-1], filtered_state_covariances[:-
                                                          1], observation=ts.values[1:]
)

predicted_means = np.insert(predicted_means, 0, filtered_state_means[0])
predicted_covs = np.insert(
    predicted_covs, 0, filtered_state_covariances[0], axis=0)
std_devs = np.sqrt(np.squeeze(predicted_covs))

residuals = ts.values - predicted_means
nis = (residuals ** 2) / (std_devs ** 2)

# Chi-square cutoff for anomaly detection
cutoff = chi2.ppf(1, df=1)
is_anomaly = nis > cutoff

# Return anomalies as Series aligned to original index
anomalies = pd.Series(is_anomaly, index=ts.index)
filtered_series = pd.Series(filtered_state_means.flatten(), index=ts.index)

plt.plot(ts, label='Observations', alpha=0.5)
plt.plot(filtered_series, label='Kalman Filter Estimate')
plt.legend()
plt.show()
