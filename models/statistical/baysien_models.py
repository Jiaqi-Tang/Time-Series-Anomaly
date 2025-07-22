import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import changepoint as cpt
from scipy.stats import norm


def constant_hazard(lam):
    return lambda r: 1/lam * np.ones_like(r, dtype=float)


def bocpd_gaussian(data, hazard_fn, mu0=0, kappa0=1, alpha0=1, beta0=1):
    n = len(data)
    max_r = n + 1
    log_R = -np.inf * np.ones((n + 1, n + 1))
    log_R[0, 0] = 0.0  # log(1)

    mu_t = np.zeros((n + 1, n + 1))
    kappa_t = np.zeros((n + 1, n + 1))
    alpha_t = np.zeros((n + 1, n + 1))
    beta_t = np.zeros((n + 1, n + 1))

    mu_t[0, 0] = mu0
    kappa_t[0, 0] = kappa0
    alpha_t[0, 0] = alpha0
    beta_t[0, 0] = beta0

    pred_probs = np.zeros(n)
    run_probs = np.zeros(n)

    for t in range(1, n + 1):
        xt = data[t - 1]

        # Predictive probabilities
        pred_prob = np.zeros(t)
        for r in range(t):
            mu = mu_t[t - 1, r]
            kappa = kappa_t[t - 1, r]
            alpha = alpha_t[t - 1, r]
            beta = beta_t[t - 1, r]

            scale = np.sqrt((beta * (kappa + 1)) / (alpha * kappa))
            pred_prob[r] = norm.pdf(xt, loc=mu, scale=scale)

        # Update growth probabilities
        growth_probs = pred_prob * \
            np.exp(log_R[t - 1, :t]) * (1 - hazard_fn(np.arange(t)))

        # Update changepoint probability
        cp_prob = np.sum(
            pred_prob * np.exp(log_R[t - 1, :t]) * hazard_fn(np.arange(t)))

        # Update run-length distribution
        log_R[t, 1:t+1] = np.log(growth_probs + 1e-200)  # prevent log(0)
        log_R[t, 0] = np.log(cp_prob + 1e-200)

        # Normalize
        log_R[t, :t+1] -= np.log(np.sum(np.exp(log_R[t, :t+1])) + 1e-200)

        # Update parameters for each run-length
        for r in range(t+1):
            if r == 0:
                mu_t[t, r] = mu0
                kappa_t[t, r] = kappa0
                alpha_t[t, r] = alpha0
                beta_t[t, r] = beta0
            else:
                kappa = kappa_t[t - 1, r - 1] + 1
                mu = (kappa_t[t - 1, r - 1] * mu_t[t - 1, r - 1] + xt) / kappa
                alpha = alpha_t[t - 1, r - 1] + 0.5
                beta = beta_t[t - 1, r - 1] + (kappa_t[t - 1, r - 1]
                                               * (xt - mu_t[t - 1, r - 1]) ** 2) / (2 * kappa)

                mu_t[t, r] = mu
                kappa_t[t, r] = kappa
                alpha_t[t, r] = alpha
                beta_t[t, r] = beta

        # Save MAP run-length
        run_probs[t - 1] = np.exp(log_R[t, :t+1]).argmax()
        pred_probs[t - 1] = np.max(pred_prob)

    return run_probs
