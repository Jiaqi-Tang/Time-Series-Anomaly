import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from bayesian_changepoint_detection.bayesian_models import online_changepoint_detection
import bayesian_changepoint_detection.online_likelihoods as online_ll

from src.visualization.plots import plot_state_space, plot_ts


# Wrapper function for BOCPD model
def bocpd(ts, hazard_function, epsilon=1e-7, sparsity=5, Nw=10):
    R, maxes = online_changepoint_detection(
        ts, hazard_function, online_ll.StudentT(
            alpha=0.1, beta=.01, kappa=1, mu=0)
    )

    density_matrix = -np.log(R[0:-1:sparsity, 0:-1:sparsity]+epsilon)
    plot_state_space(R, sparsity, density_matrix)
    plot_ts(R[Nw, Nw:-1])
