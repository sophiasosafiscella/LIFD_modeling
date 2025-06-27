import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import pandas as pd
from pint.models import get_model
from pint.toa import get_TOAs
from utils import filter_observations, corner_plot, find_a0a2a4, plot_a0a2a4
from mcmc_likelihood import lnprob, compute_mcmc
import pickle
import os.path

# Global parameters
PSR_name: str = "J1643-1224"
weight: bool = False

# Input files
parfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/par/{PSR_name}_PINT_*.nb.par")[0]
timfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/tim/{PSR_name}_PINT_*.nb.tim")[0]
pickle_file: str = f"./results/{PSR_name}/{PSR_name}_filtered_obs.pkl"
samples_file: str = f"./results/{PSR_name}/{PSR_name}_samples.npy"
a0a2a4_file = f"./results/{PSR_name}/{PSR_name}_a0a2a4.pkl"

# Load the timing model and TOAs
timing_model = get_model(parfile)  # Ecliptical coordiantes
toas = get_TOAs(timfile, planets=True, ephem=timing_model.EPHEM.value)

# Get the TOAs obtained with GUPPI and that are in DMX windows with observations in both frequency bands
if os.path.exists(pickle_file):
    with open(pickle_file, "rb") as f:
        filtered_obs = pickle.load(f)
else:
    filtered_obs = filter_observations(toas, timing_model)
    with open(pickle_file, "wb") as f:
        pickle.dump(filtered_obs, f)

# Initial position in the 3D space of (C1, C3, C5) from where the walkers will start. I got the values from the
# plots I created previously
#pinit = np.array([350.0, 5.0, 0.0])
pinit = np.array([342.9607408562299, 3.6656647122634305, -0.21600401837611255])

# Run the MCMC sampler
if os.path.exists(samples_file):
    samples = np.load(samples_file)
else:
    samples = compute_mcmc(lnprob, (filtered_obs, weight), pinit)
    np.save(samples_file, samples)

# Present the results
#corner_plot(samples, PSR_name)

param_labels = ["$a_1$", "$a_3$", "$a_5$"]
quantiles = [16, 50, 84]  # For 68% credible interval

# Find the medians of a1, a3, a5
a1a3a5 = np.empty(3)
for i in range(samples.shape[1]):
    q16, q50, q84 = np.percentile(samples[:, i], quantiles)
    a1a3a5[i] = q50
    median = q50
    minus = q50 - q16
    plus = q84 - q50
#    print(f"{param_labels[i]} = {median:.4f} (+{plus:.4f}/-{minus:.4f})")

# For the median values of a1, a3, a5, find the fitted valus of a0, a2, a4 in each DMX window
if os.path.exists(a0a2a4_file):
    a0a2a4 = pd.read_pickle(a0a2a4_file)
else:
    a0a2a4 = find_a0a2a4(PSR_name, filtered_obs, a1a3a5)
    a0a2a4.to_pickle(a0a2a4_file)

# Plot the results
plot_a0a2a4(PSR_name, filtered_obs, a0a2a4)



