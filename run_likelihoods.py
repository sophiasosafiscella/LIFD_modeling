import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
from glob import glob

import pandas as pd
from pint.models import get_model
from pint.toa import get_TOAs
from utils import get_data, corner_plot, find_a0a2a4, plot_a0a2a4, reverse_mapping, get_FD_curve_values
from mcmc_likelihood import compute_mcmc, lnprob, load_pinit
from pypulse.par import Par
import pickle
import os.path
import sys

# Global parameters
PSR_name: str = "J1643-1224"
#PSR_name: str = "J1024-0719"
#PSR_name: str = "J1903+0327"

print(f"Running {PSR_name}...")

weight: bool = False

# Input files
parfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/par/{PSR_name}_PINT_*.nb.par")[0]
timfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/tim/{PSR_name}_PINT_*.nb.tim")[0]
pickle_file: str = f"./results/{PSR_name}/{PSR_name}_data_obj.pkl"
samples_file: str = f"./results/{PSR_name}/{PSR_name}_samples.npy"
a0a2a4_file: str = f"./results/{PSR_name}/{PSR_name}_a0a2a4.pkl"

# Load the timing model and TOAs
timing_model = get_model(parfile)  # Ecliptical coordiantes
toas = get_TOAs(timfile, planets=True, ephem=timing_model.EPHEM.value)

# Get the TOAs obtained with GUPPI and that are in DMX windows with observations in both frequency bands
if os.path.exists(pickle_file):
    print("Observations already filtered. Now loading them...")
    with open(pickle_file, "rb") as f:
        data_obj = pickle.load(f)
else:
    print("Filtering observations...")
    data_obj = get_data(PSR_name, toas, timing_model)
    with open(pickle_file, "wb") as f:
        pickle.dump(data_obj, f)
    print("Done!")

sys.exit()

# Initial position in the 3D space of (C1, C3, C5) from where the walkers will start. I got the values from the
# plots I created previously
pinit = load_pinit(PSR_name)

# Run the MCMC sampler
if os.path.exists(samples_file):
    samples = np.load(samples_file)
else:
    print("MCMC sampling...")
    samples = compute_mcmc(lnprob, (data_obj, weight), pinit)
    np.save(samples_file, samples)
    print("Done!")

# Present the results
corner_plot(samples, PSR_name)

param_labels = ["$a_1$", "$a_3$", "$a_5$"]
quantiles = [16, 50, 84]  # For 68% credible interval

# Find the medians of a1, a3, a5
#a1a3a5 = np.empty(3)
for i in range(samples.shape[1]):
    q16, q50, q84 = np.percentile(samples[:, i], quantiles)
#    a1a3a5[i] = q50
    median = q50
    minus = q50 - q16
    plus = q84 - q50
#    print(f"{param_labels[i]} = {median:.4f} (+{plus:.4f}/-{minus:.4f})")

# Plot the FD curves
fig, ax = plt.subplots()
fig.suptitle(PSR_name)
a1a3a5 = np.median(samples, axis=0)                                  # Calculate the maximum posterior coefficients
poly = Polynomial([0.0, a1a3a5[0], 0.0, a1a3a5[1], 0.0, a1a3a5[2]])  # Construct the power series polynomial
xvals = np.arange(-1.0, 1.0, 0.001)           # Create values of the normalized inverse frequency between -1 and 1
ys = poly(xvals)                              # Evaluate the power series polynomial at those inverse frequencies
ys -= np.mean(ys)

# Transform the inverse frequencies to normal frequencies (in GHz)
freqs = reverse_mapping(xvals, data_obj.max_inv_freq, data_obj.min_inv_freq)
ax.plot(freqs, ys, label="$a_1 x + a_3 x^3 + a_5 x^5$")

# FD model
p = Par(parfile, numwrap=float)
DM = p.getDM()

# Frequencies and delays for the model as it is
fs, ys_FD = get_FD_curve_values(p, freqs, DM0=DM)
ax.plot(fs, ys_FD, 'k', label="NG15's FD model")
F1, F2 = freqs[0], freqs[-1]
Fdiff = F2 - F1
ax.set_xlim(F1 - 0.1 * Fdiff, F2 + 0.1 * Fdiff)

ax.set_xlabel(r"Frequency (GHz)")
ax.set_ylabel(r"Residual ($\mu$s)")

plt.legend()
plt.tight_layout()
plt.savefig(f"./results/{PSR_name}/{PSR_name}_FD_curve.png")
plt.show()

# For the median values of a1, a3, a5, find the fitted valus of a0, a2, a4 in each DMX window
if os.path.exists(a0a2a4_file):
    a0a2a4 = pd.read_pickle(a0a2a4_file)
else:
    a0a2a4 = find_a0a2a4(PSR_name, data_obj, a1a3a5)
    a0a2a4.to_pickle(a0a2a4_file)

# Plot the results
plot_a0a2a4(PSR_name, data_obj, a0a2a4)



