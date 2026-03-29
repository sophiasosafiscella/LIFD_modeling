import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import astropy.units as u
from pint.toa import get_TOAs
from uncertainties import ufloat
from uncertainties import unumpy as unp
import sys
from glob import glob
from IFD_class import IFD
from LIFD_class import LIFD
from numpy.polynomial.legendre import legval
from numpy.polynomial.polynomial import polyval

def get_DMX_info(method):
    # FD
    if method == "FD":
        df = pd.read_pickle(f"../results/{PSR_name}/FD_DMX.pkl")

    # IFD
    elif method == "IFD":
        df = pd.read_pickle(f"../results/{PSR_name}/IFD_DMX.pkl")

    # LIFD
    elif method == "LIFD":
        df = pd.read_pickle(f"../results/{PSR_name}/LIFD_DMX.pkl")
    else:
        print(f"Method {method} not implemented")
        sys.exit(1)

    DMX = df[df.index.str.startswith('DMX_')]
    DMXs = [ufloat(nominal_value=x, std_dev=x_err.value) for x, x_err in
            zip(DMX.Value.to_numpy(), DMX.Error.to_numpy())]

    DMXR1 = df[df.index.str.startswith('DMXR1_')].Value.to_numpy()
    DMXR2 = df[df.index.str.startswith('DMXR2_')].Value.to_numpy()

    return DMXs, DMXR1, DMXR2



def get_prof_evolution_delays(method, coeffs, freqs_GHz, lmin=None, lmax=None):

    if method == "FD":

        FD_params = coeffs[::-1]  # We need to invert the order because np.polyval uses p[0]*x**(N-1)+p[1]*x**(N-2)+...
        FD_params = np.concatenate((FD_params, [0]))  # Adding 0 at the end ensures constant term exists.
        FD_func = lambda nu: 1e6 * np.polyval(FD_params, np.log(nu))  # Function that gives the timing delay
        delay_us = FD_func(freqs_GHz) * u.us  # nu in GHz, returns values in microseconds

    elif method == "IFD":

        lambdas_ns = IFD.get_lambda_ns_from_freq_GHz(freqs_GHz)  # Inverse frequencies
        # This will automatically have units of seconds because the coefficients have units of seconds
        delay_us = (polyval(x=lambdas_ns, c=coeffs) * u.second).to(u.us)


    elif method == "LIFD":

        lambdas = LIFD.get_lambda_from_freq_hz((freqs_GHz * u.GHz).to(u.Hz))
        x_vals = np.sort(LIFD.map_lambda_to_unit(lambdas, lmin, lmax))  # fixed mapping set in setup()
        # This will automatically have units of seconds because the coefficients have units of seconds
        delay_us = (legval(x_vals, coeffs) * u.second).to(u.us)

    return delay_us


PSR_name: str = "B1937+21"
#PSR_name: str = "J1024-0719"
#PSR_name: str = "J1643-1224"
#PSR_name: str = "J2145-0750"
K = 4.149e3
fig, ax = plt.subplots(1,1)

# Input files
timfile: str = glob(f"../NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/tim/{PSR_name}_PINT_*.nb.tim")[0]

# Get the TOAs, MJDs, and observing frequencies
toas = get_TOAs(timfile)
mjds = toas.get_mjds().value
freqs = toas.get_freqs()
freqs_GHz = freqs.to(u.GHz).value  # strip units once

for method in ["FD", "IFD", "LIFD"]:

    # DMX contributions
    DMX_values, DMXR1, DMXR2 = get_DMX_info(method)

    mask = (DMXR1[:, None] <= mjds[None, :]) & (mjds[None, :] <= DMXR2[:, None])
    has_match = mask.any(axis=0)
    if not has_match.all():
        print(f"Warning: {(~has_match).sum()} TOAs fall outside all DMX windows")

    window_indices = np.argmax(mask, axis=0)
    dmx_per_toa = np.array(DMX_values)[window_indices]  # cast to numpy first

    dmx_nominal = unp.nominal_values(dmx_per_toa)  # plain numpy array
    dmx_std = unp.std_devs(dmx_per_toa)  # plain numpy array

    dmx_delays = K * dmx_nominal / freqs_GHz ** 2  # µs
    dmx_delays_err = K * dmx_std / freqs_GHz ** 2  # µs

    # Round frequencies to nearest channel to group identical frequencies
    freq_rounded = np.round(freqs_GHz, decimals=3)
    unique_freqs = np.unique(freq_rounded)

    dmx_delays_means = np.array([dmx_delays[freq_rounded == f].mean() for f in unique_freqs])
    dmx_delays_stds = np.array([dmx_delays[freq_rounded == f].std() for f in unique_freqs])

    # Profile evolution contributions
    if method == "FD":
        coeffs = np.load(f"../results/{PSR_name}/{method}_coeffs.npy")
    else:
        coeffs = np.load(f"../results/{PSR_name}/new_{method}_coeffs.npy")

    lmin, lmax = None, None
    if method == "LIFD":
        lambdas = LIFD.get_lambda_from_freq_hz((freqs_GHz * u.GHz).to(u.Hz))
        lmin, lmax = lambdas.min(), lambdas.max()

    prof_evolution_delay = get_prof_evolution_delays(method, coeffs, unique_freqs, lmin=lmin, lmax=lmax)

    # Plot
    total_delay = dmx_delays_means + prof_evolution_delay.value
    total_delay -= np.mean(total_delay)  # We are comparing total delays across methods, but each has an arbitrary vertical offset
    ax.plot(unique_freqs, total_delay, 'o', label=f"{method}")
    ax.fill_between(unique_freqs, total_delay - dmx_delays_stds, total_delay + dmx_delays_stds, alpha=0.3)

plt.legend()
plt.tight_layout()
plt.show()


