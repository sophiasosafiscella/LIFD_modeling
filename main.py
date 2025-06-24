import numpy as np

import numpy.polynomial.legendre as leg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from lmfit import Model, Parameters
from numpy.polynomial.legendre import Legendre
from numpy.polynomial.polynomial import Polynomial
from pint.models import get_model
from pint.residuals import Residuals
from pint.toa import get_TOAs
from utils import get_dmx_ranges, get_dmx_observations, epoch_scrunch, broadband_observations, map_domain, make_plot
from fit_coefficients import my_legfit
import astropy.units as u
from glob import glob
import os
import sys


def find_dispersion_coefficients(psr_name:str, fixed_coeffs:bool=True, plot:bool=True):

    # Input files
    parfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/par/{psr_name}_PINT_*.nb.par")[0]
    timfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/tim/{psr_name}_PINT_*.nb.tim")[0]

    # Load the timing model and TOAs
    timing_model = get_model(parfile)  # Ecliptical coordiantes
    toas = get_TOAs(timfile, planets=True, ephem=timing_model.EPHEM.value)

    # Get the TOAs obtained with GUPPI
    GUPPI_toas = broadband_observations(toas)

    if False:
        original_residuals = Residuals(GUPPI_toas, timing_model).time_resids.to(u.us).value
        epochs, retval, retvalerrs = epoch_scrunch(GUPPI_toas.get_mjds().value, data=original_residuals, errors=GUPPI_toas.get_errors().to(u.us).value, weighted=True)

        sns.set_style('darkgrid')
        plt.errorbar(epochs, retval, yerr=retvalerrs, fmt='x')
        plt.xlabel('MJD')
        plt.ylabel('Residuals [$\mu \mathrm{s}$]')
        plt.title('Original Residuals vs. MJD')
        plt.show()

    # Find the DMX windows
    dmx_ranges = get_dmx_ranges(timing_model, GUPPI_toas)

    # Get rid of the DMX and FD parameters
    timing_model.remove_component("DispersionDMX")
    timing_model.remove_component("FD")

    # Arrays to store the coefficients
    coeffs_arr = np.full((len(dmx_ranges), 6), np.nan)

    # Iterate over the DMX windows
    for n, window in enumerate(dmx_ranges):
        print("Working on " + str(window[0]) + " to " + str(window[1]) + " (index " + str(n) + ")...")

        # We find the observations in the windows and the corresponding frequencies
        observations_in_window = get_dmx_observations(GUPPI_toas, window[0], window[1])
        frequencies = np.array(observations_in_window.get_freqs().to(u.GHz).value)  # Convert frequencies to GHz

        # We make sure there are observations in both bands
        lowerband_ok = np.any((0.725 <= frequencies) & (frequencies <= 0.916))
        upperband_ok = np.any((1.156 <= frequencies) & (frequencies <= 1.882))
        if (not lowerband_ok) or (not upperband_ok):
            print("Skipped a window because it didn't have observations in both bands")
            continue

        # Calculate residuals (in microseconds) with the simplified model
        residuals = Residuals(observations_in_window, timing_model).time_resids.to(u.us).value
        frequencies, residuals = zip(*sorted(zip(frequencies, residuals)))
        frequencies, residuals = np.array(frequencies), np.array(residuals)

        # Map the frequencies to the range [-1, 1]
        x_values = map_domain(frequencies)

        # Calculate the coefficients for the unscaled and unshifted Legendre basis polynomials

        if fixed_coeffs:
            a1a3a5 = np.array([357.33928150837346, 6.78687187489919, -2.510308474769396])
            c1c3c5 = leg.poly2leg([0.0, a1a3a5[0], 0.0, a1a3a5[1], 0.0, a1a3a5[2]])[[1,3,5]]
            c0c2c4 = my_legfit(x=x_values, y=residuals.astype(np.float64), deg=5, coeffs=c1c3c5, full=False)
            leg_pfit_coef = np.array([c0c2c4[0], c1c3c5[0], c0c2c4[1], c1c3c5[1], c0c2c4[2], c1c3c5[2]])
            pfit = Legendre(leg_pfit_coef)

        else:
            pfit = leg.Legendre.fit(x=x_values, y=residuals.astype(np.float64), deg=5, full=False)
            leg_pfit_coef = pfit.coef

#        power_series_pol = np.polynomial.polynomial.Polynomial(leg_pfit_coef)

        #  Coefficients of the equivalent polynomial (relative to the “standard” basis) ordered from lowest to highest degree.
        coeffs_arr[n, :] = leg.leg2poly(leg_pfit_coef)

        if plot:

            at = AnchoredText(f"Power series coefficients: \n $C_0$ = {coeffs_arr[n, 0]} \n $C_1$ = {coeffs_arr[n, 1]} \n $C_2$ = {coeffs_arr[n, 2]} \n $C_3$ = {coeffs_arr[n, 3]} \n $C_4$ = {coeffs_arr[n, 4]} \n $C_5$ = {coeffs_arr[n, 5]}",
                              prop=dict(size=10), frameon=True, loc='upper left')

#            at = AnchoredText(f"Legendre series coefficients: \n $C_0$ = {leg_pfit_coef[0]} \n $C_1$ = {leg_pfit_coef[1]} \n $C_2$ = {leg_pfit_coef[2]} \n $C_3$ = {leg_pfit_coef[3]} \n $C_4$ = {leg_pfit_coef[4]} \n $C_5$ = {leg_pfit_coef[5]}",
#                              prop=dict(size=10), frameon=True, loc='upper left')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")

            # Make some cool plots
            '''
            fig, ax = plt.subplots()
            ax.plot(frequencies, residuals, "o")
            ax.set_title("MJD " + str(window[0]) + " to " + str(window[1]))
            ax.set_xlabel("Frequency [GHz]")
            ax.set_ylabel(r'Residuals [$\mu s$]')
            ax.axvspan(0.725, 0.916, alpha=0.4, color='C2', label="GUPPI - Revr_800")
            ax.axvspan(1.156, 1.882, alpha=0.4, color='C3', label="GUPPI - Revr1_2")
            ax.legend(loc="upper right")
            '''

            fig2, ax2 = plt.subplots()
            ax2.plot(x_values, residuals, "o")
            ax2.plot(x_values, pfit(x_values), lw=2, label="Fitted Legendre Polynomial")
#            ax2.plot(x_values, power_series_pol(x_values), "-", lw=2, label="Power Series Polynomial fit")
            ax2.set_title("MJD " + str(window[0]) + " to " + str(window[1]))
            ax2.set_xlabel("Normalized Inverse Frequency")
            ax2.set_ylabel(r'Residuals [$\mu s$]')
            ax2.add_artist(at)
            ax2.legend(loc="upper right")
            plt.tight_layout()

#            plt.savefig('./figures/NG15/fit_' + str(n) + '.png')
            plt.show()

    # Create a DataFrame to hold all the information and drop the windows where any of the coefficients is a Nan,
    # because that means that that particular window was skipped due to not having observations in both bands
    df = pd.DataFrame(np.hstack((dmx_ranges, coeffs_arr)), columns=["DMXR1", "DMXR2", "C0", "C1", "C2", "C3", "C4", "C5"])

    return df.dropna(axis='index', how='any')


if __name__ == "__main__":

    PSR_name: str = "J1643-1224"
    if not os.path.exists('./' + PSR_name + '_results.csv'):
        df = find_dispersion_coefficients(PSR_name)
        df.to_csv('./' + PSR_name + '_results.csv')
    else:
        df = pd.read_csv('./' + PSR_name + '_results.csv')

    make_plot(PSR_name, df)



