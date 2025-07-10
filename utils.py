import numpy as np
import numpy.polynomial.legendre as leg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.linalg import cho_factor, cho_solve
import corner
import pint
from pint.residuals import Residuals
from pypulse.utils import weighted_moments
import astropy.units as u
from dataclasses import dataclass

from uncertainties import unumpy, ufloat

from fit_coefficients import my_legfit_full


@dataclass
class DataObject:
    PSR_name: str
    dmx_ranges: np.matrix
    max_inv_freq: float
    min_inv_freq: float
    Cinv: np.matrix
    logdet_C: float
    Ndiag: np.array
    U: np.matrix
    Sigma_cf: np.matrix
    xvals: list[np.ndarray]
    resids: list[np.ndarray]


def get_dmx_ranges(model, observations):
    """
    Return an array of the MJD range for each DMX parameter corresponding to a given set of observations_in_window
    """

    mjds = observations.get_mjds().value
    dmx_parameters = model.components['DispersionDMX']  # names of the DMX_xxxx parameters

    DMXR1_names = [a for a in dir(dmx_parameters) if a.startswith('DMXR1')]
    DMXR2_names = [a for a in dir(dmx_parameters) if a.startswith('DMXR2')]

    DMXR1_values = [getattr(dmx_parameters, par).value for par in DMXR1_names]
    DMXR2_values = [getattr(dmx_parameters, par).value for par in DMXR2_names]

    DMXR = np.column_stack((DMXR1_values, DMXR2_values))  # Zip the two arrays

    # Out of all the DMX windows, we will only keep those with at least one TOA inside it
    mask = (DMXR1_values < max(mjds)) & (DMXR2_values > min(mjds))

    return DMXR[mask]


def get_dmx_observations(observations, low_mjd, high_mjd):
    """
    Return an array for selecting TOAs from toas in a DMX range.

    toas is a PINT TOA object of TOAs in the DMX bin.
    low_mjd is the left edge of the DMX bin.
    high_mjd is the right edge of the DMX bin.
    strict_inclusion=True if TOAs exactly on a bin edge are not in the bin for
        the implemented DMX model.
    """

    mjds = observations.get_mjds().value
    mask = (low_mjd < mjds) & (mjds < high_mjd)

    return observations[mask]


def map_domain(frequencies, max_inv_freq, min_inv_freq):

    lambdas = np.power(frequencies, -1.0)                                # Inverse of the frequencies
    x_aux_values = (lambdas - min_inv_freq)/(max_inv_freq-min_inv_freq)  # Between 0 and 1
    x_values = np.subtract(np.multiply(x_aux_values, 2.0), 1.0)          # Between -1 and 1

    return x_values

def reverse_mapping(x_values, max_inv_freq, min_inv_freq):

    aux = np.divide(np.add(x_values, 1.0), 2.0)                 # Between 0 and 1
    lambdas = aux * (max_inv_freq-min_inv_freq) + min_inv_freq  # Inverse of the frequencies, in GHz^(-1)
    frequencies = np.power(lambdas, -1.0)                       # In GHz

    return frequencies


def get_data(toas, timing_model):
        """Given TOAs, extract broadband observations and select DMX windows with both frequency bands."""

        # Filter for GUPPI backend only
        backends = np.array([toas.table["flags"][obs]["be"] for obs in range(len(toas.table["flags"]))])
        broadband_TOAs = toas[np.isin(backends, ["GUPPI"])]
        #broadband_TOAs = toas
        # Find the DMX windows
        dmx_ranges = get_dmx_ranges(timing_model, broadband_TOAs)

        # Get rid of the DMX and FD parameters to create the simplified timing model
        timing_model.remove_component("DispersionDMX")
        timing_model.remove_component("FD")

        # Precompute outputs
        valid_dmx_ranges = []
        valid_resids = []
        valid_xvals = []

        mjds = broadband_TOAs.get_mjds().value
        freqs_GHz = broadband_TOAs.get_freqs().to(u.GHz).value
        max_freq, min_freq = np.amax(freqs_GHz), np.amin(freqs_GHz)      # Maximum and minimum frequencies
        max_inv_freq, min_inv_freq = max_freq ** (-1), min_freq ** (-1)  # Inverse of max and min frequencies
        valid_toas_mask = np.full(broadband_TOAs.ntoas, False)

        for window in dmx_ranges:

            in_window = (mjds > window[0]) & (mjds < window[1])
            freqs_in_window = freqs_GHz[in_window]

            has_lower = np.any((0.725 <= freqs_in_window) & (freqs_in_window <= 0.916))
            has_upper = np.any((1.156 <= freqs_in_window) & (freqs_in_window <= 1.882))

#            has_lower, has_upper = True, True
            if has_lower and has_upper:
                valid_toas_mask |= in_window

                res_object = Residuals(broadband_TOAs[in_window], timing_model)

                valid_dmx_ranges.append(window)
                valid_resids.append(res_object.time_resids.to(u.us).value)
    #            valid_resids_errs.append(res_object.get_data_error().value)  # TODO: we are assuming there's no correlation (for now)
                valid_xvals.append(map_domain(freqs_in_window, max_inv_freq, min_inv_freq))

        valid_toas = broadband_TOAs[valid_toas_mask]
        valid_res_object = Residuals(valid_toas, timing_model)

        Ndiag = valid_res_object.model.scaled_toa_uncertainty(valid_toas).to_value(u.s) ** 2
        U = valid_res_object.model.noise_model_designmatrix(valid_res_object.toas)
        Phidiag = valid_res_object.model.noise_model_basis_weight(valid_res_object.toas)

        Ninv = np.diag(1.0/ Ndiag)
        Ninv_U = np.diag(1.0 / Ndiag) @ U
        Sigma = np.diag(1.0 / Phidiag) + (U.T / Ndiag) @ U
        Sigma_cf = cho_factor(Sigma)

        Cinv = Ninv - Ninv_U @ cho_solve(Sigma_cf, Ninv_U.T)

        logdet_N = np.sum(np.log(Ndiag))
        logdet_Phi = np.sum(np.log(Phidiag))
        _, logdet_Sigma = np.linalg.slogdet(Sigma.astype(float))

        logdet_C = logdet_N + logdet_Phi + logdet_Sigma

        return DataObject(PSR_name=timing_model.PSR.value, dmx_ranges=np.array(valid_dmx_ranges),
                        max_inv_freq=max_inv_freq, min_inv_freq=min_inv_freq,
                        Cinv=Cinv, logdet_C=logdet_C, Ndiag=Ndiag, U=U, Sigma_cf=Sigma_cf,
                        xvals=valid_xvals, resids=valid_resids)  # , resids_errs=valid_resids_errs)


def make_plot(PSR_name, df):
    windows_centers = df["DMXR1"] + (df["DMXR2"] - df["DMXR1"]) / 2.0

    sns.set_style("ticks")
    sns.set_context("paper", font_scale=3.0)
    fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(12, 24), sharex=True,
                           gridspec_kw={'hspace': 0})
    fig.suptitle(PSR_name + " - Monomial Coefficients")

    means = df[['a0', 'a1', 'a2', 'a3', 'a4', 'a5']].mean(axis=0)
    print(f"a1 = {means['a1']}")
    print(f"a3 = {means['a3']}")
    print(f"a5 = {means['a5']}")

    # Plot and label each subplot
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    for i in range(6):
        ax[i].scatter(windows_centers, df[f'a{i}'], color=colors[i])
        ax[i].axhline(y=means[f'a{i}'], color='black', lw=4, linestyle='--')
        ax[i].set_ylabel(f"$a_{i}$")
        ax[i].grid(True)  # Add grid
        ax[i].label_outer()  # Hide inner x labels and ticks

        ax[i].text(0.2, 0.1, f'Mean = {round(means[f"a{i}"], 4)}', horizontalalignment='center', verticalalignment='center',
                 transform=ax[i].transAxes)

    ax[5].set_xlabel("Window Center [MJD]")

    plt.tight_layout()
    plt.savefig('./' + PSR_name + '_results.png')
    plt.show()

    return


def epoch_scrunch(toas, data=None, errors=None, epochs=None, decimals=0, getdict=False, weighted=False, harmonic=False):
    if epochs is None:
        epochsize = 10 ** (-decimals)
        bins = np.arange(np.around(min(toas), decimals=decimals) - epochsize,
                         np.around(max(toas), decimals=decimals) + 2 * epochsize,
                         epochsize)  # 2 allows for the extra bin to get chopped by np.histogram
        freq, bins = np.histogram(toas, bins)
        validinds = np.where(freq != 0)[0]

        epochs = np.sort(bins[validinds])
        diffs = np.array(list(map(lambda x: np.around(x, decimals=decimals), np.diff(epochs))))
        epochs = np.append(epochs[np.where(diffs > epochsize)[0]], [epochs[-1]])
    else:
        epochs = np.array(epochs)
    reducedTOAs = np.array(list(map(lambda toa: epochs[np.argmin(np.abs(epochs - toa))], toas)))

    if data is None:
        return epochs

    Nepochs = len(epochs)

    if weighted and errors is not None:
        averaging_func = lambda x, y: weighted_moments(x, 1.0 / y ** 2, unbiased=True, harmonic=harmonic)
    else:
        averaging_func = lambda x, y: (np.mean(x), np.std(y))  # is this correct?

    if getdict:
        retval = dict()
        retvalerrs = dict()
    else:
        retval = np.zeros(Nepochs)
        retvalerrs = np.zeros(Nepochs)
    for i in range(Nepochs):
        epoch = epochs[i]
        inds = np.where(reducedTOAs == epoch)[0]
        if getdict:
            retval[epoch] = data[inds]
            if errors is not None:
                retvalerrs[epoch] = errors[inds]
        else:
            if errors is None:
                retval[i] = np.mean(data[inds])  # this is incomplete
                retvalerrs[i] = np.std(data[inds])  # temporary
            else:
                retval[i], retvalerrs[i] = averaging_func(data[inds], errors[inds])
    #            print data[inds],errors[inds]
    if getdict and errors is None:  # is this correct?
        return epochs, retval
    return epochs, retval, retvalerrs


def corner_plot(samples, PSR_name):
    fig = corner.corner(
        samples,
        labels=["$a_1$", "$a_3$", "$a_5$"],
        show_titles=True,
        title_fmt=".4f",
        quantiles=[0.16, 0.5, 0.84],  # 68% credible interval
        plot_datapoints=True,
        color="blue",  # Default for contours
        hist_kwargs={"color": "gray"},
        label_kwargs={"fontsize": 14},
        title_kwargs={"fontsize": 12},
        weights=None,  # Don't weight by anything
        truths=np.median(samples, axis=0),  # Draw crosshairs at medians
        plot_density=True,
        fill_contours=True,
        contour_kwargs={"colors": ["black"]}
    )

    fig.savefig(f"./results/{PSR_name}_corner_plot.png", dpi=300)
    plt.show()

def my_leg2poly(u_leg_coeffs):

    '''
    Transform the Legendre coefficients of a degree=5 Legendre polynomial (with uncertainties in the coefficients) to
    the corresponding coefficients in the monomial base
    :param u_leg_coeffs:
    :return: u_poly_coeffs:
    '''

    # Define the transformation matrix
    T = np.array([
        [1, 0, -1 / 2, 0, 3 / 8, 0],
        [0, 1, 0, -3 / 2, 0, 15 / 8],
        [0, 0, 3 / 2, 0, -15 / 4, 0],
        [0, 0, 0, 5 / 2, 0, -35 / 4],
        [0, 0, 0, 0, 35 / 8, 0],
        [0, 0, 0, 0, 0, 63 / 8]
    ])

    # Make the conversion

    return np.dot(T, u_leg_coeffs.T)


def find_a0a2a4(PSR_name, filtered_obs, a1a3a5, plot=False):

    a0a2a4_arr = np.empty((len(filtered_obs.dmx_ranges), 3))
    a0a2a4_err_arr = np.empty((len(filtered_obs.dmx_ranges), 3))

    for n, (x, y) in enumerate(zip(filtered_obs.xvals, filtered_obs.resids)):

        window = filtered_obs.dmx_ranges[n]
        _, sorted_freqs = zip(*sorted(zip(x, filtered_obs.freqs[n])))
        sorted_freqs = np.array(sorted_freqs)
        x, y = zip(*sorted(zip(x, y)))
        x, y = np.array(x), np.array(y)

        # Calculate the coefficients for the unscaled and unshifted Legendre basis polynomials
        c1c3c5 = leg.poly2leg([0.0, a1a3a5[0], 0.0, a1a3a5[1], 0.0, a1a3a5[2]])[[1, 3, 5]]
        c0c2c4, c0c2c4_errs, _ = my_legfit_full(x=x, y=y.astype(np.float64), deg=5, coeffs=c1c3c5)

        # Assemble the Legendre series
        leg_coeffs = np.array([c0c2c4[0], c1c3c5[0], c0c2c4[1], c1c3c5[1], c0c2c4[2], c1c3c5[2]])
        pfit = leg.Legendre(leg_coeffs)

        # Find the coefficients and corresponding uncertaintites in the monomial base
        u_leg_coeffs = unumpy.umatrix(leg_coeffs, [c0c2c4_errs[0], 0.0, c0c2c4_errs[1], 0.0, c0c2c4_errs[2], 0.0])
        u_poly_coeffs = my_leg2poly(u_leg_coeffs)

        a0a2a4_arr[[n], :] = u_poly_coeffs[[0, 2, 4]].nominal_values.T
        a0a2a4_err_arr[[n], :] = u_poly_coeffs[[0, 2, 4]].std_devs.T

        if plot:
            sns.set_style("ticks")
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [1, 2], 'hspace': 0})
            ax2.plot(sorted_freqs, y, "o")
            ax2.plot(sorted_freqs, pfit(x), lw=2, label="Fitted Legendre Polynomial")
    #        ax2.set_xlabel("Legendre-Mapped Inverse Frequency")
            ax2.set_xlabel("Frequency [MHz]")
            ax2.set_ylabel(r'Residuals [$\mu s$]')
            at = AnchoredText(
                f"Power series coefficients: \n $a_0$ = {u_poly_coeffs[0].nominal_value} \n $a_1$ = {u_poly_coeffs[1].nominal_value} \n $a_2$ = {u_poly_coeffs[2].nominal_value} \n $a_3$ = {u_poly_coeffs[3].nominal_value} \n $a_4$ = {u_poly_coeffs[4].nominal_value} \n $a_5$ = {u_poly_coeffs[5].nominal_value}",
                prop=dict(size=10), frameon=True, loc='upper right')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax2.add_artist(at)
            ax2.grid()

            ax1.scatter(sorted_freqs, y - pfit(x))
            ax1.set_ylabel(r'Fit $-$ Residuals [$\mu s$]')
            ax1.grid()

            plt.suptitle("MJD " + str(window[0]) + " to " + str(window[1]))
            plt.tight_layout()
            plt.savefig(f"./results/final_fits/{PSR_name}/diffs_{n}.png")
            plt.show()

    return pd.DataFrame(np.hstack((a0a2a4_arr, a0a2a4_err_arr)), columns=['a0', 'a2', 'a4', 'a0_err', 'a2_err', 'a4_err'])


def plot_a0a2a4(PSR_name, filtered_obs, a0a2a4):

    DMXR1, DMXR2 = filtered_obs.dmx_ranges[:, 0], filtered_obs.dmx_ranges[:, 1]
    windows_centers = DMXR1 + (DMXR2 - DMXR2) / 2.0

    sns.set_style("ticks")
    sns.set_context("paper", font_scale=3.0)
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 16), sharex=True,
                           gridspec_kw={'hspace': 0})
    fig.suptitle(PSR_name + " - Monomial Coefficients")

    # Plot and label each subplot
    colors = ['C0', 'C1', 'C2']
    for i in range(3):
        ax[i].errorbar(x=windows_centers, y=a0a2a4[f'a{2*i}'], yerr=a0a2a4[f'a{2*i}_err'], color=colors[i], fmt='o')
        ax[i].set_ylabel(f"$a_{2*i}~[\mu$s]")
        ax[i].grid(True)  # Add grid
        ax[i].label_outer()  # Hide inner x labels and ticks


    # Convert a2 to more natural units
    '''
    u_a2 = unumpy.umatrix(a0a2a4['a2'], a0a2a4['a2_err'])
    D = ufloat(4.148808e9, 0.000003e9)
    u_a2_new = u_a2 / D
    a2_new = np.squeeze(np.asarray(u_a2_new.nominal_values))
    a2_new_err = np.squeeze(np.asarray(u_a2_new.std_devs))
    '''
    D = 4.148808e9
    def a2_natural(y):
        return y / D

    def a2_natural_inverse(y):
        return y * D

    secax = ax[1].secondary_yaxis('right', functions=(a2_natural, a2_natural_inverse))
    secax.set_ylabel('$a_2~[\mathrm{pc}~\mathrm{cm}^3]$')

    ax[0].errorbar(x=windows_centers, y=a0a2a4['a0'], yerr=a0a2a4[f'a0_err'], color=colors[0], fmt='o')
#    ax[1].errorbar(x=windows_centers, y=a2_new, yerr=a2_new_err, color=colors[1], fmt='o')
    ax[2].errorbar(x=windows_centers, y=a0a2a4['a4'], yerr=a0a2a4['a4_err'], color=colors[2], fmt='o')

#    ax[0].set_ylabel('$a_0~[\mu \mathrm{s}]$')
#    ax[1].set_ylabel('$a_2~[\mathrm{pc}~\mathrm{cm}^3]$')
#    ax[2].set_ylabel('$a_4~[\mu \mathrm{s}]$')
    ax[0].set_ylim([-100.0,120.0])
#    ax[1].set_ylim([0.0,150.0])
#    ax[2].set_ylim([-80.0,80.0])


    ax[2].set_xlabel("Window Center [MJD]")

    plt.tight_layout()
#    plt.savefig(f'./results/{PSR_name}/{PSR_name}_a0a2a4_results.png')
    plt.show()

    return


def get_FD_curve_values(p, freqs, DM0=0.0):

    FDfunc = p.getFDfunc()
    if FDfunc is None:
        return

    DM = p.getDM()
    ts, dmx, errs, R1s, R2s, _, _ = p.getDMX(full_output=True)
    F1s = np.amin(freqs)
    F2s = np.amax(freqs)
    F1 = np.min(F1s)/1000.0 #in GHz
    F2 = np.max(F2s)/1000.0

    fs = np.arange(F1, F2, 0.001)

#    shift = -K*((DM+dmx[0])-DM0)/fs**2
    shift = 0.0

    ys = FDfunc(fs) + shift
    ys -= np.mean(ys)

    return fs, ys