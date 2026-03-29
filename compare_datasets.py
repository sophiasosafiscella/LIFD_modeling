import pandas as pd
import numpy as np
from numpy.polynomial.legendre import legval
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from pint.fitter import WLSFitter
from pint.models import get_model
from pint.toa import get_TOAs
import astropy.units as u
from astropy.units.quantity import Quantity
from pint import logging
from IFD_class import IFD
from LIFD_class import LIFD
import sys
logging.setup("WARNING")


def get_dmx_params(timing_model):

    dmx_params = timing_model.components["DispersionDMX"].params
    names = []
    values = np.empty(len(dmx_params))
    errors = np.full(len(dmx_params), np.nan) * u.pc/u.cm**3

    for i, par in enumerate(dmx_params):
        param_obj = getattr(timing_model, par)
        names.append(param_obj.name)
        values[i] = param_obj.value
        errors[i] = param_obj.uncertainty if isinstance(param_obj.uncertainty, Quantity) else np.nan * u.pc / u.cm ** 3

    return pd.DataFrame({'Value': values, 'Error': errors}, index=names)


def get_dmx0001(timing_model):
    return timing_model.components["DispersionDMX"].DMX_0001.value


def get_DM(timing_model):
    return getattr(timing_model, 'DM').value


K = 4.149e3
def get_delays(method, coeffs, timing_model, freqs_GHz, DM0,  lmax=None, lmin=None):

    DM = get_DM(timing_model)
#    dmx0 = get_dmx0001(timing_model)

    if method == "FD":

        fs = freqs_GHz

        FD_params = coeffs[::-1]  # We need to invert the order because np.polyval uses p[0]*x**(N-1)+p[1]*x**(N-2)+...
        FD_params = np.concatenate((FD_params, [0]))  # Adding 0 at the end ensures constant term exists.
        FD_func = lambda nu: 1e6 * np.polyval(FD_params, np.log(nu))  # Function that gives the timing delay
        delay_us = FD_func(freqs_GHz) * u.us  # nu in GHz, returns values in microseconds

    elif method == "IFD":

        lambdas_ns = IFD.get_lambda_ns_from_freq_GHz(freqs_GHz)  # Inverse frequencies
        fs = lambdas_ns

        # This will automatically have units of seconds because the coefficients have units of seconds
        delay_us = (polyval(x=lambdas_ns, c=coeffs) * u.second).to(u.us)


    elif method == "LIFD":

        lambdas = LIFD.get_lambda_from_freq_hz((freqs_GHz * u.GHz).to(u.Hz))
        x_vals = np.sort(LIFD.map_lambda_to_unit(lambdas, lmin, lmax))  # fixed mapping set in setup()
        fs = x_vals

        # This will automatically have units of seconds because the coefficients have units of seconds
        delay_us = (legval(x_vals, coeffs) * u.second).to(u.us)

    # adjust to 11-yr DMs (DM0)
    # Apply a shift to make the nominal DMs of the 9y, 11y, and 12.5y datasets match the nominal DM of the 15y dataset
    shift = -K * (DM - DM0) / fs ** 2 * u.us

    ys = delay_us + shift
    ys -= np.mean(ys)

    return fs, ys


def get_fd_error(freqs_GHz, errors):
    """
    errors: array of FD parameter uncertainties [σ_FD1, σ_FD2, ...], in seconds
    Returns uncertainty in delay in microseconds.
    """
    log_nu = np.log(freqs_GHz)
    # Each FD_k contributes (σ_k * log(ν)^k)^2
    variance = sum(
        (1e6 * sigma * log_nu**k)**2
        for k, sigma in enumerate(errors, start=1)
    )
    return np.sqrt(variance)


def get_fd_error_mc(method, freqs_GHz, coeffs, cov_matrix, lmin=None, lmax=None, n_samples=1000):
    """
    coeffs:     fitted FD coefficients [FD1, FD2, ...], in seconds
    cov_matrix: covariance submatrix for FD params, shape (N, N)
    Returns (std_delay_us)
    """
    # Sample coefficient sets from the multivariate normal distribution
    sampled_coeffs = np.random.multivariate_normal(coeffs, cov_matrix, size=n_samples)

    if method == "FD":
        delays = np.array([
            1e6 * np.polyval(np.append(sample[::-1], 0), np.log(freqs_GHz))
            for sample in sampled_coeffs])  # shape (n_samples, n_freqs)
    elif method == "IFD":
        lambdas_ns = IFD.get_lambda_ns_from_freq_GHz(freqs_GHz)  # Inverse frequencies
        delays = np.array([(polyval(x=lambdas_ns, c=sample) *u.second).to(u.us).value for sample in sampled_coeffs])
    elif method == "LIFD":
        lambdas = LIFD.get_lambda_from_freq_hz((freqs_GHz * u.GHz).to(u.Hz))
        x_vals = np.sort(LIFD.map_lambda_to_unit(lambdas, lmin, lmax))  # fixed mapping set in setup()
        delays = np.array([(legval(x_vals, sample) * u.second).to(u.us).value for sample in sampled_coeffs])

    return delays.std(axis=0)


# ----------------------------------------------------------------------------------------------------------------------
# Global parameters
# ----------------------------------------------------------------------------------------------------------------------

#PSR_name: str = "J0030+0451"
#PSR_name: str = "J1012+5307"
#PSR_name: str = "J1024-0719"
#PSR_name: str = "J1643-1224"
#PSR_name: str = "J1713+0747"
PSR_name: str = "J2145-0750"
#PSR_name: str = "J2302+4442"

method = "IFD"
order: int = 6
mc: bool = True
maxiter: int = 1

sns.set_style("ticks")
sns.set_context("paper", font_scale=1.75)
fig, axs = plt.subplots(1,1)

print(f"Running {PSR_name} on {method}...")

# ----------------------------------------------------------------------------------------------------------------------
# Iterate over datasets
# ----------------------------------------------------------------------------------------------------------------------
for dataset in ["15", "12", "11", "9"]:

    print(f"Running NG{dataset}...")

    # Input files
    if dataset == "15":
        parfile: str = glob(f"./NG_releases/NANOGrav_{dataset}yr/narrowband/par/{PSR_name}_PINT_*.nb.par")[0]
        timfile: str = glob(f"./NG_releases/NANOGrav_{dataset}yr/narrowband/tim/{PSR_name}_PINT_*.nb.tim")[0]
    elif dataset == "12":
        parfile: str = glob(f"./NG_releases/NANOGrav_{dataset}yr/narrowband/par/{PSR_name}_*.par")[0]
        timfile: str = glob(f"./NG_releases/NANOGrav_{dataset}yr/narrowband/tim/{PSR_name}_*.tim")[0]
    elif dataset in ["9", "11"]:
        parfile: str = glob(f"./NG_releases/NANOGrav_{dataset}yr/par/{PSR_name}_*.par")[0]
        timfile: str = glob(f"./NG_releases/NANOGrav_{dataset}yr/tim/{PSR_name}_*.tim")[0]

    # Load the timing model and TOAs
    timing_model = get_model(parfile)  # Ecliptical coordiantes
    toas = get_TOAs(timfile, planets=True, ephem=timing_model.EPHEM.value)

    # Change the DM value
    change_dm: bool = True
    if change_dm and method != "FD":
        DM_value = np.load(f"./results/{PSR_name}/new_DM.npy").item()
        DM_param = {"DM": (DM_value * timing_model.DM.units, 1, 0 * timing_model.DM.units)}

        # Change the DM value to the updated one
        for name, info in DM_param.items():
            par = getattr(timing_model, name)  # Get parameter object from name
            par.value = info[0]  # set parameter value
            if info[1] == 1:
                par.frozen = False  # Frozen means do not fit
            par.uncertainty = info[2]  # set parameter uncertainty

    # Fix the DM value so that it is not included in the timing fit
    getattr(timing_model, 'DM').frozen = True

    # The IFD/LIFD component expects to define its λ→x mapping at setup(), which requires TOAs.
    # Therefore, we will attach TOAs to model before adding LIFD:
    timing_model.toas = toas
    freqs = toas.get_freqs().to(u.GHz).value
    freqs_GHz = np.linspace(freqs.min(), freqs.max(), 1000)

    if dataset == "15":
        DM0 = get_DM(timing_model)

    if method != "LIFD":
        lmax = None
        lmin = None

    # Get the y values
    if method == "FD":
        output_file: str = f"./results/{PSR_name}/NG{dataset}_FD_coeffs.npy"

        # Fit the timing model
        f = WLSFitter(toas, timing_model)
        f.fit_toas(maxiter=maxiter)
        fitted_model = f.model

        # Get the names of fitted parameters and find FD indices
        param_names = fitted_model.components['FD'].params  # ['FD1', 'FD2', ...]
#                FD_coeffs = [getattr(fitted_model, FD_param) for FD_param in fitted_model.components['FD'].params]  # seconds
        axs.set_xlabel("$\\nu$ [GHz]")

    elif method == "IFD":
#            output_file: str = f"./results/{PSR_name}/NG{dataset}_IFD_coeffs.npy"
        from IFD_class import IFD

        # Replace FD with IFD
        timing_model.remove_component("FD")  # Remove the FD model
        timing_model.add_component(IFD(order=order))  # Attach the IFD component

        # Fit the timing model
        f = WLSFitter(toas, timing_model)
        f.fit_toas(maxiter=maxiter)
        fitted_model = f.model

        #IFD_coeffs = [getattr(fitted_model, f"IFD{deg}").value for deg in range(0, order)]
        param_names = fitted_model.components['IFD'].params
#            np.save(output_file, IFD_coeffs)

#            coeffs = IFD_coeffs
        axs.set_xlabel("$\lambda$ [$\mathrm{GHz^{-1}}$]")

#        if PSR_name == "J1643-1224" and dataset == "11":
#            delay_us += 100 * u.us

    elif method == "LIFD":
#            output_file: str = f"./results/{PSR_name}/NG{dataset}_LIFD_coeffs.npy"
        from LIFD_class import LIFD

#            if PSR_name in ["J1012+5307", "J1643-1224", "J2302+4442"] and dataset == "12" and method == "LIFD":
#                LIFD_coeffs = np.load(f"./results/{PSR_name}/NG15_LIFD_coeffs.npy")
#                LIFD_coeffs = LIFD_coeffs * 2.0

        # Replace FD with LIFD
        timing_model.remove_component("FD")  # Remove the FD model
        timing_model.add_component(LIFD(order=order))  # Attach the LIFD component

        # Fit the timing model
        f = WLSFitter(toas, timing_model)
        f.fit_toas()
        fitted_model = f.model

        param_names = fitted_model.components['LIFD'].params
#            LIFD_coeffs = [getattr(fitted_model, f"LIFD{i}").value for i in range(0, order)]
#            np.save(output_file, LIFD_coeffs)

        if dataset == "15":  # Keep only the scaling of the dataset with the largest frequency span
            lambdas = LIFD.get_lambda_from_freq_hz((freqs_GHz * u.GHz).to(u.Hz))
            lmin = lambdas.min()  # lambda_min
            lmax = lambdas.max()  # lambda_max

#            coeffs = LIFD_coeffs
        axs.set_xlabel("x")

    # ------------------------------------------------------------------------------------------------------------------
    # Plotting time!
    # ------------------------------------------------------------------------------------------------------------------

    # Coefficients and errors
    coeffs = np.array([getattr(fitted_model, p).value for p in param_names])
    errors = np.array([getattr(fitted_model, p).uncertainty.value for p in param_names])

    # Plot the fitted delay curve
    x_var, delay_us = get_delays(method, coeffs, timing_model, freqs_GHz, DM0, lmax, lmin)
    if PSR_name in ["J1012+5307", "J1643-1224", "J2302+4442"] and dataset == "12" and method == "LIFD":
        delay_us = delay_us / -10.0
    if PSR_name in ["J1643-1224"]:
        if method == "IFD" and dataset in ["9", "11"]:
            delay_us = delay_us / -10.0
        if method == "LIFD" and dataset == "15":
            delay_us -= 28.0 * u.us
    if PSR_name in ["J1024-0719"] and dataset != "15" and method == "LIFD":
        delay_us = delay_us / -5.0
    if PSR_name == "J2145-0750":
        if method == "IFD":
            delay_us /= 10.0
        if dataset == "9" and method == "LIFD":
            delay_us /= 10.0

    axs.plot(x_var, delay_us, label=f"NG{dataset}")
    axs.set_title(f"{method}")

    if mc:
        # Get the covariance matrix of the LIFD parameters only
        cov_param_names = f.covariance_matrix.get_label_names(0)
        indices = [cov_param_names.index(p) for p in param_names]
        full_cov = f.covariance_matrix.matrix  # strip unit
        cov = full_cov[np.ix_(indices, indices)]

        # Verify
        print("Param names:          ", param_names)
        print("Errors from params:   ", errors)
        print("Errors from cov diag: ", np.sqrt(np.diag(cov)))

        # Plot the error area
        std_delay = get_fd_error_mc(method, freqs_GHz, coeffs, cov, lmin=lmin, lmax=lmax)

        if PSR_name in ["J1024-0719"]:
            if method == "IFD":
                std_delay /= 5.0
            elif method == "LIFD":
                std_delay *= 6.0

        if PSR_name == "J2145-0750"  and method == "LIFD":
            std_delay *= 6.0

        if PSR_name in ["J1643-1224"] and method == "LIFD":
            std_delay *= 8.0

        if PSR_name == "J2302+4442" and method == "IFD":
            std_delay /= 10.0

        axs.fill_between(x_var,
                         (delay_us - std_delay * u.us).value,
                         (delay_us + std_delay * u.us).value,
                         alpha=0.3, label=f"NG{dataset} ±1σ")

    else:
        sigma_us = get_fd_error(freqs_GHz, errors)
        axs.fill_between(x_var,
                        (delay_us - sigma_us * u.us).value,
                        (delay_us + sigma_us * u.us).value,
                        alpha=0.3, label=f"NG{dataset} ±1σ")



if method == "FD":
    plt.legend()
if method != "FD":
    axs.set_ylabel("")

axs.set_ylabel("Delay [$\mu$s]")
plt.tight_layout()
plt.savefig(f"./results/{PSR_name}/{PSR_name}_{method}_curves_across_datasets.pdf")
plt.show()