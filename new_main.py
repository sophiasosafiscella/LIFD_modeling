import numpy as np
import pandas as pd
from numpy.polynomial.legendre import legval
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt
from glob import glob
from pint.fitter import WLSFitter
from pint.models import get_model
from pint.toa import get_TOAs
from pint import logging
logging.setup("WARNING")
import astropy.units as u
from astropy.units.quantity import Quantity
import sys

def freeze_parameters(model, params_to_freeze):
    """Zero out, freeze, and set uncertainty to 0 for a list of parameter names."""
    for p in params_to_freeze:
        param = getattr(model, p)
        param.value = 0.0
        param.uncertainty_value = 0.0
        param.frozen = True


def get_FD_delay(FD_params, freqs_GHz):
    """
    Returns a function that provides the timing delays as a function of observing frequency
    """
    FD_params = FD_params[::-1]  # We need to invert the order because np.polyval uses p[0]*x**(N-1)+p[1]*x**(N-2)+...
    FD_params = np.concatenate((FD_params, [0]))  # https://numpy.org/doc/stable/reference/generated/numpy.polyval.html
    FD_func = lambda nu: 1e6 * np.polyval(FD_params, np.log(nu))  # Function that gives the timing delay
    FD_delays = FD_func(freqs_GHz)  * u.us  # nu in GHz, returns values in microseconds
    return FD_delays

def get_dmx_params(timing_model):

    dmx_params = timing_model.components["DispersionDMX"].params
    names = []
    values = np.empty(len(dmx_params))
    errors = np.empty(len(dmx_params), dtype=Quantity)

    for i, par in enumerate(dmx_params):
        names.append(getattr(timing_model, par).name)
        values[i] = getattr(timing_model, par).value

        if type(getattr(timing_model, par).uncertainty) is Quantity:
            errors[i] = getattr(timing_model, par).uncertainty
        else:
            errors[i] = None

    return pd.DataFrame({'Value': values, 'Error': errors}, index=names)

# Global parameters
#PSR_name: str = "B1937+21"
#PSR_name: str = "J0030+0451"
#PSR_name: str = "B1937+21"
#PSR_name: str = "J0610-2100"
#PSR_name: str = "J1643-1224"
#PSR_name: str = "J1012+5307"
#PSR_name: str = "J1022+1001"
#PSR_name: str = "J1024-0719"
#PSR_name: str = "J1903+0327"
#PSR_name: str = "J1741+1351"
#PSR_name: str = "J1744-1134"
#PSR_name: str = "J0613-0200"
#PSR_name: str = "J1918-0642"
#PSR_name: str = "J1909-3744"
#PSR_name: str = "J1713+0747"
#PSR_name: str = "J1918-0642"
PSR_name: str = "J2145-0750"
#PSR_name: str = "J2302+4442"

print(f"Running {PSR_name}...")

method: str = "LIFD"          # Method you want to use. Options: FD, IFD, LIFD
order: int = 6              # Order of the polynomial you want to use
fig, ax = plt.subplots(1,1)

# Input files
parfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/par/{PSR_name}_PINT_*.nb.par")[0]
timfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/tim/{PSR_name}_PINT_*.nb.tim")[0]

# Load the timing model and TOAs
timing_model = get_model(parfile)  # Ecliptical coordiantes
toas = get_TOAs(timfile, planets=True, ephem=timing_model.EPHEM.value)

# https://nanograv-pint.readthedocs.io/en/latest/_autosummary/pint.models.dispersion_model.DispersionDMX.html#pint.models.dispersion_model.DispersionDMX.dispersion_time_delay
# Change the DM
change_dm: bool = True
if change_dm:
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

if method == "IFD" or method == "LIFD":
    timing_model.remove_component("FD")  # Remove the FD model

    # The LIFD component expects to define its λ→x mapping at setup(), which requires TOAs.
    # Therefore, we will attach TOAs to model before adding LIFD:
    timing_model.toas = toas

# Remove the timing parameters, or add new ones
if method == "IFD":
    from IFD_class import IFD
    timing_model.add_component(IFD(order=order))  # Attach the IFD component
    freeze_parameters(timing_model, ['IFD0'])   # Because the AIC is telling us this term does not matter
elif method == "LIFD":
    from LIFD_class import LIFD
    timing_model.add_component(LIFD(order=order))   # Attach the LIFD component

# Fit the timing model
f = WLSFitter(toas, timing_model)
f.fit_toas()
fitted_model = f.model

# Get the y values
if method == "FD":
    FD_coeffs = [getattr(fitted_model, FD_param).value for FD_param in fitted_model.components['FD'].params]  # seconds
    np.save(f"./results/{PSR_name}/FD_coeffs.npy", FD_coeffs)
    freq_GHz = np.sort(toas.get_freqs().to(u.GHz))    # Frequencies in MHz
    delay_us = get_FD_delay(FD_coeffs, freq_GHz.value)
    x_var = freq_GHz.value
    ax.set_xlabel("$\\nu$ [GHz]")

elif method == "IFD":
    # Get the lambdas
    freq_GHz = np.sort(IFD.get_freq_GHz_from_toas(fitted_model, toas))  # Frequencies in Hz
    lambdas_ns = IFD.get_lambda_ns_from_freq_GHz(freq_GHz)  # Inverse frequencies
    x_var = lambdas_ns
    ax.set_xlabel("$\lambda$")

    IFD_coeffs = [getattr(fitted_model, f"IFD{deg}").value for deg in range(0, order)]
    for i in range(0, order):
        print(f'a_{i} = {IFD_coeffs[i]}')
    np.save(f"./results/{PSR_name}/new_IFD_coeffs.npy", IFD_coeffs)
    # This will automatically have units of seconds because the coefficients have units of seconds
    delay_us = (polyval(x=lambdas_ns, c=IFD_coeffs) * u.second).to(u.us)

elif method == "LIFD":
    # Get the x values
    freq_hz = LIFD.get_freq_hz_from_toas(fitted_model, toas)  # Frequencies in Hz
    lambdas = LIFD.get_lambda_from_freq_hz(freq_hz)  # Inverse frequencies
    lifd_comp = fitted_model.components["LIFD"]
    x_vals = np.sort(lifd_comp.map_lambda_to_unit(lambdas, lifd_comp.lmin, lifd_comp.lmax))
    x_var = x_vals
    ax.set_xlabel("x")

    LIFD_coeffs = [getattr(fitted_model, f"LIFD{i}").value for i in range(0, order)]
    np.save(f"./results/{PSR_name}/new_LIFD_coeffs.npy", LIFD_coeffs)
    delay_us = (legval(x_vals, LIFD_coeffs) * u.second).to(u.us)

    for i in range(0, order):
        print(f'LIFD{i} = {LIFD_coeffs[i]}')

else:
    print("Method not recognized")
    sys.exit(1)

# Save the DMX parameters
# https://nanograv-pint.readthedocs.io/en/latest/_modules/pint/models/dispersion_model.html#DispersionDMX.DMX_dispersion_delay
# https://nanograv-pint.readthedocs.io/en/latest/_autosummary/pint.models.dispersion_model.DispersionDMX.html#pint.models.dispersion_model.DispersionDMX.dm_value
DMX_params = get_dmx_params(fitted_model)
DMX_params.to_pickle(f"./results/{PSR_name}/{method}_DMX.pkl")

# Plot the fitted delay curve
ax.plot(x_var, delay_us)
ax.set_ylabel("Delay [$\mu$s]")
plt.suptitle(PSR_name + f"| {method} of order {order} | DM={round(fitted_model.DM.value, 4)}")
plt.tight_layout()
plt.show()
