import numpy as np
from numpy.polynomial.legendre import legval
import matplotlib.pyplot as plt
from glob import glob
from pint.fitter import WLSFitter
from pint.models import get_model
from pint.toa import get_TOAs
from new_LIFD import LIFD


# Global parameters
#PSR_name: str = "J0030+0451"
#PSR_name: str = "B1937+21"
#PSR_name: str = "J1643-1224"
#PSR_name: str = "J1024-0719"
#PSR_name: str = "J1903+0327"
#PSR_name: str = "J1741+1351"
#PSR_name: str = "J1744-1134"
#PSR_name: str = "J0613-0200"
#PSR_name: str = "J2145-0750"
#PSR_name: str = "J1909-3744"
PSR_name: str = "J1918-0642"

print(f"Running {PSR_name}...")

output_par: str = f"./results/{PSR_name}/new_LIFD.par"
n_LIFD: int = 5

# Input files
parfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/par/{PSR_name}_PINT_*.nb.par")[0]
timfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/tim/{PSR_name}_PINT_*.nb.tim")[0]
# Load the timing model and TOAs
timing_model = get_model(parfile)  # Ecliptical coordiantes
toas = get_TOAs(timfile, planets=True, ephem=timing_model.EPHEM.value)

# Change the DM
print(timing_model.DM)

change_dm: bool = True
if change_dm:
    if PSR_name == "J2145-0750":
        params = {"DM": (8.999792308264002 * timing_model.DM.units, 1, 0 * timing_model.DM.units)}
    elif PSR_name == "J1909-3744":
        params = {"DM": (10.391378695991417 * timing_model.DM.units, 1, 0 * timing_model.DM.units)}
    elif PSR_name == "J1918-0642":
        #params = {"DM": (26.5889 * timing_model.DM.units, 1, 0 * timing_model.DM.units)}
        #params = {"DM": (26.589879853155644 * timing_model.DM.units, 1, 0 * timing_model.DM.units)}
        params = {"DM": (26.9 * timing_model.DM.units, 1, 0 * timing_model.DM.units)}
    elif PSR_name == "J1744-1134":
        params = {"DM": (3.139 * timing_model.DM.units, 1, 0 * timing_model.DM.units)}
        #params = {"DM": (3.14021942492064 * timing_
        # model.DM.units, 1, 0 * timing_model.DM.units)}

    for name, info in params.items():
        par = getattr(timing_model, name)  # Get parameter object from name
        par.value = info[0]  # set parameter value
        if len(info) > 1:
            if info[1] == 1:
                par.frozen = True  # Frozen means do not fit
            par.uncertainty = info[2]  # set parameter uncertainty

    print(timing_model.DM)

# Remove the FD model
timing_model.remove_component("FD")

# The LIFD component expects to define its λ→x mapping at setup(), which requires TOAs.
# Therefore, we will attach TOAs to model before adding LIFD:
timing_model.toas = toas

# Attach the LIFD component
lifd = LIFD(order=n_LIFD)
timing_model.add_component(lifd)

# Now LIFD1, LIFD2, … LIFD4 are free parameters
#print(timing_model.params)
#print(timing_model.free_params)

# Fitting
f = WLSFitter(toas, timing_model)
f.fit_toas()
#    f.model.write_parfile(output_par, "wt")
new_model = f.model
print(new_model.free_params)

# Get the x values
freq_hz = LIFD.get_freq_hz_from_toas(new_model, toas)  # Frequencies in Hz
lambdas = LIFD.get_lambda_from_freq_hz(freq_hz)  # Inverse frequencies
lmin = lambdas.min()  # lambda_min
lmax = lambdas.max()  # lambda_max
x_vals = np.sort(LIFD.map_lambda_to_unit(lambdas, lmin, lmax))  # fixed mapping set in setup()

# Get the y values
coeffs = [getattr(new_model, f"LIFD{i}").value for i in range(1, n_LIFD+1)]
y_vals = legval(x_vals, coeffs)
y_vals_musec = y_vals * 1e6

fig, ax = plt.subplots(1,1)
ax.plot(x_vals, y_vals_musec)
ax.set_xlabel("x")
ax.set_ylabel("Delay [$\mu$s]")
plt.suptitle(PSR_name + f" | DM={timing_model.DM.value}")
plt.tight_layout()
plt.show()

print(new_model.LIFD1)
print(new_model.LIFD2)
print(new_model.LIFD3)
print(new_model.LIFD4)
print(new_model.LIFD5)
