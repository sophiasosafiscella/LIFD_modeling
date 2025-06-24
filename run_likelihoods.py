import numpy as np
from glob import glob
from pint.models import get_model
from pint.toa import get_TOAs
import utils
from utils import broadband_observations, get_dmx_ranges
PulsarData = utils.PulsarData
from mcmc_likelihood import lnprob, compute_mcmc

class PulsarData:
    def __init__(self, name, toas, tm, dmx_ranges):
        self.name = name
        self.toas = toas
        self.tm = tm
        self.dmx_ranges = dmx_ranges

    # The __str__() function controls what should be returned when the class object is represented as a string.
    def __str__(self):
        return f"{self.name}"


def save_samples(samples,filename="samples.npy"):
    np.save("samples/"+filename,samples)


# Global parameters
PSR_name: str = "J1643-1224"
weight: bool = False

# Input files
parfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/par/{PSR_name}_PINT_*.nb.par")[0]
timfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/tim/{PSR_name}_PINT_*.nb.tim")[0]

# Load the timing model and TOAs
timing_model = get_model(parfile)  # Ecliptical coordiantes
toas = get_TOAs(timfile, planets=True, ephem=timing_model.EPHEM.value)

# Get the TOAs obtained with GUPPI
GUPPI_toas = broadband_observations(toas)

# Find the DMX windows
dmx_ranges = get_dmx_ranges(timing_model, GUPPI_toas)

# Get rid of the DMX and FD parameters to create the simplified timing model
timing_model.remove_component("DispersionDMX")
timing_model.remove_component("FD")

# Build the PulsarData object
pulsar_data = PulsarData(name=PSR_name, toas=GUPPI_toas, tm=timing_model, dmx_ranges=dmx_ranges)

# Initial position in the 3D space of (C1, C3, C5) from where the walkers will start. I got the values from the
# plots I created previously
pinit = np.array([350.0, 5.0, 0.0])

# Run the MCMC sampler
samples = compute_mcmc(lnprob, (pulsar_data, weight), pinit)
save_samples(samples,filename=f"{PSR_name}_samples")
