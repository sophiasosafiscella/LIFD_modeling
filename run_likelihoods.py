import numpy as np
from glob import glob
from pint.models import get_model
from pint.toa import get_TOAs
from utils import filter_observations
from mcmc_likelihood import lnprob, compute_mcmc
import pickle
import os.path


def save_samples(samples,filename="samples.npy"):
    np.save("samples/"+filename,samples)

# Global parameters
PSR_name: str = "J1643-1224"
weight: bool = False

# Input files
parfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/par/{PSR_name}_PINT_*.nb.par")[0]
timfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/tim/{PSR_name}_PINT_*.nb.tim")[0]
pickle_file: str = f"./results/{PSR_name}_filtered_obs.pkl"

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
pinit = np.array([350.0, 5.0, 0.0])

# Run the MCMC sampler
samples = compute_mcmc(lnprob, (filtered_obs, weight), pinit)
save_samples(samples,filename=f"{PSR_name}_samples")
