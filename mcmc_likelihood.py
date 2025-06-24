import numpy as np
import numpy.polynomial.legendre as leg
import emcee
from pint.residuals import Residuals
import pint.logging
pint.logging.setup(level="ERROR")
import astropy.units as u
from utils import get_dmx_observations, map_domain
from fit_coefficients import my_legfit

def log_prior(theta):
    a1, a3, a5 = theta  # These are priors on the MONIMIAl coefficients, not on the LEGENDRE coefficients
    if 250 < a1 < 450 and -30.0 < a3 < 30.0 and -40.0 < a5 < 40.0:
        return 0.0
    return -np.inf


def lnlike(theta, pulsar_data, weight=False):

    a1, a3, a5 = theta

    # Array to store the differences between the fitted Legendre series and the residuals
    diffs_arr = np.full(pulsar_data.toas.ntoas, np.nan)

    # Array to store the TOA errors
    sig_tot = np.full(pulsar_data.toas.ntoas, np.nan)

    N: int = 0
    for n, window in enumerate(pulsar_data.dmx_ranges):

        # We find the observations in the windows and the corresponding frequencies
        observations_in_window = get_dmx_observations(pulsar_data.toas, window[0], window[1])
        n_observations_in_window = observations_in_window.ntoas
        frequencies = np.array(observations_in_window.get_freqs().to(u.GHz).value)  # Convert frequencies to GHz

        # We make sure there are observations in both bands
        lowerband_ok = np.any((0.725 <= frequencies) & (frequencies <= 0.916))
        upperband_ok = np.any((1.156 <= frequencies) & (frequencies <= 1.882))
        if (not lowerband_ok) or (not upperband_ok):
#            print("Skipped a window because it didn't have observations in both bands")
            continue

        # Calculate residuals (in microseconds) with the simplified model
        res_object = Residuals(observations_in_window, pulsar_data.tm)
        residuals = res_object.time_resids.to(u.us).value
        frequencies, residuals = zip(*sorted(zip(frequencies, residuals)))
        frequencies, residuals = np.array(frequencies), np.array(residuals)

        # Map the frequencies to the range [-1, 1]
        x_values = map_domain(frequencies)

        # Calculate the coefficients for the unscaled and unshifted Legendre basis polynomials
        c1c3c5 = leg.poly2leg([0.0, a1, 0.0, a3, 0.0, a5])[[1, 3, 5]]
        c0c2c4 = my_legfit(x=x_values, y=residuals.astype(np.float64), deg=5, coeffs=c1c3c5, full=False)
        leg_pfit_coef = np.array([c0c2c4[0], c1c3c5[0], c0c2c4[1], c1c3c5[1], c0c2c4[2], c1c3c5[2]])
        pfit = leg.Legendre(leg_pfit_coef)

        # Calculate the difference between the fitted Legendre series and the residuals
        diffs_arr[N: N+n_observations_in_window] = residuals - pfit(x_values)
        sig_tot[N: N+n_observations_in_window] = res_object.get_data_error()    # TODO: we are assuming there's no correlation (for now)
        N += n_observations_in_window

    # Use these differences to calculate the ln(likelihood)
    diffs_arr = diffs_arr[np.logical_not(np.isnan(diffs_arr))]
    sig_tot = sig_tot[np.logical_not(np.isnan(sig_tot))]

    lnL = -(N / 2) * np.log(2 * np.pi) - np.log(sig_tot).sum() - 0.5 * np.power(diffs_arr/ sig_tot,
                                                                                2).sum()  # can speed up the square root of squaring

    return lnL


def lnprob(theta, pulsar_data, weight=False):
    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, pulsar_data, weight=weight)


# http://jakevdp.github.io/blog/2015/08/07/frequentism-and-bayesianism-5-model-selection/
def compute_mcmc(lnprob, args, pinit, nwalkers=10, niter=500, threads=4):

    # Number of variables we're MCMCing over
    ndim = len(pinit)

    # Initial position in the 3D space of (C1, C3, C5) from where the walkers will start
    nwalkers = 3 * ndim            # emcee requires nwalkers > ndim
    p0 = pinit + 1e-4 * np.random.randn(nwalkers, ndim)
#    p0 = [pinit + 1e-4*pinit*np.random.randn(ndim) for i in range(nwalkers)]

    # Set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args, threads=threads)
    sampler.run_mcmc(p0, niter, progress=True)

    # Burn the ends of the samples chains
    if niter <= 50000:
        burn = niter/10
    if burn > 5000:
        burn = 5000

    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))  #can also just use flatchain
    return samples
