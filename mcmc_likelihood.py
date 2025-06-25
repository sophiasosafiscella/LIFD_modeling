import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import numpy.polynomial.legendre as leg
import emcee
import pint.logging
pint.logging.setup(level="ERROR")
from fit_coefficients import my_legfit
from tqdm import tqdm

def log_prior(theta):
    a1, a3, a5 = theta  # These are priors on the MONIMIAl coefficients, not on the LEGENDRE coefficients
    if 250 < a1 < 450 and -30.0 < a3 < 30.0 and -40.0 < a5 < 40.0:
        return 0.0
    return -np.inf


def lnlike(theta, filtered_obs, weight=False):

    a1, a3, a5 = theta
    diffs_arr = []

    for x, y in zip(filtered_obs.xvals, filtered_obs.resids):

        # Calculate the coefficients for the unscaled and unshifted Legendre basis polynomials
        c1c3c5 = leg.poly2leg([0.0, a1, 0.0, a3, 0.0, a5])[[1, 3, 5]]
        c0c2c4 = my_legfit(x=x, y=y.astype(np.float64), deg=5, coeffs=c1c3c5, full=False)
        pfit = leg.Legendre(np.array([c0c2c4[0], c1c3c5[0], c0c2c4[1], c1c3c5[1], c0c2c4[2], c1c3c5[2]]))

        # Calculate the difference between the fitted Legendre series and the residuals
        diffs_arr.append(y - pfit(x))

        '''
        fig2, ax2 = plt.subplots()
        ax2.plot(x, y, "o")
        ax2.plot(x, pfit(x), lw=2, label="Fitted Legendre Polynomial")
        ax2.set_xlabel("Normalized Inverse Frequency")
        ax2.set_ylabel(r'Residuals [$\mu s$]')
        at = AnchoredText(
        f"Legendre series coefficients: \n $C_0$ = {c0c2c4[0]} \n $C_1$ = {c1c3c5[0]} \n $C_2$ = {c0c2c4[1]} \n $C_3$ = {c1c3c5[1]} \n $C_4$ = {c0c2c4[2]} \n $C_5$ = {c1c3c5[2]}",
                                      prop=dict(size=10), frameon=True, loc='upper left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax2.add_artist(at)
        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
        '''

    # Use these differences to calculate the ln(likelihood)
    diffs_arr = np.concatenate(diffs_arr)
    sig_tot = np.concatenate(filtered_obs.resids_errs)

    lnL = -0.5 * np.sum(np.log(2 * np.pi) + 2 * np.log(sig_tot) + (diffs_arr / sig_tot) ** 2)
    return lnL


def lnprob(theta, filtered_obs, weight=False):
    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, filtered_obs, weight=weight)


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
