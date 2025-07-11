import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import numpy.polynomial.legendre as leg
import emcee
from scipy.linalg import cho_solve
import pint.logging
import astropy.units as u
pint.logging.setup(level="ERROR")
from fit_coefficients import my_legfit_minimal
import sys

def load_pinit(psr_name):

    if psr_name == "J1643-1224":
        return np.array([342.9607408562299, 3.6656647122634305, -0.21600401837611255])
    elif psr_name == "J1024-0719":
        return np.array([-6226.9, -0.6, -6.0])
    elif psr_name == "J1903+0327":
        return np.array([-84.1345, 35.3294, -16.9185])


def log_prior(theta, psr_name):
    a1, a3, a5 = theta  # These are priors on the MONIMIAl coefficients, not on the LEGENDRE coefficients
    if psr_name == "J1643-1224":
        if 50 < a1 < 450 and -60.0 < a3 < 30.0 and -80.0 < a5 < 40.0:
            return 0.0
    elif psr_name == "J1024-0719":
        if -7000 < a1 < -4000 and -1000.0 < a3 < 1500.0 and -100.0 < a5 < 800.0:
            return 0.0
    elif psr_name == "J1903+0327":
        if -1000 < a1 < 1000 and -1000.0 < a3 < 1000.0 and -1000.0 < a5 < 1000.0:
            return 0.0
    return -np.inf


def fit_and_eval(args):
    x, y, c1c3c5 = args
    c0c2c4 = my_legfit_minimal(x, y, deg=5, coeffs=c1c3c5)
    coeffs = [c0c2c4[0], c1c3c5[0], c0c2c4[1], c1c3c5[1], c0c2c4[2], c1c3c5[2]]
    pfit = leg.Legendre(coeffs)
    return y - pfit(x)


def lnlike(theta, data_obj, weight=False):

    a1, a3, a5 = theta
    c1c3c5 = leg.poly2leg([0.0, a1, 0.0, a3, 0.0, a5])[[1, 3, 5]]

    diffs_arr = []

    for x, y in zip(data_obj.xvals, data_obj.resids):

        # Calculate the coefficients for the unscaled and unshifted Legendre basis polynomials
        c0c2c4 = my_legfit_minimal(x=x, y=y.astype(np.float64), deg=5, coeffs=c1c3c5)
        coeffs = np.array([
            c0c2c4[0], c1c3c5[0],
            c0c2c4[1], c1c3c5[1],
            c0c2c4[2], c1c3c5[2]
        ])
        pfit = leg.Legendre(coeffs)

        # Calculate the difference between the fitted Legendre series and the residuals
        diffs_arr.append(y - pfit(x))

    # Use these differences to calculate the ln(likelihood)
    diffs_arr = (np.concatenate(diffs_arr) * u.us).to(u.s).value

#    sig_tot = np.concatenate(data_obj.resids_errs)
#    lnL = -0.5 * np.sum(np.log(2 * np.pi) + 2 * np.log(sig_tot) + (diffs_arr / sig_tot) ** 2)
#    lnlikelihood = -0.5 * (diffs_arr @ data_obj.Cinv @ diffs_arr.T + data_obj.logdet_C)

    return lnlikelihood(data_obj, diffs_arr)


def lnlikelihood(pre_calcs, s) -> float:

    """
    Compute the chi2 when correlated noise is present in the timing model.
    If the system is not singular, it uses Cholesky factorization to evaluate this.
    If the system is singular, it uses singular value decomposition instead.
    """

    x_Ninv_y = np.sum(s * s / pre_calcs.Ndiag)
    x_Ninv_U = (s / pre_calcs.Ndiag) @ pre_calcs.U
    y_Ninv_U = (s / pre_calcs.Ndiag) @ pre_calcs.U

    x_Cinv_y = x_Ninv_y - x_Ninv_U @ cho_solve(pre_calcs.Sigma_cf, y_Ninv_U)

    return -(x_Cinv_y / 2 + pre_calcs.logdet_C / 2)


def lnprob(theta, data_obj, weight=False):
    lp = log_prior(theta, data_obj.PSR_name)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, data_obj, weight=weight)


# http://jakevdp.github.io/blog/2015/08/07/frequentism-and-bayesianism-5-model-selection/
def compute_mcmc(lnprob, args, pinit, nwalkers=10, niter=20000):

    # Number of variables we're MCMCing over
    ndim = len(pinit)

    # Initial position in the 3D space of (C1, C3, C5) from where the walkers will start
    nwalkers = 3 * ndim            # emcee requires nwalkers > ndim
    p0 = pinit + 1e-4 * np.random.randn(nwalkers, ndim)

    # Set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args, threads=8)
    sampler.run_mcmc(p0, niter, progress=True)

    # Burn the ends of the samples chains
    if niter <= 50000:
        burn = int(niter/10)
#        burn = int(niter / 5)
    if burn > 5000:
        burn = 5000

    samples = sampler.get_chain(discard=burn, flat=True)
    return samples
