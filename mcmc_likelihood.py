import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import numpy.polynomial.legendre as leg
import emcee
from scipy.linalg import cho_factor, cho_solve
from pint.residuals import Residuals
import pint.logging
import astropy.units as u
pint.logging.setup(level="ERROR")
from fit_coefficients import my_legfit
from typing import (Tuple)
import sys

def log_prior(theta):
    a1, a3, a5 = theta  # These are priors on the MONIMIAl coefficients, not on the LEGENDRE coefficients
    if 250 < a1 < 450 and -30.0 < a3 < 30.0 and -40.0 < a5 < 40.0:
        return 0.0
    return -np.inf


def lnlike(theta, filtered_obs, weight=False):

    a1, a3, a5 = theta
    c1c3c5 = leg.poly2leg([0.0, a1, 0.0, a3, 0.0, a5])[[1, 3, 5]]

    diffs_arr = []

    for x, y in zip(filtered_obs.xvals, filtered_obs.resids):

        # Calculate the coefficients for the unscaled and unshifted Legendre basis polynomials
        c0c2c4 = my_legfit(x=x, y=y.astype(np.float64), deg=5, coeffs=c1c3c5, full=False)
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

#    sig_tot = np.concatenate(filtered_obs.resids_errs)
#    lnL = -0.5 * np.sum(np.log(2 * np.pi) + 2 * np.log(sig_tot) + (diffs_arr / sig_tot) ** 2)

    return lnlikelihood(filtered_obs.res_object, diffs_arr)


def lnlikelihood(res_object, s) -> float:
    """Compute the log-likelihood for the model and TOAs."""

    """
    Compute the chi2 when correlated noise is present in the timing model.
    If the system is not singular, it uses Cholesky factorization to evaluate this.
    If the system is singular, it uses singular value decomposition instead.

    If `lognorm=True` is given, the log-normalization-factor of the likelihood
    function will also be returned.
    """

    Ndiag = res_object.get_data_error().to_value(u.s) ** 2
    U = res_object.model.noise_model_designmatrix(res_object.toas)
    Phidiag = res_object.model.noise_model_basis_weight(res_object.toas)

    U = np.append(U, np.ones((len(res_object.toas), 1)), axis=1)
    Phidiag = np.append(Phidiag, [1e40])

    # See Eq. (11) in https://iopscience.iop.org/article/10.3847/1538-4357/ad59f7/pdf
    chi2, logdet_C = woodbury_dot(Ndiag, U, Phidiag, s, s)

    return -(chi2 / 2 + logdet_C / 2)


def calc_gls_chi2(res_object, s, lognorm: bool = False) -> float:
    """Compute the chi2 when correlated noise is present in the timing model.
    If the system is not singular, it uses Cholesky factorization to evaluate this.
    If the system is singular, it uses singular value decomposition instead.

    If `lognorm=True` is given, the log-normalization-factor of the likelihood
    function will also be returned.
    """

#    s = res_object.time_resids.to_value(u.s)  # See Eq. (8) in https://iopscience.iop.org/article/10.3847/1538-4357/ad59f7/pdf
    Ndiag = res_object.get_data_error().to_value(u.s) ** 2
    U = res_object.model.noise_model_designmatrix(res_object.toas)
    Phidiag = res_object.model.noise_model_basis_weight(res_object.toas)

    U = np.append(U, np.ones((len(res_object.toas), 1)), axis=1)
    Phidiag = np.append(Phidiag, [1e40])

    # See Eq. (11) in https://iopscience.iop.org/article/10.3847/1538-4357/ad59f7/pdf
    chi2, logdet_C = woodbury_dot(Ndiag, U, Phidiag, s, s)

    return (chi2, logdet_C / 2) if lognorm else chi2


def woodbury_dot(
    Ndiag: np.ndarray, U: np.ndarray, Phidiag: np.ndarray, x: np.ndarray, y: np.ndarray
) -> Tuple[float, float]:
    """
    Compute an inner product of the form
        (x| C^-1 |y)
    where
        C = N + U Phi U^T ,
    N and Phi are diagonal matrices, using the Woodbury
    identity
        C^-1 = N^-1 - N^-1 - N^-1 U Sigma^-1 U^T N^-1
    where
        Sigma = Phi^-1 + U^T N^-1 U

    Additionally,
        det[C] = det[N] * det[Phi] * det[Sigma]

    Paremeters
    ----------
    Ndiag: array-like
        Diagonal elements of the diagonal matrix N
    U: array-like
        A matrix that represents a rank-n update to N
    Phidiag: array-like
        Weights associated with the rank-n update
    x: array-like
        Vector 1 for the inner product
    y: array-like
        Vector 2 for the inner product

    Returns
    -------
    result: float
        The inner product
    logdetC: float
        log-determinant of C
    """

    x_Ninv_y = np.sum(x * y / Ndiag)
    x_Ninv_U = (x / Ndiag) @ U
    y_Ninv_U = (y / Ndiag) @ U
    Sigma = np.diag(1 / Phidiag) + (U.T / Ndiag) @ U
    Sigma_cf = cho_factor(Sigma)

    x_Cinv_y = x_Ninv_y - x_Ninv_U @ cho_solve(Sigma_cf, y_Ninv_U)

    logdet_N = np.sum(np.log(Ndiag))
    logdet_Phi = np.sum(np.log(Phidiag))
    _, logdet_Sigma = np.linalg.slogdet(Sigma.astype(float))

    logdet_C = logdet_N + logdet_Phi + logdet_Sigma

    return x_Cinv_y, logdet_C



def lnprob(theta, filtered_obs, weight=False):
    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, filtered_obs, weight=weight)


# http://jakevdp.github.io/blog/2015/08/07/frequentism-and-bayesianism-5-model-selection/
def compute_mcmc(lnprob, args, pinit, nwalkers=10, niter=5000, threads=4):

    # Number of variables we're MCMCing over
    ndim = len(pinit)

    # Initial position in the 3D space of (C1, C3, C5) from where the walkers will start
    nwalkers = 3 * ndim            # emcee requires nwalkers > ndim
    p0 = pinit + 1e-4 * np.random.randn(nwalkers, ndim)

    # Set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args, threads=threads)
    sampler.run_mcmc(p0, niter, progress=True)

    # Burn the ends of the samples chains
    if niter <= 50000:
        burn = int(niter/10)
    if burn > 5000:
        burn = 5000

    samples = sampler.get_chain(discard=burn, flat=True)
    return samples
