import numpy as np
import numpy.polynomial.legendre as leg
import emcee
from scipy.linalg import cho_solve
from numba import njit
import pint.logging
import astropy.units as u
pint.logging.setup(level="ERROR")
from numpy.polynomial.legendre import legval
import sys

def load_pinit(psr_name):

    if psr_name == "B1937+21":
        return np.array([179.5, -0.136, -0.19])
    elif psr_name == "J1643-1224":
        return np.array([342.9607408562299, 3.6656647122634305, -0.21600401837611255])
    elif psr_name == "J1024-0719":
        return np.array([-4000.0, -0.6, -6.0])
    elif psr_name == "J1903+0327":
        return np.array([-84.1345, 35.3294, -16.9185])
    elif psr_name == "J1741+1351":
        return np.array([-44.4992, 87.0703, -45.7222])
    elif psr_name == "J1744-1134":
        return np.array([-102.3, -7.7, 5.6])
    elif psr_name == "J0613-0200":
        return np.array([-4.0882, 3.016, -0.6471])
    elif psr_name == "J2145-0750":
        return np.array([-203.0, 2.0, 0.35])

@njit
def log_prior(theta, psr_name):
    a1, a3, a5 = theta  # These are priors on the MONIMIAl coefficients, not on the LEGENDRE coefficients
    if psr_name == "B1937+21":
        if 0.0 < a1 < 400.0 and -50.0 < a3 < 50.0 and -50.0 < a5 < 50.0:
            return 0.0
    elif psr_name == "J1643-1224":
        if -450 < a1 < 450 and -120.0 < a3 < 30.0 and -160.0 < a5 < 40.0:
            return 0.0
    elif psr_name == "J1024-0719":
        if -5000 < a1 < 7000 and -1000.0 < a3 < 5500.0 and -200.0 < a5 < 3800.0:
            return 0.0
    elif psr_name == "J1903+0327":
        if -1000 < a1 < 1000 and -1000.0 < a3 < 1000.0 and -1000.0 < a5 < 1000.0:
            return 0.0
    elif psr_name == "J1741+1351":
        if -200.0 < a1 < 200.0 and -500.0 < a3 < 500.0 and -500.0 < a5 < 500.0:
            return 0.0
    elif psr_name == "J1744-1134":
        if -300.0 < a1 < 200.0 and -400.0 < a3 < 400.0 and -250.0 < a5 < 250.0:
            return 0.0
    elif psr_name == "J0613-0200":
        if -100.0 < a1 < 100.0 and -500.0 < a3 < 500.0 and -250.0 < a5 < 2500.0:
            return 0.0
    elif psr_name == "J2145-0750":
        if -300.0 < a1 < 10.0 and -50.0 < a3 < 120.0 and -100.0 < a5 < 100.0:
            return 0.0

    return -np.inf

'''
@njit
def eval_legendre_basis(x):
    """
    Returns a matrix with columns [P_0(x), P_1(x), ..., P_5(x)]
    """
    coeffs = np.eye(6)  # Create an identity matrix (which acts as a unit vector in coefficient space)

    # Loop through each row of that identity matrix and evaluate the corresponding Legendre polynomial
    return np.column_stack([legval(x, c) for c in coeffs])

@njit
def compute_fixed_part(x, c1, c3, c5):
    """
    Compute the fixed part of the model (from MCMC sample c1, c3, c5)
    """
    P = eval_legendre_basis(x)
    return c1 * P[:, 1] + c3 * P[:, 3] + c5 * P[:, 5]
'''


@njit
def eval_legendre_0_to_5(x):
    n = len(x)
    P = np.empty((n, 6), dtype=np.float64)

    for i in range(n):
        xi = x[i]
        P[i, 0] = 1.0
        P[i, 1] = xi
        P[i, 2] = 0.5 * (3.0 * xi**2 - 1.0)
        P[i, 3] = 0.5 * (5.0 * xi**3 - 3.0 * xi)
        P[i, 4] = (35.0 * xi**4 - 30.0 * xi**2 + 3.0) / 8.0
        P[i, 5] = (63.0 * xi**5 - 70.0 * xi**3 + 15.0 * xi) / 8.0

    return P


@njit
def solve_c024(P, resids_minus_fixed):
    """
    Set up and solve the 3×3 normal equations for c0, c2, c4
    """
    #A = np.column_stack([P[:, 0], P[:, 2], P[:, 4]])  # columns for c0, c2, c4
    n = len(resids_minus_fixed)
    A = np.empty((len(resids_minus_fixed), 3), dtype=np.float64)

    for i in range(n):
        A[i, 0] = P[i, 0]  # P0
        A[i, 1] = P[i, 2]  # P2
        A[i, 2] = P[i, 4]  # P4

    ATA = A.T @ A       # 3x3
    ATy = A.T @ resids_minus_fixed # 3x1

    # see https://textbooks.math.gatech.edu/ila/least-squares.html or https://see.stanford.edu/materials/lsoeldsee263/06-ls-app.pdf
    return np.linalg.solve(ATA, ATy)  # returns [c0, c2, c4]


@njit
def fit_and_eval(x, resids, c1, c3, c5):
    '''
    For a Legendre polynomial
                           L(x) = c0 * L0(x) + c1 L1(x) + c2 L2(X) + c3 L3(x) + c4 L4(x) + c5 L5(x)
    given values of c1, c3, c5, find the values of c0, c2, c4 that
    xvals: values of x between -1 and 1
    resids: timing residuals
    '''
    L = eval_legendre_0_to_5(x)    # Legendre polynomials up to order 5 evaluated at the given values of x in [-1,1]
    n = len(x)                     # Number of values of x

    resids_minus_fixed = np.empty(n, dtype=np.float64)
    for i in range(n):
        resids_minus_fixed[i] = resids[i] - (c1 * L[i, 1] + c3 * L[i, 3] + c5 * L[i, 5])  # Subtract fixed part from the residuals

    c0, c2, c4 = solve_c024(L, resids_minus_fixed)

    y_legendre = np.empty(n, dtype=np.float64)
    for i in range(n):
        y_legendre[i] = c0 * L[i, 0] + c1 * L[i, 1] + c2 * L[i, 2] + c3 * L[i, 3] + c4 * L[i, 4] + c5 * L[i, 5]

    return resids - y_legendre

def batched_fit_and_eval(xvals, resids, c1c3c5):
    diffs_arr = []
    for x, y in zip(xvals, resids):
        diffs_arr.append(fit_and_eval(x, y, *c1c3c5))
    return (np.concatenate(diffs_arr) * u.us).to(u.s).value


def lnlike(theta, data_obj):

    a1, a3, a5 = theta                                                # Coefficients of the power series polynomial
    c1c3c5 = leg.poly2leg([0.0, a1, 0.0, a3, 0.0, a5])[[1, 3, 5]]     # Coefficients of the Legendre polynomial

    # Batch-process all pulsars at once, instead of looping
    diffs_arr = batched_fit_and_eval(data_obj.xvals, data_obj.resids, c1c3c5)

    return lnlikelihood(data_obj.Ndiag, data_obj.U, data_obj.Sigma_cf, data_obj.logdet_C, diffs_arr)


@njit
def cholesky_solve_upper(U, y):
    # Solves C⁻¹ y when C = U.T @ U (upper-triangular Cholesky factor U)
    z = np.linalg.solve(U.T, y)  # U.T is lower-triangular
    return np.linalg.solve(U, z)  # U is upper-triangular


#@njit
def lnlikelihood(Ndiag, U, Sigma_cf, logdet_C, s) -> float:

    """
    Compute the chi2 when correlated noise is present in the timing model.
    If the system is not singular, it uses Cholesky factorization to evaluate this.
    If the system is singular, it uses singular value decomposition instead.
    """

    x_Ninv_y = np.sum(s * s / Ndiag)
    x_Ninv_U = (s / Ndiag) @ U
    y_Ninv_U = (s / Ndiag) @ U

    x_Cinv_y = x_Ninv_y - x_Ninv_U @ cho_solve(Sigma_cf, y_Ninv_U)
#    x_Cinv_y = x_Ninv_y - x_Ninv_U @ cholesky_solve_upper(Sigma_cf[0], y_Ninv_U)

    return -(x_Cinv_y / 2 + logdet_C / 2)


def lnprob(theta, data_obj):
    lp = log_prior(theta, data_obj.PSR_name)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, data_obj)


# http://jakevdp.github.io/blog/2015/08/07/frequentism-and-bayesianism-5-model-selection/
def compute_mcmc(lnprob, args, pinit, nwalkers=10, niter=50000, threads=8):

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
#        burn = int(niter / 10)
        burn = int(niter / 8)
    if burn > 5000:
        burn = 5000

    samples = sampler.get_chain(discard=burn, flat=True)
    return samples
