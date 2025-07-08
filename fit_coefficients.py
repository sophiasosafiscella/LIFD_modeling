import numpy as np
from numba import njit

#@njit
def leg_vander_ordinate(x, y, deg, coeffs):
    """Pseudo-Vandermonde matrix of given degree.

    And ordinate matrix for the Legendre fit, i.e., the "b" vector in the least-square fit for
    a @ x = b, when only C0, C2, C4 are actually fitted and all other coefficients are MCMCed

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = L_i(x)

    where `0 <= i <= deg`. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Legendre polynomial.

    If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
    array ``V = legvander(x, n)``, then ``np.dot(V, c)`` and
    ``legval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Legendre series of the same degree and sample points.

    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.

    Returns
    -------
    vander : ndarray
        The pseudo-Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where The last index is the degree of the
        corresponding Legendre polynomial.  The dtype will be the same as
        the converted `x`.

    """
    ideg = deg.item()
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.asarray(x) + 0.0
    dims = (ideg + 1,) + x.shape   # Row = fitted parameter, column = observation
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)

    # Use forward recursion to generate the entries. This is not as accurate
    # as reverse recursion in this application but it is more efficient.
    v[0] = x*0 + 1
    if ideg > 0:
        v[1] = x
        for i in range(2, ideg + 1):
            # This is where the Legendre polynomials come into play! See Section 3 of the following link, with n
            # replaced with n-1: # http://nsmn1.uh.edu/hunger/class/fall_2012/lectures/lecture_8.pdf
            v[i] = (v[i-1]*x*(2*i - 1) - v[i-2]*(i - 1))/i

    # We will fit the coefficients corresponding to i = 0, 2, 4, (epoch-dependent)
    fitted_v = v[np.array([0, 2, 4]), :]

    # For all the terms OTHER than i = 0, 2, 4, move that term into the independent part of the least squares a @ x = b
    idx_not_fitted = np.array([i for i in range(np.shape(v)[0]) if i not in [0, 2, 4]])
    not_fitted_v = v[idx_not_fitted, :]

    b = y - coeffs @ not_fitted_v

    return fitted_v.T, b

#@njit
def fit_minimal(vander_f, x, y, deg, coeffs):
    """
    Helper function used to implement the ``<type>fit`` functions.

    Parameters
    ----------
    vander_f : function(array_like, int) -> ndarray
        The 1d vander function, such as ``polyvander``
    c1, c2
        See the ``<type>fit`` functions for more detail
    """

    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0
    deg = np.asarray(deg)

    if deg.ndim == 0:
        lmax = deg
        van, ord = vander_f(x, y, deg, coeffs)
    else:
        print("deg.nim > 0")
#        deg = np.sort(deg)
#        lmax = deg[-1]
#        order = len(deg)
#        van = vander_f(x, y, deg, coeffs)[:, deg]

    # set up the least squares matrices in transposed form
    lhs = van.T
    rhs = ord.T

    # set rcond
    rcond = len(x)*np.finfo(x.dtype).eps

    # Determine the norms of the design matrix columns.
    if lhs.dtype.kind == "c":  # 'c' means complex in NumPy dtype kinds
        scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
    else:
        scl = np.sqrt(np.square(lhs).sum(1))
    scl[scl == 0] = 1

    # Solve the least squares problem.
    A = lhs.T/scl
    c, resids, rank, s = np.linalg.lstsq(A, rhs.T, rcond)
    c = (c.T/scl).T

    return c

#@njit
def my_legfit_minimal(x, y, deg, coeffs):

    return fit_minimal(leg_vander_ordinate, x, y, deg, coeffs)



def fit(vander_f, x, y, deg, coeffs, rcond=None, full=False, w=None):
    """
    Helper function used to implement the ``<type>fit`` functions.

    Parameters
    ----------
    vander_f : function(array_like, int) -> ndarray
        The 1d vander function, such as ``polyvander``
    c1, c2
        See the ``<type>fit`` functions for more detail
    """

    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0
    deg = np.asarray(deg)

    # check arguments.
    if deg.ndim > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
        raise TypeError("deg must be an int or non-empty 1-D array of int")
    if deg.min() < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if len(x) != len(y):
        raise TypeError("expected x and y to have same length")

    if deg.ndim == 0:
        lmax = deg
        order = lmax + 1
        van, ord = vander_f(x, y, deg, coeffs)
    else:
        deg = np.sort(deg)
        lmax = deg[-1]
        order = len(deg)
        van = vander_f(x, y, deg, coeffs)[:, deg]

    # set up the least squares matrices in transposed form
    lhs = van.T
    rhs = ord.T
    if w is not None:
        w = np.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected 1D vector for w")
        if len(x) != len(w):
            raise TypeError("expected x and w to have same length")
        # apply weights. Don't use inplace operations as they
        # can cause problems with NA.
        lhs = lhs * w
        rhs = rhs * w

    # set rcond
    if rcond is None:
        rcond = len(x)*np.finfo(x.dtype).eps

    # Determine the norms of the design matrix columns.
    if issubclass(lhs.dtype.type, np.complexfloating):
        scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
    else:
        scl = np.sqrt(np.square(lhs).sum(1))
    scl[scl == 0] = 1

    # Solve the least squares problem.
    A = lhs.T/scl
    c, resids, rank, s = np.linalg.lstsq(A, rhs.T, rcond)
    c = (c.T/scl).T

    # Now compute the covariance matrix and standard errors
    # Number of data points and number of coefficients
    n = len(y)
    p = A.shape[1]

    # Estimate variance of the residuals
    if resids.size > 0:
        sigma2 = resids[0] / (n - p)  # unbiased estimate of variance
    else:
        # If residuals is empty (perfect fit or underdetermined), estimate it manually
        y_fit = A @ coeffs
        sigma2 = np.sum((y - y_fit) ** 2) / (n - p)

    # Compute the covariance matrix
    cov = sigma2 * np.linalg.inv(A.T @ A)

    # Standard deviation (uncertainty) of the coefficients
    uncertainties = np.sqrt(np.diag(cov))

    # Expand c to include non-fitted coefficients which are set to zero
    if deg.ndim > 0:
        if c.ndim == 2:
            cc = np.zeros((lmax+1, c.shape[1]), dtype=c.dtype)
        else:
            cc = np.zeros(lmax+1, dtype=c.dtype)
        cc[deg] = c
        c = cc

    # warn on rank reduction
#    if rank != order and not full:
#        msg = "The fit may be poorly conditioned"
#        warnings.warn(msg, np.RankWarning, stacklevel=2)

    if full:
        return c, uncertainties, [resids, rank, s, rcond]
    else:
        return c


def my_legfit_full(x, y, deg, coeffs, rcond=None, full=True, w=None):
    """
    Least squares fit of Legendre series to data.

    Return the coefficients of a Legendre series of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * L_1(x) + ... + c_n * L_n(x),

    where `n` is `deg`.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer
        all terms up to and including the `deg`'th term are included in the
        fit. For NumPy versions >= 1.11.0 a list of integers specifying the
        degrees of the terms to include may be used instead.
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, shape (`M`,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.

        .. versionadded:: 1.5.0

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        Legendre coefficients ordered from low to high. If `y` was
        2-D, the coefficients for the data in column k of `y` are in
        column `k`. If `deg` is specified as a list, coefficients for
        terms not included in the fit are set equal to zero in the
        returned `coef`.

    [residuals, rank, singular_values, rcond] : list
        These values are only returned if ``full == True``

        - residuals -- sum of squared residuals of the least squares fit
        - rank -- the numerical rank of the scaled Vandermonde matrix
        - singular_values -- singular values of the scaled Vandermonde matrix
        - rcond -- value of `rcond`.

        For more details, see `numpy.linalg.lstsq`.

    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if ``full == False``.  The
        warnings can be turned off by

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.RankWarning)

    See Also
    --------
    numpy.polynomial.polynomial.polyfit
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.laguerre.lagfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.hermite_e.hermefit
    legval : Evaluates a Legendre series.
    legvander : Vandermonde matrix of Legendre series.
    legweight : Legendre weight function (= 1).
    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the Legendre series `p` that
    minimizes the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where :math:`w_j` are the weights. This problem is solved by setting up
    as the (typically) overdetermined matrix equation

    .. math:: V(x) * c = w * y,

    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.

    If some of the singular values of `V` are so small that they are
    neglected, then a `RankWarning` will be issued. This means that the
    coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using Legendre series are usually better conditioned than fits
    using power series, but much can depend on the distribution of the
    sample points and the smoothness of the data. If the quality of the fit
    is inadequate splines may be a good alternative.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------

    """
    return fit(leg_vander_ordinate, x, y, deg, coeffs, rcond, full, w)