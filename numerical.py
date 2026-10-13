"""Numerical helpers used by the executable coursework workflow."""
import numpy as np
from scipy.special import log_ndtr
from scipy.spatial import cKDTree


def match_mahalanobis(treated, controls, inverse_covariance):
    treated=np.asarray(treated,dtype=float)
    controls=np.asarray(controls,dtype=float)
    whitening=np.linalg.cholesky(inverse_covariance)
    tree=cKDTree(controls @ whitening)
    if len(controls)==1: return np.zeros(len(treated),dtype=int)
    distances, nearest=tree.query(treated @ whitening,k=2,eps=0)
    indices=nearest[:,0].copy()
    # Check floating-point near-ties with the original scalar distance operation,
    # sorting control indices so equal minima retain pandas idxmin semantics.
    ambiguous=np.flatnonzero(np.isclose(distances[:,0],distances[:,1],rtol=1e-10,atol=1e-12))
    for i in ambiguous:
        candidates=sorted(tree.query_ball_point(treated[i] @ whitening,distances[i,0]*(1+1e-10)+1e-12))
        scalar_distances=[]
        for j in candidates:
            delta=treated[i]-controls[j]
            scalar_distances.append(np.sqrt(np.dot(np.dot(delta,inverse_covariance),delta.T)))
        indices[i]=candidates[int(np.argmin(scalar_distances))]
    return indices


def probit_generalized_residual(linear_index, treatment):
    """Return E[epsilon | treatment, Z] for a standard-normal probit.

    ``linear_index`` must be Z @ gamma, not the predicted probability
    Phi(Z @ gamma).  Log-CDF arithmetic keeps the two Mills ratios finite in
    tails where direct division by ``cdf`` or ``1 - cdf`` underflows.
    """
    eta = np.asarray(linear_index, dtype=float)
    observed = np.asarray(treatment)
    eta, observed = np.broadcast_arrays(eta, observed)
    if not np.all(np.isfinite(eta)):
        raise ValueError("linear_index must contain only finite values")
    if not np.all(np.isin(observed, (0, 1))):
        raise ValueError("treatment must contain only 0 and 1")

    log_pdf = -0.5 * eta**2 - 0.5 * np.log(2 * np.pi)
    selected = np.exp(log_pdf - log_ndtr(eta))
    not_selected = -np.exp(log_pdf - log_ndtr(-eta))
    return np.where(observed == 1, selected, not_selected)
