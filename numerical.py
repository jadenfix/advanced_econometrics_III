"""Exact-distance matching with original first-control tie behavior."""
import numpy as np
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
