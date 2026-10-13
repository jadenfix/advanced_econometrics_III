import json, unittest
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

class NumericalParity(unittest.TestCase):
    def test_whitened_matching_equals_pairwise_distance(self):
        rng=np.random.default_rng(42)
        controls=rng.normal(size=(25,4)); treated=rng.normal(size=(12,4))
        inv=np.linalg.inv(np.cov(np.vstack([controls,treated]),rowvar=False))
        expected=cdist(treated,controls,metric="mahalanobis",VI=inv).argmin(axis=1)
        whitening=np.linalg.cholesky(inv)
        _,actual=cKDTree(controls@whitening).query(treated@whitening)
        np.testing.assert_array_equal(actual,expected)
    def test_weighted_quantile_minimizes_original_check_loss(self):
        d=json.loads((Path(__file__).resolve().parents[1]/"econometrics_III.ipynb").read_text())
        source="".join(d["cells"][40]["source"])
        fragment=source[source.index("def weighted_quantile("):source.index("for tau in quantiles:",source.index("def weighted_quantile("))]
        ns={"np":np};exec(fragment,ns)
        y=np.array([0.,1.,1.,5.,9.]);w=np.array([1.,2.,4.,1.,3.])
        for tau in [.1,.25,.5,.75,.9]:
            q=ns["weighted_quantile"](y,w,tau)
            def loss(q):
                r=y-q;return np.sum(w*np.where(r>=0,tau*r,(tau-1)*r))
            self.assertAlmostEqual(loss(q),min(loss(v) for v in y))
    def test_matching_ties_preserve_minimum_distance(self):
        controls=np.array([[1.,0.],[-1.,0.],[2.,0.]])
        distance,index=cKDTree(controls).query([[0.,0.]])
        self.assertEqual(distance[0],1.)
        self.assertIn(index[0],[0,1])

if __name__=="__main__": unittest.main()
