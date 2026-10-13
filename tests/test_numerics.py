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
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from numerical import match_mahalanobis
        actual=match_mahalanobis(treated,controls,inv)
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
    def test_tied_matching_preserves_original_outcomes_and_att(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from numerical import match_mahalanobis
        controls=np.array([[1.,0.],[-1.,0.],[1.,0.],[2.,0.]])
        treated=np.array([[0.,0.],[1.,0.],[2.,0.]])
        inv=np.eye(2)
        expected=[]
        for row in treated:
            distances=[]
            for control in controls:
                delta=row-control
                distances.append(np.sqrt(np.dot(np.dot(delta,inv),delta.T)))
            expected.append(int(np.argmin(distances)))
        actual=match_mahalanobis(treated,controls,inv)
        np.testing.assert_array_equal(actual,expected)
        outcomes=np.array([10.,90.,500.,20.])
        np.testing.assert_array_equal(outcomes[actual],outcomes[expected])
        self.assertEqual(float(np.mean(np.array([30.,40.,50.])-outcomes[actual])),float(np.mean(np.array([30.,40.,50.])-outcomes[expected])))

if __name__=="__main__": unittest.main()
