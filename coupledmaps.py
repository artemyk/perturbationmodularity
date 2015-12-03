import dynpy
import numpy as np
from scipy.sparse import issparse

class CoupledMaps(dynpy.dynsys.VectorDynamicalSystem, dynpy.dynsys.DeterministicDynamicalSystem):
    def __init__(self, coupling_matrix, mapping_func, noise=0.0):
        # TODO: Document

        dynpy.dynsys.VectorDynamicalSystem.__init__(self, num_vars=coupling_matrix.shape[0])
        dynpy.dynsys.DeterministicDynamicalSystem.__init__(self, discrete_time=True)
        self.mapping_func    = mapping_func
        self.coupling_matrix = coupling_matrix
        self.issparse        = issparse(coupling_matrix)
        self.noise           = noise

    def _iterate_1step_discrete(self, start_state):
        r  = self.mapping_func(self, start_state)
        if self.noise > 0:
            r += self.noise * np.random.rand(self.num_vars)
        return r

    def perturb(self, base_vec, pert_vec):
        return base_vec + pert_vec
        
    def get_random_initial(self):
        return np.random.rand(self.num_vars)*2.0-1.0

# From sklearn.utils
def safe_sparse_dot(a, b, dense_output=False):
    if issparse(a) or issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)

def get_logistic_map_update_function(alpha, eps):
    def updatefunction(obj, prev_state):
        updstates = (1 - alpha*(prev_state*prev_state))
        if obj.issparse:
            n = safe_sparse_dot(obj.coupling_matrix, updstates)
        else:
            n = obj.coupling_matrix.dot(updstates)
        return (1-eps) * updstates + eps * n
    return updatefunction

