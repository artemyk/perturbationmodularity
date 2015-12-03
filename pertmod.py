import graphy
import numpy as np

import nmi

class PerturbedStates(object):
    def __init__(self, dynsys, init_time, unperturbed_state, perturbed_states, 
        time_lag=0, pnorm=1):
        """
        Class for storing perturbed trajectories of dynamical systems.
        
        Parameters
        ----------
        dynsys : dynpy.dynsys.DynamicalSystem
            Dynamical system object to perturb
        init_time : int or float
            Initial time at which system is started.
        unperturbed_state : np.array
            Unperturbed initial system state
        perturbed_states : list of np.array 
            Initial system state with perturbations.
        time_lag : int or float
            Timescale considered
        pnorm : int (default 1)
            Which norm to consider
        """

        self.dynsys = dynsys
        self.unperturbed_state = unperturbed_state
        self.perturbed_states = perturbed_states
        self.is_bad_pert = np.zeros(len(perturbed_states), dtype='bool')
        self.pnorm = pnorm
        self.time = init_time
        self.diffs = None
        self.advance(time_lag)

    @staticmethod
    def convtype(v):
        if str(v.dtype).startswith('uint'):
            return v.astype('double')
        else:
            return v

    def advance(self, t):
        self.unperturbed_state = self.dynsys.iterate(self.unperturbed_state, t)
        self.perturbed_states = np.vstack([self.dynsys.iterate(ps, t) for ps in self.perturbed_states])

        self.is_bad_pert[:] = 0
        new_diffs = self.convtype(self.unperturbed_state) - self.convtype(self.perturbed_states)

        # Normalize differences
        norms = np.linalg.norm(new_diffs, ord=self.pnorm, axis=1)
        ixs = norms==0 
        new_diffs[ixs] = 0.0
        self.is_bad_pert[ixs] = True
        norms[ixs] = 1.0

        new_diffs = new_diffs / norms[:,None]

        self.diffs = new_diffs
        self.time += t


class PertModLouvain(graphy.qualityfuncs.QualityFunction):
    def __init__(self, dynsys, start_state, time_lag, perts=None, pnorm=1):
        """
        Class for computing optimal perturbation modularity decomposition using the Louvain algorithm.

        Parameters
        ----------
        dynsys : dynpy.dynsys.DynamicalSystem
            Dynamical system object to perturb
        start_state : np.array
            Which state to start with.
        time_lag : int or float
            Timescale considered
        perts : list (default: single node perturbations)
            List of initial perturbations. By default, all single node perturbations are used,
            of size 1 for integer-based systems and 0.0001 otherwise.
        pnorm : int (default 1)
            Which norm to consider (only pnorm=1 currently supported)
        """

        if pnorm != 1:
            raise ValueError("Can only use pnorm=1 for Louvain")
        
        self.dynsys = dynsys
        self.pnorm = pnorm
        
        # Number of dimensions 
        self.N = dynsys.num_vars
        
        if perts is None:
            pertsize = 1 if issubclass(start_state.dtype.type, np.integer) else 0.0001
            perts = get_all_perts(dynsys.num_vars, 1, pertsize)

        # Uniform probability of perturbations
        self.init_pert_probs = np.ones(len(perts)) / len(perts)
        
        pert_states = np.vstack([self.dynsys.perturb(start_state, pert) for pert in perts])
        baseopts = dict(dynsys=dynsys, init_time=0,  unperturbed_state=start_state, 
            perturbed_states=pert_states, pnorm=pnorm)
        self.start = PerturbedStates(**baseopts)
        self.end   = PerturbedStates(time_lag=time_lag, **baseopts)

        self.recalc_probs()

    def recalc_probs(self):
        self.pert_probs = self.init_pert_probs.copy()
        self.pert_probs[np.logical_or(self.start.is_bad_pert, self.end.is_bad_pert)] = 0.0
        s = self.pert_probs.sum()
        if s > 0:
            self.pert_probs /= s

    def end_state_advance(self, t):
        self.end.advance(t)
        self.recalc_probs()

    def quality(self, membership):
        # Return quality of partition specified by membership

        def coarse_grain(membership, comms, vec):
            # Coarse grain perturbed differences (in vec) as relative perturbation magnitudes
            r = np.array([np.linalg.norm(vec[:,membership == c], ord=self.pnorm, axis=1) for c in comms])
            return r
        
        # List of communities in membership vector
        comms = sorted(list(set(membership)))

        cgrainedS = coarse_grain(membership, comms, self.start.diffs)
        cgrainedE = coarse_grain(membership, comms, self.end.diffs )

        ps = self.pert_probs
        prods = (cgrainedS * cgrainedE).sum(axis=0)

        q = ps.dot(prods) - ps.dot(cgrainedS.T).T.dot(ps.dot(cgrainedE.T))

        return q

    def find_optimal(self, initial_membership=None, num_runs=1, debug_level=0):
        mx = np.multiply(np.abs(self.start.diffs).T, self.pert_probs[None,:]).dot(np.abs(self.end.diffs))
        best_membership, q = graphy.louvain.optimize_modularity(mx, num_runs=num_runs, errortol=1e-2)
        return best_membership, q


def find_optimal_across_time(qualityObj, timepoints, num_runs=1, debug_level=0):
    saved_best = []
    best_membership, last_best_membership = None, None
    last_time = 0
    for t in sorted(timepoints):
        qualityObj.end_state_advance(t - last_time)
        last_time = t

        best_membership, best_membership_q = qualityObj.find_optimal(debug_level=0, 
            initial_membership=last_best_membership,
            num_runs=num_runs)

        nmival = 0.0
        if last_best_membership is not None:
            best_membership = graphy.partitions.remap2match(best_membership, last_best_membership)
            nmival = nmi.norm_mutual_info(best_membership, last_best_membership)

        if debug_level > 0:
            print('time=%2d nmi=%0.4f #modules=%2d Q=%0.4f %s' % (t, nmival, len(set(best_membership)), best_membership_q, graphy.partitions.to_alphanum_str(best_membership)))

        saved_best.append( (best_membership_q, nmival, best_membership) )
        last_best_membership = best_membership

    return saved_best

