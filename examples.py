# Code to findoptimal perturbation modularity decompositions
# See for details :
#   A Kolchinsky, AJ Gates, LM Rocha, "Modularity and the Spread of 
#     Perturbations in Complex Dynamical Systems", Physical Review E, 2015.

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import scipy.sparse as ss

import dynpy
import graphy

import pertmod
import coupledmaps

# This determines how many times Louvain algorithm is run (the best 
# decomposition is selected).  Higher number give better quality results
# but increase computational time
NUM_RUNS = 5

def get_optimal(conn_mx, alpha, eps, timesteps):
    """
    Get optimal perturbation modularity decompositions across a range of
    timescales for a system of coupled logistic maps.

    Parameters
    ----------
    conn_mx : np.array
        Coupling matrix
    alpha : float
        Chaoticity parameter value
    eps : float
        Coupling parameter value
    timesteps : list
        List of timescales to sweep.
    """
    N=conn_mx.shape[0]
    conn_mx = conn_mx / conn_mx.sum(axis=0)
    np.fill_diagonal(conn_mx, 0)
    conn_mx = ss.csr_matrix(conn_mx)

    dynsys = coupledmaps.CoupledMaps(conn_mx, coupledmaps.get_logistic_map_update_function(alpha, eps))
    start_state = dynsys.iterate(dynsys.get_random_initial(), 0)
    start_state=dynsys.iterate(start_state, 1e5)

    perts = np.eye(N)*1e-4
    qObj = pertmod.PertModLouvain(dynsys=dynsys, time_lag=0, perts=perts, start_state=start_state)
    return pertmod.find_optimal_across_time(qObj, timesteps, num_runs=NUM_RUNS, debug_level=1)



# **** Hierarchical coupling matrix example *****
alpha, eps = 2.0, 0.04
print "Running decomposition of hiearchically-coupled logistic maps (alpha=%0.2f, eps=%0.2f)" % (alpha, eps)
a = 0.01
conn_mx = graphy.graphgen.gen_hierarchical_weighted_block_matrix(10, 2, 3, [a**0, a**1, a**2, a**3])
timesteps = np.arange(15, 60, dtype='int')
r = get_optimal(conn_mx, alpha=alpha, eps=eps, timesteps=timesteps)

Qs, NMIs, _ = zip(*r)

fig, ax1 = plt.subplots()
ax1.plot(timesteps, Qs, 'k')
ax1.set_xlabel('Timesteps $t$')
ax1.set_ylabel('Perturbation modularity')

ax2 = ax1.twinx()
ax2.plot(timesteps, NMIs, 'b--')
ax2.set_ylabel('NMI', color='b')
plt.title('Hierarchical coupling matrix')
plt.show()
print 
print
# **** End hierarchical coupling matrix example *****


# **** Example using coupled map lattices in two regimes: one modular, one diffusive ****

timesteps = np.linspace(2, 400, 20, dtype='int')

for regime, alpha, eps in \
    (('Modular', 1.7, 0.1),
     ('Diffusive', 1.9, 0.6)):
    print "Running decomposition of coupled map lattice (CML) in %s regime (alpha=%0.2f, eps=%0.2f)" % (regime, alpha, eps)

    r = get_optimal(graphy.graphgen.gen_ring_matrix(N=100), alpha=alpha, eps=eps, timesteps=timesteps)

    Qs, NMIs, partitions = zip(*r)

    plt.figure(figsize=(14,5))
    gs = gridspec.GridSpec(1, 2, wspace=0.6)
    ax1 = plt.subplot(gs[0, 0])

    ax1.plot(timesteps, Qs, 'k')
    ax1.set_ylabel('Perturbation modularity')
    ax1.set_ylim([0, 1])
    ax1.set_xlim([timesteps.min(), timesteps.max()])

    ax2 = ax1.twinx()
    ax2.plot(timesteps, NMIs, 'b--')
    ax2.set_ylabel('NMI', color='b')
    ax2.set_ylim([0, 1])

    plt.xlabel('Timestep $t$')
    plt.title('CML in %s Regime' % regime)

    ax3 = plt.subplot(gs[0, 1])
    plt.imshow(np.vstack(partitions).T, interpolation='nearest', aspect='auto', cmap='Accent', 
        extent=[timesteps.min(), timesteps.max(), 0, 1])
    plt.yticks([])
    plt.xlabel('Timestep $t$')
    plt.title('Optimal Decompositions')
    plt.show()

    print
    print




