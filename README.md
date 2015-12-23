# Perturbation Modularity for Dynamical Systems

This repository provides sample Python code for decomposing dynamical systems into weakly-coupled modules using the concept of *perturbation modularity*. This is defined and explained in the following paper:

> A Kolchinsky, AJ Gates, LM Rocha, "Modularity and the Spread of Perturbations in Complex Dynamical Systems", *Physical Review E*, 2015. [link](http://journals.aps.org/pre/abstract/10.1103/PhysRevE.92.060801) [arxiv](http://arxiv.org/abs/1509.04386)

Four Python files are included:
* `examples.py`: This includes code to run some example decompositions.  The first one is Example 1 from the paper (system of logistic maps coupled in a hierarchically-modular manner).  The second and third examples in the code are from Example 2 in the paper (one coupled map lattice [CML] in the modular regime, and one in the diffusive regime).  Please note that due to randomization of initial conditions and minor code differences, the output results will not be numerically equal to those reported in the paper.
* `pertmod.py`: Library code to find optimal perturbation modularity decompositions.
* `coupledmaps.py`: Library code to define dynamical systems from coupled logistic maps.
* `nmi.py`: Library code to compute normalize mutual information (NMI) between different decompositions.

In order to run the code, the following libraries need to be installed:
* `numpy`
* `scipy` for the `scipy.sparse` sparse matrix library
* `matplotlib` for plotting
* `dynpy`, a library for running dynamical systems in Python. Installable from PyPi as `pip install dynpy` or from GitHub as `pip install https://github.com/artemyk/dynpy/archive/master.zip`. For reference, see the [Dynpy GitHub](https://github.com/artemyk/dynpy).
* `graphy`, a library for generating graphs and running Louvain.  Installable from GitHub as `pip install https://github.com/artemyk/graphy/archive/master.zip`.  For reference, see the [Graphy GitHub](https://github.com/artemyk/graphy).

