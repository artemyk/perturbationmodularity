# Module for calculating Normalized Mutunq1l Information between two
# partitions

# See Danon et al., Comparing community structure identification, 
#      Journal of Statistical Mechanics: Theory and Experiment, 2005.

import numpy as np

def norm_mutual_info(membership1, membership2):
    def entropy(p):
        p = p[p!=0]
        return -np.sum(p * np.log(p))

    unq1 = np.unique(membership1)
    unq2 = np.unique(membership2)
    N = len(membership1)
    cmx = np.zeros((len(unq1),len(unq2)), dtype='float')

    for i, val1 in enumerate(unq1):
        for j, val2 in enumerate(unq2):
            cmx[i,j] = np.sum(np.logical_and(membership1 == val1, membership2 == val2))
    cmx /= cmx.sum()

    pi = np.sum(cmx,1)
    pj = np.sum(cmx,0)

    mi = 0
    for i, val1 in enumerate(unq1):
        for j, val2 in enumerate(unq2):
            if cmx[i,j] != 0:
                mi += cmx[i,j] * np.log(cmx[i,j]/(pi[i]*pj[j]))

    return 2*mi / (entropy(pi)+entropy(pj))
