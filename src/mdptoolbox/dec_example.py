# -*- coding: utf-8 -*-

import numpy as _np
import scipy.sparse as _sp

def multi_forest(num = 2, S = 3, r1 = 4, r2 = 2, p = 0.1):
    assert S > 1, "The number of states S must be greater than 1."
    assert (r1 > 0) and (r2 > 0), "The rewards must be non-negative."
    assert 0 <= p <= 1, "The probability p must be in [0; 1]."
    assert num == 2, "only tested with num = 2"

    """
    States = {00, 01, ..., 0{S-1}, 10, 11, ..., 1{S-1}, ..., {S-1}{S-1}}
    """
    # build transition matrix P
    P = _np.zeros((4, S*S, S*S))

    # build P[00, :, :], which is when no 'cut' is happening, only burn
    P[0, :, 0] = 2*p - p*p
    for i in range(S):
        for j in range(S):
            if i != (S-1) and j != (S-1):
                P[0, i * S + j, (i+1) * S + j + 1] = (1-p) * (1-p)
            elif i == (S-1) and j != (S-1):
                P[0, i * S + j, S*S - 1] = (1-p) * (1-p)
            elif i != (S-1) and j == (S-1):
                P[0, i * S + j, (i+1) * S + j] = (1-p) * (1-p)
            else:
                P[0, i * S + j, i * S + j] = (1-p) * (1-p)

    # build P[01, :, :]
    for i in range(S):
        for j in range(S):
            P[1, i*S + j, i*S] = 1

    # build P[10, :, :]
    for i in range(S):
        for j in range(S):
            P[2, i*S + j, j] = 1

    # build P[11, :, :]
    P[3, :, 0] = 1

    #  print P

    # build reward matrix R
    R = _np.zeros((S*S, 4))
    for i in range(S):
        for j in range(S):
            # reward for no cut: R[:, 00]
            if (i == (S-1)) or (j == (S-1)):
                R[i*S + j, 0] = r1
            # reward for 1 cut: R[:, 01]
            if j != 0:
                R[i*S + j, 1] = 1
            elif j == (S-1):
                R[i*S + j, 1] = r2
            # reward for 1 cut: R[:, 10]
            if i != 0:
                R[i*S + j, 2] = 1
            elif i == (S-1):
                R[i*S + j, 2] = r2
            # reward for 2 cut: R[:, 11]
            if (i != 0) and (i != (S-1)):
                R[i*S + j, 3] += 1
            if (j != 0) and (j != (S-1)):
                R[i*S + j, 3] += 1
            if i == (S-1):
                R[i*S + j, 3] += r2
            if j == (S-1):
                R[i*S + j, 3] += r2
    
    return (P, R)
