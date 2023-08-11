"""
FNNLS vs NNLS
=============

Comparing the ``scipy.optimize.nnls`` function with a better method: "Fast Nonnegative Least Squares (FNNLS)
"""

import numpy as np
from scipy.optimize import nnls

import pyspaceaware as ps

test_sizes = np.arange(100, 2001, 200, dtype=np.int16)
t_old = np.zeros_like(test_sizes, dtype=np.float64)
t_new = np.zeros_like(t_old)
n_each = 10

for i, s in enumerate(test_sizes):
    n, m = s, s

    ps.tic()
    for j in range(n_each):
        Z = np.abs(np.random.rand(n, m))
        x = np.abs(np.random.rand(n))
        fres = ps.vendrow_fnnls(Z, x)[0]
    t_new[i] = ps.toc(return_elapsed_seconds=True) / n_each

    ps.tic()
    for j in range(n_each):
        Z = np.abs(np.random.rand(n, m))
        x = np.abs(np.random.rand(n))
        nres = nnls(Z, x)[0]
    t_old[i] = ps.toc(return_elapsed_seconds=True) / n_each

import matplotlib.pyplot as plt

plt.plot(test_sizes, t_old)
plt.plot(test_sizes, t_new)
ps.texit("scipy NNLS vs FNNLS", "Matrix size", "Time elapsed [sec]", ["scipy", "FNNLS"])
plt.show()
