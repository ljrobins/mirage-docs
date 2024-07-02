"""
Support Optimization Comparison
===============================

Comparing Durech and Kaasalainen's light curve inversion code to mine
"""

import numpy as np

import mirage as mr

num = 100
ns = mr.spiral_sample_sphere(num).reshape(-1, 3)
ns = ns[np.random.permutation(num), :]
az = np.random.random(num) ** 2

egi = ns * az[:, None]
egi -= np.sum(egi, axis=0) / num
ns = mr.hat(egi)

mr.tic()
obj = mr.construct_mesh_from_egi(egi, implementation='fortran')
mr.toc()

import pyvista as pv

pl = pv.Plotter()
pl.add_mesh(obj._mesh)
pl.show()
