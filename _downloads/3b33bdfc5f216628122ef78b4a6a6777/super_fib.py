"""
Quaternion Sampling Efficiency
==============================

Comparing methods for sampling quaternions, with an emphasis on distributing the quaternions uniformly throughout orientation space (in the cosine distance sense)
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import BallTree

import mirage as mr
import mirage.vis as mrv

n = int(1e4)

Q = mr.quat_upper_hemisphere(mr.quaternion_fibonacci_sample(n))
cq = BallTree(Q).query(Q, k=2)[1][:, 1]

ang1 = np.rad2deg(mr.quat_ang(Q, Q[cq, :]))

Q = mr.quat_upper_hemisphere(mr.rand_quaternions(n))
cq = BallTree(Q).query(Q, k=2)[1][:, 1]
ang2 = np.rad2deg(mr.quat_ang(Q, Q[cq, :]))


plt.hist(ang1, bins=30, label='Fibonacci', density=True)
plt.hist(ang2, bins=30, alpha=0.7, label='Random', density=True)
mrv.texit(
    'Quaternion Sampling Comparison',
    'Angle to nearest neighbor [deg]',
    'Probability density',
)
plt.legend()
plt.show()
