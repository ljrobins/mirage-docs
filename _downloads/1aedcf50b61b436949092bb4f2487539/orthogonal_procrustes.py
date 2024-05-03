"""
Orthogonal Procrustes
=====================

Solving the orthogonal procrustes problem via the SVD and Davenport's q-method
"""

import numpy as np
from scipy.linalg import orthogonal_procrustes

import mirage as mr

rhat_ref = mr.rand_cone_vectors(np.array([1.0, 0.0, 0.0]), 0.1, 5)
q_true = mr.rand_quaternions(1)
A_true = mr.quat_to_dcm(q_true)
rhat_body = mr.stack_mat_mult_vec(A_true, rhat_ref)

A_opro = orthogonal_procrustes(rhat_body, rhat_ref)[0]
q_opro = mr.dcm_to_quat(A_opro)

print(
    mr.wrap_to_180(mr.quat_ang(q_true, q_opro) * 180 / np.pi).squeeze()
)  # Error in degrees

# %%
# Davenport's q-method

q_davenport = mr.davenport(rhat_body, rhat_ref)
print(
    mr.wrap_to_180(mr.quat_ang(q_true, q_davenport) * 180 / np.pi).squeeze()
)  # Error in degrees

# %%
# Let's iteratively apply Davenport's q-method, for fun! We see that we get the same final quaternion

rbt = rhat_body.copy()
q_total = np.array([0.0, 0.0, 0.0, 1.0])
for i in range(10):
    dq = mr.davenport(rbt, rhat_ref)
    rbt = mr.stack_mat_mult_vec(mr.quat_to_dcm(dq).T, rbt)
    q_total = mr.quat_add(dq, q_total)

print(q_total)
print(q_true)
