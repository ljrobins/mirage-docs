"""
Convex Light Curves
===================

Simulates torque-free rigid body motion for a simple object and computes the light curve
"""

import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pyspaceaware as ps

# %%
# To keep things simple, we'll just use a cube for this demo
obj = ps.SpaceObject("cube.obj")
brdf = ps.Brdf("diffuse", cd=1.0, cs=0.0)

# %%
# Computing the quaternion time history of the object
teval = np.linspace(0, 10, 1000)
q, _ = ps.propagate_attitude_torque_free(
    quat0=np.array([0.0, 0.0, 0.0, 1.0]),
    omega0=np.array([1.0, 1.0, 1.0]),
    itensor=np.diag([1, 2, 3]),
    teval=teval,
)
dcm = ps.quat_to_dcm(q)  # Converting to quaternion

# %%
# Transforming fixed inertial Sun and Observer vectors into the body frame
svi = np.array([[1, 0, 0]])
# Sun vector in the inertial frame
svb = ps.stack_mat_mult_vec(dcm, svi)
# Sun vector in the body frame
ovi = np.array([[0, 1, 0]])
# Observer vector in the inertial frame
ovb = ps.stack_mat_mult_vec(dcm, ovi)
# Observer vector in the body frame

lc = obj.convex_light_curve(brdf, svb, ovb)

plt.figure()
sns.lineplot(x=teval, y=lc)
plt.title("Convex Light Curves")
plt.xlabel("Time [s]")
plt.ylabel("Normalized brightness")
plt.grid()
plt.show()
