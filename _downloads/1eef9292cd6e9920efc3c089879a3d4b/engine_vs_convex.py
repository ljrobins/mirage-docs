"""
Convex vs Engine Light Curves
=============================

Comparing light curves produced by the LightCurveEngine renderer and a simplified convex method
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import mirage as mr

# %%
# Defining the object and BRDF
obj = mr.SpaceObject("cylinder.obj")
del obj.file_name
brdf = mr.Brdf("phong", cd=0.1, cs=0.9, n=30)

# %%
# Defining the attitude profile

t_eval = np.linspace(0, 10, int(1e3) + 1)
q, _ = mr.propagate_attitude_torque_free(
    mr.hat(np.array([0.0, 0.0, 0.0, 1.0])),
    np.array([1.0, 1.0, 1.0]),
    np.diag([1, 2, 3]),
    t_eval,
)

dcm = mr.quat_to_dcm(q)
ovb = mr.stack_mat_mult_vec(dcm, np.array([[1, 0, 0]]))
svb = mr.stack_mat_mult_vec(dcm, np.array([[0, 1, 0]]))

# %%
# We can now run the engine and compute a convex light curve:

b_non_convex = mr.run_light_curve_engine(
    brdf,
    obj,
    svb,
    ovb,
    instance_count=9,
    silent=False,
)
b_convex = obj.convex_light_curve(brdf, svb, ovb)

# %%
# And plot the results

plt.figure()
sns.lineplot(x=t_eval, y=b_non_convex, errorbar=None)
sns.lineplot(x=t_eval, y=b_convex, errorbar=None)
plt.title(f"Light Curves for {obj.file_name}")
plt.xlabel("Time [s]")
plt.ylabel("Normalized brightness")
plt.legend(["LightCurveEngine", "Convex"])
plt.grid()
plt.show()


# %%
# Plotting the error instead of the brightness values

plt.figure()
sns.lineplot(x=t_eval, y=b_non_convex - b_convex, errorbar=None)
plt.title(f"Light Curves Error for {obj.file_name}")
plt.xlabel("Time [s]")
plt.ylabel("Normalized brightness error")
plt.grid()
plt.show()

# %%
# This is nice and small, which we like to see. If we repeat this process for a non-convex object, the error quickly becomes clear

obj = mr.SpaceObject("tess.obj")
brdf = mr.Brdf("phong", cd=0.5, cs=0.5, n=5)

b_non_convex = mr.run_light_curve_engine(brdf, obj, svb, ovb, instance_count=1)
b_non_convex_rotating = mr.run_light_curve_engine(
    brdf, obj, svb, ovb, instance_count=1, rotate_panels=True
)
b_convex = obj.convex_light_curve(brdf, svb, ovb)


plt.figure()
sns.lineplot(x=t_eval, y=b_non_convex, errorbar=None)
sns.lineplot(x=t_eval, y=b_non_convex_rotating, errorbar=None)
sns.lineplot(x=t_eval, y=b_convex, errorbar=None)
plt.title(f"Light Curves for {obj.file_name}")
plt.xlabel("Time [s]")
plt.ylabel("Normalized brightness")
plt.legend(["LightCurveEngine", "LightCurveEngine - rotating panels", "Convex"])
plt.grid()
plt.show()
