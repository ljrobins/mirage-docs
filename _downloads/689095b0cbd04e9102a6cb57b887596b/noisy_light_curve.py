"""
Noisy Light Curves
==================

Simulates torque-free rigid body motion for a simple object and computes the full light curve, informed by station constraints and a high-fidelity background signal model
"""


import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pyspaceaware as ps

# %%
# Setting up analysis times
date_start = ps.utc(2023, 5, 20, 20, 45, 0)
(dates, epsecs) = ps.date_linspace(
    date_start - ps.days(1), date_start, 1e3, return_epsecs=True
)
ephr = epsecs / 3600  # Epoch hours

# %%
# Setting up the scenario objects
obj = ps.SpaceObject("tess.obj", identifier="goes 15")
brdf = ps.Brdf("phong")
station = ps.Station(preset="pogs")
# Observing from the Purdue Optical Ground Station in New Mexico

# %%
# Defining observation constraints on the station
station.constraints = [
    ps.SnrConstraint(3),
    ps.ElevationConstraint(10),
    ps.TargetIlluminatedConstraint(),
    ps.ObserverEclipseConstraint(station),
    ps.VisualMagnitudeConstraint(20),
    ps.MoonExclusionConstraint(10),
    ps.HorizonMaskConstraint(station),
]

# %%
# Defining the object's attitude profile and mass properties
obj_attitude = ps.RbtfAttitude(
    w0=0.000 * np.array([0, 1, 1]),
    q0=ps.hat(np.array([0, 0, 0, 1])),
    itensor=obj.principal_itensor,
)

# %%
# Computing the full noisy light curve
(lc_noisy_sampler, aux_data) = station.observe_light_curve(
    obj, obj_attitude, brdf, dates, use_engine=True
)
lc_noisy = lc_noisy_sampler()

# %%
# Extracting data and plotting results
lc_clean = aux_data["lc_clean"]

sns.scatterplot(x=ephr, y=lc_noisy, linewidth=0.1, size=0.5)
sns.scatterplot(x=ephr, y=lc_clean, linewidth=0.1, size=0.5)
plt.xlim((0, np.max(ephr)))
ps.texit(
    f"Light Curves for {obj.satnum}",
    "Epoch hours",
    "[e-]",
    ["Noisy", "Clean"],
)
plt.grid()
plt.show()
