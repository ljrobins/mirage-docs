"""
Noisy Light Curves
==================

Simulates torque-free rigid body motion for a simple object and computes the full light curve, informed by station constraints and a high-fidelity background signal model
"""

import sys

sys.path.append(".")

import pyspaceaware as ps
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %%
# Setting up analysis times
date_start = datetime.datetime(2023, 5, 20, 20, 45, 0, tzinfo=datetime.timezone.utc)
(dates, epsecs) = ps.date_linspace(
    date_start - ps.days(1), date_start, 1e3, return_epsecs=True
)

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
(lc_noisy, aux_data) = station.observe_light_curve(
    obj, obj_attitude, brdf, dates, use_engine=True
)

# %%
# Extracting data and plotting results
lc_clean = aux_data["lc_clean"]

sns.scatterplot(x=epsecs, y=lc_clean)
sns.scatterplot(x=epsecs, y=lc_noisy)
plt.xlim((0, np.max(epsecs)))
ps.texit(
    f"Light Curves for {obj.satnum}",
    "EpSec",
    "[e-]",
    ["Clean", "Noisy"],
)
plt.grid()
plt.show()
