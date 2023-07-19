"""
Propagating single TLE
======================

Propagates a single TLE for a given NORAD ID and propagates the trajectory
"""

import pyspaceaware as ps
import pyvista as pv
import numpy as np

date_space = ps.date_linspace(ps.today(), ps.today() + ps.hours(3), 1000)
# Propagate out one day
r_eci = ps.propagate_satnum_to_dates(dates=date_space, satnum=25544)
# Propagates ISS, note that output is technically in TEME frame, but we'll treat it as if it's just ECI

# %%
# Let's scatter plot the object positions we've propagated
pl = pv.Plotter()
ps.plot_earth(pl, date=date_space[0], night_lights=True, atmosphere=True)
ps.plot3(
    pl,
    r_eci,
    line_width=4,
    lighting=False,
)
pl.camera.focal_point = (0.0, 0.0, 0.0)
pl.camera.position = 25e3 * np.array([1, 1, 0.3])
pl.show()
