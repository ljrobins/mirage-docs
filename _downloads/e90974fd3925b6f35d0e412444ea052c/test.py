"""
This is my example script
=========================

This example doesn't do much, it just makes a simple plot
"""


# %%
# This is title

import pyspaceaware as ps
import seaborn as sns
import matplotlib.pyplot as plt
import pyvista as pv
import datetime
import numpy as np

date_start = datetime.datetime(
    2023, 12, 21, 4, 0, 0, tzinfo=datetime.timezone.utc
)  # Fig 5.38

(r_eci, _) = ps.propagate_catalog_to_dates(date_start)
r_eci = r_eci.squeeze()
invalid = ps.vecnorm(r_eci).flatten() > 1e5
r_eci = np.delete(r_eci, np.argwhere(invalid), axis=0)
ps.tic()
irrad_frac = ps.sun_irradiance_fraction(date_start, r_eci)
ps.toc()

pl = pv.Plotter()
ps.plot_earth(
    pl,
    mode="eci",
    high_def=False,
    night_lights=False,
    date=date_start,
)
ps.scatter3(pl, r_eci, scalars=1 - irrad_frac, cmap="hot_r")
pl.camera.focal_point = (0.0, 0.0, 0.0)
pl.show()
