"""
Propagating the catalog
=========================

Load the full TLE catalog and propagate all objects to a given epoch
"""


import datetime

import numpy as np
import pyvista as pv

import pyspaceaware as ps
import pyspaceaware.vis as psv


# %%
# First, let's define a function that plots the catalog at a given date
def plot_catalog_at_date(date: datetime.datetime):
    r_eci = ps.propagate_catalog_to_dates(date)
    pl = pv.Plotter()
    psv.plot_earth(
        pl,
        date=date,
        atmosphere=True,
        night_lights=True,
    )
    psv.scatter3(
        pl, r_eci, show_scalar_bar=False, point_size=5, lighting=False, color="lime"
    )
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.position = 90e3 * np.array([1, 1, 0.3])
    pl.show()


# %%
# Space in 2023
# ~~~~~~~~~~~~~
plot_catalog_at_date(ps.today())

# %%
# Space in 2000
# ~~~~~~~~~~~~~
plot_catalog_at_date(ps.today() - ps.years(23))

# %%
# Space in 1980
# ~~~~~~~~~~~~~
plot_catalog_at_date(ps.today() - ps.years(43))
