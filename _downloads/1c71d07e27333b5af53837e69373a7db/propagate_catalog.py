"""
Propagating the catalog
=========================

Load the full TLE catalog and propagate all objects to a given epoch
"""


import datetime

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv


# %%
# First, let's define a function that plots the catalog at a given date
def plot_catalog_at_date(pl: pv.Plotter, date: datetime.datetime):
    r_eci = mr.propagate_catalog_to_dates(date)
    mrv.plot_earth(
        pl,
        date=date,
        atmosphere=True,
        night_lights=True,
    )
    mrv.scatter3(
        pl, r_eci, show_scalar_bar=False, point_size=3, lighting=False, color="k"
    )
    pl.add_text(f"{date.day}/{date.month}/{date.year}", font="courier")
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.position = 180e3 * np.array([0, 0.01, 1])


# %%
# Space in 2023 compared with space in 2000
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pl = pv.Plotter(shape=(1, 2), window_size=(2000, 1000))
pl.subplot(0, 1)
plot_catalog_at_date(pl, mr.today())
pl.disable_anti_aliasing()
pl.subplot(0, 0)
plot_catalog_at_date(pl, mr.today() - mr.years(23))
pl.disable_anti_aliasing()
pl.link_views()
pl.background_color = "white"
pl.show()


# %%
# Space in 1980
# ~~~~~~~~~~~~~
pl = pv.Plotter()
plot_catalog_at_date(pl, mr.today() - mr.years(43))
pl.show()
