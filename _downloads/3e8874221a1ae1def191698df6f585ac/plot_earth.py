"""
Plotting Earth
==================

Plotting the Earth with a variety of options
"""

import datetime

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

date = datetime.datetime(2022, 6, 4, 12, 0, 0, tzinfo=datetime.timezone.utc)
date_space_day = date + mr.days(np.linspace(0, 1, 100, endpoint=False))

# %%
# Just so that the thumbnail of this example is exciting, let's animate a full photorealistic Earth over the course of a day
pl = pv.Plotter()
pl.open_gif("earth_day.gif", fps=20)
for date in date_space_day:
    mrv.plot_earth(pl, date=date, night_lights=True, atmosphere=True)
    pl.camera.position = (40e3, 0.0, 0.0)
    pl.write_frame()
pl.close()

# %%
# We can also plot over the course of the year to show the variation of the Sun
date_space_year = date + mr.days(np.round(np.linspace(0, 365.25, 100, endpoint=False)))
pl = pv.Plotter()
pl.open_gif("earth_year.gif", fps=20)
for date in date_space_year:
    mrv.plot_earth(pl, date=date, night_lights=True, atmosphere=True)
    pl.camera.position = (40e3, 0.0, 0.0)
    pl.write_frame()
pl.close()


# %%
# Elevation data and texture map
pl = pv.Plotter()
mrv.plot_earth(
    pl,
    date=date,
    elevation=True,
    use_elevation_texture=True,
    lighting=False,
)
pl.show()

# %%
# Night lights with latitude and longitude lines
pl = pv.Plotter()
mrv.plot_earth(
    pl,
    date=date,
    night_lights=True,
)
grid = mrv.celestial_grid(15, 15)
mrv.plot3(
    pl,
    mr.AstroConstants.earth_r_eq * grid,
    color="cornflowerblue",
    line_width=5,
    lighting=False,
    opacity=0.2,
)
pl.show()

# %%
# Star background
pl = pv.Plotter()
mrv.plot_earth(
    pl,
    date=date,
    stars=True,
)
pl.show()

# %%
# Country borders
pl = pv.Plotter()
mrv.plot_earth(
    pl,
    date=date,
    borders=True,
)
pl.show()

# %%
# All photorealistic settings
pl = pv.Plotter()
mrv.plot_earth(
    pl,
    date=date,
    stars=True,
    night_lights=True,
    atmosphere=True,
    high_def=True,
)
pl.show()
