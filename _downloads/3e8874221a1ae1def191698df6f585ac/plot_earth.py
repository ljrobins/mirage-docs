"""
Plotting Earth
==================

Plotting the Earth with a variety of options
"""

import pyspaceaware as ps
import pyvista as pv
import datetime

date = datetime.datetime(
    2022, 9, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
)

# %%
# Night lights
pl = pv.Plotter()
ps.plot_earth(pl, date=date, night_lights=True)
pl.show()

# %%
# Elevation data and texture map
pl = pv.Plotter()
ps.plot_earth(
    pl,
    date=date,
    elevation=True,
    use_elevation_texture=True,
    lighting=False,
    atmosphere=False,
)
pl.show()

# %%
# Night lights
pl = pv.Plotter()
ps.plot_earth(
    pl,
    date=date,
    night_lights=True,
)
pl.show()

# %%
# Star background
pl = pv.Plotter()
ps.plot_earth(
    pl,
    date=date,
    stars=True,
)
pl.show()

# %%
# All photorealistic settings
pl = pv.Plotter()
ps.plot_earth(
    pl,
    date=date,
    stars=True,
    night_lights=True,
    atmosphere=True,
    high_def=True,
)
pl.show()
