"""
Solar System
============

Plotting the solar system planet directions with respect to the Earth at a given time
"""

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

date = mr.now()

fcns = np.array([
    mr.venus,
    mr.mars,
    mr.jupiter,
    mr.saturn,
    mr.uranus,
    mr.neptune,
    mr.moon,
    mr.sun,
    mr.pluto,
])

labels = np.array([
    "Venus",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Moon",
    "Sun",
    "Pluto",
])

r = np.zeros((len(fcns), 3))
for i,fcn in enumerate(fcns):
    r[i,:] = fcn(date)

pl = pv.Plotter()
mrv.plot_earth(pl, date=date, night_lights=True, elevation=True, atmosphere=True, lighting=False)

mag = 20000
planet_dirs = mr.hat(r)

for pi,labeli in zip(planet_dirs,labels):
    mrv.plot_arrow(pl, np.zeros(3), pi, color="lime", label=labeli, name=labeli, scale=mag * 0.8,
                   font_size=15
    )
grid = mrv.celestial_grid()
mrv.plot3(
    pl,
    mag * grid,
    color="cornflowerblue",
    line_width=5,
    lighting=False,
    opacity=0.2,
)

mrv.orbit_plotter(pl)