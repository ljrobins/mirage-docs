"""
Shapefiles
==================

Plotting the Earth with a variety of options
"""
# https://hub.arcgis.com/datasets/esri::world-countries-generalized/explore?location=-0.247398%2C0.000000%2C1.51

import sys

sys.path.append("./src")
import shapefile
import pyspaceaware as ps
import os
import numpy as np


import pyvista as pv

pl = pv.Plotter()
ps.plot_earth(
    pl,
    date=ps.now(),
    mode="eci",
    night_lights=True,
    atmosphere=True,
    borders=True,
)

pl.show()

# print(sf.shapeType, len(shapes))
