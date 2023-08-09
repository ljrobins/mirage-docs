"""
Global Elevation
================

Plots global elevations
"""


import os

import rasterio

import pyspaceaware as ps

# %%
# Let's open the GeoTIFF file for the geoid heights
with rasterio.open(
    os.path.join(os.environ["DATADIR"], "us_nga_egm96_15.tif"), "r"
) as f:
    ps.tic()
    x = f.read().squeeze()  # Geoid heights in [m]
    ps.toc()

# %%
# And plot with a map of the Earth below

ps.plot_map_with_grid(
    x,
    "EGM96 Geoid Undulations",
    "Height above WGS84 ellipsoid [m]",
    alpha=0.6,
    cmap="plasma",
    borders=True,
)

# %%
# Repeated with PyGMT
# ===================

import pygmt

# %%
# Plotting terrain elevation above the EGM96 geoid
region = [-179, 179, -89, 89]
projection = "Cyl_stere/30/-20/12c"
cmap = "haxby"
grid = pygmt.datasets.load_earth_relief(resolution="01d", region=region)
fig = pygmt.Figure()

fig.basemap(region=region, projection=projection, frame=["a"])

fig.grdimage(
    grid=grid,
    cmap=cmap,
    projection=projection,
)

fig.grdcontour(
    annotation=None,
    interval=2000,
    grid=grid,
    pen="0.5p,black",
)

fig.colorbar(frame=["x+lTerrain elevation to EGM96 geoid [m]"])
fig.show()

# %%
# Plotting terrain elevation above the EGM96 geoid
grid_geoid = pygmt.datasets.load_earth_geoid(resolution="01d", region=region)
fig = pygmt.Figure()

fig.basemap(region=region, projection=projection, frame=["a"])

fig.grdimage(
    grid=grid_geoid,
    cmap=cmap,
    projection=projection,
)

fig.grdcontour(
    grid=grid_geoid,
    interval=30,
    annotation=30,
)

fig.coast(land="white", transparency=30)


fig.colorbar(frame=["x+lEGM96 Geoid elevation to WGS84 ellipsoid [m]"])
fig.show()
