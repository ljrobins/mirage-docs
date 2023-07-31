"""
Geoid Heights
=============

Display grid of EGM96 geoid height above the WGS84 ellipsoid
"""


import pyspaceaware as ps
import os
import rasterio

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
