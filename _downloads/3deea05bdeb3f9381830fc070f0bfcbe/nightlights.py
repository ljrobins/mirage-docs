"""
Zenith Light Pollution
======================

Plotting and querying a large dataset of zenith light pollution from `here <https://www.lightpollutionmap.info/help.html#FAQ31>`_, with the raw file `here <https://www2.lightpollutionmap.info/data/viirs_2022_raw.zip>`_ 
"""

import rasterio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import os
import sys

sys.path.append("./src")
import pyspaceaware as ps

sz = 20
lat_space = np.linspace(-90, 90, sz)
lon_space = np.linspace(-180, 180, sz)

lats, lons = np.meshgrid(lat_space, lon_space, indexing="ij")

with rasterio.open(
    os.path.join(os.environ["DATADIR"], "viirs_2022_raw.tif"), "r"
) as f:
    ps.tic()
    x = f.read().squeeze()  # nW / cm^2 / sr
    ps.toc()
    x = x[::10, ::10] * 1e-9  # W / cm^2 / sr
    x[np.isinf(np.abs(x))] = np.nan
    x *= 1e4  # W / m^2 / sr
    x = ps.irradiance_to_apparent_magnitude(
        x / ps.AstroConstants.steradian_to_arcsecond2
    )  # mag / arcsec^2
    x[np.isnan(x) | np.isinf(x)] = 22.0
    x = gaussian_filter(x, sigma=0.5)

# %%
# Plotting zenith light pollution
cmap = plt.get_cmap("gnuplot_r")
plt.imshow(x, extent=(-180, 180, -65, 75), cmap=cmap)
plt.xlim(-102, -65)
plt.ylim(22, 54)
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
plt.title("Zenith Light Pollution in Eastern US")
plt.colorbar(
    label=r"$log_{10}$ VIIRS/NPP Lunar BRDF-Adjusted Night Lights $\left[\frac{nW}{cm^2sr}\right]$"
)
plt.show()
