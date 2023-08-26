"""
Zenith Light Pollution
======================

Plotting and querying a large dataset of zenith light pollution from the `Light Pollution Map <https://www.lightpollutionmap.info/help.html#FAQ31>`_, with the raw file `found here <https://www2.lightpollutionmap.info/data/viirs_2022_raw.zip>`_
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter

import pyspaceaware as ps
import pyspaceaware.vis as psv

# %%
# Let's plot the zenith sky radiances as reported by the World Atlas 2015 dataset.

with rasterio.open(os.path.join(os.environ["DATADIR"], "World_Atlas_2015/World_Atlas_2015.tif"), "r") as f:
    ps.tic()
    art_brightness = f.read().squeeze()  # mcd / cm^2
    ps.toc()

    total_brightness = art_brightness + 0.171168465 # mcd/m2
    sqm = np.log10(total_brightness/108000000)/-0.4


# %%
# Plotting zenith light pollution
plt.xlim(-102, -65)
plt.ylim(22, 54)
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
psv.plot_map_with_grid(sqm, "Zenith Light Pollution in Eastern US", r"World Atlas 2015 Zenith Light Pollution $\left[\frac{mag}{arcsec^2}\right]$", cmap=plt.get_cmap("gnuplot_r"), borders=True, border_color='w', extent=(-180, 180, -60, 85))
plt.show()

# %%
# Let's plot the ground-level radiances as observed by the VIIRS satellite

with rasterio.open(os.path.join(os.environ["DATADIR"], "viirs_2022_raw.tif"), "r") as f:
    ps.tic()
    print(f.bounds)
    x = f.read().squeeze()  # nW / cm^2 / sr
    ps.toc()
    x = x[::10, ::10] * 1e-9  # W / cm^2 / sr
    x[np.isinf(np.abs(x))] = np.nan
    x *= 1e4  # W / m^2 / sr
    x = ps.irradiance_to_apparent_magnitude(
        x / ps.AstroConstants.steradian_to_arcsecond2
    )  # mag / arcsec^2
    x[np.isnan(x) | np.isinf(x)] = 22.0

# %%
# Plotting ground light pollution sources
plt.xlim(-102, -65)
plt.ylim(22, 54)
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
psv.plot_map_with_grid(x, "Ground Sources of Light Pollution in Eastern US", r"VIIRS/NPP Lunar BRDF-Adjusted Night Lights $\left[\frac{mag}{arcsec^2}\right]$", cmap=plt.get_cmap("gnuplot_r"), borders=True, border_color='w', extent=(-180, 180, -65, 75))
plt.show()

