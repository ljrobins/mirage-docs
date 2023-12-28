"""
Zenith Light Pollution
======================

Plotting and querying a large dataset of zenith light pollution from the `Light Pollution Map <https://www.lightpollutionmap.info/help.html#FAQ31>`_, with the raw file `found here <https://www2.lightpollutionmap.info/data/viirs_2022_raw.zip>`_
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio

import mirage as mr
import mirage.vis as mrv

# %%
# Let's plot the zenith sky radiances as reported by the World Atlas 2015 dataset.

data_file = os.path.join(os.environ["DATADIR"], "World_Atlas_2015.tif")

if not os.path.exists(data_file):
    mr.save_file_from_url(
        "https://filebin.net/v3ja2gt5jrifqsc6/World_Atlas_2015.zip",
        os.environ["DATADIR"],
    )

with rasterio.open(data_file, "r") as f:
    mr.tic()
    art_brightness = f.read().squeeze()  # mcd / cm^2
    mr.toc()

    mpsas = mr.mcd_per_m2_to_mpsas((art_brightness + 0.0173) * 1e1)


# %%
# Plotting zenith light pollution in MPSAS
plt.xlim(-102, -65)
plt.ylim(22, 54)
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
mrv.plot_map_with_grid(
    mpsas,
    "Light Pollution in Eastern US",
    r"World Atlas 2015 Zenith Brightness $\left[\frac{mag}{arcsec^2}\right]$",
    cmap=plt.get_cmap("gnuplot_r"),
    borders=True,
    border_color="w",
    extent=(-180, 180, -60, 85),
    set_plot_size=False,
)
plt.show()

# %%
# Let's plot the ground-level radiances as observed by the VIIRS satellite

data_file = os.path.join(os.environ["DATADIR"], "viirs_2022_raw.tif")

if not os.path.exists(data_file):
    mr.save_file_from_url(
        "ps://www2.lightpollutionmap.info/data/viirs_2022_raw.zip",
        os.environ["DATADIR"],
    )

with rasterio.open(data_file, "r") as f:
    mr.tic()
    x = f.read().squeeze()  # nW / cm^2 / sr
    mr.toc()
    x = x[::10, ::10] * 1e-9  # W / cm^2 / sr
    x[np.isinf(np.abs(x))] = np.nan
    x *= 1e4  # W / m^2 / sr
    x = mr.irradiance_to_apparent_magnitude(
        x / mr.AstroConstants.steradian_to_arcsecond2
    )  # mag / arcsec^2
    x[np.isnan(x) | np.isinf(x)] = 22.0

# %%
# Plotting ground light pollution sources
plt.xlim(-102, -65)
plt.ylim(22, 54)
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
mrv.plot_map_with_grid(
    x,
    "Ground Light Sources in Eastern US",
    r"VIIRS/NPP Lunar BRDF-Adjusted Zenith Radiance $\left[\frac{mag}{arcsec^2}\right]$",
    cmap=plt.get_cmap("gnuplot_r"),
    borders=True,
    border_color="w",
    extent=(-180, 180, -65, 75),
    set_plot_size=False,
)
plt.show()
