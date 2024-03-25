"""
Mapping Model
=============

Modeling the shape of the CCD image
"""

# %%
# It's important to understand where each pixel in the CCD originates from in the far field

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr

station = mr.Station()
x, y = np.meshgrid(
    np.arange(station.telescope.sensor_pixels),
    np.arange(station.telescope.sensor_pixels),
)
mr.tic("Mapping")
xd, yx = station.telescope.pixel_distortion(
    x, y, station.telescope.sensor_pixels // 2, station.telescope.sensor_pixels // 2
)
mr.toc()

dist = np.sqrt((x - xd) ** 2 + (y - yx) ** 2)

plt.imshow(dist, cmap="cool")
plt.colorbar(label="Apparent Distance from Pinhole Model [pix]")
cp = plt.contour(dist, levels=[0.01, 0.1, 1, 2, 4, 7], colors="k")
plt.clabel(cp, inline=True, fontsize=14)
plt.title("POGS Pixel Distortion")
plt.tight_layout()
plt.show()
