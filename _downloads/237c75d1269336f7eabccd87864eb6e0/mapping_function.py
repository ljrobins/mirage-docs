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

plt.figure()
plt.imshow(dist, cmap="cool")
plt.colorbar(label="Apparent Distance from Pinhole Model [pix]")
cp = plt.contour(dist, levels=[0.01, 0.1, 1, 2, 4, 7], colors="k")
plt.clabel(cp, inline=True, fontsize=14)
plt.title("POGS Pixel Distortion")
plt.tight_layout()

# %%
# Corner of the image principal lines
look_dir = np.array([[1.0, 0.0, 0.0]])
up_dir = np.array([[0.0, 0.0, 1.0]])

plt.figure()
for v in np.arange(0, 101, 10):
    px = np.arange(0, 101, 10)
    py = np.full_like(px, v)

    for x, y in zip([px, py], [py, px]):
        uvs = station.telescope.pixels_to_j2000_unit_vectors(
            look_dir, up_dir, np.vstack((x, y)).T
        )
        pxd, pyd = station.telescope.j2000_unit_vectors_to_pixels(
            look_dir, up_dir, uvs, add_distortion=True
        )
        kwargs = {}
        if v != 0 and v != 100:
            kwargs["alpha"] = 0.15
            plt.scatter(pxd, pyd, c="m", marker="+", **kwargs)
            plt.scatter(x, y, c="k", marker="+", **kwargs)
        plt.plot(x, y, c="k", **kwargs)
        plt.plot(pxd, pyd, c="m", **kwargs)

plt.gca().invert_yaxis()
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.legend(
    ["Distorted", "Undistorted"],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.10),
    ncol=2,
    fancybox=True,
    shadow=True,
)
plt.show()
