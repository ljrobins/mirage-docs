"""
Star Matching
=============

Given star centroid locations and an initial estimate of the look direction and tracking rate, fit the catalog
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv

station = mr.Station()
date = mr.now()
mr.tic("Loading star catalog")
catalog = mr.StarCatalog("gaia", station, date, limiting_magnitude=13)
mr.toc()

# %%
# Let's figure out the streak direction

fits_path = os.path.join(os.environ["SRCDIR"], "..", "00161295.48859.fit")
fits_info = mr.info_from_fits(fits_path)

img = fits_info["ccd_adu"]
img_raw = img.copy()
img_log10 = np.log10(img)
img = np.log10(img - mr.image_background_parabola(img))
img[img < 1] = 0
img[np.isnan(img) | np.isinf(np.abs(img))] = 0

up_dir_eci = mr.fits_up_direction(fits_info)

station.telescope.fwhm = 4
mr.tic()
adu_grid_streaked_sampled = station.telescope.ccd.generate_ccd_image(
    fits_info["dates"],
    station,
    fits_info["look_dirs_eci"],
    [1e4],
    catalog,
    scope_up_hat_eci=up_dir_eci,
    hot_pixel_probability=0,
    dead_pixel_probability=0,
    add_parabola=False,
    scintillation=False,
)
mr.toc()

adu_grid_streaked_sampled = np.log10(
    adu_grid_streaked_sampled - mr.image_background_naive(adu_grid_streaked_sampled)[1]
)
adu_grid_streaked_sampled[adu_grid_streaked_sampled < 1] = 0
adu_grid_streaked_sampled[
    np.isnan(adu_grid_streaked_sampled) | np.isinf(np.abs(adu_grid_streaked_sampled))
] = 0

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
mrv.texit("True Image", "", "", grid=False)
plt.subplot(1, 2, 2)
plt.imshow(adu_grid_streaked_sampled, cmap="gray")
mrv.texit("Synthetic Image", "", "", grid=False)
plt.show()
