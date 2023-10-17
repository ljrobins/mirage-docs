"""
Integration Time
================
Comparing the effects of different integration times on the shape and level of the background
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import seaborn as sns
# %%
# Loading a fits image from the Purdue Optical Ground Station
from astropy.io import fits

import mirage as mr
import mirage.vis as mrv

# %%
# Defining functions that compute the background of a CCD image


# %%
# Loading the CCD image
ccd_dir = os.path.join(os.environ["SRCDIR"], "..", "data")
fit_files = ["00147020.fit", "00130398.fit"]

ccd_paths = [os.path.join(ccd_dir, f) for f in fit_files]

integration_time_seconds = np.zeros(2)
ccd_images = []
for i, ccd_path in enumerate(ccd_paths):
    with fits.open(ccd_path) as hdul:
        ccd_images.append(hdul[0].data)
        integration_time_seconds[i] = hdul[0].header["EXPTIME"]

# %%
# Plotting the two CCD images side by side
plt.figure(figsize=(10, 5))
for i in range(len(ccd_images)):
    plt.subplot(1, 2, i + 1)
    plt.imshow(np.log10(ccd_images[i]), cmap="gist_stern")
    mrv.texit(f"Integration Time: {integration_time_seconds[i]} s", "", "", grid=False)
    plt.colorbar(cax=mrv.get_cbar_ax(), label="$\log_{10}{ADU}$")
    plt.clim(3, 3.5)
plt.tight_layout()
plt.show()

# %%
# Printing the background level of both images
for ccd_adu, int_time in zip(ccd_images, integration_time_seconds):
    ccd_adu_minus_br = mr.ChargeCoupledDevice().subtract_parabola(ccd_adu)
    _, background_mean = mr.image_background_naive(ccd_adu_minus_br)
    print(f"Integration time: {int_time} s: {background_mean} ADU")
