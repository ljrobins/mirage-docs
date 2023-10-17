"""
Characterizing the CCD Background
=================================

Saving background parabola information for the POGS CCD
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import mirage as mr
import mirage.vis as mrv

# %%
# Loading the CCD image
with fits.open(
    os.path.join(os.environ["SRCDIR"], "..", "examples/10-ccd/00095337.fit")
) as hdul:
    ccd_adu = hdul[0].data

# %%
# Plotting the background values as a function of the distance from the center of the image
im_br_parabola, eq = mr.image_background_parabola(ccd_adu, return_eq_str=True)
ccd_adu_minus_parabola = ccd_adu - im_br_parabola
im_br_mask = mr.image_background_naive(ccd_adu_minus_parabola)[0]
var_br = np.var(ccd_adu_minus_parabola[im_br_mask][::10])
ccd_adu_minus_parabola_poisson = ccd_adu_minus_parabola + var_br

plt.imshow(np.log10(ccd_adu_minus_parabola_poisson), cmap="plasma")
mrv.texit("Image Background", "", "", grid=False)
plt.colorbar(cax=mrv.get_cbar_ax(), label="$\log_{10}(ADU)$")
plt.show()
