"""
CCD Scintillation
=================
Generating convolution kernels that take into account atmospheric turbulence
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv

# %%
# Plotting the convolution kernel with no atmospheric scintillation
direction = [1, 0.4]
length = 300
kernel_no_scint = mr.streak_convolution_kernel(direction, length)
kernel_scint = mr.streak_convolution_kernel(
    direction, length, position_turbulence=0.1, intensity_turbulence=0.1
)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(kernel_no_scint, cmap="hot")
mrv.texit("Kernel Without Scintillation", "", "", grid=False)
plt.colorbar(cax=mrv.get_cbar_ax(), label="ADU")
plt.clim(0, np.max(kernel_no_scint))
plt.subplot(1, 2, 2)
plt.imshow(kernel_scint, cmap="hot")
mrv.texit("Kernel With Scintillation", "", "", grid=False)
plt.colorbar(cax=mrv.get_cbar_ax(), label="ADU")
plt.clim(0, np.max(kernel_scint))
plt.tight_layout()
plt.show()

# %%
# Let's make sure that the volume of these kernels is one:

print(f"Kernel volume without scintillation: {np.sum(kernel_no_scint):.4f}")
print(f"Kernel volume with scintillation: {np.sum(kernel_scint):.4f}")


# %%
# Plotting a point after applying the streak filter
import itertools

telescope = mr.Telescope(preset="pogs")
ccd_adu = np.zeros((600, 600))
ccd_adu[300:310, 300:310] = 1

turbs = [0.0, 0.1, 0.3]

plt.figure(figsize=(5, 5))
for i, turb in enumerate(itertools.product(turbs, repeat=2)):
    plt.subplot(len(turbs), len(turbs), i + 1)
    ccd_adu_scint = mr.streak_convolution(
        ccd_adu,
        (1.0, 0.0),
        200,
        position_turbulence=turb[0],
        intensity_turbulence=turb[1],
    )

    plt.imshow(ccd_adu_scint, cmap="hot")
    plt.xticks([])
    plt.yticks([])

plt.suptitle("Streaks with Atmospheric Scintillation")
plt.gcf().supxlabel("Increasing Intensity Turbulence")
plt.gcf().supylabel("Increasing Position Turbulence")
plt.tight_layout()
plt.show()
