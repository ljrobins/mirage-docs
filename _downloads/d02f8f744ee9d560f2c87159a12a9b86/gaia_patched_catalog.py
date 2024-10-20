"""
GAIA Patched Catalog
====================

Displays the patched GAIA catalog
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv

# %%
# Let's set up a grid of directions to plot the starlight signal at in J2000

dec_grid, ra_grid = np.meshgrid(
    np.linspace(-np.pi / 2, np.pi / 2, 180),
    np.linspace(-np.pi, np.pi, 360),
    indexing='ij',
)
look_dir_grid = mr.ra_dec_to_eci(ra_grid.flatten(), dec_grid.flatten())

# %%
# Conversion from :math:`S_{10}` to irradiance

solid_angle_sterad = np.deg2rad(1) ** 2
lambdas = np.linspace(1e-8, 1e-6, int(1e2))
strint = mr.proof_zero_mag_stellar_spectrum(lambdas)  # Approximately same as STRINT
s10_to_irrad = (
    10**-4 * solid_angle_sterad * np.rad2deg(1) ** 2 * np.trapz(strint, lambdas)
)
m_s10 = mr.irradiance_to_apparent_magnitude(s10_to_irrad)
s10_to_irrad_true = np.rad2deg(1) ** 2 * mr.apparent_magnitude_to_irradiance(10)

# %%
# Let's first display the raw :math:`S_{10}` brightness of the patched catalog

x = np.load(os.path.join(os.environ['SRCDIR'], '..', 'patched6.npz'))
signal_3d = x['isl']
lims = x['mag_lims']

start_ind = np.argwhere(lims[:, 0] == 14).squeeze()
signal_2d = np.sum(signal_3d[:, :, start_ind:], axis=2)

xx, yy = np.meshgrid(
    np.linspace(-180, 180, signal_2d.shape[1]), np.linspace(-90, 90, signal_2d.shape[0])
)

plt.figure(figsize=(9, 4))
plt.imshow(
    np.flipud(np.log10(signal_2d)),
    cmap='plasma',
    extent=(-180, 180, -90, 90),
)

mrv.texit(
    'Patched GAIA Catalog $m_{G} \geq 14$',
    'Right Ascension [deg]',
    'Declination [deg]',
    grid=False,
)
plt.colorbar(
    label=r'Zenith signal $\log_{10} \: \frac{W}{m^2\cdot \text{deg}^2}$',
    cax=mrv.get_cbar_ax(),
)
plt.tight_layout()
plt.show()
