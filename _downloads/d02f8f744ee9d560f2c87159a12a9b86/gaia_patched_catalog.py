"""
GAIA Patched Catalog
====================

Displays the patched GAIA catalog
"""

import matplotlib.pyplot as plt
import numpy as np

# %%
# Let's set up a grid of directions to plot the starlight signal at in J2000
import mirage as mr
import mirage.vis as mrv

dec_grid, ra_grid = np.meshgrid(
    np.linspace(-np.pi / 2, np.pi / 2, 180),
    np.linspace(-np.pi, np.pi, 360),
    indexing="ij",
)

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

f_star = mr.catalog_starlight_signal(
    ra_grid, dec_grid
)  # Units [10th magnitude stars / deg^2] = S_10

plt.imshow(
    np.flipud(f_star),
    cmap="hot",
    extent=(-180, 180, -90, 90),
)

mrv.texit(
    "Patched GAIA Catalog $m \geq 16$", "Right Ascension [deg]", "Declination [deg]"
)
plt.colorbar(
    label="Surface brightness $\\left[ S_{10} \\right]$", cax=mrv.get_cbar_ax()
)
plt.show()

# %%
# Now we define the telescope we want to perform the observations with, we'll use the Purdue Optical Ground Station (POGS)
x, y, z = mr.sph_to_cart(ra_grid.flatten(), dec_grid.flatten())
sample_dirs_eci = np.vstack((x, y, z)).T

ts = mr.Telescope(preset="pogs")
sig = mr.integrated_starlight_signal(
    dates=None,
    look_dirs_eci_eq=sample_dirs_eci,
    obs_pos_eci_eq=sample_dirs_eci,
    t_int=ts.integration_time,
    scale=ts.pixel_scale,
    d=ts.aperture_diameter,
)

# %%
# Now we reshape the signal into the original grid and display the plot as an image
# We'll also overlay the Tycho 2 RA/Dec coordinates to confirm that both overlap correctly

plt.imshow(
    np.flipud(sig.reshape(dec_grid.shape)),
    cmap="hot",
    extent=(-180, 180, -90, 90),
)

t2 = mr.load_json_data("tycho2.json")
tycho2_ra_rad = t2["j2000_ra"][::10]
tycho2_dec_rad = t2["j2000_dec"][::10]
vm = t2["visual_magnitude"][::10]

# plt.scatter(
#     x=np.rad2deg(tycho2_ra_rad),
#     y=np.rad2deg(tycho2_dec_rad),
#     marker=",",
#     s=0.01,
#     alpha=1 - vm / np.max(vm),
# )
mrv.texit(
    "Patched GAIA Catalog $m \geq 16$", "Right Ascension [deg]", "Declination [deg]"
)
plt.colorbar(label="Total signal [e-/pix]", cax=mrv.get_cbar_ax())
plt.show()

# %%
# We can also display the GAIA patched catalog and the Tycho 2 unit vectors on the ECI unit sphere:
import pyvista as pv

tycho2_unit_vectors = np.vstack(mr.sph_to_cart(az=tycho2_ra_rad, el=tycho2_dec_rad)).T

pl = pv.Plotter()
pl.set_background("black")
mrv.scatter3(
    pl,
    sample_dirs_eci,
    scalars=sig,
    point_size=10,
    cmap="fire",
    opacity=sig / np.max(sig),
)
mrv.scatter3(
    pl,
    tycho2_unit_vectors,
    scalars=1 - vm / np.max(vm),
    point_size=0.05,
    cmap="cool",
)
mrv.plot_basis(pl, np.eye(3), ["x", "y", "z"], scale=1.3, color="cyan")
pl.view_isometric()
pl.show()
