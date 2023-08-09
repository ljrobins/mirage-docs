"""
GAIA Patched Catalog
====================

Displays the patched GAIA catalog
"""

import matplotlib.pyplot as plt
import numpy as np

# %%
# Let's set up a grid of directions to plot the starlight signal at in J2000
import pyspaceaware as ps

dec_grid, ra_grid = np.meshgrid(
    np.linspace(-np.pi / 2, np.pi / 2, 180),
    np.linspace(-np.pi, np.pi, 360),
    indexing="ij",
)
x, y, z = ps.sph_to_cart(ra_grid.flatten(), dec_grid.flatten())
sample_dirs_eci = np.vstack((x, y, z)).T

# %%
# Now we define the telescope we want to perform the observations with, we'll use the Purdue Optical Ground Station (POGS)
ts = ps.Telescope(preset="pogs")
sig = ps.integrated_starlight_signal(
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

t2 = ps.load_json_data("tycho2.json")
tycho2_ra_rad = t2["j2000_ra"][::10]
tycho2_dec_rad = t2["j2000_dec"][::10]
vm = t2["visual_magnitude"][::10]

plt.scatter(
    x=np.rad2deg(tycho2_ra_rad),
    y=np.rad2deg(tycho2_dec_rad),
    marker=",",
    s=0.01,
    alpha=1 - vm / np.max(vm),
)
plt.colorbar(label="Total signal [e-/pix]")
plt.title("Patched GAIA Catalog Above Magnitude 16")
plt.xlabel("Inertial Right Ascension [deg]")
plt.ylabel("Inertial Declination [deg]")
plt.show()

# %%
# We can also display the GAIA patched catalog and the Tycho 2 unit vectors on the ECI unit sphere:
import pyvista as pv

tycho2_unit_vectors = np.vstack(ps.sph_to_cart(az=tycho2_ra_rad, el=tycho2_dec_rad)).T

pl = pv.Plotter()
pl.set_background("black")
ps.scatter3(
    pl,
    sample_dirs_eci,
    scalars=sig,
    point_size=10,
    cmap="fire",
    opacity=sig / np.max(sig),
)
ps.scatter3(
    pl,
    tycho2_unit_vectors,
    scalars=1 - vm / np.max(vm),
    point_size=0.05,
    cmap="cool",
)
ps.plot_basis(pl, np.eye(3), ["x", "y", "z"], scale=1.3, color="cyan")
pl.view_isometric()
pl.show()
