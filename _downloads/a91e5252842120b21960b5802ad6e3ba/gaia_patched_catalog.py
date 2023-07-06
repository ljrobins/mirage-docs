"""
GAIA Patched Catalog
====================

Displays the patched GAIA catalog
"""

# %%
# Let's set up a grid of directions to plot the starlight signal at in J2000
import pyspaceaware as ps
import matplotlib.pyplot as plt
import numpy as np

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
    look_dirs_eci_eq=sample_dirs_eci,
    obs_pos_eci_eq=sample_dirs_eci,
    t_int=ts.integration_time,
    scale=ts.pixel_scale,
    d=ts.aperture_diameter,
)

# %%
# Now we reshape the signal into the original grid and display the plot as an image
plt.imshow(
    np.flipud(sig.reshape(dec_grid.shape)),
    cmap="hot",
    extent=(-180, 180, -90, 90),
)
plt.colorbar(label="Total signal [e-/pix]")
plt.title("Patched GAIA Catalog Above Magnitude 16")
plt.xlabel("Inertial Right Ascension [deg]")
plt.xlabel("Inertial Declination [deg]")
plt.show()

# %%
# We can also display this on the ECI unit sphere:
import pyvista as pv

pl = pv.Plotter()
pl.set_background("black")
ps.scatter3(
    pl,
    sample_dirs_eci,
    scalars=sig,
    point_size=10,
    cmap="hot",
    opacity=sig / np.max(sig),
)
ps.plot_basis(pl, np.eye(3), ["x", "y", "z"], scale=1.3, color="cyan")
pl.view_isometric()
pl.show()
