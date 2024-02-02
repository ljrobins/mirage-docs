"""
Star Catalogs
=============

Initializing and querying star catalogs
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import mirage as mr
import mirage.vis as mrv

station = mr.Station()
date = mr.now()
mr.tic("Loading Gaia")
gaia = mr.StarCatalog("gaia", station, date, limiting_magnitude=15)
mr.toc()

mr.tic("Loading Tycho-2")
tycho2 = mr.StarCatalog("tycho2", station, date, limiting_magnitude=15)
mr.toc()

eci_look_dir = mr.hat(np.array([1, 1, 0]))
look_ra, look_dec = mr.eci_to_ra_dec(eci_look_dir)
scope_up_initial = np.array([0, 1, 0])
telescope = mr.Telescope(preset="pogs")
mr.tic("Finding stars in frame for Tycho-2")
if_uvs_tycho2, if_mags_tycho2 = tycho2.in_fov(eci_look_dir, scope_up_initial)
mr.toc()

print(f"Tycho-2 found {if_uvs_tycho2.shape[0]} stars in frame")

mr.tic("Finding stars in frame for Gaia")
if_uvs_gaia, if_mags_gaia = gaia.in_fov(eci_look_dir, scope_up_initial)
mr.toc()

print(f"Gaia found {if_uvs_gaia.shape[0]} stars in frame")

# %%
# Plotting the FOV stars
gaia_pix_x, gaia_pix_y = telescope.j2000_unit_vectors_to_pixels(
    eci_look_dir, scope_up_initial, if_uvs_gaia
)
tycho_pix_x, tycho_pix_y = telescope.j2000_unit_vectors_to_pixels(
    eci_look_dir, scope_up_initial, if_uvs_tycho2
)

plt.figure()
plt.scatter(gaia_pix_x, gaia_pix_y, s=2 * if_mags_gaia, c="black")
plt.scatter(tycho_pix_x, tycho_pix_y, s=if_mags_tycho2 / 10, c="cyan")
plt.title("Tycho-2 vs Gaia up close")
plt.xlabel("RA (pixels)")
plt.ylabel("Dec (pixels)")
plt.gca().set_aspect("equal")
plt.legend(["Gaia", "Tycho-2"])
plt.show()

# %%
# Plotting a heatmap of the RA/Dec of both catalogs with seaborn

fig, axs = plt.subplots(2, 1, figsize=(8, 8))
tycho_weights = mr.apparent_magnitude_to_irradiance(tycho2._mags)
gaia_weights = mr.apparent_magnitude_to_irradiance(gaia._mags)

sns.histplot(
    x=tycho2._alpha,
    y=tycho2._delta,
    ax=axs[0],
    bins=50,
    cbar=False,
    weights=tycho_weights,
    cbar_kws={"label": "Cummulative irradiance [W/m^2]"},
)
axs[0].set_title("Tycho-2")
axs[0].set_xlabel("RA (rad)")
axs[0].set_ylabel("Dec (rad)")
axs[0].set_aspect("equal")

sns.histplot(
    x=gaia._alpha,
    y=gaia._delta,
    ax=axs[1],
    bins=50,
    cbar=True,
    cbar_kws={"label": "Cummulative irradiance [W/m^2]"},
    weights=gaia_weights,
    cbar_ax=mrv.get_cbar_ax(),
)
axs[1].set_title("Gaia")
axs[1].set_xlabel("RA (rad)")
axs[1].set_ylabel("Dec (rad)")
axs[1].set_aspect("equal")

plt.tight_layout()
plt.show()
