"""
Star Catalogs
=============

Initializing and querying star catalogs
"""

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv

station = mr.Station()
date = mr.now()
mr.tic('Loading Gaia')
gaia = mr.GaiaStarCatalog(station, date)
mr.toc()

mr.tic('Loading Tycho-2')
tycho2 = mr.Tycho2StarCatalog(station, date)
mr.toc()

eci_look_dir = mr.hat(np.array([1, 1, 0]))
look_ra, look_dec = mr.eci_to_ra_dec(eci_look_dir)
scope_up_initial = np.array([0, 1, 0])
telescope = mr.Telescope(preset='pogs')
mr.tic('Finding stars in frame for Tycho-2')
if_uvs_tycho2, if_spec_tycho2 = tycho2.in_fov(eci_look_dir, scope_up_initial)
mr.toc()

print(f'Tycho-2 found {if_uvs_tycho2.shape[0]} stars in frame')

mr.tic('Finding stars in frame for Gaia')
if_uvs_gaia, if_spec_gaia = gaia.in_fov(eci_look_dir, scope_up_initial)
mr.toc()

print(f'Gaia found {if_uvs_gaia.shape[0]} stars in frame')

# %%
# Plotting the FOV stars
gaia_pix_x, gaia_pix_y = telescope.j2000_unit_vectors_to_pixels(
    eci_look_dir, scope_up_initial, if_uvs_gaia
)
tycho_pix_x, tycho_pix_y = telescope.j2000_unit_vectors_to_pixels(
    eci_look_dir, scope_up_initial, if_uvs_tycho2
)

plt.figure()
plt.scatter(gaia_pix_x, gaia_pix_y, s=20, c='black')
plt.scatter(tycho_pix_x, tycho_pix_y, s=1, c='cyan')
plt.title('Tycho-2 vs Gaia up close')
plt.xlabel('RA (pixels)')
plt.ylabel('Dec (pixels)')
plt.gca().set_aspect('equal')
plt.legend(['Gaia', 'Tycho-2'])
plt.show()

# %%
# Star Aberration

t2 = mr.load_json_data('tycho2.json')
alpha_rad, delta_rad = t2['j2000_ra'], t2['j2000_dec']
mr.tic()
alpha_rad_app, delta_rad_app = mr.apparent_star_positons(
    mr.now(), station.lat_geod_rad, alpha_rad, delta_rad
)
mr.toc()

# %%
# Plotting the resulting right ascensions and declinations

plt.scatter(np.rad2deg(alpha_rad), np.rad2deg(delta_rad), s=2)
plt.scatter(np.rad2deg(alpha_rad_app), np.rad2deg(delta_rad_app), s=2)
plt.xlim(0, 1)
plt.ylim(0, 1)
mrv.texit(
    'Apparent and True Star Positions',
    'Right ascension [deg]',
    'Declination [deg]',
    ['True', 'Apparent'],
)
plt.show()
