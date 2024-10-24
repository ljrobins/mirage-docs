"""
Stars in FOV
============

Computing which stars are in the field of view
"""

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv


def plot_telescope_fov(station: mr.Station, look_dir: np.ndarray, up_dir: np.ndarray):
    sp = station.telescope.ccd.sensor_pixels
    xy1 = np.vstack((np.arange(sp), np.full(sp, 0))).T
    xy2 = np.vstack((np.arange(sp), np.full(sp, sp))).T
    xy3 = np.vstack((np.full(sp, 0), np.arange(sp))).T
    xy4 = np.vstack((np.full(sp, sp), np.arange(sp))).T

    for xy in [xy1, xy2, xy3, xy4]:
        uvs = station.telescope.pixels_to_j2000_unit_vectors(look_dir, up_dir, xy)
        ras, decs = mr.eci_to_ra_dec(uvs)
        plt.plot(ras, decs, linewidth=2, color='k')


station = mr.Station()
catalog = mr.GaiaStarCatalog(station, date=mr.now())

print(f'Number of stars in catalog: {catalog._alpha.size}')

look_dir = mr.hat(np.array([[1.0, 1.0, 0.0]]))
up_dir = np.array([[0.0, 0.0, 1.0]])

fov_rad = station.telescope.fov_deg * np.pi / 180

mr.tic()
brute_inds = np.argwhere(
    mr.dot(catalog._uvs, look_dir).flatten() > np.cos(1.2 * fov_rad)
).flatten()
brute_time = mr.toc(return_elapsed_seconds=True)

mr.tic()
tree_inds = catalog._tree.query_radius(look_dir, 1.2 * fov_rad)[0]
t1 = mr.toc(return_elapsed_seconds=True)

assert (
    brute_inds.size == np.intersect1d(brute_inds, tree_inds).size
), 'Brute force and tree search returned different numbers of stars!'

mr.tic()
tree_inds_in_fov = mr.unit_vector_inds_in_fov(
    station.telescope, look_dir, up_dir, catalog._uvs[tree_inds]
)
t2 = mr.toc(return_elapsed_seconds=True)

print(f'Brute time total: {brute_time:.2e}')
print(f'Tree time total: {t1+t2:.2e}')
print(f'Factor speedup: {brute_time/(t1+t2):.1f}')

# %%
# Let's plot the tree solution
plt.scatter(
    catalog._alpha[tree_inds][~tree_inds_in_fov],
    catalog._delta[tree_inds][~tree_inds_in_fov],
    s=3,
)
plt.scatter(
    catalog._alpha[tree_inds][tree_inds_in_fov],
    catalog._delta[tree_inds][tree_inds_in_fov],
    s=3,
)
plot_telescope_fov(station, look_dir, up_dir)
mrv.texit('Stars in FOV', 'Right ascension [rad]', 'Declination [rad]')
plt.gca().set_aspect('equal')
plt.show()
