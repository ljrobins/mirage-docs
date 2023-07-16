"""
Tycho 2 Catalog
===============

Visualizing the Tycho 2 star catalog efficiently
"""
import os
import sys
import pyvista as pv
import numpy as np

sys.path.append("./src")
import pyspaceaware as ps

from scipy.io import loadmat

ps.tic()
mat = loadmat(
    os.path.join(os.environ["DATADIR"], "Tycho_2_fullcatalog.mat")
)
ps.toc()

uvs = mat["Tycho2_full"][0][0][0].T
ra = mat["Tycho2_full"][0][0][1].T
dec = mat["Tycho2_full"][0][0][2].T
vm = mat["Tycho2_full"][0][0][-1].T.flatten()

pl = pv.Plotter()
ps.plot_earth(pl)
# ps.scatter3(pl, uvs, scalars=vm.flatten())
irrad = ps.apparent_magnitude_to_irradiance(vm)
irrad /= np.max(irrad)

use_inds = (
    ~np.isnan(ra.flatten())
    & ~np.isnan(dec.flatten())
    & ~np.isnan(vm.flatten())
)
save_dict = {
    "j2000_ra": ps.wrap_to_pi(np.deg2rad(ra).flatten()[use_inds]),
    "j2000_dec": np.deg2rad(dec).flatten()[use_inds],
    "visual_magnitude": vm.flatten()[use_inds],
}

ps.save_json_data(save_dict, "tycho2.json", 8)
enddd

add_stars = irrad > 0.01
ps.tic()
actor = pl.add_points(
    ps.AstroConstants.earth_r_eq * uvs[add_stars, :],
    render_points_as_spheres=True,
    color="y",
    opacity=vm[add_stars],
    point_size=1,
)
ps.toc()

pl.show()
