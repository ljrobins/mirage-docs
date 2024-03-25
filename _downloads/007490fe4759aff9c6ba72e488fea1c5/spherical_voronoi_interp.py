"""
Spherical Voronoi Interpolation
===============================
"""

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

nref = int(1e2)
ref_pts = mr.spiral_sample_sphere(nref)
ref_weights = np.random.rand(nref)

query_pts = mr.rand_points_in_shell((0, 1), int(1e6))

w1 = mr.SphericalWeight(ref_pts, ref_weights)
mr.tic()
query_weights = w1.query_points(query_pts)
mr.toc()

pl = pv.Plotter()
mrv.scatter3(
    pl,
    query_pts,
    scalars=query_weights,
    point_size=10,
    scalar_bar_args=dict(title="Weight"),
)
mrv.scatter3(pl, w1.ref_pts, color="r", point_size=20)
pl.camera.zoom(1.3)
pl.show()

# %%
# Defining a new set of weights and merging the two

w2 = mr.SphericalWeight(mr.rand_unit_vectors(nref), np.random.rand(nref))
m12 = w1 + w2

pl = pv.Plotter()
mrv.scatter3(
    pl,
    query_pts,
    scalars=m12.query_points(query_pts),
    point_size=10,
    scalar_bar_args=dict(title="Weight"),
)
pl.show()

# %%
# Bilinear interpolation sampling

pl = pv.Plotter()
mrv.scatter3(
    pl,
    query_pts,
    scalars=m12.query_points(query_pts, method="bilinear"),
    point_size=10,
    scalar_bar_args=dict(title="Weight"),
)
mrv.scatter3(pl, m12.ref_pts, color="r", point_size=20)
pl.show()
