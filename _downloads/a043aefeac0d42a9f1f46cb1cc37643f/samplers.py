"""
Samplers
========

Various strategies for sampling vectors and quaternions
"""

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

# %%
# Sampling uniform unit vectors
n = 1_000
pl = pv.Plotter()
uv = mr.rand_unit_vectors(n)
mrv.scatter3(pl, uv, point_size=10, color="c")
mrv.two_sphere(pl)
pl.show()

# %%
# Sampling uniform vectors in a cone
v = mr.rand_unit_vectors(1)
pl = pv.Plotter()
uv = mr.rand_cone_vectors(v, np.pi / 10, n)
mrv.scatter3(pl, uv, point_size=10, color="c")
mrv.scatter3(pl, v, point_size=20, color="r")
mrv.two_sphere(pl)
pl.view_isometric()
pl.show()

# %%
# Sampling uniform vectors in multiple cone
v = mr.rand_unit_vectors(100)
pl = pv.Plotter()
uv = mr.rand_cone_vectors(v, np.pi / 30, 100)
mrv.scatter3(pl, uv, point_size=10, color="c")
mrv.scatter3(pl, v, point_size=20, color="r")
mrv.two_sphere(pl)
pl.view_isometric()
pl.show()

# %%
# Sampling uniformly in the volume of a ball
pl = pv.Plotter()
uv = mr.rand_points_in_ball(1.0, n)
mrv.scatter3(pl, uv, point_size=10, cmap="cool", scalars=mr.vecnorm(uv))
mrv.two_sphere(pl)
pl.view_isometric()
pl.show()

# %%
# Sampling uniformly in a spherical shell
pl = pv.Plotter()
uv = mr.rand_points_in_shell((0.6, 0.7), n)
mrv.scatter3(pl, uv, point_size=10, cmap="cool", scalars=mr.vecnorm(uv))
mrv.two_sphere(pl)
pl.view_isometric()
pl.show()
