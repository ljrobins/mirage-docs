"""
Merging with Bias
=================

Merging trimeshes using their SDFs with spherical bias weighting
"""

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

obj1 = mr.SpaceObject('icosahedron.obj')
obj2 = mr.SpaceObject('cube.obj')
sdf1 = obj1.get_sdf()
sdf2 = obj2.get_sdf()


# The bias should be a function of azimuth and elevation, returning on [0,1]
weighting1 = lambda az, el: 10
weighting2 = lambda az, el: 10

obj_merged = mr.merge_shapes([obj1, obj2], [weighting1, weighting2])

pl = pv.Plotter()
mrv.render_spaceobject(pl, obj1, color='r', style='wireframe', line_width=5)
mrv.render_spaceobject(pl, obj2, color='b', style='wireframe', line_width=5)
mrv.render_spaceobject(pl, obj_merged, opacity=0.7)
pl.show()

# %%
# Let's visualize the gradient of the SDF

grid = mr.r3_grid(np.max(mr.vecnorm(obj1.v)), 10)
gaz, gel, _ = mr.cart_to_sph(*grid.points.T)

sdfs = sdf1.query_grid(1.3 * np.max(mr.vecnorm(obj1.v)), 150)
pdata = pv.PolyData(grid.points)
pdata['SDF Gradient'] = sdf1.gradient(grid.points)
pdata['SDF Gradient'] = pdata['SDF Gradient'] * grid.spacing
pdata.active_vectors_name = 'SDF Gradient'
arrows = pdata.arrows

pl = pv.Plotter()
mrv.render_spaceobject(pl, obj1, color='r', style='wireframe', line_width=5)
pl.add_mesh(arrows)
pl.show()
