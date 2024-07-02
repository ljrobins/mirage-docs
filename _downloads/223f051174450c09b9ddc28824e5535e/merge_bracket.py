"""
Shape Interpolation Bracket
===========================

Shape interpolating using SDFs for large numbers of input shapes
"""

# %%
# Problem: Merging a bunch of objects all at once leads to terrible results

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

objs = [mr.SpaceObject(x) for x in ['duck.obj', 'cylinder.obj']]

w2 = mr.SphericalWeight(mr.spiral_sample_sphere(5), np.random.rand(5))
weights = np.array([10, w2])
obj_merged = mr.merge_shapes(objs, weights, grid_density=200)

pl = pv.Plotter()
mrv.render_spaceobject(pl, obj_merged)
pl.show()
