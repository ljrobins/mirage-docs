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

objs = [
    mr.SpaceObject(x)
    for x in ["duck.obj", "cube.obj", "icosahedron.obj", "cylinder.obj"]
]

weights = np.array([10, 2, 3, 1])
obj_merged = mr.merge_shapes(objs, weights, grid_density=150)

pl = pv.Plotter()
mrv.render_spaceobject(pl, obj_merged)
pl.show()
