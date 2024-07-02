"""
Perfect Information Inversion
=============================

Providing a near-perfect brightness function, let's see how well we can recover the object's shape.
"""

import pyvista as pv

import mirage as mr
import mirage.vis as mrv

n = int(1e4)
svb = mr.rand_unit_vectors(n)
ovb = mr.rand_unit_vectors(n)

brdf = mr.Brdf('phong', 0.5, 0.5, 5)

obj = mr.SpaceObject('cube.obj')

bf = obj.convex_light_curve(brdf, svb, ovb)

egi = mr.optimize_egi(bf, svb, ovb, brdf)

obj_rec = mr.construct_mesh_from_egi(egi)

pl = pv.Plotter()
mrv.render_spaceobject(pl, obj)
mrv.render_spaceobject(pl, obj_rec, color='red')

pl.show()
