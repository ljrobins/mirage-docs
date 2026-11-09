# /// script
# requires-python = ">=3.11"
# dependencies = ["mirage"]
#
# [tool.uv.sources]
# mirage = { path = "../../", editable = true }
# ///
"""
Sampling Uniform Surface Points
===============================
"""

import pyvista as pv

import mirage as mr
import mirage.vis as mrv

obj = mr.SpaceObject('stanford_dragon.obj')

n_to_sample = int(100000)
mr.tic()
pts, _ = obj.sample_points(n_to_sample)
mr.toc()

p = pv.Plotter()
mrv.render_spaceobject(p, obj)
mrv.scatter3(p, pts, color='k', point_size=10)
p.show()
