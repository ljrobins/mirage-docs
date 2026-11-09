# /// script
# requires-python = ">=3.11"
# dependencies = ["mirage"]
#
# [tool.uv.sources]
# mirage = { path = "../../", editable = true }
# ///
"""
Introducing Concavities
=======================
This example shows how to introduce concavities in a model using the method from :cite:p:`robinson2022`.
"""

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

obj = mr.SpaceObject('icosahedron.obj', identifier=36411)
disp_dir = np.array([[1.0, 1.0, 1.0]]) / np.sqrt(3)
psi_est = 45
obj = obj.introduce_concavity(disp_dir, psi_est, linear_iter=3)

pl = pv.Plotter()
mrv.render_spaceobject(pl, obj)
pl.camera.position = (4.0, 0.0, 0.0)
pl.show()
