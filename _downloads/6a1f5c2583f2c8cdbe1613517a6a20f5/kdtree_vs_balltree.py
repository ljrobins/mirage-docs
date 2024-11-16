"""
KDTree vs BallTree
==================

Benchmarking tree searches in :math:`R^3`
"""

import pyvista as pv
from pykdtree.kdtree import KDTree as pyKD
from scipy.spatial import KDTree as ssKD
from sklearn.neighbors import BallTree as skBT
from sklearn.neighbors import KDTree as skKD

import mirage as mr
import mirage.vis as mrv

obj = mr.SpaceObject('stanford_bunny.obj')
pts = obj.face_centroids

mr.tic('sklearn KDTree build')
kt = skKD(pts, 10)
mr.toc()

mr.tic('sklearn BallTree build')
bt = skBT(pts, 10)
mr.toc()

mr.tic('scipy KDTree build')
kt2 = ssKD(pts, 10)
mr.toc()

mr.tic('pykdtree KDTree build')
kt3 = pyKD(pts, 10)
mr.toc()

# %%
# Querying benchmarks

qpts = mr.rand_points_in_ball(1.0, int(1e5))
mr.tic('sklearn kdtree query')
kt.query(qpts)
mr.toc()

mr.tic('sklearn balltree query')
bt.query(qpts)
mr.toc()

mr.tic('scipy kdtree query')
kt2.query(qpts)
mr.toc()

mr.tic('pykdtree kdtree query')
kt3.query(qpts)
mr.toc()

# %%
# KDTree

pl = pv.Plotter(window_size=(2 * 1080, 2 * 720))
pl.camera.zoom(4.0)
mrv.render_spaceobject(pl, obj)
mrv.plot_kdtree(pl, kt)
pl.show()

# $$
# BallTree

pl = pv.Plotter(window_size=(2 * 1080, 2 * 720))
pl.camera.zoom(4.0)
mrv.render_spaceobject(pl, obj)
mrv.plot_balltree(pl, bt)
pl.show()
