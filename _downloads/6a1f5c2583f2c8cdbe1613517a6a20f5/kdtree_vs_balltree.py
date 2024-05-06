"""
KDTree vs BallTree
==================

Benchmarking tree searches in :math:`R^3`
"""

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree as ssKD
from sklearn.neighbors import BallTree as skBT
from sklearn.neighbors import KDTree as skKD

import mirage as mr
import mirage.vis as mrv


def plot_kdtree(pl: pv.Plotter, tree):
    tree_data, index, tree_nodes, node_bounds = tree.get_arrays()

    bounds_boxes = np.zeros((node_bounds.shape[1], 6)) # xmin, xmax, ymin, ymax
    bounds_boxes[:,[0,2,4]] = node_bounds[0,:,:]
    bounds_boxes[:,[1,3,5]] = node_bounds[1,:,:]

    pl.add_mesh(
        pv.MultiBlock([pv.Box(bound) for bound in bounds_boxes]),
        line_width=1,
        style="wireframe",
        color="k",
    )

def plot_balltree(pl: pv.Plotter, tree):
    tree_data, index, tree_nodes, node_bounds = tree.get_arrays()
    centers = node_bounds.squeeze()

    pl.add_mesh(
        pv.MultiBlock([pv.Sphere(radius=r[-1], 
                                 center=b,
                                 theta_resolution=30,
                                 phi_resolution=30) for r,b in zip(tree_nodes, centers)]),
        opacity=0.1,
        style='wireframe',
        color="k",
        line_width=0.1,
    )

obj = mr.SpaceObject("stanford_bunny.obj")
pts = obj.face_centroids

mr.tic("sklearn KDTree build")
kt = skKD(pts, 10)
mr.toc()

mr.tic("sklearn BallTree build")
bt = skBT(pts, 10)
mr.toc()

mr.tic("scipy KDTree build")
kt2 = ssKD(pts, 10)
mr.toc()

# %%
# Querying benchmarks

qpts = mr.rand_points_in_ball(1.0, int(1e5))
mr.tic()
kt.query(qpts)
mr.toc()

mr.tic()
bt.query(qpts)
mr.toc()

mr.tic()
kt2.query(qpts)
mr.toc()

# %%
# KDTree

pl = pv.Plotter(window_size=(2*1080, 2*720))
pl.camera.zoom(4.0)
mrv.render_spaceobject(pl, obj)
plot_kdtree(pl, kt)
pl.show()

# $$
# BallTree

pl = pv.Plotter(window_size=(2*1080, 2*720))
pl.camera.zoom(4.0)
mrv.render_spaceobject(pl, obj)
plot_balltree(pl, bt)
pl.show()