"""
Shape Interpolation
===================

Given two shapes as triangulated 3D models, how can we smoothly interpolate another model between them using signed distance fields
"""

import numpy as np
import pyvista as pv

import mirage as mr

# %%
# Animating the entire interpolation

obj1 = mr.SpaceObject("icosahedron.obj").clean()
obj2 = mr.SpaceObject("duck.obj").clean()

pl = pv.Plotter()
pl.open_gif("shape_interpolation.gif")

for frac1 in np.concatenate((np.linspace(0, 1, 20), np.linspace(1, 0, 20))):
    weights = np.array([1 - frac1, frac1]).astype(float)
    mr.tic()
    obj_merged = mr.merge_shapes([obj1, obj2], weights)
    mr.toc()
    pl.add_mesh(obj_merged._mesh, color="lightblue", name="mesh", smooth_shading=True)
    pl.add_text(
        f"{weights[0]*100:3.0f}% Icosahedron \n{weights[1]*100:3.0f}% Duck",
        font="courier",
        name="label",
    )
    pl.write_frame()
pl.close()

# %%
# Individual interpolation steps in a grid

pl = pv.Plotter(shape=(2, 2))

for i, weight1 in enumerate(np.linspace(0, 1, 4)):
    weights = np.array([1 - weight1, weight1]).astype(float)
    obj_merged = mr.merge_shapes(
        [
            mr.SpaceObject("icosahedron.obj").clean(),
            mr.SpaceObject("torus.obj").clean(),
        ],
        weights,
    )
    pl.subplot(i // 2, i % 2)
    pl.add_mesh(obj_merged._mesh, color="lightblue", name="mesh", smooth_shading=True)
    pl.add_text(
        f"{weights[0]*100:3.0f}% Icosahedron \n{weights[1]*100:3.0f}% Torus",
        font="courier",
        name="label",
    )
pl.show()
