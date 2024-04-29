"""
Satellite Lineup
================

Plotting a variety of space objects against a soccer field background for size reference
"""

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

mr.set_model_directory("/Users/liamrobinson/Documents/mirage-models/accurate_sats")
space_objects = [
    mr.SpaceObject("matlib_tdrs.obj", identifier=19548),
    mr.SpaceObject("matlib_astra.obj", identifier=26853),  # ASTRA 2C
    mr.SpaceObject("matlib_hylas4.obj", identifier=44333),  # AT&T T-16
    mr.SpaceObject("matlib_hispasat_30w-6.obj", identifier=44333),  # AT&T T-16
    mr.SpaceObject("matlib_telstar19v.obj", identifier=44333),  # AT&T T-16
    mr.SpaceObject("matlib_tess.obj", identifier=43435),
    mr.SpaceObject("matlib_landsat8.obj", identifier=39084),
    mr.SpaceObject("matlib_saturn_v_sii.obj", identifier=43652),  # ATLAS 5 CENTAUR DEB
    mr.SpaceObject("matlib_starlink_v1.obj", identifier=44743),  # STARLINK 1038
]

for i in range(len(space_objects)):
    so = space_objects[i]
    rotm = np.eye(3)
    if so.file_name in ["matlib_hispasat_30w-6.obj", "matlib_landsat8.obj"]:
        rotm = mr.r3(-np.pi / 2)
    if so.file_name in ["matlib_saturn_v_sii.obj"]:
        rotm = mr.r1(np.pi / 2)
    if so.file_name in ["matlib_starlink_v1.obj"]:
        rotm = mr.r3(np.pi / 2) @ mr.r2(np.pi / 2)
    so._mesh.points = (rotm @ so._mesh.points.T).T
    so.v = (rotm @ so.v.T).T


space_objects = sorted(space_objects, key=lambda so: np.max(mr.vecnorm(so.v)))
print([x.file_name for x in space_objects])

# %%
# Let's plot the sorted set of objects

pl = pv.Plotter()

x_shift = 0
for i, so in enumerate(space_objects):
    x_shift += 5 + 1.4 * i
    mrv.render_spaceobject(pl, so, origin=np.array([x_shift, 0, 0]), color="linen")

field_length_m = 110
image = pv.read("field.jpg")
field_width_m = field_length_m * 1800 / 2880
image.spacing = (field_length_m / 2880, field_width_m / 1800, 0.0)
image.origin = ((x_shift - field_length_m + 5) / 2, -field_width_m / 2, -5)
pl.add_mesh(image, rgb=True)

pl.enable_anti_aliasing("ssaa")
pl.view_xy()
pl.camera.zoom(1.5)
pl.show()
