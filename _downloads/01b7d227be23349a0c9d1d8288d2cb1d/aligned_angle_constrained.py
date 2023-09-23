"""
Aligned and Constrained
=======================

Simulates and animates an aligned and constrained attitude profile

.. note:: If you want to record a .mp4 video instead, try ``pl.open_movie("aligned_and_constrained.mov", framerate=30, quality=9)``
"""


import datetime

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

data_points = 100
obj = mr.SpaceObject("tess.obj", identifier="INTELSAT 511")
date = mr.utc(2022, 12, 9, 14)
(date_space, epsec_space) = mr.date_linspace(
    date,
    date + datetime.timedelta(hours=24),
    data_points,
    return_epsecs=True,
)
(r, v) = obj.propagate(date_space, return_velocity=True)

orbit_normal = mr.hat(np.cross(r, v))
sat_nadir = -mr.hat(r)
t = epsec_space / np.max(epsec_space) * 4 * np.pi

sat_sun = mr.hat(mr.sun(date_space))
att = mr.AlignedAndConstrainedAttitude(
    sat_nadir, sat_sun, date_space, axis_order=(2, 0, 1)
)
c = att.dcms_at_dates(date_space)
quat = mr.dcm_to_quat(c)
(v1, v2, v3) = att.basis_vectors_at_dates(date_space)

sun_in_body = mr.stack_mat_mult_vec(c, sat_sun)
obs_in_body = mr.stack_mat_mult_vec(c, sat_nadir)

pl = pv.Plotter()
pl.open_gif("aligned_and_constrained.gif")

mrv.plot3(pl, r, color="cyan")

omesh = obj._mesh.copy()
cdist = 300
pdist = cdist / 4
psize = 30
pl._on_first_render_request()
pl.render()
for i in range(data_points - 1):
    pl.camera.position = (
        r[i, :] - cdist * sat_nadir[i, :] + cdist / 10 * orbit_normal[i, :]
    )
    pl.camera.focal_point = r[i, :]
    mrv.render_spaceobject(
        pl, obj, origin=r[i, :], scale=10, opacity=1.0, quat=quat[i, :]
    )
    mrv.plot_arrow(pl, r[i, :], v1[i, :], scale=pdist, name="arr_v1")
    mrv.plot_arrow(pl, r[i, :], v2[i, :], scale=pdist, name="arr_v2")
    mrv.plot_arrow(pl, r[i, :], v3[i, :], scale=pdist, name="arr_v3")
    mrv.plot_arrow(
        pl,
        r[i, :],
        sat_sun[i, :],
        scale=pdist,
        name="arr_sun",
        color="y",
        label="Sun",
    )
    mrv.plot_earth(pl, date=date_space[i], atmosphere=True, night_lights=True)
    pl.write_frame()
    obj._mesh.copy_from(omesh)
pl.close()
