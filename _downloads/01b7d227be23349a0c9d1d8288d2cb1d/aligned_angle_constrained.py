"""
Aligned and Constrained
=======================

Simulates and animates an aligned and constrained attitude profile

.. note:: If you want to record a .mp4 video instead, try ``pl.open_movie("aligned_and_constrained.mov", framerate=30, quality=9)``
"""


import pyspaceaware as ps
import numpy as np
import pyvista as pv
import datetime

data_points = 100
obj = ps.SpaceObject("tess.obj", identifier="INTELSAT 511")
date = ps.utc(2022, 12, 9, 14)
(date_space, epsec_space) = ps.date_linspace(
    date,
    date + datetime.timedelta(hours=24),
    data_points,
    return_epsecs=True,
)
(r, v) = obj.propagate(date_space, return_velocity=True)

orbit_normal = ps.hat(np.cross(r, v))
sat_nadir = -ps.hat(r)
t = epsec_space / np.max(epsec_space) * 4 * np.pi
jd_space = ps.date_to_jd(date_space)

sat_sun = ps.hat(ps.sun(date_space))
att = ps.AlignedAndConstrainedAttitude(
    sat_nadir, sat_sun, jd_space, axis_order=(2, 0, 1)
)
c = att.dcm_at_jds(jd_space)
quat = ps.dcm_to_quat(c)
(v1, v2, v3) = att.basis_vectors_at_jds(jd_space)

sun_in_body = ps.stack_mat_mult_vec(c, sat_sun)
obs_in_body = ps.stack_mat_mult_vec(c, sat_nadir)

pl = pv.Plotter()
pl.open_gif("aligned_and_constrained.gif")

ps.plot3(pl, r, color="cyan")

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
    obj.render(pl, origin=r[i, :], scale=10, opacity=1.0, quat=quat[i, :])
    ps.plot_arrow(pl, r[i, :], v1[i, :], scale=pdist, name="arr_v1")
    ps.plot_arrow(pl, r[i, :], v2[i, :], scale=pdist, name="arr_v2")
    ps.plot_arrow(pl, r[i, :], v3[i, :], scale=pdist, name="arr_v3")
    ps.plot_arrow(
        pl,
        r[i, :],
        sat_sun[i, :],
        scale=pdist,
        name="arr_sun",
        color="y",
        label="Sun",
    )
    ps.plot_earth(pl, date=date_space[i], atmosphere=True, night_lights=True)
    pl.write_frame()
    obj._mesh.copy_from(omesh)
pl.close()
