"""
Classical IOD Methods
=====================

"""

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

station = mr.Station()
obj = mr.SpaceObject("cube.obj", identifier=36411)
dates = mr.date_linspace(mr.now(), mr.now() + mr.seconds(200), 3)
o_pos = obj.propagate(dates)
s_pos = station.j2000_at_dates(dates)

look_dirs_eci = mr.hat(
    o_pos
    - s_pos
    + np.random.multivariate_normal(
        mean=[0.0, 0.0, 0.0], cov=1e-5 * np.eye(3), size=o_pos.shape
    )
)

ras, decs = mr.eci_to_ra_dec(look_dirs_eci)

# Position vector methods
mr.tic("Gibbs")
state2_gibbs = mr.gibbs_iod(o_pos)
mr.toc()
mr.tic("Herrick-Gibbs")
state2_herrick_gibbs = mr.herrick_gibbs_iod(o_pos, dates)
mr.toc()
# Angles-only methods
mr.tic("Laplace")
state2_laplace = mr.laplace_iod(station, dates, ras, decs)
mr.toc()
mr.tic("Gauss")
state2_gauss = mr.gauss_iod(station, dates, ras, decs)
mr.toc()

dense_dates = mr.date_linspace(mr.now(), mr.now() + mr.days(1), 1_000)
obj_pos = obj.propagate(dense_dates)

pl = pv.Plotter()
mrv.plot3(pl, obj_pos, line_width=10, lighting=False, color="r")

# iod_pos_laplace = mr.integrate_orbit_dynamics(state2_laplace, dense_dates)[:,:3]
# mrv.plot3(pl, iod_pos_laplace, line_width=10, lighting=False, color='c')

iod_pos_gauss = mr.integrate_orbit_dynamics(state2_gauss, dense_dates)[:, :3]
mrv.plot3(pl, iod_pos_gauss, line_width=10, lighting=False, color="lime")

iod_pos_gibbs = mr.integrate_orbit_dynamics(state2_gibbs, dense_dates)[:, :3]
mrv.plot3(pl, iod_pos_gibbs, line_width=10, lighting=False, color="white")

iod_pos_herrick_gibbs = mr.integrate_orbit_dynamics(state2_herrick_gibbs, dense_dates)[
    :, :3
]
mrv.plot3(pl, iod_pos_herrick_gibbs, line_width=10, lighting=False, color="b")

mrv.plot_earth(pl)
pl.show()
