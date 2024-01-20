"""
Propagating with Perturbations
==============================

Propagating an orbit with spherical harmonics and third body effects
"""


import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

# %%
# Defining our initial condition and the propagation times
n = 1
rv0 = np.random.uniform(
    low=[7_000, 0, -1_000, 0, 2.5, 7],
    high=[8_000, 1_000, 0, 0.3, 2.8, 7.3],
    size=(n, 6),
)
dates, epsecs = mr.date_linspace(
    mr.now(), mr.now() + mr.days(50), 100_000, return_epsecs=True
)

# %%
# Propagating with the full EGM96 gravity model and Sun/Moon third-body accelerations
mr.tic()
rv = mr.integrate_orbit_dynamics(
    rv0,
    dates,
    gravity_harmonics_degree=4,
    third_bodies=["sun", "moon", "jupiter"],
    int_tol=1e-9,  # Because I want this example to render quickly
)
rv = rv.reshape((dates.size, 6, -1))
mr.toc()

pl = pv.Plotter()
mrv.plot_earth(pl, date=dates[0], night_lights=True, atmosphere=True)
for i in range(rv.shape[-1]):
    r = rv[:, :3, i]
    mrv.plot3(
        pl,
        r,
        lighting=True,
        scalars=epsecs,
        cmap="twilight",
        show_scalar_bar=False,
        line_width=15,
    )
pl.show()
