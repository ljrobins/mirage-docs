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
rv0 = np.array([8000, 0, 0, 0, 2, 7])
dates = mr.date_linspace(mr.now(), mr.now() + mr.days(10), 10_000)

# %%
# Propagating with the full EGM96 gravity model and Sun/Moon third-body accelerations
mr.tic()
rv = mr.integrate_orbit_dynamics(
    rv0,
    dates,
    gravity_harmonics_degree=360,
    moon_third_body=True,
    sun_third_body=True,
    int_tol=1e-6,  # Because I want this example to render quickly
)
mr.toc()
r = rv[:, :3]

pl = pv.Plotter()
mrv.plot_earth(pl, date=mr.now(), night_lights=True, elevation=True, atmosphere=True)
mrv.plot3(pl, r, lighting=True)
pl.show()
