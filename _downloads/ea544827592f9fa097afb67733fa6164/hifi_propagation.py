"""
Propagating with Perturbations
==============================

Propagating an orbit with spherical harmonics and third body effects
"""


import numpy as np
import pyvista as pv

import pyspaceaware as ps

# %%
# Defining our initial condition and the propagation times
rv0 = np.array([8000, 0, 0, 0, 2, 7])
dates = ps.date_linspace(ps.now(), ps.now() + ps.days(10), 10_000)

# %%
# Propagating with the full EGM96 gravity model and Sun/Moon third-body accelerations
ps.tic()
rv = ps.integrate_orbit_dynamics(
    rv0,
    dates,
    gravity_harmonics_degree=360,
    moon_third_body=True,
    sun_third_body=True,
    int_tol=1e-6,  # Because I want this example to render quickly
)
ps.toc()
r = rv[:, :3]

pl = pv.Plotter()
ps.plot_earth(pl, date=ps.now(), night_lights=True, elevation=True, atmosphere=True)
ps.plot3(pl, r, lighting=True)
pl.show()
