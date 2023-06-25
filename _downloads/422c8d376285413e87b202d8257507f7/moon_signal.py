"""
Moonlight Signal
================

The background signal model due to scattered moonlight
"""

import pyspaceaware as ps
import numpy as np
import pyvista as pv

pl = pv.Plotter()
station = ps.Station()
date = ps.now()
station_pos_eci = station.eci_at_dates(date)
(g_az, g_el) = np.meshgrid(
    np.linspace(0, 2 * np.pi, 200),
    np.linspace(0.1, np.pi / 2 - 0.1, 200),
)
look_dirs_eci_eq = np.array(
    [
        station.az_el_to_eci(az, el, date)
        for az, el in zip(g_az.flatten(), g_el.flatten())
    ]
).squeeze()
obs_pos_eci_eq = np.tile(station_pos_eci, (g_az.size, 1))

dates = np.tile(date, (g_az.size, 1))
sm = ps.moonlight_signal(
    dates,
    look_dirs_eci_eq,
    obs_pos_eci_eq,
    t_int=station.telescope.integration_time,
    scale=station.telescope.pixel_scale,
    d=station.telescope.aperture_diameter,
).reshape(g_el.shape, order="f")

obs_to_moon = ps.hat(
    ps.moon(ps.date_to_jd(dates.flatten())) - station_pos_eci
)

ps.scatter3(
    pl,
    look_dirs_eci_eq,
    scalars=sm.T.flatten(),
    cmap="plasma",
    point_size=10,
)
ps.scatter3(pl, obs_to_moon, color="grey", point_size=25)
ps.two_sphere(pl)
pl.show()
