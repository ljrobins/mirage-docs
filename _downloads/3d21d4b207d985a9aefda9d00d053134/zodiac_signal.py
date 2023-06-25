"""
Zodiac Signal
=============

The background signal model due to zodiac light
"""

import pyspaceaware as ps
import numpy as np
import pyvista as pv

pl = pv.Plotter()
station = ps.Station()
date = ps.now() + ps.hours(8)
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
sz = ps.zodiacal_signal(
    dates,
    look_dirs_eci_eq,
    obs_pos_eci_eq,
    t_int=station.telescope.integration_time,
    scale=station.telescope.pixel_scale,
    d=station.telescope.aperture_diameter,
)
sz = np.reshape(sz, g_el.shape, order="f")

ps.scatter3(
    pl,
    look_dirs_eci_eq,
    scalars=sz.T.flatten(),
    cmap="plasma",
    point_size=10,
)
ps.scatter3(
    pl,
    ps.hat(ps.sun(ps.date_to_jd(dates.flatten()))),
    color="yellow",
    point_size=50,
)
ps.two_sphere(pl)
pl.show()
