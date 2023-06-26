"""
Background Signals
==================

The signal mean model due to various sources
"""

# %%
# Defining a function we can use to plot various background signals

import pyspaceaware as ps
import numpy as np
import pyvista as pv
import datetime


def hemisphere_signal(
    station: ps.Station, date: datetime.datetime, signal_kwargs: dict
) -> None:
    pl = pv.Plotter()
    (g_az, g_el) = np.meshgrid(
        np.linspace(0, 2 * np.pi, 250),
        np.linspace(np.deg2rad(10), np.pi / 2, 250),
    )

    look_dirs_eci_eq = np.array(
        [
            station.az_el_to_eci(az, el, date)
            for az, el in zip(g_az.flatten(), g_el.flatten())
        ]
    ).squeeze()

    dates = np.tile(date, (g_az.size, 1))
    stat_eci = station.eci_at_dates(date)

    ps.tic()
    sb = station.sky_brightness(
        dates, look_dirs_eci_eq, **signal_kwargs
    )
    ps.toc()
    ps.plot_earth(
        pl,
        date=date,
        atmosphere=False,
        lighting=True,
        night_lights=True,
    )
    r_dome = 500  # km
    view_dist = 40e3  # km
    zoom = 3.5
    ps.scatter3(
        pl,
        stat_eci + r_dome * look_dirs_eci_eq,
        scalars=sb.flatten(),
        cmap="hot",
        point_size=10,
        lighting=False,
        clim=[0, np.max(sb)],
    )

    pl.camera.focal_point = stat_eci.flatten()
    cpos = (stat_eci + ps.hat(stat_eci) * view_dist).flatten()
    pl.camera.position = cpos
    pl.camera.zoom(zoom)
    pl.show()


# %%
# Setting up observation conditions using an example Liquid Mirror Telescope preset
station = ps.Station(preset="lmt")
date = datetime.datetime(
    2023, 10, 1, 5, 45, 0, tzinfo=datetime.timezone.utc
)  # Fig 5.38

# %%
# Plotting the background signal for scattered moonlight
signal_kwargs = {
    "atmos_scattered": False,
    "moonlight": True,
    "airglow": False,
    "integrated_starlight": False,
    "zodiac": False,
}
hemisphere_signal(station, date, signal_kwargs)

# %%
# Plotting the background signal for integrated starlight
signal_kwargs = {
    "atmos_scattered": False,
    "moonlight": False,
    "airglow": False,
    "integrated_starlight": True,
    "zodiac": False,
}
hemisphere_signal(station, date, signal_kwargs)

# %%
# Plotting the background signal for zodiac light
signal_kwargs = {
    "atmos_scattered": False,
    "moonlight": False,
    "airglow": False,
    "integrated_starlight": False,
    "zodiac": True,
}
hemisphere_signal(station, date, signal_kwargs)

# %%
# Plotting the background signal for airglow
signal_kwargs = {
    "atmos_scattered": False,
    "moonlight": False,
    "airglow": True,
    "integrated_starlight": False,
    "zodiac": False,
}
hemisphere_signal(station, date, signal_kwargs)

# %%
# Plotting the background signal for atmospherically scattered light
signal_kwargs = {
    "atmos_scattered": True,
    "moonlight": False,
    "airglow": False,
    "integrated_starlight": False,
    "zodiac": False,
}
hemisphere_signal(station, date, signal_kwargs)

# %%
# Plotting the full background signal with all sources enabled
signal_kwargs = {
    "atmos_scattered": True,
    "moonlight": True,
    "airglow": True,
    "integrated_starlight": True,
    "zodiac": True,
}
hemisphere_signal(station, date, signal_kwargs)
