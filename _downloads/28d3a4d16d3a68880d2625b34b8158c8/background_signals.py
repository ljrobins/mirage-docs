"""
Background Signals
==================

The mean background signal model due to various sources
"""


import datetime

import numpy as np
import pyvista as pv
import pyspaceaware as ps
import pyspaceaware.vis as psv

# %%
# Defining a function we can use to plot various background signals


def hemisphere_signal(
    station: ps.Station,
    date: datetime.datetime,
    signal_kwargs: dict,
) -> None:
    pl = pv.Plotter()
    c_grid = psv.celestial_grid(30, 30)
    (g_az, g_el) = np.meshgrid(
        np.linspace(0, 2 * np.pi, 250),
        np.linspace(np.deg2rad(10), np.pi / 2, 250),
    )

    ps.tic()
    look_dirs_eci_eq = station.az_el_to_eci(g_az.flatten(), g_el.flatten(), date)
    ps.toc()

    dates = np.tile(date, (g_az.size, 1))
    stat_eci = station.j2000_at_dates(date)

    ps.tic()
    sb = station.sky_brightness(dates, look_dirs_eci_eq, **signal_kwargs)
    ps.toc()
    psv.plot_earth(
        pl,
        date=date,
        atmosphere=False,
        lighting=True,
        night_lights=True,
        borders=True,
    )
    r_dome = 500  # km
    view_dist = 20e3  # km
    zoom = 4.5
    sargs = dict(
        height=0.75,
        vertical=True,
        position_x=0.05,
        position_y=0.05,
        title_font_size=24,
        label_font_size=20,
        shadow=True,
        n_labels=4,
        fmt="%.3e",
        font_family="courier",
        color="white",
    )

    psv.plot3(
        pl,
        stat_eci + 1.03 * r_dome * c_grid,
        line_width=4,
        color="linen",
        lighting=False,
    )

    psv.scatter3(
        pl,
        stat_eci + r_dome * look_dirs_eci_eq,
        scalars=sb.flatten(),
        cmap="bmy",
        point_size=10,
        lighting=False,
        clim=[0, np.max(sb)],
        scalar_bar_args=sargs,
    )
    pl.scalar_bar.SetTitle("[e-/pix]")
    pl.camera.focal_point = stat_eci.flatten()
    cpos = (stat_eci + ps.hat(stat_eci) * view_dist).flatten() - np.array([0, 0, 1e4])
    pl.camera.position = cpos
    pl.camera.zoom(zoom)

    pl.add_text(
        f'{date.strftime("%m/%d/%Y, %H:%M:%S")} UTC',
        name="utc_str",
        font="courier",
        color="white",
    )

    pl.show()


# %%
# Setting up observation conditions using an example Space Debris Telescope preset from Krag2003
# station = ps.Station(preset="lmt", lat_deg=33.776864, lon_deg=-84.363777) # Atlanta, GA
station = ps.Station(preset="pogs")
station.telescope = ps.Telescope(preset="sdt")
date = ps.utc(2023, 10, 1, 5, 45, 0)  # Fig 5.38

# %%
# Plotting the background signal for scattered moonlight
signal_kwargs = {
    "moonlight": True,
    "airglow": False,
    "integrated_starlight": False,
    "zodiac": False,
    "pollution": False,
    "twilight": False,
}
hemisphere_signal(station, date, signal_kwargs)

# %%
# Plotting the background signal for integrated starlight
signal_kwargs = {
    "moonlight": False,
    "airglow": False,
    "integrated_starlight": True,
    "zodiac": False,
    "pollution": False,
    "twilight": False, 
}
hemisphere_signal(station, date, signal_kwargs)

# %%
# Plotting the background signal for light pollution
signal_kwargs = {
    "moonlight": False,
    "airglow": False,
    "integrated_starlight": False,
    "zodiac": False,
    "pollution": True,
    "twilight": False,
}
hemisphere_signal(station, date, signal_kwargs)

# %%
# Plotting the background signal for zodiac light
signal_kwargs = {
    "moonlight": False,
    "airglow": False,
    "integrated_starlight": False,
    "zodiac": True,
    "pollution": False,
    "twilight": False,
}
hemisphere_signal(station, date, signal_kwargs)

# %%
# Plotting the background signal for airglow
signal_kwargs = {
    "moonlight": False,
    "airglow": True,
    "integrated_starlight": False,
    "zodiac": False,
    "pollution": False,
    "twilight": False,
}
hemisphere_signal(station, date, signal_kwargs)

# %%
# Plotting the background signal for scattered twilight
signal_kwargs = {
    "moonlight": False,
    "airglow": False,
    "integrated_starlight": False,
    "zodiac": False,
    "pollution": False,
    "twilight": True,
}
hemisphere_signal(station, date - ps.hours(4), signal_kwargs)

# %%
# Plotting the full background signal with all sources enabled
signal_kwargs = {
    "moonlight": True,
    "airglow": True,
    "integrated_starlight": True,
    "zodiac": True,
    "pollution": True,
    "twilight": True,
}
hemisphere_signal(station, date, signal_kwargs)