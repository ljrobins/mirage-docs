"""
Plotting Earth
==================

Plotting the Earth with a variety of options
"""

import pyspaceaware as ps
import pyvista as pv
import datetime

date = datetime.datetime(
    2022, 9, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
)

# %%
# Night lights
pl = pv.Plotter()
ps.plot_earth(pl, date=date, night_lights=True)
pl.show()

# %%
# Elevation data and texture map
pl = pv.Plotter()
ps.plot_earth(
    pl,
    date=date,
    elevation=True,
    use_elevation_texture=True,
    lighting=False,
    atmosphere=False,
)
pl.show()

# %%
# Night lights
pl = pv.Plotter()
ps.plot_earth(
    pl,
    date=date,
    night_lights=True,
)
pl.show()

# %%
# Star background
pl = pv.Plotter()
ps.plot_earth(
    pl,
    date=date,
    stars=True,
)
pl.show()

# %%
# All photorealistic settings
pl = pv.Plotter()
ps.plot_earth(
    pl,
    date=date,
    stars=True,
    night_lights=True,
    atmosphere=True,
    high_def=True,
)
pl.show()

# %%
# We can also plot a range of dates and save the result as a movie

import numpy as np

dates = ps.now() + ps.days(np.linspace(0, 1, 80, endpoint=False))

pre_render_fcn = lambda pl: (
    ps.plot_earth(pl, mode="eci", night_lights=True, date=dates[0]),
    pl.enable_anti_aliasing("msaa"),
)


def render_fcn(
    pl: pv.Plotter,
    i: int,
    dates: datetime.datetime = None,
):
    ps.plot_earth(
        pl,
        mode="eci",
        night_lights=True,
        high_def=True,
        stars=True,
        atmosphere=True,
        date=dates[i],
    )
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.position = (20000.0, -20000.0, 10000.0)
    pl.camera.up = (0.0, 0.0, 1.0)
    pl.add_text(
        f'{dates[i].strftime("%m/%d/%Y, %H:%M:%S")} UTC',
        name="utc_str",
        font="courier",
    )


ps.render_video(
    pre_render_fcn,
    lambda pl, i: render_fcn(pl, i, dates),
    lambda pl, i: None,
    dates.size,
    "earth_with_nightlights.gif",
    framerate=24,
)
