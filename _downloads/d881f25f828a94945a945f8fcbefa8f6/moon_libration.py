"""
Moon Librations
===============

Animations the librations of the Moon: the apparent motion of the Moon as viewed from the Earth
"""

import sys

sys.path.append("./src")
import pyspaceaware as ps
import pyvista as pv

# %%
# Let's animate the librations of the Moon over the course of a synodic month (the time required for the Moon to complete an orbit with respect to the Sun-Earth line).
dates, _ = ps.date_linspace(
    ps.now(),
    ps.now() + ps.days(ps.AstroConstants.moon_synodic_period_days),
    100,
)
pl = pv.Plotter()
pl.open_gif("moon_librations.gif", fps=20)
for i, date in enumerate(dates[:-1]):
    ps.plot_moon(pl, mode="eci", date=date)
    pl.camera.view_angle = 1.0  # The Moon's angular size from the Earth is about 0.5 deg, so let's double that FOV
    pl.camera.focal_point = pl.actors["moon"].user_matrix[
        :3, -1
    ]  # Focal point must be set before position, for some reason
    pl.camera.position = (0.0, 0.0, 0.0)
    pl.write_frame()
pl.close()
