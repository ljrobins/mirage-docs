"""
Propagating the catalog
=========================

Load the full TLE catalog and propagate all objects to a given epoch
"""

import datetime
# %%
# First, let's define a function that plots the catalog at a given date
from typing import Callable

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv


def plot_catalog_at_date(
    pl: pv.Plotter,
    date: datetime.datetime,
    color: str = "k",
    scalars: Callable = None,
    cmap: str = "viridis",
    point_size: int = 3,
) -> None:
    r_eci = mr.propagate_catalog_to_dates(date)
    mrv.plot_earth(
        pl,
        date=date,
        atmosphere=True,
        night_lights=True,
    )
    mrv.scatter3(
        pl,
        r_eci,
        show_scalar_bar=False,
        point_size=point_size,
        lighting=False,
        color=color if scalars is None else None,
        scalars=scalars(r_eci) if scalars is not None else None,
        cmap=cmap,
    )
    pl.add_text(f"{date.day}/{date.month}/{date.year}", font="courier")
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.position = 180e3 * np.array([0, 0.01, 1])


# %%
# Space in 2023 compared with space in 2000
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pl = pv.Plotter(shape=(1, 2), window_size=(2000, 1000))
pl.subplot(0, 1)
plot_catalog_at_date(pl, mr.today())
pl.disable_anti_aliasing()
pl.subplot(0, 0)
plot_catalog_at_date(pl, mr.today() - mr.years(23))
pl.disable_anti_aliasing()
pl.link_views()
pl.background_color = "white"
pl.show()


# %%
# Space in 1980
# ~~~~~~~~~~~~~
pl = pv.Plotter()
plot_catalog_at_date(
    pl,
    mr.today(),
    scalars=lambda r: (mr.vecnorm(r) < 42100)
    + (mr.vecnorm(r) < 21_000)
    + (mr.vecnorm(r) < 42190),
    cmap="glasbey",
    point_size=10,
)

pl.open_gif("test.gif", fps=20)
nframes = 150
t = np.linspace(0, 2 * np.pi, nframes, endpoint=False)
path_pts = np.array([np.sin(t), np.cos(t), np.zeros_like(t)]).T
path_pts[:, 2] += 0.5
path_pts = mr.hat(path_pts) * 18e4
pl.camera.center = (0.0, 0.0, 0.0)
for pt in path_pts:
    pl.camera.position = pt
    pl.write_frame()
pl.close()
