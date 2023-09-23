"""
Precession and Nutation
=======================

Viewing the evolution of the terrestrial
"""

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

# %%
# Let's use the full range of the datetime module, from near year 0 to near year 9999
date0 = mr.utc(2023, 12, 9) - mr.years(2020)
dates, ep = mr.date_linspace(
    date0, date0 + mr.years(9990), int(1e5), return_epsecs=True
)

# %%
# We then transform the true terrestrial pole in TOD to J2000 to look at the effect of precession and nutation with respect to the J2000 epoch
pole_nominal = np.tile(
    np.array([[0, 0, mr.AstroConstants.earth_r_eq * 1.3]]), (dates.size, 1)
)
mr.tic()
pole_instant = mr.EarthFixedFrame("tod", "j2000").vecs_at_dates(dates, pole_nominal)
mr.toc()

# %%
# We can view this data from a distance to view precession
pl = pv.Plotter()
mrv.plot_earth(pl, lighting=False, high_def=True)
mrv.scatter3(pl, pole_instant, point_size=10)
pl.camera.focal_point = np.mean(pole_instant, axis=0)
pl.camera.position = 6 * pole_nominal[0, :] + np.array([1e-4, 1e-4, 0])
pl.show()


# %%
# And from close up to show nutation
z = 20
pl = pv.Plotter()
mrv.plot_earth(pl, lighting=False, high_def=True)
mrv.plot3(pl, pole_instant, line_width=10, color="m")
pl.camera.focal_point = pole_instant[0, :]
pl.camera.position = pole_instant[0, :] + np.array([1e-2, 1e-2, 25_000 / z])
pl.camera.zoom(z)
pl.show()

# %%
# Animating a full zoom sequence

pl = pv.Plotter()
pl.open_gif("precession_nutation_zoom.gif")
mrv.plot_earth(pl, lighting=False, high_def=True)
mrv.plot3(pl, pole_instant, line_width=10, color="m")
for z in np.logspace(-10, 30, 50, base=1.2):
    pl.camera = pv.Camera()
    pl.camera.focal_point = pole_instant[0, :]
    pl.camera.position = pole_instant[0, :] + np.array([1e-2, 1e-2, 2_000])
    pl.camera.zoom(z)
    pl.write_frame()
pl.close()
