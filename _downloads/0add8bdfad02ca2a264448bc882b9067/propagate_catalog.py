"""
Propagating the catalog
=========================

Load the full TLE catalog and propagate all objects to a given epoch
"""


import pyspaceaware as ps
import pyvista as pv
import datetime
import numpy as np

date_start = datetime.datetime(
    2023, 12, 21, 4, 0, 0, tzinfo=datetime.timezone.utc
)

(r_eci, _) = ps.propagate_catalog_to_dates(date_start)
r_eci = r_eci.squeeze()

# %%
# Sometimes, propagating a TLE too far past its last collection point leads to enormous position magnitudes, let's filter those out
invalid = ps.vecnorm(r_eci).flatten() > 1e5
r_eci = np.delete(r_eci, np.argwhere(invalid), axis=0)

# %%
# Let's scatter plot the object positions we've propagated
pl = pv.Plotter()
ps.plot_earth(
    pl,
    mode="eci",
    date=date_start,
)
ps.scatter3(pl, r_eci, show_scalar_bar=False, point_size=1)
pl.camera.focal_point = (0.0, 0.0, 0.0)
# Otherwise it'll use the mean of all vertices, including the sats
pl.camera.position = 70e3 * np.array([1, -1, 0.3])
pl.show()
