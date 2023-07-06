"""
SPICE vs Vallado
=================

Computes the difference between the Vallado approximations of the Moon ephemerides and the JPL NAIF SPICE Toolkit results
"""

import sys

sys.path.append("./src")

# %%
# Let's set up the dates we want to evaluate at, here choosing the next year
import pyspaceaware as ps

dates, _ = ps.date_linspace(ps.now(), ps.now() + ps.days(365), int(1e4))

# %%
# Compute the position of the Moon relative to the Earth using SPICE
ps.tic()
spice_moon_state_eci = ps.spice_moon(dates)
ps.toc()

# %%
# And using Vallado's approximation
ps.tic()
ps_moon_state_eci = ps.moon(ps.date_to_jd(dates))
ps.toc()

# %%
# And plot the results
import pyvista as pv

pl = pv.Plotter()
pl.set_background("k")
ps.scatter3(
    pl,
    spice_moon_state_eci - ps_moon_state_eci,
    scalars=ps.vecnorm(
        spice_moon_state_eci - ps_moon_state_eci
    ).flatten(),
    cmap="cividis",
)
pl.view_xy()
pl.show()

# %%
# As we can see, the Vallado approximation is usually a few thousand kilometers off from SPICE, and takes about half the time to evaluate
