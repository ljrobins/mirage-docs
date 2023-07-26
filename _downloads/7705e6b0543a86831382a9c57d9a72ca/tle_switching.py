"""
TLE Switching
=============

Various methods to switch between TLEs for more accurate long-term propagation
"""
import sys

sys.path.append(".")
import pyspaceaware as ps
import pyvista as pv
import numpy as np

# %%
# Let's use the SUPERBIRD 6 satellite
satdefs = ps.load_satdef_array()
satnum = satdefs.get_satnum_by_name("SUPERBIRD 6")

# %%
# And propagate for the previous 30 days
dtimes, epsec_space = ps.date_linspace(
    ps.now() - ps.days(30), ps.now(), int(1e4), return_epsecs=True
)

# %%
# We can then propagate with three switching strategies:
#   - ``closest`` choses the closest TLE epoch to the current time
#   - ``newest`` choses the most previous recent TLE at each time
#   - ``interp`` choses the most recent and next TLEs and linearly interpolates between their propogated positions
r_closest = ps.tle_propagate_with_switching(
    satnum, dtimes, switch_strategy="closest", frame="ecef"
)
r_interp = ps.tle_propagate_with_switching(
    satnum, dtimes, switch_strategy="interp", frame="ecef"
)
r_newest = ps.tle_propagate_with_switching(
    satnum, dtimes, switch_strategy="newest", frame="ecef"
)

# %%
# We can plot these trajectories to show that they result in similar trajectories
pl = pv.Plotter()
ps.plot_earth(pl, date=dtimes[0], mode="eci", night_lights=True, atmosphere=True)
lw = 6
ps.plot3(pl, r_closest, color="c", lighting=False, line_width=lw)
# ps.plot3(pl, r_newest, color="m", lighting=False, line_width=lw)
# ps.plot3(pl, r_interp, color="lime", lighting=False, line_width=lw)
mid_point = r_interp[r_interp.shape[0] // 2, :]
pl.camera.focal_point = mid_point
pl.camera.position = (np.linalg.norm(mid_point) + 100_000) * (
    ps.hat(mid_point) + np.array([0.0, 0.0, 0.4])
)
pv.rcParams["transparent_background"] = True
pl.show()

# %%
# We can also plot the error between these switching methods. Clearly, the interpolated switching strategy is the most accurate choice

import matplotlib.pyplot as plt

plt.plot(epsec_space / 86400, ps.vecnorm(r_interp - r_closest))
plt.plot(epsec_space / 86400, ps.vecnorm(r_newest - r_closest))
plt.xlabel("Elapsed time [days]")
plt.ylabel("Position error [km]")
plt.legend(["Interp - Closest", "Newest - Closest"])
plt.grid()
plt.show()
