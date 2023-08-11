"""
Background Signals at Zenith
============================
Plotting the general behavior of background signals for zodiac light, moonlight, and integrated starlight at zenith
"""

import numpy as np
import pyvista as pv
import pyspaceaware as ps

# %%
# Let's choose a point after sunset on the US east coast
date = ps.today() + ps.hours(4)

# %%
# We can then generate the background signals for a set of spiral points
npts = int(1e5)
pts = 1e4 * ps.spiral_sample_sphere(npts)
sv = np.tile(ps.hat(ps.sun(date)), (npts, 1))
pts = pts[ps.angle_between_vecs(sv, pts).flatten() > np.pi / 4, :]
tdargs = (
    np.tile([[date]], (pts.shape[0], 1)),
    pts,
    pts / 2 + 0.01,
    1,
    1,
    1,
)
ss = ps.integrated_starlight_signal(*tdargs)
sm = ps.moonlight_signal(*tdargs)
sz = ps.zodiacal_signal(*tdargs)


def plot_sig(pl, s, cmap, scale=1):
    ps.scatter3(
        pl,
        scale * pts,
        scalars=s,
        cmap=cmap,
        opacity=(s - np.min(s)) / (np.max(s) - np.min(s)) / 2,
        point_size=15,
        show_scalar_bar=False,
        lighting=False,
    )


pl = pv.Plotter()
ps.plot_earth(pl, mode="eci", night_lights=True, atmosphere=True, date=date),
plot_sig(pl, ss, "fire", scale=1.2)
plot_sig(pl, sm, "bone", scale=1.2)
plot_sig(pl, sz, "cividis", scale=1.0)
pl.camera.position = (35e3, 35e3, -8e3)
pl.show()
