"""
Ground Tracks
=============

Plots the ground track for a GPS satellite
"""

import sys

sys.path.append(".")

import pyspaceaware as ps
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# %%
# Let's set up a space of dates to operate on

dates = ps.date_linspace(ps.now(), ps.now() + ps.days(10), 8640) - ps.days(100)

# %%
# And propagate one of the NAVSTAR satellites to all the dates
obj = ps.SpaceObject("cube.obj", identifier="NAVSTAR 81 (USA 319)")
r_eci = obj.propagate(dates)

# %%
# Converting the propagated result into ECEF, then LLA
r_ecef = ps.stack_mat_mult_vec(ps.j2000_to_itrf(dates), r_eci)
lla = ps.itrf_to_lla(r_ecef)

# %%
# Finally, plotting the resulting Earth-fixed trajectory with the Earth in the background
im = Image.open(
    os.path.join(os.environ["TEXDIR"], "world.topo.bathy.200412.3x5400x2700.jpg")
)
plt.imshow(im, extent=(-180, 180, -90, 90))
plt.scatter(np.rad2deg(lla[1]), np.rad2deg(lla[0]), s=1, c="m")
plt.title(f"{obj.identifier} Ground Track")
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
plt.show()
