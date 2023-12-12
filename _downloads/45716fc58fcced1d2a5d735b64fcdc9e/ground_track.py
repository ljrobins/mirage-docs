"""
Ground Tracks
=============

Plots the ground track for a GPS satellite
"""


import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import mirage as mr

# %%
# Let's set up a space of dates to operate on

dates = mr.date_linspace(mr.now(), mr.now() + mr.days(0.5), 8640) - mr.days(100)

# %%
# And propagate one of the NAVSTAR satellites to all the dates
obj = mr.SpaceObject("cube.obj", identifier="NAVSTAR 81 (USA 319)")
r_eci = obj.propagate(dates)

# %%
# Converting the propagated result into ECEF, then LLA
r_ecef = mr.stack_mat_mult_vec(mr.j2000_to_itrf(dates), r_eci)
lla = mr.itrf_to_lla(r_ecef)

# %%
# Finally, plotting the resulting Earth-fixed trajectory with the Earth in the background
im = Image.open(
    os.path.join(os.environ["TEXDIR"], "world.topo.bathy.200412.3x5400x2700.jpg")
)
plt.imshow(im, extent=(-180, 180, -90, 90))
plt.scatter(np.rad2deg(lla[1]), np.rad2deg(lla[0]), s=1, c="m")
plt.title(f"{obj.satnum} Ground Track")
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
plt.show()

# %%
# Repeating with PyGMT
# ====================

import pygmt

# Create a new instance or object of the pygmt.Figure() class
fig = pygmt.Figure()
projection = "N12c"
# projection = "G-90/10/12c"
# Orthographic projection (G) with projection center at 0° East and
# 15° North and a width of 12 centimeters
fig.coast(
    projection=projection,
    region="g",  # global
    frame="g30",  # Add frame and gridlines in steps of 30 degrees on top
    land="gray",  # Color land masses in "gray"
    water="lightblue",  # Color water masses in "lightblue"
    # Add coastlines with a 0.25 points thick pen in "gray50"
    shorelines="1/0.25p,gray50",
)

x, y = np.rad2deg(lla[1]), np.rad2deg(lla[0])
fig.plot(x=x, y=y, pen="1.5p,red")

sl = slice(0, len(dates), len(dates) // 10)
rg = range(0, len(dates), len(dates) // 10)

fig.text(
    text=[d.strftime("%H:%M UTC") for d in dates[sl]],
    x=x[sl],
    y=y[sl],
    font="9p,Courier-Bold,black",
)

fig.show()
