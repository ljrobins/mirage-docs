"""
Azimuth/Elevation Conversion
============================

Given a station and a target in inertial space we can compute the azimuth and elevation of the object, or invert an azimuth and elevation into a new look direction
"""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import pyspaceaware as ps

# %%
# Let's use the Purdue Optical Ground Station for this example
station = ps.Station(preset="pogs")
dates = ps.date_linspace(ps.now(), ps.now() + ps.days(1), int(1e4)) + ps.hours(12)

# %%
# Let's extract the station's position in J2000 and create an ECI look direction which is just outwards and down towards the equator
station_j2k = station.j2000_at_dates(dates)

eci_pos = station_j2k * 2
eci_pos[:, -1] = 0

target_dir_eci = ps.hat(eci_pos - station_j2k)


# %%
# We can then turn this target direction into an azimuth/elevation pair
az, el = station.eci_to_az_el(dates, target_dir_eci)
print(f"{az[0]=} [rad]")
print(f"{el[0]=} [rad]")

# %%
# And easily invert that into the same unit vector direction
eci_hat = station.az_el_to_eci(az, el, dates)
print(
    f"ECI vector reconstruction error: {np.linalg.norm(target_dir_eci[0,:] - eci_hat[0,:])}"
)

# %%
# We can also display this resulting unit vector as a ray cast out from the observer, hitting the target
s = 10
pl = pv.Plotter()
stat_origin0 = station.j2000_at_dates(dates[0])
ps.plot_earth(pl, date=dates[0], high_def=True, atmosphere=False, borders=True)
rotm = ps.EarthFixedFrame("itrf", "j2000").rotms_at_dates(dates[0]) @ ps.enu_to_ecef(
    station.itrf
)
ps.plot_basis(
    pl, rotm, labels=["E", "N", "U"], origin=stat_origin0, scale=s * 100, color="linen"
)
ps.plot_arrow(
    pl,
    origin=stat_origin0,
    direction=eci_hat[0, :],
    scale=s * 100,
    color="c",
    label="To object",
)
pl.camera.focal_point = stat_origin0
pl.camera.position = pl.camera.focal_point + s * np.array([300, 400, -200])
pl.show()
