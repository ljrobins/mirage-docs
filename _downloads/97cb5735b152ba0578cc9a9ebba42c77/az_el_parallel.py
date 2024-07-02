"""
Azimuth/Elevation Conversion
============================

Given a station and a target in inertial space we can compute the azimuth and elevation of the object, or invert an azimuth and elevation into a new look direction
"""

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

# %%
# Let's use the Purdue Optical Ground Station for this example
station = mr.Station(preset='pogs')
dt0 = mr.utc(2023, 8, 13)
dates = mr.date_linspace(dt0, dt0 + mr.days(1), int(1e4))

# %%
# Let's extract the station's position in J2000 and create an ECI look direction which is just outwards and down towards the equator
station_j2k = station.j2000_at_dates(dates)

eci_pos = station_j2k * 2
eci_pos[:, -1] = 0

target_dir_eci = mr.hat(eci_pos - station_j2k)


# %%
# We can then turn this target direction into an azimuth/elevation pair
az, el = station.eci_to_az_el(dates, target_dir_eci)
print(f'{az[0]=} [rad]')
print(f'{el[0]=} [rad]')

# %%
# And easily invert that into the same unit vector direction
eci_hat = station.az_el_to_eci(az, el, dates)
print(
    f'ECI vector reconstruction error: {np.linalg.norm(target_dir_eci[0,:] - eci_hat[0,:])}'
)

# %%
# We can also display this resulting unit vector as a ray cast out from the observer, hitting the target
pl = pv.Plotter()
stat_origin0 = station.j2000_at_dates(dates[0])
mrv.plot_earth(pl, date=dates[0], high_def=True, atmosphere=False, borders=True)
rotm = mr.EarthFixedFrame('itrf', 'j2000').rotms_at_dates(dates[0]) @ mr.enu_to_ecef(
    station.itrf
)
mrv.plot_basis(
    pl, rotm, labels=['E', 'N', 'U'], origin=stat_origin0, scale=1000, color='linen'
)
mrv.plot_arrow(
    pl,
    origin=stat_origin0,
    direction=eci_hat[0, :],
    scale=1000,
    color='c',
    label='To object',
)
pl.camera.focal_point = stat_origin0
pl.camera.position = np.array(pl.camera.focal_point) + 10 * np.array([-500, -200, 200])
print(pl.camera.focal_point)
pl.show()
