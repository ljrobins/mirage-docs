"""
Local Sidereal Time
===================
Studying the yearly variations in sidereal time
"""

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv

stations = [mr.Station(lon_deg=0.0), mr.Station(preset='pogs')]
dates, epsecs = mr.date_arange(
    mr.utc(2020, 1, 1, 12),
    mr.utc(2020, 1, 1, 12) + mr.years(1),
    mr.days(1),
    return_epsecs=True,
)

# %%
# We can see that as expected, Greenwich passes LST = 0 at noon on the vernal equinox (~80 days into the year). Rephrased, this means that at noon on the vernal equinox, the local meridian at Greenwich is pointing in the direction of inertial :math:`\\hat{x}`, the first point of Aires, zero right ascension.
# On the same date, the station in New Mexico is always facing 105 degrees earlier.

for station in stations:
    sid_time = np.rad2deg(mr.sidereal_hour_angle(station.lon_rad, dates)) / 360 * 24
    plt.plot(epsecs / mr.AstroConstants.earth_sec_in_day, sid_time)
mrv.texit(
    'Local Sidereal Time at 12:00 UTC',
    'Day of the year',
    'Sidereal time [hr]',
    [f'$\lambda = {s.lon_deg:.2f}^\circ$' for s in stations],
)
plt.ylim(0, 24)
plt.xlim(0, 366)
plt.show()

# %%
# For example, on June 2nd (day 153 of the year), the LST at 12:00 UTC is approximately 21.67/24 * 360 = 325 degrees. For an observation at ~6:00 UTC, we subtract another 6/24*360 degrees to yield LST = 235 degrees.
