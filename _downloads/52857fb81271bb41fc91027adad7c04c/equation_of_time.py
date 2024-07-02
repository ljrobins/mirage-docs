"""
The Equation of Time
====================

The equation of time is the difference between apparent solar time and mean solar time. It is caused by the eccentricity of the Earth's orbit and the tilt of the Earth's axis. The equation of time is a periodic function with a period of one year.
"""

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv

date0 = mr.utc(2023, 1, 1, 12)
dates, epsecs = mr.date_arange(
    date0, date0 + mr.years(1), mr.days(1), return_epsecs=True
)
year_frac = epsecs / epsecs[-1]

sun_pos = mr.sun(dates)
sun_pos = mr.stack_mat_mult_vec(mr.j2000_to_itrf(dates), sun_pos)
sun_lon = np.arctan2(sun_pos[:, 1], sun_pos[:, 0])


plt.plot(
    12 * year_frac, -sun_lon / (np.pi * 2) * 86400 / 60, label='Apparent solar time'
)
mrv.texit('The Equation of Time', 'Month of the Year', 'Minutes')
plt.legend()
plt.show()
