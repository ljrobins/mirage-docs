"""
Time Systems
============

Uses Astropy to extract exact offsets between various time systems
"""

import matplotlib.pyplot as plt

import pyspaceaware as ps

dates = ps.date_linspace(ps.now() - ps.years(50), ps.now(), int(1e4))

tai_minus_utc = ps.tai_minus_utc(dates)
tt_minus_utc = ps.tt_minus_utc(dates)
ut1_minus_utc = ps.ut1_minus_utc(dates)

plt.plot(dates, tt_minus_utc - tai_minus_utc)
plt.plot(dates, ut1_minus_utc - tai_minus_utc)
plt.plot(dates, tai_minus_utc - tai_minus_utc)
plt.plot(dates, -tai_minus_utc)
plt.legend(["TT", "UT1", "TAI", "UTC"])
plt.ylabel("Difference to TAI [sec]")
plt.xlabel("Date")
plt.show()
