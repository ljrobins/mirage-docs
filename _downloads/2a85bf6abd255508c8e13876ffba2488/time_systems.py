"""
Time Systems
============

Uses Astropy to extract exact offsets between various time systems
"""

import matplotlib.pyplot as plt

import mirage as mr

dates = mr.date_linspace(mr.now() - mr.years(50), mr.now(), int(1e4))

tai_minus_utc = mr.tai_minus_utc(dates)
tt_minus_utc = mr.tt_minus_utc(dates)
ut1_minus_utc = mr.ut1_minus_utc(dates)

plt.plot(dates, tt_minus_utc - tai_minus_utc)
plt.plot(dates, tai_minus_utc - tai_minus_utc)
plt.plot(dates, -tai_minus_utc)
plt.plot(dates, ut1_minus_utc - tai_minus_utc)

plt.legend(["TT", "TAI", "UTC", "UT1"])
plt.ylabel("Difference to TAI [sec]")
plt.xlabel("Date")
plt.show()
