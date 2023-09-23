"""
Local Weather Data
==================
Local pressure and temperature data for a given station
"""

import matplotlib.pyplot as plt

import mirage as mr
import mirage.vis as mrv

station = mr.Station(preset='pogs')
dates = mr.date_linspace(mr.utc(2020, 11, 30), mr.utc(2020, 12, 9), 100)
jds = mr.date_to_jd(dates)
interp_pressure, interp_temp = mr.pressure_and_temp_at_station(station, dates)

plt.subplot(1,2,1)
plt.plot(jds, interp_temp)
mrv.texit("Station Temperature", "Julian date", "Temperature [${}^{\circ}C$]")
plt.subplot(1,2,2)
plt.plot(jds, interp_pressure)
mrv.texit("Station Pressure", "Julian date", "Pressure [mbar]")
plt.tight_layout()
plt.show()