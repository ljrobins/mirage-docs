"""
Total Solar Irradiance
======================

Modeling the variations in solar energy output at 1 AU. This example explains the rationale behind the function ``pyspaceaware.total_solar_irradiance_at_dates``

"""

import sys

sys.path.append("./src")
import pyspaceaware as ps
from terrainman import TsiDataHandler
import datetime
import matplotlib.pyplot as plt
import numpy as np

# %%
# Let's plot the variation in the total solar irradiance from the beginning of the J2000 epoch till now

date = datetime.datetime(
    2000, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
)
dates, _ = ps.date_linspace(date, ps.now(), 10_000)

# %%
# The ``terrainman.TsiDataHandler`` class deals with downloading the relevant netCDF4 files from `here <https://www.ncei.noaa.gov/data/total-solar-irradiance/access/daily/>`_. Outside of the interval covered by this dataset (1882-current_year) :math:`1361 \frac{W}{m^2}` is used as a default.
tsi_dh = TsiDataHandler()
ps.tic()
sc_at_one_au = tsi_dh.eval(dates)
ps.toc()

# %%
# Plotting the irradiance over time
sz = 0.5
plt.scatter(dates, sc_at_one_au, s=sz)
plt.title("Total Solar Irradiance")
plt.xlabel("Year")
plt.ylabel(r"Irradiance at 1 AU $\left[\frac{W}{m^2}\right]$")
plt.show()

# %%
# This isn't the end of the story, as the distance to the Sun changes over the course of the year. Let's compute the distance from the Sun to the Earth in AU over this time period we just plotted

earth_to_sun = ps.sun(dates)
earth_to_sun_dist_km = ps.vecnorm(earth_to_sun).flatten()
earth_to_sun_dist_au = earth_to_sun_dist_km / ps.AstroConstants.au_to_km

plt.plot(dates, earth_to_sun_dist_au)
plt.title("Distance from Earth to Sun")
plt.xlabel("Year")
plt.ylabel(r"[AU]")
plt.show()

# %%
# With this distance information, we can augment the Total Solar Irradiance plot to show the actual irradiance felt by a shell at Earth's instantaneous orbital radius. We can do this by noting that doubling the radius of a sphere squares its area, so we just have to divide by the square of the ``earth_to_sun_dist_au``

sc_at_earth_radius = sc_at_one_au / earth_to_sun_dist_au**2
plt.scatter(dates, sc_at_one_au, s=sz, alpha=0.5)
plt.scatter(dates, sc_at_earth_radius, s=sz, color="r")
plt.title("Solar Irradiance At Earth Radius")
plt.xlabel("Year")
plt.ylabel(r"Irradiance $\left[\frac{W}{m^2}\right]$")
plt.show()

# %%
# This has all been packaged up into a single function which evaluates the Total Solar Irradiance at Earth's radius at an array of dates:
# .. autofunction:: pyspaceaware.total_solar_irradiance_at_dates

# %%
# We can prove that this function produces identical outputs to the implementation above:

tsi = ps.total_solar_irradiance_at_dates(dates)
print(
    f"Implemented function max error: {np.max(np.abs(tsi - sc_at_earth_radius))}"
)
