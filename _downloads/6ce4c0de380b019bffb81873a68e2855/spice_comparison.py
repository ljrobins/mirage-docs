"""
SPICE vs Vallado
=================

Computes the difference between the Vallado approximations of the Moon ephemerides and the JPL NAIF SPICE Toolkit results
"""


# %%
# Let's set up the dates we want to evaluate at, here choosing the next year
import mirage as mr
import mirage.vis as mrv

dates = mr.date_linspace(mr.now(), mr.now() + mr.days(365), int(1e3))

# %%
# Compute the position of the Moon relative to the Earth using SPICE
mr.tic()
spice_moon_state_eci = mr.moon(dates, "spice")
mr.toc()

# %%
# And using Vallado's approximation
mr.tic()
ps_moon_state_eci = mr.moon(dates, "vallado")
mr.toc()

# %%
# And plot the results
import pyvista as pv

pl = pv.Plotter()
pl.set_background("k")
mrv.plot3(
    pl,
    spice_moon_state_eci - ps_moon_state_eci,
    scalars=mr.vecnorm(spice_moon_state_eci - ps_moon_state_eci).flatten(),
    cmap="isolum",
    lighting=False,
    line_width=3,
)
pl.view_isometric()

# mr.plot_moon(
#     pl, date=dates[0], mode="mci"
# )  # Display the Moon centered in inertial coordinates
pl.show()

# %%
# As we can see, the Vallado approximation is usually a few thousand kilometers off from SPICE, and takes about half the time to evaluate.
# Let's get a better intuition for the magnitude of this discrepancy by computing the center of the totality of a lunar eclipse that occured on November 8th, 2022

import matplotlib.pyplot as plt

date = mr.utc(2022, 11, 8)
dates, epsecs = mr.date_arange(
    date, date + mr.days(1), mr.seconds(10), return_epsecs=True
)

# %%
# Computing the Moon position with each method:
moon_pos_spice = mr.moon(dates)
irrad_frac_spice = mr.sun_irradiance_fraction(dates, moon_pos_spice)

moon_pos_vallado = mr.moon(dates, method="vallado")
irrad_frac_vallado = mr.sun_irradiance_fraction(dates, moon_pos_vallado)

# %%
# And plotting the eclipses:
plt.plot(epsecs / 3600, irrad_frac_spice)
plt.plot(epsecs / 3600, irrad_frac_vallado)
old_ylim = plt.ylim()
plt.vlines(10 + 59.5 / 60, *old_ylim, colors="lime")
plt.ylim(*old_ylim)
plt.xlim(0, 24)
plt.legend(["SPICE", "Vallado approx.", "True totality center"])
plt.xlabel("Hours (UTC)")
plt.ylabel("Fraction of Sun visible from Moon center")
plt.title("Nov 08, 2022 Lunar Eclipse")
plt.grid()

plt.show()

# %%
# We can refer to `a NASA article <https://moon.nasa.gov/news/185/what-you-need-to-know-about-the-lunar-eclipse/#:~:text=The%20last%20total%20lunar%20eclipse,Moon%20passes%20into%20Earth%27s%20shadow.>`_ for more info about this eclipse, which proves that the SPICE solution is almost exactly on top of the true center, but the Vallado approximation is only a few minutes different
#
# .. note:: Ironically, the "true" totality time reported by NASA was probably just computed with SPICE in the first place
