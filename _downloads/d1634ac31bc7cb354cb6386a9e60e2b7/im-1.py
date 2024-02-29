"""IM-1
"""

# %%
# Loading the JPL Horizons ephemeris data
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

with open(
    os.path.join(
        os.environ["SRCDIR"],
        "..",
        "examples",
        "000-work-in-progress",
        "horizons_results.txt",
    )
) as f:
    ephemeris = f.read().splitlines()

start_index = ephemeris.index([x for x in ephemeris if "$$SOE" in x][0]) + 1
end_index = ephemeris.index([x for x in ephemeris if "$$EOE" in x][0])

jds = []
pos = []
vel = []

for line in ephemeris[start_index:end_index]:
    if "A.D." in line:
        jds.append(float(line.split()[0]))
    elif "X =" in line:
        floats = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[Ee][-+]?[0-9]+)?", line)
        pos.append([float(x) for x in floats])
    elif "VX=" in line:
        floats = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[Ee][-+]?[0-9]+)?", line)
        vel.append([float(x) for x in floats])

jds = np.array(jds)
vel = np.array(vel)
pos = np.array(pos)

# %%
# Propagating from the first ephemeris data point

initial_state = np.hstack((pos[0], vel[0]))
dates = mr.jd_to_date(jds)

mr.tic("Propagating IM-1")
rv = mr.integrate_orbit_dynamics(
    initial_state, dates, gravity_harmonics_degree=4, third_bodies=["moon", "sun"]
)
mr.toc()

# %%
# Plotting the position error between the propagated and Horizons data
# plt.figure()
# plt.plot(jds, rv[:,:3] - pos)
# mrv.texit("ICRF Position Comparison", "Julian date (UTC)", "(Propagated - Horizons) (km)")
# plt.tight_layout()
# plt.show()


# %%
# Propagating till lunar periapsis
# pl = pv.Plotter()

dates_long = mr.date_arange(dates[0], dates[0] + mr.days(5.5), mr.minutes(15))
mr.tic("Propagating IM-1, long")
rv_long = mr.integrate_orbit_dynamics(
    initial_state, dates_long, gravity_harmonics_degree=4, third_bodies=["moon", "sun"]
)
mr.toc()


pl = pv.Plotter()
mrv.plot_earth(pl, date=dates_long[-1], lighting=False)
mrv.plot_moon(pl, date=dates_long[-1], lighting=False)
mrv.plot3(pl, rv_long[:, :3], color="m", line_width=10)
pl.show()

# %%
# Propagating to now

dates_now = mr.date_linspace(dates[0], mr.now(), 2)
rv_now = mr.integrate_orbit_dynamics(
    initial_state, dates_now, gravity_harmonics_degree=4, third_bodies=["moon", "sun"]
)[-1, :]

# %%
# Determining if IM-1 is visible from POGS
station = mr.Station()
station_j2000 = station.j2000_at_dates(dates_long)
station_to_im1 = mr.hat(rv_long[:, :3] - station_j2000)
elevation = (
    90
    - np.rad2deg(mr.angle_between_vecs(station_to_im1, mr.hat(station_j2000))).flatten()
)

# plt.plot(dates_long, elevation, 'k', label="Elevation")
# mrv.texit("IM-1 Elevation from POGS", "Date (UTC)", "Elevation (deg)")
# plt.fill_between(dates_long, elevation, 0, where=elevation>20, color='g', alpha=0.3, label="Observable")
# plt.fill_between(dates_long, elevation, 0, where=elevation<20, color='r', alpha=0.3, label="Not observable")
# plt.legend(loc='upper right')
# plt.ylim([-45, 90])
# plt.show()

rv_tod = mr.EarthFixedFrame("j2000", "tod").vecs_at_dates(dates_now[-1], rv_now[:3])
ra_tod_now, dec_tod_now = mr.eci_to_ra_dec(rv_tod[:3])

dates_now_plus = mr.date_linspace(dates[0], mr.now() + mr.seconds(1), 2)
rv_now_plus = mr.integrate_orbit_dynamics(
    initial_state,
    dates_now_plus,
    gravity_harmonics_degree=4,
    third_bodies=["moon", "sun"],
)[-1, :]
rv_tod_plus = mr.EarthFixedFrame("j2000", "tod").vecs_at_dates(
    dates_now_plus[-1], rv_now_plus[:3]
)
ra_tod_now_plus, dec_tod_now_plus = mr.eci_to_ra_dec(rv_tod_plus[:3])
ra_rate = (ra_tod_now_plus - ra_tod_now)[0] * mr.AstroConstants.rad_to_arcsecond
dec_rate = (dec_tod_now_plus - dec_tod_now)[0] * mr.AstroConstants.rad_to_arcsecond

ra_tod_now = np.rad2deg(ra_tod_now[0])
dec_tod_now = np.rad2deg(dec_tod_now[0])
print(f"RA: {ra_tod_now}, Dec: {dec_tod_now}")
print(f"RA rate: {ra_rate}, Dec rate: {dec_rate}")

# %%
# Saving out a file of jds, RA/Dec and rates

jds_long = mr.date_to_jd(dates_long)
dates_long_plus = dates_long + mr.seconds(1)
rv_long_tod = mr.EarthFixedFrame("j2000", "tod").vecs_at_dates(
    dates_long, rv_long[:, :3]
)
ra_tod, dec_tod = mr.eci_to_ra_dec(rv_long_tod[:, :3])
ra_tod_deg = np.rad2deg(ra_tod)
dec_tod_deg = np.rad2deg(dec_tod)
rv_long_plus = mr.integrate_orbit_dynamics(
    initial_state,
    dates_long_plus,
    gravity_harmonics_degree=4,
    third_bodies=["moon", "sun"],
)
rv_long_plus_tod = mr.EarthFixedFrame("j2000", "tod").vecs_at_dates(
    dates_long_plus, rv_long_plus[:, :3]
)
ra_tod_plus, dec_tod_plus = mr.eci_to_ra_dec(rv_long_plus_tod[:, :3])
ra_rate = (ra_tod_plus - ra_tod) * mr.AstroConstants.rad_to_arcsecond
dec_rate = (dec_tod_plus - dec_tod) * mr.AstroConstants.rad_to_arcsecond
# print(dec_rate)
# enddd

with open(
    os.path.join(
        os.environ["SRCDIR"], "..", "examples", "000-work-in-progress", "im-1.txt"
    ),
    "w",
) as f:
    f.write(
        f"# JD (UTC), RA (deg, TOD), Dec (deg, TOD), RA rate (arcsec/s), Dec rate (arcsec/s)\n"
    )
    for i in range(len(dates_long)):
        f.write(
            f"{jds_long[i]}, {ra_tod_deg[i]}, {dec_tod_deg[i]}, {ra_rate[i]}, {dec_rate[i]}\n"
        )
