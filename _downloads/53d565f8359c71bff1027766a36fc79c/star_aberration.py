"""
Star Aberration
===============
Computing and plotting the daily and yearly aberration of the stars due to Earth's motion through space
"""

import os

import numpy as np

import mirage as mr
import mirage.vis as mrv

mr.save_file_from_url(
    "https://github.com/liamrobinson1/pyspaceaware-resources/raw/main/tycho2.json",
    os.environ["DATADIR"],
)

station = mr.Station(preset="pogs")
t2 = mr.load_json_data("tycho2.json")
alpha_rad, delta_rad = t2["j2000_ra"], t2["j2000_dec"]
mr.tic()
alpha_rad_app, delta_rad_app = mr.apparent_star_positons(
    mr.now(), station.lat_geod_rad, alpha_rad, delta_rad
)
mr.toc()

# %%
# Plotting the resulting right ascensions and declinations
import matplotlib.pyplot as plt

plt.scatter(np.rad2deg(alpha_rad), np.rad2deg(delta_rad), s=2)
plt.scatter(np.rad2deg(alpha_rad_app), np.rad2deg(delta_rad_app), s=2)
plt.xlim(0, 1)
plt.ylim(0, 1)
mrv.texit(
    "Apparent and True Star Positions",
    "Right ascension [deg]",
    "Declination [deg]",
    ["True", "Apparent"],
)
plt.show()
