"""
Space Weather
=============

Plotting the :math:`K_p`, :math:`A_p`, and F10.7 space weather indices
"""


import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd

import mirage as mr
import mirage.vis as mrv

# %%
# Loading the space weather file and extracting the dates and indices
sw_file_path = os.path.join(os.environ["DATADIR"], "SW-Last5Years.csv")
sw_df = pd.read_csv(sw_file_path, header=0)
dates = [datetime.datetime.strptime(x, "%Y-%m-%d") for x in sw_df["DATE"]]
f107_1au = sw_df["F10.7_ADJ"]
f107_obs = sw_df["F10.7_OBS"]
ap = sw_df["AP_AVG"]
kp = sw_df["KP_SUM"]

# %%
# Plotting F10.7 radio flux
plt.scatter(dates, f107_1au, s=1)
plt.scatter(dates, f107_obs, s=1)
mrv.texit("F10.7 Radio Flux", "Date", "F10.7", ["1 AU", "Observed"])
plt.show()

# %%
# Plotting the geomagnetic index :math:`K_p`
plt.scatter(dates, ap, s=1)
mrv.texit("Equivalent Amplitude $A_p$", "Date", "$A_p$")
plt.show()

# %%
# Plotting the geomagnetic index :math:`K_p`
plt.scatter(dates, kp, s=1)
mrv.texit("Range Index $K_p$", "Date", "$K_p$")
plt.show()

# %%
# Reference for the CSV format found `at CelesTrak <https://celestrak.org/SpaceData/SpaceWx-format.php>`_
