# /// script
# requires-python = ">=3.11"
# dependencies = ["mirage"]
#
# [tool.uv.sources]
# mirage = { path = "../../", editable = true }
# ///
"""
Marini-Murray Atmospheric Correction
====================================
"""

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr

P_H = 1000
T_H = 300
R_H = 50
elevs = np.linspace(20, 90, 100)
sc = mr.marini_murray_atmosphere_correction(
    lambda_=0.532, E=np.deg2rad(elevs), phi=0.0, H=0.0, P_H=P_H, T_H=T_H, R_H=R_H
)

plt.figure(figsize=(5, 4))
plt.plot(elevs, sc)
plt.xlabel('Target elevation [deg]')
plt.ylabel('Additional light travel time [m]')
plt.title('Marini-Murray Model')
plt.grid()
plt.xlim(elevs.min(), elevs.max())
plt.ylim(0, 8)
plt.tight_layout()
plt.show()
