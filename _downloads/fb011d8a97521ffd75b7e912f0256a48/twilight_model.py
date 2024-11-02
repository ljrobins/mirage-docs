"""
Twilight Model
==============

Plotting the zenith surface brightness of the ESO-Parnal twilight model
"""

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv

gammas = np.linspace(90, 130, 1000)

mpsas = mr.twilight_zenith_mpsas(gammas)

plt.plot(gammas, mpsas)
plt.gca().invert_yaxis()
mrv.texit(
    'Twilight Model Brightness',
    'Solar zenith angle [deg]',
    'Surface brightness in $\\left[ \\frac{mag}{arcsecond^2} \\right]$',
)
plt.show()
