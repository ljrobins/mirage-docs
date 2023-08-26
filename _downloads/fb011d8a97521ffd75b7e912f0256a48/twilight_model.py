"""
Twilight Model
==============

Plotting the zenith surface brightness of the ESO-Parnal twilight model
"""

import pyspaceaware as ps
import pyspaceaware.vis as psv
import matplotlib.pyplot as plt
import numpy as np

gammas = np.linspace(90, 130, 100)

sqm = ps.twilight_zenith_sqm(gammas)

plt.plot(gammas, sqm)
plt.gca().invert_yaxis()
plt.grid()
psv.texit('Twilight Model Brightness', 'Solar zenith angle [deg]', 'Surface brightness in $\\left[ \\frac{mag}{arcsecond^2} \\right]$')
plt.show()