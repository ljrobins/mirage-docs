"""
Airmass Function Comparison
===========================
Comparing the approximate and true airmass functions
"""
import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv

theta = np.linspace(0, np.pi / 2, int(1e3))

plt.plot(theta, mr.relative_airmass(theta, approx=True))
plt.plot(theta, mr.relative_airmass(theta, approx=False))
plt.yscale("log")
mrv.texit(
    "Airmass Functions", "Zenith Angle [rad]", "Airmass", [r"$\sec\theta$", "Pickering"]
)
plt.grid()
plt.show()
