"""
Zodiacal Light
==============
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import mirage as mr  # noqa

x = np.loadtxt(os.path.join(os.environ['DATADIR'], 'proof.zod'))

ecliptic_lats = np.flip(np.unique(x[:, 0]))

ecliptic_lons = np.unique(x[:, 1])
vals = x[:, 2].reshape(ecliptic_lats.size, ecliptic_lons.size, order='f')
f = RegularGridInterpolator((ecliptic_lats, ecliptic_lons), vals)

plt.imshow(np.log10(vals), origin='lower', extent=[0, 180, -90, 90])
plt.show()
