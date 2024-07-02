"""
Luminous Efficiency Function
============================

Plotting the luminous efficiency function used to convert from candela to watts per steradian
"""

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv

lambdas = np.linspace(350, 800)  # [nm]
kcd = 1 / mr.candela_to_watt_per_sr(np.ones_like(lambdas), lambdas=lambdas)

plt.plot(lambdas, kcd)
mrv.texit(
    'Luminous Efficiency Function $K_{cd}$', 'Wavelength $\lambda$ in [nm]', '$K_{cd}$'
)

plt.show()
