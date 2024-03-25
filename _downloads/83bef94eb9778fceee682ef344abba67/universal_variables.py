"""
Universal Variables
===================
"""

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv

# %%
Psi = np.linspace(-40, 180, 1000)
c2c3 = mr.c2_c3(Psi)

fig, ax = plt.subplots()
ax.plot(Psi, c2c3)
mrv.texit(f"", "$\Psi$ [$rad^2$]", "$c(\Psi)$", legend=["$c_2(\Psi)$", "$c_3(\Psi)$"])
plt.ylim(0, 0.6)
plt.show()
