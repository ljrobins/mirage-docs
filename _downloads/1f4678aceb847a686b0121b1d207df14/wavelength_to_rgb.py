"""
RGB to Wavelength
=================
An approximate conversion from RGB values to an equivalent wavelength
"""

import numpy as np

import mirage as mr

wavelengths = np.linspace(400, 700, 1000) * 1e-9
rgbs = mr.wavelength_to_rgb(wavelengths).reshape(1, -1, 3)

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 2))
plt.imshow(rgbs, extent=[wavelengths.min(), wavelengths.max(), 0, 50])
plt.yticks([])
plt.xlabel("Wavelength (nm)")
plt.title("Wavelength to RGB")
plt.gca().invert_xaxis()
plt.show()
