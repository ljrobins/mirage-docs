"""
Atmospheric Transmission
========================

Transmission spectra using the LOWTRAN atmosphere model
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr

# %%
# Variation due to zenith angle

observer_altitude_km = 2.206
lambdas = np.linspace(200, 1200, 200).astype(np.float32)
zenith_angles = np.linspace(0, np.pi / 2.3, 5)
trans_interp = mr.individual_atmospheric_transmission(
    lambdas, zenith_angles, observer_altitude_km
)

plt.figure()
plt.plot(lambdas, trans_interp.T)

plt.xlabel('Wavelength [nm]')
plt.ylabel('Transmission (unitless)')
plt.title('Atmospheric Transmission')
plt.grid()
plt.ylim(0, 1)
plt.legend([f'{x:.1f} deg' for x in np.rad2deg(zenith_angles)])
plt.show()

# %%
# Variation due to observer altitude
zenith_angle = 0

plt.figure()

for h in np.linspace(0, 5, 5):
    trans_interp = mr.individual_atmospheric_transmission(lambdas, zenith_angle, h)
    plt.plot(lambdas, trans_interp.T, label=f'{h:.1f} km')

plt.xlabel('Wavelength [nm]')
plt.ylabel('Transmission (unitless)')
plt.title('Atmospheric Transmission')
plt.grid()
plt.ylim(0, 1)
plt.legend()

plt.show()

# %%
# Now let's interpolate the spectrum as a function of altitude and zenith angle

trans_grid = []
zenith_angles = np.linspace(0, np.pi / 2, 20).astype(np.float32)
altitudes = np.linspace(0, 3, 12).astype(np.float32)
for h in altitudes:
    trans_grid.append(mr.individual_atmospheric_transmission(lambdas, zenith_angles, h))
trans_grid = np.array(trans_grid).astype(np.float32)

# %%
# Let's save this grid so it can be used to interpolate transmission values elsewhere without calling LOWTRAN

np.savez(
    os.path.join(os.environ['DATADIR'], 'atmos_trans.npz'),
    altitudes=altitudes,
    zenith_angles=zenith_angles,
    lambdas=lambdas,
    trans_grid=trans_grid,
)

gl, gz = np.meshgrid(lambdas, zenith_angles)
mr.tic()
t = mr.atmospheric_transmission(gl, gz, 0)  # using this npz file
mr.toc()
