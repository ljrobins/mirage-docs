"""
Astronomical Spectra
====================

A few useful spectra for simulating CCD measurements
"""

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv

ltest = np.linspace(370, 1050, int(1e4))

tee = mr.atmospheric_transmission(ltest, 0, 0)
ccd = mr.ChargeCoupledDevice(preset='pogs')
qe = ccd.quantum_efficiency(ltest)
zm = mr.proof_zero_mag_stellar_spectrum(ltest)
sm = mr.sun_spectrum(ltest)


spectra_cols = ['C0', 'C1', 'C2', 'C3']

labels = [
    ('Atmospheric transmission', 'nondim'),
    ('Solar spectrum', r'$\left[ \frac{W}{m^2 \cdot nm} \right]$'),
    ('Quantum efficiency', r'$\left[ \frac{W}{m^2 \cdot nm} \right]$'),
    ('Zero magnitude spectrum', r'$\left[ \frac{e^{-}}{\text{photon}} \right]$'),
]

plt.figure(figsize=(8, 6))
for i, (col, y) in enumerate(zip(spectra_cols, [tee, sm, qe, zm])):
    plt.subplot(2, 2, i + 1)
    plt.plot(ltest, y, color=col)
    plt.fill_between(ltest, y, color=col, alpha=0.1, label='_nolegend_')
    mrv.texit(labels[i][0], 'Wavelength [nm]', labels[i][1])

plt.tight_layout()
plt.show()
