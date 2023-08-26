"""
Astrometric Spectra
===================

A few useful spectra for simulating CCD measurements
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pyspaceaware as ps
import pyspaceaware.vis as psv

ltest = np.linspace(200, 1000, int(1e4)) * 1e-9
isun = ps.sun_spectrum(ltest) * 1e-10
isuntot = np.trapz(isun, ltest)
tee = ps.atmospheric_transmission(ltest, 0 * ltest + np.pi / 4)
qe = ps.quantum_efficiency(ltest)
zm = ps.proof_zero_mag_stellar_spectrum(ltest)

sns.set_theme(style="whitegrid")
sns.lineplot(x=ltest * 1e9, y=tee, ci=None)
sns.lineplot(x=ltest * 1e9, y=qe, ci=None)
sns.lineplot(x=ltest * 1e9, y=zm, ci=None)
sns.lineplot(x=ltest * 1e9, y=isun, ci=None)
plt.legend(
    title="",
    loc="upper left",
    labels=[
        "Atmospheric transmission [-]",
        "Quantum efficiency [-]",
        "Zero magnitude irradiance [W/cm^2/$\mu$m]",
        "Solar irradiance [W/cm^2/$\mu$m]",
    ],
)
psv.texit("", "Wavelength [nm]", "")
plt.show()
