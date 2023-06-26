"""
Astrometric Spectra
===================

A few useful spectra for simulating CCD measurements
"""

import seaborn as sns
import pyspaceaware as ps
import numpy as np
import matplotlib.pyplot as plt

ltest = np.linspace(200, 1000, int(1e4)) * 1e-9
isun = ps.sun_spectrum(ltest) * 1e-10
isuntot = np.trapz(isun, ltest)
tee = ps.atmospheric_transmission(ltest, 0 * ltest + np.pi / 4)
qe = ps.quantum_efficiency(ltest)
zm = ps.proof_zero_mag_stellar_spectrum(ltest)

sns.set_theme(style="whitegrid")
sns.lineplot(x=ltest, y=isun, ci=None)
sns.lineplot(x=ltest, y=qe, ci=None)
sns.lineplot(x=ltest, y=tee, ci=None)
sns.lineplot(x=ltest, y=zm, ci=None)
plt.legend(
    title="",
    loc="upper left",
    labels=[
        "Sun irradiance [W/cm^2/um]",
        "Quantum efficiency [-]",
        "Atmospheric transmission [-]",
        "Zero magnitude source [-]",
    ],
)
plt.show()
