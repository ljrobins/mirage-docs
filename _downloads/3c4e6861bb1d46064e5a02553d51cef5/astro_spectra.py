"""
Astrometrical Spectra
=====================

A few useful spectra for simulating CCD measurements
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import mirage as mr
import mirage.vis as mrv

ltest = np.linspace(200, 1000, int(1e4)) * 1e-9
isun = mr.sun_spectrum(ltest) * 1e-10
isuntot = np.trapz(isun, ltest)
tee = mr.atmospheric_transmission(ltest, 0 * ltest + np.pi / 4)
ccd = mr.ChargeCoupledDevice(preset='pogs')
qe = ccd.quantum_efficiency(ltest)
zm = mr.proof_zero_mag_stellar_spectrum(ltest)

sns.set_theme(style="whitegrid")
sns.lineplot(x=ltest * 1e9, y=tee, ci=None)
sns.lineplot(x=ltest * 1e9, y=qe, ci=None)
sns.lineplot(x=ltest * 1e9, y=zm, ci=None)
sns.lineplot(x=ltest * 1e9, y=isun, ci=None)
plt.legend(
    loc="upper left",
    labels=[
        "Atmospheric transmission [-]",
        "Quantum efficiency [-]",
        "Zero magnitude irradiance [W/cm^2/$\mu$m]",
        "Solar irradiance [W/cm^2/$\mu$m]",
    ],
)
mrv.texit("", "Wavelength [nm]", "")
plt.show()

# %%
# UBVRI Passbands
import os

BANDS = ["u", "b", "v", "r", "i"]
BAND_COLS = ["violet", "c", "g", "r", "maroon"]

for i, (band, col) in enumerate(zip(BANDS, BAND_COLS)):
    band_path = os.path.join(os.environ["DATADIR"], f"bess-{band}.pass")
    band_data = np.loadtxt(
        band_path
    )  # first col angstroms, second col transmission fraction
    x, y = band_data[:, 0] * 1e-1, band_data[:, 1]
    ax = sns.lineplot(x=x, y=y, color=col, alpha=0.5)
    ax.fill_between(x, y, color=col, alpha=0.2)

plt.xlim(200, 1000)
mrv.texit(
    "Bessel (Johnson/Cousins) UBVRI Passbands",
    "Wavelength in [nm]",
    "Transmission fraction",
)
ax.legend(handles=ax.lines, labels=[band.upper() for band in BANDS])
plt.show()
