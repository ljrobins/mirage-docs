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

ltest = np.linspace(100, 2600, int(1e4)) * 1e-9

tee = mr.atmospheric_transmission(ltest, 0 * ltest + np.pi / 4)
ccd = mr.ChargeCoupledDevice(preset="pogs")
qe = ccd.quantum_efficiency(ltest)
zm = mr.proof_zero_mag_stellar_spectrum(ltest)
sm = mr.sun_spectrum(ltest) * 1e-9

spectra_cols = ["C0", "C1", "C2", "C3"]

for i, (col, y) in enumerate(zip(spectra_cols, [tee, sm, qe, zm])):
    plt.plot(ltest * 1e9, y, color=col)
    plt.fill_between(ltest * 1e9, y, color=col, alpha=0.1, label="_nolegend_")

mrv.texit("", "Wavelength [nm]", "")
plt.legend(
    loc="upper right",
    labels=[
        "Atmospheric transmission [-]",
        r"Solar spectrum $\left[ \frac{W}{cm^2 \cdot \mu m} \right]$",
        "Quantum efficiency [-]",
        r"Zero magnitude irradiance $\left[ \frac{W}{cm^2 \cdot \mu m} \right]$",
    ],
)
plt.tight_layout()
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
