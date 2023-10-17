"""
Atmospheric Transmission
========================
Using Sun spectra to determine the atmospheric transmission spectrum
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv

sun_spectra = np.loadtxt(
    os.path.join(os.environ["DATADIR"], "ASTMG173.csv"), skiprows=2, delimiter=","
)
lambdas_nm = sun_spectra[:, 0]
sun_vacuum_w_m3 = sun_spectra[:, 1] * 1e9
sun_global_w_m3 = sun_spectra[:, 2] * 1e9
sun_direct_w_m3 = sun_spectra[:, 3] * 1e9
ss_proof = mr.sun_spectrum(lambdas_nm * 1e-9)

i0 = np.trapz(sun_vacuum_w_m3, lambdas_nm * 1e-9)
print(i0)
plt.plot(lambdas_nm, sun_vacuum_w_m3)
plt.plot(lambdas_nm, sun_global_w_m3)
plt.plot(lambdas_nm, sun_direct_w_m3)
plt.plot(lambdas_nm, ss_proof)
mrv.texit(
    "Sun Irradiance Spectra",
    "Wavelength [nm]",
    r"$\left[ \frac{W}{m^2 \cdot m} \right]$",
    ["AM0", "AM1.5 Global", "AM1.5 Direct", "PROOF zero airmass"],
)
plt.show()

# %%
# We can divide the direct spectrum by the vacuum spectrum to get the fraction of light at that wavelength transmitted by the atmosphere. Note that since the AM1.5 Direct spectrum is for a relative airmass of 1.5, we have to adjust for the optical path length difference

plt.plot(lambdas_nm, mr.atmospheric_transmission_spectrum(lambdas_nm * 1e-9))
mrv.texit("Atmospheric Absorption Spectrum", "Wavelength [nm]", "Fraction absorbed")
plt.show()

# %%
# Let's define a function that can do all of this stuff for us
