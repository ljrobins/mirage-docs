"""
Astronomical Spectra
====================

A few useful spectra for simulating CCD measurements
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import mirage as mr

ltest = np.linspace(300, 1050, int(1e4))

tee = mr.atmospheric_transmission(ltest, 0, 0)
ccd_pogs = mr.ChargeCoupledDevice(preset='pogs')
qe_pogs = ccd_pogs.quantum_efficiency(ltest)
ccd_zim = mr.ChargeCoupledDevice(preset='zimmain')
qe_zim = ccd_zim.quantum_efficiency(ltest)
zm = mr.proof_zero_mag_stellar_spectrum(ltest)
sm = mr.sun_spectrum(ltest)

ar = mr.airglow_radiance(ltest) * 1e-9


labels = [
    ('Atmospheric Transmission', 'nondim'),
    ('Solar Spectrum', r'$\left[ \frac{W}{m^2 \cdot nm} \right]$'),
    ('Airglow Radiance', r'$\left[ \frac{W}{m^2 \cdot nm \cdot \text{ster}} \right]$'),
    ('POGS Quantum Efficiency', r'nondimensional'),
    ('ZimMAIN Quantum Efficiency', r'nondimensional'),
    ('Zero Magnitude Spectrum', r'$\left[ \frac{W}{m^2 \cdot nm } \right]$'),
]

specs = [tee, sm, ar, qe_pogs, qe_zim, zm]
spectra_cols = [f'C{i}' for i in range(len(specs))]


for i, (col, y) in enumerate(zip(spectra_cols, specs)):
    fig = plt.figure(figsize=(4, 4))
    plt.plot(ltest, y, color=col)
    plt.fill_between(ltest, y, color=col, alpha=0.1, label='_nolegend_')
    plt.title(labels[i][0])
    plt.xlabel('Wavelength [nm]')
    plt.ylabel(labels[i][1])
    plt.xlim(ltest.min(), ltest.max())
    plt.grid()
    plt.ylim(0, plt.ylim()[1])
    plt.tight_layout()
    plt.show()

# %%
# Loading and plotting the Vega (Alpha Lyrae) spectrum

vega_spec_path = os.path.join(os.environ['DATADIR'], 'stars', 'alpha_lyr_mod_004.fits')
if not os.path.exists(vega_spec_path):
    mr.save_file_from_url(
        'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/alpha_lyr_mod_004.fits',
        os.path.join(os.environ['DATADIR'], 'stars'),
    )

with fits.open(vega_spec_path) as hdul:
    # https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/
    data = np.vstack(hdul[1].data)
    lambdas = data[:, 0] / 10
    flux = data[:, 1]
    valid = (lambdas > 300) & (lambdas < 1200)
    flux = (
        flux[valid] / 1e2
    )  # converting from erg / s / cm^2 / Angstrom to W / m^2 / nm
    lambdas = lambdas[valid]
    np.savez(
        os.path.join(os.environ['DATADIR'], 'stars', 'vega_spectrum.npz'),
        lambdas=lambdas,
        flux=flux,
    )

y = mr.vega_spectrum(lambdas)
plt.plot(lambdas, y)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Flux [W / m^2 / nm]')
plt.title('Vega Spectrum')
plt.grid()
plt.xlim(300, 1200)
plt.ylim(0, 1.02 * np.max(y))
plt.tight_layout()
plt.show()
