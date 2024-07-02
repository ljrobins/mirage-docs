"""
Vega and Apparent Magnitudes
============================

Computing the apparent magnitude of a star from its absolute flux spectrum in a given passband
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from gaiaxpy import calibrate

import mirage as mr

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

# %%
# Using precise fluxes for a given reference star
# http://gaiaextra.ssdc.asi.it:8080/SPSSV2extendedfluxtable_user_show.php?sizelimit=150


example_spec_path = os.path.join(os.environ['DATADIR'], 'stars', 'V2.SPSS001.ascii')
if not os.path.exists(example_spec_path):
    mr.save_file_from_url(
        'http://gaiaextra.ssdc.asi.it:8080/reduced/2/SPSSpublic/V2.SPSS001.ascii',
        os.path.join(os.environ['DATADIR'], 'stars'),
    )

star_id = 266077145295627520
lam_true, flux_true, flux_err_true = np.loadtxt(example_spec_path).T
mags = [
    ('V', 11.78),
    ('B', 11.46),
    ('R', 11.93098),
    ('I', 12.11202),
    ('G', 11.72116),
    ('G_BP', 11.54143),
    ('G_RP', 12.06725),
]

plt.figure(figsize=(4, 6))
for band_name, mag_true in mags:
    passband = mr.passband_by_name(lam_true, band_name)
    band_mag_from_true_spec = mr.apparent_magnitude_in_band(
        lam_true, flux_true, band_name
    )
    print(
        f'{band_name} magnitude: {band_mag_from_true_spec:.2f}, error {band_mag_from_true_spec-mag_true:.3f}'
    )

    plt.subplot(2, 1, 1)
    plt.plot(lam_true, passband, label=band_name)
    plt.subplot(2, 1, 2)
    plt.plot(lam_true, flux_true * passband, label=band_name)

plt.subplot(2, 1, 1)
plt.title('G191-B2B Spectra')
plt.ylabel('Irradiance spectrum [W/m^2/nm]')
plt.grid()
plt.subplot(2, 1, 2)
plt.grid()
plt.plot(lam_true, flux_true, label='CALSPEC Raw')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Irradiance spectrum [W/m^2/nm]')

plt.legend()
plt.tight_layout()
plt.show()

# %%
# Now let's use GAIA's calibrated spectra on a different star

star_id = 5937173300407375616
g_mag_true = 12.136271
plt.figure(figsize=(4, 6))
lam_cal = np.linspace(350, 1020, 20)
spec = calibrate(
    [star_id],
    lam_cal,
    username=os.environ['COSMOS_USERNAME'],
    password=os.environ['COSMOS_PASSWORD'],
    save_file=False,
)
flux_cal = spec[0]['flux'][0]

passband = mr.passband_by_name(lam_cal, 'G')
band_mag_from_cal_spec = mr.apparent_magnitude_in_band(lam_cal, flux_cal, 'G')
print(
    f'{band_name} magnitude: {band_mag_from_cal_spec:.2f}, error {band_mag_from_cal_spec-g_mag_true:.3f}'
)

plt.subplot(2, 1, 1)
plt.plot(lam_cal, passband, label=band_name)
plt.subplot(2, 1, 2)
plt.plot(lam_cal, flux_cal * passband, label=band_name)


plt.subplot(2, 1, 1)
plt.title('G191-B2B Spectra')
plt.ylabel('Irradiance spectrum [W/m^2/nm]')
plt.grid()
plt.subplot(2, 1, 2)
plt.grid()
plt.plot(lam_cal, flux_cal, label='GAIA XP Raw')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Irradiance spectrum [W/m^2/nm]')

plt.legend()
plt.tight_layout()
plt.show()
