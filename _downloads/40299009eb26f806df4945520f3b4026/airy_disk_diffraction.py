"""
Airy Disk Diffraction
=====================
Plotting the Airy disk diffraction pattern
"""

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv

# Parameters
telescope = mr.Telescope(preset='pogs')
telescope.sensor_pixels = np.array([1000, 750])
telescope.pixel_scale = 0.05 / 20

obj_pos = (
    telescope.sensor_pixels[0] // 2 + np.random.rand(),
    telescope.sensor_pixels[1] // 2 + np.random.rand(),
)
x_pix, y_pix = np.meshgrid(
    np.arange(telescope.sensor_pixels[0]), np.arange(telescope.sensor_pixels[1])
)

r_dist = np.sqrt((x_pix - obj_pos[0]) ** 2 + (y_pix - obj_pos[1]) ** 2)
theta_grid_rad = mr.dms_to_rad(0, 0, r_dist * telescope.pixel_scale)

mr.tic()
total_counts = 1
pattern = telescope.airy_disk_pattern(total_counts, theta_grid_rad, 550)
mr.toc()


# Visualize the Airy disk on the CCD grid
plt.imshow(
    np.log10(pattern),
    cmap='gray',
    extent=[x_pix.min(), x_pix.max(), y_pix.min(), y_pix.max()],
)
plt.title('Airy Disk on CCD Grid')
plt.xlabel('Pixels (x)')
plt.ylabel('Pixels (y)')
plt.colorbar(label='Logarithmic Intensity', cax=mrv.get_cbar_ax())
plt.clim(-8, np.max(np.log10(pattern)))
plt.show()

# %%
# Plotting the same pattern at a range of wavelengths

wavelengths = np.linspace(400, 700, 10)
plt.figure(figsize=(6, 6))
pattern = np.zeros((pattern.shape[0], pattern.shape[1], 3))
for wavelength in wavelengths:
    rgb_at_wavelength = mrv.wavelength_to_rgb(wavelength)
    wavelength_pattern = telescope.airy_disk_pattern(1e5, theta_grid_rad, wavelength)
    pattern += (
        wavelength_pattern[..., np.newaxis]
        * rgb_at_wavelength.reshape(1, 1, 3).astype(float)
        / wavelengths.size
    )

plt.imshow(
    np.log10(pattern) / np.max(np.log10(pattern)),
    extent=[x_pix.min(), x_pix.max(), y_pix.min(), y_pix.max()],
)
mrv.texit(
    'CCD Diffraction by Wavelength - Log Brightness',
    'Pixels (x)',
    'Pixels (y)',
    grid=False,
)
plt.show()

# %%
# Plotting the same diffraction pattern at a range of wavelengths, weighted by the CCD quantum efficiency and Sun irradiance spectrum

plt.figure(figsize=(6, 6))
pattern = np.zeros((pattern.shape[0], pattern.shape[1], 3))
for wavelength in wavelengths:
    rgb_at_wavelength = mrv.wavelength_to_rgb(wavelength)
    wavelength_pattern = (
        telescope.ccd.quantum_efficiency(wavelength)
        * telescope.airy_disk_pattern(1, theta_grid_rad, wavelength)
        * mr.sun_spectrum(wavelength)
    )
    pattern += (
        wavelength_pattern[..., np.newaxis]
        * rgb_at_wavelength.reshape(1, 1, 3)
        / wavelengths.size
    )
pattern /= pattern.max()
plt.imshow(pattern, extent=[x_pix.min(), x_pix.max(), y_pix.min(), y_pix.max()])
mrv.texit('CCD Diffraction - Sun Spectrum', 'Pixels (x)', 'Pixels (y)', grid=False)
plt.show()
