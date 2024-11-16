"""
Digital Twin Animated
=====================

An animated gif of the real and simulated images taken by POGS
"""

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr

# info_path = "/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/2022-09-18_GPS_PRN14/ObservationData.mat"
# info_path = '/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/2023-05-29 Telstar 19V/ObservationData.mat'
info_path = '/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/pogs/2023-07-30/ObservationData.mat'

add_distortion = True
add_refraction = True
station = mr.Station(preset='pogs')
station.telescope.gain = 0.25
station.telescope.fwhm = 3.0
mr.tic('Loading star catalog')
catalog = mr.GaiaStarCatalog(station, mr.now() - mr.years(1))
mr.toc()

fig = plt.figure()
plt.subplot(1, 2, 1)
im_obs = plt.imshow(np.eye(4096), cmap='gray')
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
plt.title('Observed')
plt.subplot(1, 2, 2)
im_synth = plt.imshow(np.eye(4096), cmap='gray')
plt.title('Synthetic')
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
plt.tight_layout()

i = 100
res = mr.generate_matched_image(
    info_path,
    i,
    station,
    catalog,
    add_distortion,
    add_refraction,
    bias_variance=175,
    noise=True,
)
img_synth = np.log10(np.clip(res['img_sym'], 1, np.inf))
img = np.log10(np.clip(res['img'] - int(1e3), 1, np.inf))

plt.subplot(1, 2, 1)
im_obs.set_data(img)
plt.clim(img.min(), img.max())
plt.subplot(1, 2, 2)
im_synth.set_data(img_synth)
plt.clim(img.min(), img.max())
plt.show()
