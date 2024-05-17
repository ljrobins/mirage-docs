"""
Digital Twin Animated
=====================

An animated gif of the real and simulated images taken by POGS
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import mirage as mr
from digitaltwin import generate_matched_image

# info_path = "/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/2022-09-18_GPS_PRN14/ObservationData.mat"
# info_path = '/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/2023-05-29 Telstar 19V/ObservationData.mat'
info_path = "/Users/liamrobinson/Documents/mirage/digitaltwin/data/ObservationData.mat"
add_distortion = True
add_refraction = True
limiting_magnitude = 14.0
station = mr.Station()
station.telescope.fwhm = 3.0
mr.tic("Loading star catalog")
catalog = mr.StarCatalog("gaia", station, mr.now(), aberration=False)
mr.toc()

fig = plt.figure()
plt.subplot(1, 2, 1)
im_obs = plt.imshow(np.eye(4096), cmap="gray")
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
plt.title("Observed")
plt.subplot(1, 2, 2)
im_synth = plt.imshow(np.eye(4096), cmap="gray")
plt.title("Synthetic")
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
plt.tight_layout()


def animate(i):
    res = generate_matched_image(
        info_path,
        i,
        station,
        catalog,
        add_distortion,
        add_refraction,
        limiting_magnitude,
    )
    br_variance = 150
    img_synth = np.log10(np.clip(np.random.poisson(res['img_sym'] + br_variance) - br_variance, 1, np.inf))

    img = np.log10(np.clip(res["img"] - int(1e3), 1, np.inf))
    plt.subplot(1, 2, 1)
    im_obs.set_data(img)
    plt.clim(img.min(), img.max())
    plt.subplot(1, 2, 2)
    im_synth.set_data(img_synth)
    plt.clim(img.min(), img.max())
    return im_obs, im_synth

frames = 50
fps = 5
anim = FuncAnimation(fig, animate, frames=frames, interval=1000 / fps, blit=True)
anim.save("synth_imgs.gif")
