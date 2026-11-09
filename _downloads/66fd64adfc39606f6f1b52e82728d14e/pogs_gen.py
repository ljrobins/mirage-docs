"""
Synthetic Image Results
=======================

Comparison plots for the synthetic and real images
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.photo as mrp

np.random.seed(2)
info_path = '/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/pogs/2022/2022-09-18/ObservationData.mat'
add_distortion = True
add_refraction = True
station = mr.Station()
station.telescope.fwhm = 3.5
station.telescope.gain = 5.1
mr.tic('Loading star catalog')
catalog = mr.GaiaSpectralStarCatalog(station, mr.now())
mr.toc()

res = mr.generate_matched_image(
    info_path,
    200,
    station,
    catalog,
    add_distortion,
    add_refraction,
    noise=True,
    obj_total_brightness=1.818e4,
)

print(res.keys())
print(res['fits_info'])

n = SimpleNamespace(**res)

plt.figure()
plt.plot(n.counts_syn, n.fit_counts_obs_of_syn(n.counts_syn), c='r', markersize=7)
plt.scatter(n.counts_syn, n.counts_obs, s=5)
plt.xlabel('Synthetic counts')
plt.ylabel('Observed counts')
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.legend(['Best linear fit', 'Data'])
plt.title('Observed vs Synthetic Counts (POGS)')


plt.figure()
plt.scatter(n.err_updated[:, 0], n.err_updated[:, 1], s=5)
plt.yscale('symlog')
plt.xscale('symlog')
t = np.linspace(0, 2 * np.pi + 0.1, 1000)
plt.plot(5 * np.cos(t), 5 * np.sin(t), c='k')
plt.plot(1 * np.cos(t), 1 * np.sin(t), c='r')
plt.legend(
    ['Centroid errors', '5-pixel boundary', '1-pixel boundary'], loc='upper right'
)
plt.ylim(-100, 100)
plt.xlim(-100, 100)
plt.xlabel('$x$ pixel error')
plt.ylabel('$y$ pixel error')
plt.title('Centroiding Errors After Image Alignment')
plt.grid()
plt.show()


# %%
# Overlaying the two images

# mrp.plot_fits(n.img_syn)
# plt.show()

# Background subtraction
n.img, br_map_real, br_mask_real, br_std_img = mrp.background_subtract(
    n.img,
    reduction_factor=1,
    sigma=3.0,
    box_size=100,
)
n.img_syn, br_map_synth, br_mask_synth, br_std_synth = mrp.background_subtract(
    n.img_syn,
    reduction_factor=1,
    sigma=3.0,
    box_size=100,
)

print(f'Standard deviation of the background (observed): {br_std_img:.2f}')
print(f'Standard deviation of the background (synthetic): {br_std_synth:.2f}')

plt.figure()
mrp.plot_fits(n.img_syn)

plt.figure()
mrp.plot_fits(n.img)
plt.show()

np.savez(
    '/Users/liamrobinson/Documents/maintained-research/mirage/examples/14-digital-twin-2024/data/pogs.nogit.npz',
    img=n.img,
    img_syn=n.img_syn,
    br_std_img=br_std_img,
    br_std_synth=br_std_synth,
)
