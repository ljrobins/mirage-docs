"""
Synthetic Image Results
=======================

Comparison plots for the synthetic and real images
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cm

import mirage as mr
import mirage.vis as mrv

info_path = '/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/pogs/2022-09-18/ObservationData.mat'
add_distortion = True
add_refraction = True
station = mr.Station()
station.telescope.fwhm = 3
station.telescope.gain = 1 / 4
mr.tic('Loading star catalog')
catalog = mr.GaiaSpectralStarCatalog(station, mr.now())
mr.toc()

yaoe1 = 1000, 800
xaoe1 = 250, 490
yaoe2 = 750, 550
xaoe2 = 250, 490

res = mr.generate_matched_image(
    info_path,
    200,
    station,
    catalog,
    add_distortion,
    add_refraction,
    bias_variance=175,
)

n = SimpleNamespace(**res)

img_sym_prepared = np.log10(np.clip(n.img_sym, 1, np.inf))

plt.figure()
plt.plot(n.counts_sym, n.fit_counts_obs_of_sym(n.counts_sym), c='r', markersize=7)
plt.scatter(n.counts_sym, n.counts_obs, s=5)
plt.xlabel('Synthetic counts')
plt.ylabel('Observed counts')
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.legend(['Best linear fit', 'Data'])
plt.show()

# endd

# %%
# Overlaying the two images

br_val = 1010  # For the gps obs
n.img = n.img.astype(int)
n.img[n.img <= br_val] = br_val + 1
n.img -= br_val

img_prepared = np.log10(n.img)

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
plt.grid()

img_prepared_sub = img_prepared
img_sym_prepared_sub = img_sym_prepared

clim_obs = [np.max(img_prepared_sub), 0]
clim_sym = [np.max(img_sym_prepared_sub), 0]
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_prepared_sub, cmap='gray')
plt.title('Observed')
plt.clim(*clim_sym)
plt.colorbar(label=r'$\log_{10}\left(\text{ADU}\right)$', cax=mrv.get_cbar_ax())

plt.subplot(1, 2, 2)
plt.imshow(img_sym_prepared_sub, cmap='gray')
plt.title('Synthetic')
plt.clim(*clim_sym)
plt.colorbar(label=r'$\log_{10}\left(\text{ADU}\right)$', cax=mrv.get_cbar_ax())
plt.tight_layout()

# %%
# Subtracting the two images
adu_err = n.img_sym.astype(np.int64) - n.img.astype(np.int64)
adu_err_stdev = np.abs(adu_err) / np.sqrt(np.abs(n.img_sym.astype(np.int64)))
plt.figure(figsize=(8, 6))
cm = cm.get_cmap('plasma')
max_sigma = 30

plt.subplot(2, 2, 1)
plt.imshow(adu_err_stdev, cmap=cm)
plt.clim(0, max_sigma)
plt.xlim(*xaoe2)
plt.ylim(*yaoe2)
plt.xlabel('x [pix]')
plt.ylabel('y [pix]')
# plt.colorbar(label="ADU error standard deviations", cax=mrv.get_cbar_ax())

plt.subplot(2, 2, 3)
data = np.ceil(adu_err_stdev[yaoe2[1] : yaoe2[0], xaoe2[0] : xaoe2[1]].flatten())
n, bins, patches = plt.hist(data, bins=range(max_sigma), density=True)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

mrv.texit('', 'Error $\sigma$', 'Density')

plt.subplot(2, 2, 2)
plt.imshow(adu_err_stdev, cmap=cm)
plt.clim(0, max_sigma)
plt.xlim(*xaoe1)
plt.ylim(*yaoe1)
plt.xlabel('x [pix]')
plt.ylabel('y [pix]')
# plt.colorbar(label="ADU error standard deviations", cax=mrv.get_cbar_ax())

plt.subplot(2, 2, 4)
data = adu_err_stdev[yaoe1[1] : yaoe1[0], xaoe1[0] : xaoe1[1]].flatten()
n, bins, patches = plt.hist(data, bins=range(max_sigma), density=True)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

mrv.texit('', 'Error $\sigma$', 'Density')

plt.tight_layout()
plt.show()
