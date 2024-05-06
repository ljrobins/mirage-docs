"""
Synthetic Image Results
=======================

Comparison plots for the synthetic and real images
"""

from types import SimpleNamespace
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
from digitaltwin import generate_matched_image


def star_expected_adu(gmag: float, sint: Callable, integration_time_s: float):
    # note that sint has units of ADU / (W / m^2 * s)
    irrad = mr.apparent_magnitude_to_irradiance(gmag)
    return sint * irrad * integration_time_s


info_path = "/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/2022-09-18_GPS_PRN14/ObservationData.mat"
add_distortion = True
add_refraction = True
limiting_magnitude = 15.0
station = mr.Station()
station.telescope.fwhm = 2.5
mr.tic("Loading star catalog")
catalog = mr.StarCatalog("gaia", station, mr.now(), aberration=False)
mr.toc()

yaoe = 1000, 800
xaoe = 250, 490

res = generate_matched_image(info_path, 200, station, catalog, add_distortion, add_refraction, limiting_magnitude)
data_mat = res["data_mat"]
sint_synth = mr.sint(station, np.pi / 2 - data_mat["el_rad_true"])

n = SimpleNamespace(**res)

in_aoe = (n.expected_stars_corrected[:,0] > xaoe[0]) & (n.expected_stars_corrected[:,0] < xaoe[1]) & (n.expected_stars_corrected[:,1] > yaoe[1]) & (n.expected_stars_corrected[:,1] < yaoe[0])
star_aoe = n.expected_stars_corrected[in_aoe,:]
star_aoe = star_aoe[np.argsort(star_aoe[:,-2]), :]
gmag_aoe = star_aoe[0,-2]
irrad_aoe = mr.apparent_magnitude_to_irradiance(gmag_aoe)
print(n.fit_adu_of_irrad(irrad_aoe))
print(np.min(n.img))

img_sym_prepared = np.log10(n.img_sym)

plt.figure()
plt.scatter(n.matched_irrad, n.matched_adu, s=5)
plt.plot(n.matched_irrad, n.fit_adu_of_irrad(n.matched_irrad), c="r", markersize=7)
plt.xlabel("Irradiance [W/m^2]")
plt.ylabel("ADU")
plt.grid()
plt.xscale("log")
plt.yscale("log")
plt.legend(["Data", "Best linear fit"])

# %%
# Overlaying the two images

n.img = n.img.astype(int)
n.img[n.img <= 999] = 1000
n.img -= int(999)

# n.img_sym = n.img_sym.astype(float)
# n.img = n.img.astype(float)
# n.img_sym -= mr.image_background_parabola(n.img_sym)
# n.img -= mr.image_background_parabola(n.img)

print(n.img_sym[yaoe[1]:yaoe[0], xaoe[0]:xaoe[1]].sum())
print(n.img[yaoe[1]:yaoe[0], xaoe[0]:xaoe[1]].sum())
# endd

img_prepared = np.log10(n.img)

plt.figure()
plt.scatter(n.err_updated[:, 0], n.err_updated[:, 1], s=5)
plt.yscale("symlog")
plt.xscale("symlog")
t = np.linspace(0, 2 * np.pi + 0.1, 1000)
plt.plot(5 * np.cos(t), 5 * np.sin(t), c="k")
plt.plot(1 * np.cos(t), 1 * np.sin(t), c="r")
plt.legend(
    ["Centroid errors", "5-pixel boundary", "1-pixel boundary"], loc="upper right"
)
plt.ylim(-100, 100)
plt.xlim(-100, 100)
plt.xlabel("$x$ pixel error")
plt.ylabel("$y$ pixel error")
plt.grid()


plt.figure()
plt.imshow(img_prepared, cmap="gray")
plt.imshow(img_sym_prepared, cmap="gray_r", alpha=0.5)
plt.scatter(
    n.expected_stars_corrected[:, 0],
    n.expected_stars_corrected[:, 1],
    c="y",
    marker="+",
    s=20,
    label="Expected centroids",
)
plt.scatter(
    n.stars_found[:, 0],
    n.stars_found[:, 1],
    c="m",
    marker="o",
    s=10,
    label="Observed centroids",
)

clim_obs = [np.max(img_prepared), np.min(img_prepared)]
clim_sym = [np.max(img_sym_prepared), np.min(img_sym_prepared)]
plt.figure()
plt.imshow(img_prepared, cmap="gray")
plt.clim(np.min(img_sym_prepared), np.max(img_sym_prepared))
plt.colorbar(label=r"$\log_{10}\left(\text{ADU}\right)$")
plt.title("Observed")

plt.figure()
plt.imshow(img_sym_prepared, cmap="gray")
plt.colorbar(label=r"$\log_{10}\left(\text{ADU}\right)$")
plt.title("Synthetic")

# %%
# Subtracting the two images
adu_err = n.img_sym.astype(np.int64) - n.img.astype(np.int64)
adu_err_stdev = np.abs(adu_err) / np.sqrt(np.abs(n.img.astype(np.int64)))
plt.figure()
plt.imshow(adu_err_stdev, cmap="plasma")
plt.clim(0, 6)
plt.colorbar(label='ADU error standard deviations')
plt.xlim(*xaoe)
plt.ylim(*yaoe)
plt.show()