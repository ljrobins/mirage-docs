"""
POGS Sensitivity
================

Calibrating the gain of the POGS CCD sensor
"""

import matplotlib.pyplot as plt
import numpy as np

import mirage.photo as mrp

fits_path = '/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/pogs/misc/00161292.48859.fit'
header, img = mrp.load_fits(fits_path)
for k, v in header.items():
    print(k, v)

img = img.flatten() - 1000
br = img[img < np.percentile(img, 99.8)]


f = 2.2
median = np.median(br)
print(f'Observed std in e- from the median {np.sqrt(median/f)}')
print(f'Observed std in e- from the std {(br/f).std()}')

vp = np.random.poisson(np.full(br.shape, median / f)) * f
vg = np.random.normal(loc=median, scale=np.full(br.shape, br.std()))

bins = np.arange(br.min(), br.max() + 1.1)
bins2 = np.arange(br.min(), br.max() + 1.1, f)

# %%
# Plotting the distributions in ADU
h1 = plt.hist(br, bins, density=True)[0]
h2 = plt.hist(vp, bins2, density=True)[0]
h3 = plt.hist(vg, bins, density=True)[0]
plt.cla()
plt.step(bins[1:], h1, label=f'Data $\sigma={br.std():.1f}$')
plt.plot(
    bins[1:],
    h3,
    label=f'$N(\mu={median:.1f}, \sigma={br.std():.1f})$',
    alpha=0.9,
    linestyle='--',
)
plt.step(
    bins2[1:], h2, label=f'$Pois(\lambda={median:.1f}/{f:.2f})\cdot {f:.2f}$', alpha=0.5
)
plt.xlabel('Pixel value [ADU]')
plt.ylabel('Probility density')
plt.title('Background Distribution after Bias Subtraction [ADU]')
plt.legend()
plt.grid()
plt.show()

# %%
# # Plotting the distributions in ADU
# h1 = plt.hist(br, bins, density=True)[0]
# h2 = plt.hist(vp, bins2, density=True)[0]
# h3 = plt.hist(vg, bins, density=True)[0]
# plt.cla()
# plt.step(bins[1:], h1, label='data')
# plt.plot(
#     bins[1:],
#     h3,
#     label=f'$N(\mu={median:.1f}, \sigma={br.std():.1f})$',
#     alpha=0.9,
#     linestyle='--',
# )
# plt.step(
#     bins2[1:], h2, label=f'$Pois(\lambda={median:.1f}/{f:.2f})\cdot {f:.2f}$', alpha=0.5
# )
# plt.xlabel('Pixel value [ADU]')
# plt.ylabel('Probility density')
# plt.title('Background Distribution after Bias Subtraction [e-]')
# plt.legend()
# plt.grid()
# plt.show()
