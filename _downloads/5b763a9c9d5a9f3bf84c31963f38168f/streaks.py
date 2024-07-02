"""
Streak Detection and Centroiding
================================

Doing image processing to figure out where the streaks are on a FITS image
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import mirage as mr
import mirage.vis as mrv

fits_path = '/Volumes/Data 1/imgs/pogs/misc/00161295.48859.fit'

info = mr.info_from_fits(fits_path)
img = info['ccd_adu']
img_raw = img.copy()
img_log10 = np.log10(img)
img = np.log10(np.clip(img - mr.image_background_parabola(img), 1, np.inf))
total_pix_tracked = info['total_pix_tracked']

img[img < 1] = 0
img[np.isnan(img) | np.isinf(np.abs(img))] = 0

streak_length = 100

# %%
# Demonstrating the dependence of convolved variance on template streak direction

plt.figure(figsize=(15, 5))
thetas = np.linspace(0, np.pi, 30, endpoint=False)
vars = np.zeros_like(thetas)
img_for_gif = img[::10, ::10]


def animate(i):
    global vars
    dir = np.array([np.cos(thetas[i]), np.sin(thetas[i])])
    kernel = mr.streak_convolution_kernel(dir, streak_length)
    conv_img = mr.streak_convolution(img_for_gif, dir, streak_length)
    vars[i] = np.var(conv_img)
    plt.subplot(1, 3, 1)
    plt.gca().cla()
    plt.imshow(conv_img)
    mrv.texit('Convolved Image', '', '', grid=False)
    plt.subplot(1, 3, 2)
    plt.gca().cla()
    plt.imshow(kernel)
    mrv.texit(rf'Streak Kernel $\theta={thetas[i]:2.2f}$ [rad]', '', '', grid=False)
    plt.subplot(1, 3, 3)
    plt.gca().cla()
    plt.plot(thetas[: i + 1], vars[: i + 1])
    plt.xlim(0, np.pi)
    plt.ylim(0, 0.025)
    mrv.texit(
        'Convolved Image Variance', 'Streak angle [rad]', 'Variance [ndim]', grid=True
    )
    plt.pause(0.01)
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, repeat=True, frames=thetas.size, interval=50)
writer = PillowWriter(fps=10, bitrate=1800)
writer.setup(ani, 'streaks.gif', dpi=200)
ani.save('streaks.gif', writer=writer)

plt.show()

# %%
# Find the centroids
stars = mr.solve_star_centroids(info)

print(f'Found {len(stars)} streaks on the first pass')


# %%
# Rotating back into the original frame

plt.imshow(img, cmap='gray')
for star in stars:
    plt.plot(star['bbox'][:, 0], star['bbox'][:, 1], color='lime', linewidth=0.2)
plt.scatter(
    [star['centroid'][0] for star in stars],
    [star['centroid'][1] for star in stars],
    c=[star['brightness'] for star in stars],
    cmap='cool',
    s=10,
)
# label the colorbar with adu
plt.colorbar(label='Total Star ADU')
plt.xlim(0, img.shape[1])
plt.ylim(0, img.shape[0])
mrv.texit('True Image with Centroids', '', '', grid=False)

# plt.subplot(1, 2, 2)
# plt.imshow(np.log10(rotated_image_raw + rotated_image_raw.min() + 10), cmap="gist_stern")
# for star in stars:
#     plt.plot(star["rotated_bbox"][:, 0], star["rotated_bbox"][:, 1], color="lime", linewidth=0.2)

# plt.scatter([star["rotated_centroid"][0] for star in stars],
#             [star["rotated_centroid"][1] for star in stars],
#             c=[star["brightness"] for star in stars], cmap="plasma")

plt.show()
