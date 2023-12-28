"""
Saving a GIF
============

Saving a set of FITS images taken by POGS as a gif
"""

import os

import astropy.io.fits as fits
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import autopogs as ap

anim = None
def make_gif_from_imgs(ims: list[str]) -> None:
    global anim
    os.chdir(os.path.join(os.environ['TELE_SRC_DIR'], '..'))

    fig = plt.figure()
    im = plt.imshow(np.eye(3), cmap='gray', clim=(0, 1))
    open_imgs = [fits.open(os.path.join('imgs', im)) for im in ims]

    def update(i):
        data = np.log10(open_imgs[i][0].data)
        im.set_data(data)
        plt.clim(np.percentile(data, (5, 95)))
        plt.title(ims[i])
        open_imgs[i].close()
        return im

    anim = animation.FuncAnimation(fig, update, frames=len(ims), interval=100)
    plt.show()

# %%
# The shutter gets in the way of these observations
make_gif_from_imgs([f'00161{i}.48859.fit' for i in range(198, 208)])

# %%
# Three ASTRA satellites are visible in this gif
make_gif_from_imgs([f'00161{i}.40733.fit' for i in range(178, 188)])

# %%
# A planet goes through the frame in these images
make_gif_from_imgs([f'00161{i}.26853.fit' for i in range(158, 162)])
