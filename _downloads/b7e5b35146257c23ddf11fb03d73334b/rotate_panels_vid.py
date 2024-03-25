"""
Rotating Solar Panels
=====================

Running LightCurveEngine with rotating solar panels.
"""

import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import mirage as mr

obj = mr.SpaceObject("matlib_tdrs.obj")

t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
svb = mr.hat(
    np.array(
        [np.cos(t) - np.sin(2 * t), np.sin(t) + np.cos(t), np.sin(t) + 2 + 0 * t]
    ).T
)
ovb = mr.hat(np.array([1 + 0 * t, 1 + 0 * t, 3 + 0 * t]).T)
brdf = mr.Brdf("phong", cd=0.5, cs=0.5, n=5)

lc = mr.run_light_curve_engine(
    brdf, obj, svb, ovb, save_imgs=True, rotate_panels=True, instances=1
)

# %%
# Saves a gif of the output images, restricting the image to the red channel to get brightness without the masks


fig, ax = plt.subplots(facecolor="k", figsize=(5, 5))


imgs = []
for i in range(t.size):
    impath = os.path.join("out", f"frame{i+1}.png")
    with Image.open(impath) as img:
        imgs.append(mr.rgba_to_f32(np.array(img)))
imgs = np.nan_to_num(np.array(imgs))

actor = ax.imshow(np.ones((10, 10)), cmap="gray", vmin=0, vmax=np.max(imgs))
plt.xticks([])
plt.yticks([])


def animate(i):
    actor.set_data(imgs[i])
    plt.tight_layout()
    return (actor,)


ani = animation.FuncAnimation(fig, animate, repeat=True, frames=t.size, interval=50)

# To save the animation using Pillow as a gif
writer = animation.PillowWriter(fps=15, bitrate=1800)

plt.show()
