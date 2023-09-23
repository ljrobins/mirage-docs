"""
Synthetic RPO
=============

Generating and displaying synthetic images of a nearby space object
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mirage import synth as mrsyn

# %%
# Generating the dataset with key point selection UI enabled
mrsyn.generate_synthetic_dataset(
    "matlib_goes17.obj",
    data_points=9,
    key_point_count=10,
    manual_key_point_selection=True,
    max_phase_deg=30,
)

# %%
# Loading the resulting data

with open(os.path.join("out", "iminfo.json"), "r") as f:
    iminfo = json.load(f)
    kp_pix = np.array(iminfo["key_points"]["image_pixels"])

# %%
# Plotting a grid of rendered images
fig, axs = plt.subplots(3, 3)
for fnum in range(9):
    plt.subplot(3, 3, fnum + 1)
    with Image.open(os.path.join("out", f"frame{fnum}.png"), "r") as im:
        im_arr = np.array(im)
        plt.imshow(im_arr[:, :, 0], cmap="gray")
        plt.scatter(kp_pix[fnum, :, 0], kp_pix[fnum, :, 1], c="c", s=1)
        plt.axis("off")
fig.suptitle("Synthetic images of GOES-17 with Key Points", fontsize=12)
plt.tight_layout()
plt.show()
