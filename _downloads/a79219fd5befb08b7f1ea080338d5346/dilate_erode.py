"""
Dilation and Erosion
====================
Demonstrating how the binary image operations dilation and erosion can be used to build smooth background masks when the background is very noisy
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

freq = np.array([0.689, 0.562, 0.683]) / 1.3
noise = pv.perlin_noise(1, freq, (0, 0, 0))
n = 150
noise_power = 1e-1
dim = (n, n)
sampled = pv.sample_function(noise, bounds=(-10, 10, -10, 10, -10, 10), dim=(*dim, 1))
z = sampled.active_scalars.reshape(dim).squeeze()
z += np.random.randn(*dim) * noise_power

open_size = 5
blur_size = 5
n_frames = 10

im = (z < 0).astype(np.uint8)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(im, interpolation='nearest', cmap='gray')
plt.title('1. Naive Mask')
plt.subplot(1, 3, 2)
im = cv2.erode(im, np.ones((open_size, open_size), np.uint8), iterations=1)
im = cv2.dilate(im, np.ones((open_size, open_size), np.uint8), iterations=1)
plt.imshow(im, interpolation='nearest', cmap='gray')
plt.title('2. Erosion & Dilation')
plt.subplot(1, 3, 3)
im = cv2.medianBlur(im, ksize=open_size)
plt.imshow(im, interpolation='nearest', cmap='gray')
plt.title('3. Median filter')
plt.tight_layout()

for ax in plt.gcf().get_axes():
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
