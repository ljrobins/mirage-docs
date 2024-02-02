"""
Streak Detection and Centroiding
================================

"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import rotate
from tqdm import tqdm

import mirage as mr
import mirage.vis as mrv


def find_stars_in_rotated_image(
    img: np.ndarray,
    rotated_image_raw: np.ndarray,
    rotated_image: np.ndarray,
    theta_rad: float,
    total_pix_tracked: float,
) -> list[dict]:
    mr.tic("contours")
    contours, _ = cv2.findContours(
        rotated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    stars = []

    for i, cnt in enumerate(tqdm(contours, desc="Finding streaks")):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > total_pix_tracked / 1.15 and w < total_pix_tracked * 1.3:
            rotated_bbox = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]
            )
            rotated_centroid = np.array([x + w / 2, y + h / 2])
            unrot_coords = rotate_back(
                rotated_bbox,
                theta_rad,
                rotated_image.shape,
                img.shape,
            )
            unrot_centroid = rotate_back(
                rotated_centroid.reshape(1, 2),
                theta_rad,
                rotated_image.shape,
                img.shape,
            ).squeeze()
            brightness = np.sum(
                rotated_image_raw[
                    int(rotated_bbox[0, 1]) : int(rotated_bbox[2, 1]),
                    int(rotated_bbox[0, 0]) : int(rotated_bbox[2, 0]),
                ]
            )
            stars.append(
                {
                    "centroid": unrot_centroid,
                    "rotated_centroid": rotated_centroid,
                    "bbox": unrot_coords,
                    "rotated_bbox": rotated_bbox,
                    "brightness": brightness,
                }
            )
    mr.toc()
    return stars


def rotate_back(
    coords: np.ndarray,
    theta_rad: float,
    rot_image_shape: tuple,
    orig_image_shape: tuple,
) -> np.ndarray:
    """Rotate coordinates back to original image coordinates

    :param coords: Pixel coordinates in the rotated image
    :type coords: np.ndarray
    :param theta_rad: The streak angle w.r.t. the image horizontal plane in radians
    :type theta_rad: float
    :param rot_image_shape: The shape of the rotated image
    :type rot_image_shape: tuple
    :param orig_image_shape: The shape of the original image
    :type orig_image_shape: tuple
    :return: The rotated coordinates
    :rtype np.ndarray
    """
    coords = coords - np.array(rot_image_shape) / 2
    rotm = np.array(
        [
            [np.cos(theta_rad), np.sin(theta_rad)],
            [-np.sin(theta_rad), np.cos(theta_rad)],
        ]
    ).T
    coords_unrot = np.matmul(rotm, coords.T).T
    coords_unrot = coords_unrot + np.array(orig_image_shape) / 2
    return coords_unrot


def horizontal_line_kernel(length: int) -> np.ndarray:
    """Kernel that detects horizontal lines of a given length

    :param length: The length of lines to detect
    :type length: int
    :return: The kernel array
    :rtype: np.ndarray
    """
    kernel = np.zeros((length, length))
    kernel[length // 2, :] = 1 / length
    nval = -1 / (kernel.size - length)
    kernel[: length // 2, :] = nval
    kernel[length // 2 + 1 :, :] = nval
    return kernel


fits_path = os.path.join(os.environ["SRCDIR"], "..", "00161295.48859.fit")

info = mr.info_from_fits(fits_path)
img = info["ccd_adu"]
img_raw = img.copy()
img_log10 = np.log10(img)
img = np.log10(np.clip(img - mr.image_background_parabola(img), 1, np.inf))
total_pix_tracked = info["total_pix_tracked"]

img[img < 1] = 0
img[np.isnan(img) | np.isinf(np.abs(img))] = 0

streak_length = 100

# %%
# Demonstrating the dependence of convolved variance on template streak direction

plt.figure(figsize=(15, 5))
thetas = np.linspace(0, np.pi, 50)
vars = np.zeros_like(thetas)
img_for_gif = img[::6, ::6]

def animate(i):
    global vars
    dir = np.array([np.cos(thetas[i]), np.sin(thetas[i])])
    kernel = mr.streak_convolution_kernel(dir, streak_length)
    conv_img = mr.streak_convolution(img_for_gif, dir, streak_length)
    vars[i] = np.var(conv_img)
    plt.subplot(1,3,1)
    plt.gca().cla()
    plt.imshow(conv_img)
    mrv.texit(f"Convolved Image", "", "", grid=False)
    plt.subplot(1,3,2)
    plt.gca().cla()
    plt.imshow(kernel)
    mrv.texit(fr"Streak Kernel $\theta={thetas[i]:.2f}$ [rad]", "", "", grid=False)
    plt.subplot(1,3,3)
    plt.gca().cla()
    plt.plot(thetas[:i+1], vars[:i+1])
    plt.xlim(0, np.pi)
    plt.ylim(0, 0.025)
    mrv.texit(f"Convolved Image Variance", "Streak angle [rad]", "Variance [ndim]", grid=True)
    plt.pause(0.01)
    plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate, repeat=True, frames=thetas.size, interval=50)
writer = PillowWriter(fps=15, bitrate=1800)
writer.setup(ani, "streaks.gif", dpi=200)
ani.save("streaks.gif", writer=writer)

plt.show()

# %%
# Doing the actual optimization

# %%
# Convolve at the converged direction
theta_rad = -mr.solve_star_streak_angle(img)
rotated_image = rotate(img, np.rad2deg(theta_rad), reshape=True)
rotated_image_raw = rotate(img_raw, np.rad2deg(theta_rad), reshape=True)

# %%
# Find contours which correspond to the streaks
# First, convolve the image with a horizontal line to make centroiding easier
rotated_image = mr.streak_convolution(rotated_image, [1, 0], 20)
rotated_image = (rotated_image).astype(np.uint8)

stars = find_stars_in_rotated_image(
    img, rotated_image_raw, rotated_image, theta_rad, total_pix_tracked
)
print(f"Found {len(stars)} streaks on the first pass")
# mask out all the stars we found from the rotated_img
# for star in stars:
#     rotated_image[
#         int(star["rotated_bbox"][0, 1]) : int(star["rotated_bbox"][2, 1]),
#         int(star["rotated_bbox"][0, 0]) : int(star["rotated_bbox"][2, 0]),
#     ] = 0
# rotated_image = mr.streak_convolution(rotated_image, [1, 0], 10).astype(np.uint8)
# stars_fine = find_stars_in_rotated_image(img, rotated_image_raw, rotated_image, theta)
# print(f"Found {len(stars_fine)} streaks on the second pass")

# %%
# Rotating back into the original frame

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
for star in stars:
    plt.plot(star["bbox"][:, 0], star["bbox"][:, 1], color="lime", linewidth=0.2)
plt.scatter(
    [star["centroid"][0] for star in stars],
    [star["centroid"][1] for star in stars],
    c=[star["brightness"] for star in stars],
    cmap="cool",
    s=10,
)
# label the colorbar with adu
plt.colorbar(label="Total Star ADU")
plt.xlim(0, img.shape[1])
plt.ylim(0, img.shape[0])
mrv.texit("True Image with Centroids", "", "", grid=False)

# plt.subplot(1, 2, 2)
# plt.imshow(np.log10(rotated_image_raw + rotated_image_raw.min() + 10), cmap="gist_stern")
# for star in stars:
#     plt.plot(star["rotated_bbox"][:, 0], star["rotated_bbox"][:, 1], color="lime", linewidth=0.2)

# plt.scatter([star["rotated_centroid"][0] for star in stars],
#             [star["rotated_centroid"][1] for star in stars],
#             c=[star["brightness"] for star in stars], cmap="plasma")

plt.show()
