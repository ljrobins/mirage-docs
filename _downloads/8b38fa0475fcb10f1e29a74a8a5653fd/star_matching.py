"""
Star Matching
=============

Given star centroid locations and an initial estimate of the look direction and tracking rate, fit the catalog
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr


def expected_star_centroids_in_fits(
    fits_info: dict,
    catalog: mr.StarCatalog,
    mean_look_dir: np.ndarray,
    limiting_magnitude: float = 15.0,
) -> list[dict]:
    tracking_normal = mr.hat(
        np.cross(fits_info["look_dirs_eci"][0, :], fits_info["look_dirs_eci"][1, :])
    )
    img_prepared = mr.prepare_fits_for_plotting(fits_info)
    theta = mr.solve_star_streak_angle(img_prepared)
    dcm = mr.rv_to_dcm(mean_look_dir * (theta + np.pi / 2))
    up_dir_eci = mr.stack_mat_mult_vec(dcm, tracking_normal)

    uvs_in_frame, vm_in_frame = catalog.in_fov(
        mean_look_dir, up_dir_eci, limiting_magnitude=limiting_magnitude
    )
    star_xs, star_ys = station.telescope.j2000_unit_vectors_to_pixels(
        mean_look_dir, up_dir_eci, uvs_in_frame
    )
    return [
        {"centroid": np.array((x, y)), "brightness": m}
        for x, y, m in zip(star_xs, star_ys, vm_in_frame)
    ]


station = mr.Station()
date = mr.now()
mr.tic("Loading star catalog")
catalog = mr.StarCatalog("gaia", station, date)
mr.toc()

# %%
# Let's figure out the streak direction

fits_path = os.path.join(os.environ["SRCDIR"], "..", "00161295.48859.fit")
fits_info = mr.info_from_fits(fits_path)

img = fits_info["ccd_adu"]
mean_look_dir = mr.hat(
    fits_info["look_dirs_eci"][0, :] + fits_info["look_dirs_eci"][1, :]
)
img_raw = img.copy()
img_log10 = np.log10(img)
img = np.log10(img - mr.image_background_parabola(img))
img[img < 1] = 0
img[np.isnan(img) | np.isinf(np.abs(img))] = 0

theta_rad = -mr.solve_star_streak_angle(img)
# print(f"Streak angle: {np.rad2deg(theta_rad)} degrees")
# enddd

up_dir_eci = mr.fits_up_direction(fits_info)

station.telescope.fwhm = 4
mr.tic()
adu_grid_streaked_sampled = station.telescope.ccd.generate_ccd_image(
    fits_info["dates"],
    station,
    fits_info["look_dirs_eci"],
    [1e4],
    catalog,
    scope_up_hat_eci=up_dir_eci,
    hot_pixel_probability=0,
    dead_pixel_probability=0,
    add_parabola=False,
    scintillation=False,
)
mr.toc()

adu_grid_streaked_sampled = np.log10(
    adu_grid_streaked_sampled - mr.image_background_naive(adu_grid_streaked_sampled)[1]
)
adu_grid_streaked_sampled[adu_grid_streaked_sampled < 1] = 0
adu_grid_streaked_sampled[
    np.isnan(adu_grid_streaked_sampled) | np.isinf(np.abs(adu_grid_streaked_sampled))
] = 0

stars_expected = expected_star_centroids_in_fits(
    fits_info, catalog, mean_look_dir, limiting_magnitude=12.0
)
stars_found = stars = mr.solve_star_centroids(fits_info)

# plt.subplot(1, 2, 1)
# plt.imshow(img, cmap="gray")
# for star in stars_expected:
#     plt.plot(img.shape[0] - star["centroid"][0], star["centroid"][1], "c+")
# for star in stars_found:
#     plt.plot(star["centroid"][0], star["centroid"][1], "m*")
# mrv.texit("True Image", "", "", grid=False)
# plt.subplot(1, 2, 2)
# plt.imshow(adu_grid_streaked_sampled, cmap="gray")
# for star in stars_expected:
#     plt.plot(img.shape[0] - star["centroid"][0], star["centroid"][1], "c+")
# for star in stars_found:
#     plt.plot(star["centroid"][0], star["centroid"][1], "m*")
# mrv.texit("Synthetic Image", "", "", grid=False)
# plt.show()

# %%
# Overlaying the two images
# plt.imshow(img, cmap="gray", alpha=0.5)
# plt.imshow(adu_grid_streaked_sampled, cmap="gray_r", alpha=0.5)
# plt.show()

# %%
# We're close, but we need to solve for the slight rotation and translation between the two images

plt.imshow(adu_grid_streaked_sampled, cmap="gray")

# building a tree for the expected stars
from scipy.spatial import KDTree

expected_centroids = np.array([star["centroid"].flatten() for star in stars_expected])
expected_centroids[:, 0] = img.shape[0] - expected_centroids[:, 0]
tree = KDTree(expected_centroids)
found_centroids = np.array([star["centroid"].flatten() for star in stars_found])

# finding the nearest neighbor for each found star
nearest = [tree.query(star["centroid"].T)[1] for star in stars_found]
nearest_expected_centroid = expected_centroids[nearest]


# Rotating expected -> true 90 degrees prograde
nearest_expected_to_found = found_centroids - nearest_expected_centroid
expected_to_found_dist = mr.vecnorm(nearest_expected_to_found)
# only use the middle 50 inds to avoid outliers
use_inds = np.argsort(expected_to_found_dist.flatten())[
    expected_to_found_dist.size // 4 : -expected_to_found_dist.size // 4
]
nearest_expected_to_found = nearest_expected_to_found[use_inds]
nearest_expected_centroid = nearest_expected_centroid[use_inds]
found_centroids = found_centroids[use_inds]
nearest = [nearest[i] for i in use_inds]
expected_to_found_dist = expected_to_found_dist[use_inds]
rotated_found_centroids = found_centroids

for i in range(1):
    nearest_expected_to_found_perp = np.array(
        [nearest_expected_to_found[:, 1], -nearest_expected_to_found[:, 0]]
    ).T
    nearest_expected_to_found_perp = mr.hat(nearest_expected_to_found_perp) * 1000
    # plotting this line from the mean of the line segment
    mean_points = (rotated_found_centroids + nearest_expected_centroid) / 2

    # least squares intersection point
    int_point = mr.least_squares_line_intersection(
        mean_points, mean_points + nearest_expected_to_found_perp
    )

    means_to_int = int_point - mean_points
    means_to_int_dist = mr.vecnorm(means_to_int)
    rotation_angle = np.arctan(expected_to_found_dist / means_to_int_dist)
    med_rot_angle = np.median(rotation_angle)

    rotated_found_centroids = mr.rotate_points_about_point(
        rotated_found_centroids, med_rot_angle, int_point
    )
    nearest_expected_to_found = rotated_found_centroids - nearest_expected_centroid
    print(f"Rotation angle: {np.rad2deg(med_rot_angle)} degrees")
    print(f"Intersection point: {int_point}")
    print(
        f"Mean pixel error after rotation: {mr.vecnorm(nearest_expected_to_found).mean()}"
    )

plt.plot(int_point[0], int_point[1], "g*")

for i in range(len(found_centroids)):
    plt.plot(found_centroids[i, 0], found_centroids[i, 1], "m*")
    plt.plot(nearest_expected_centroid[i, 0], nearest_expected_centroid[i, 1], "c+")

    plt.plot(
        [nearest_expected_centroid[i, 0], found_centroids[i, 0]],
        [nearest_expected_centroid[i, 1], found_centroids[i, 1]],
        "r-",
    )
    plt.plot(
        [mean_points[i, 0], mean_points[i, 0] + nearest_expected_to_found_perp[i, 0]],
        [mean_points[i, 1], mean_points[i, 1] + nearest_expected_to_found_perp[i, 1]],
        "y-",
    )

    plt.plot(rotated_found_centroids[i, 0], rotated_found_centroids[i, 1], "b*")

plt.xlim(0, img.shape[1])
plt.ylim(0, img.shape[0])
plt.show()

# %%
# Now we solve for the translation

use_inds = np.argsort(mr.vecnorm(nearest_expected_to_found).flatten())[
    nearest_expected_to_found.shape[0] // 4 : -nearest_expected_to_found.shape[0] // 4
]
mean_error = nearest_expected_to_found[use_inds].mean(axis=0)
print(f"Mean translation error: {mean_error}")
rotated_and_translated_found_centroids = rotated_found_centroids - mean_error
nearest_expected_to_found = (
    rotated_and_translated_found_centroids - nearest_expected_centroid
)
print(
    f"Mean pixel error after translation: {mr.vecnorm(nearest_expected_to_found[use_inds]).mean()}"
)

plt.imshow(adu_grid_streaked_sampled, cmap="gray")

for i in range(len(found_centroids)):
    plt.plot(found_centroids[i, 0], found_centroids[i, 1], "m*")
    plt.plot(nearest_expected_centroid[i, 0], nearest_expected_centroid[i, 1], "c+")

    plt.plot(rotated_found_centroids[i, 0], rotated_found_centroids[i, 1], "b*")

    plt.plot(
        rotated_and_translated_found_centroids[i, 0],
        rotated_and_translated_found_centroids[i, 1],
        "y*",
    )

plt.xlim(0, img.shape[1])
plt.ylim(0, img.shape[0])
plt.show()
