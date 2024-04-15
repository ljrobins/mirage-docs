"""
Star Matching
=============

Given star centroid locations and an initial estimate of the look direction and tracking rate, fit the catalog
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

import mirage as mr


def expected_star_centroids_in_fits(
    catalog: mr.StarCatalog,
    look_dir_eci_app: np.ndarray,
    up_dir_eci_app: np.ndarray,
    limiting_magnitude: float = 15.0,
    add_distortion: bool = True,
) -> np.ndarray:

    uvs_in_frame, vm_in_frame = catalog.in_fov(
        look_dir_eci_app, up_dir_eci_app, limiting_magnitude=limiting_magnitude
    )
    star_xs, star_ys = station.telescope.j2000_unit_vectors_to_pixels(
        look_dir_eci_app, up_dir_eci_app, uvs_in_frame, add_distortion=add_distortion
    )
    return np.vstack((star_xs.flatten(), star_ys.flatten(), vm_in_frame.flatten())).T


info_path = "/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/2022-09-18_GPS_PRN14/ObservationData.mat"
img_ind = 299
add_distortion = True
limiting_magnitude = 15.0
station = mr.Station()
data_mat = mr.load_obs_data(station, info_path, img_ind)
date_mid = data_mat["date_mid"]
mr.tic("Loading star catalog")
catalog = mr.StarCatalog("gaia", station, date_mid, aberration=True)
mr.toc()

obs_look_dir_from_az_el = station.az_el_to_eci(
    data_mat["az_rad"], data_mat["el_rad_true"], date_mid
)
el_app = mr.apparent_refacted_elevation(
    data_mat["pressure_pa"] / 100, data_mat["temp_k"], data_mat["el_rad_true"]
)
obs_look_dir_from_az_el_app = station.az_el_to_eci(data_mat["az_rad"], el_app, date_mid)
angle_rot = mr.angle_between_vecs(obs_look_dir_from_az_el, obs_look_dir_from_az_el_app)
axis_rot = mr.hat(np.cross(obs_look_dir_from_az_el, obs_look_dir_from_az_el_app))
dcm_app_to_true = mr.rv_to_dcm(axis_rot * angle_rot)

fits_path = os.path.join(os.path.split(info_path)[0], data_mat["fits_file"])
fits_info = mr.info_from_fits(fits_path)

look_dir_eci = data_mat["look_dir_eci_processed"]
look_dir_eci_app = dcm_app_to_true @ look_dir_eci

scope_up_dir_eci = data_mat["up_dir_eci_processed"]
scope_up_dir_eci_app = dcm_app_to_true @ scope_up_dir_eci

sms_names = data_mat["_obs_mat"]["saveMatchedStars"][0][img_ind].dtype.names
sms = dict(zip(sms_names, data_mat["_obs_mat"]["saveMatchedStars"][0][img_ind]))
sms = {k: np.squeeze(v) for k, v in sms.items()}
print(sms.keys())
stars_found = np.vstack((4096 - sms["x0"], 4096 - sms["y0"], sms["Gmag"])).T

# print(stars_found.shape)
# endd
matched_gmag = catalog._mags[sms["idx_catMatched"] - 1]
matched_irrad = mr.apparent_magnitude_to_irradiance(matched_gmag)
matched_brightness = sms["brightness"]
coefs = np.polyfit(matched_irrad, matched_brightness, 1)
coefs = [coefs[0], 0]
sint = lambda irrad: np.polyval(coefs, irrad) / fits_info["integration_time"] / irrad

plt.figure()
plt.scatter(matched_irrad, matched_brightness)
plt.plot(matched_irrad, np.polyval(coefs, matched_irrad), c="r")
plt.xlabel("Irradiance [W/m^2]")
plt.ylabel("ADU")
# plt.show()
# enddd

img = fits_info["ccd_adu"]
img = np.fliplr(np.flipud(img))

tele = station.telescope

stars_expected = expected_star_centroids_in_fits(
    catalog,
    look_dir_eci_app,
    scope_up_dir_eci_app,
    limiting_magnitude=limiting_magnitude,
    add_distortion=add_distortion,
)

# %%
# We're close, but we need to solve for the slight rotation and translation between the two images

# building a tree for the expected stars
tree = KDTree(stars_expected[:, :2])

# finding the nearest neighbor for each found star
expected_to_found_dist, nearest = tree.query(stars_found[:, :2])
nearest_expected_centroid = stars_expected[nearest, :2]
# only use lowest 25% dist pairs to avoid outliers
use_inds = np.argsort(expected_to_found_dist.flatten())[
    : expected_to_found_dist.size // 4
]

nearest_expected_uvs = tele.pixels_to_j2000_unit_vectors(
    look_dir_eci_app,
    scope_up_dir_eci_app,
    nearest_expected_centroid[use_inds],
    input_is_distorted=add_distortion,
)
stars_found_uvs = tele.pixels_to_j2000_unit_vectors(
    look_dir_eci_app,
    scope_up_dir_eci_app,
    stars_found[use_inds, :2],
    input_is_distorted=add_distortion,
)

print(nearest_expected_centroid[use_inds] - stars_found[use_inds, :2])

q_davenport = mr.davenport(nearest_expected_uvs, stars_found_uvs)
print(mr.wrap_to_pi(mr.vecnorm(mr.quat_to_rv(q_davenport))) * 180 / np.pi)

A_davenport = mr.quat_to_dcm(q_davenport)

look_dir_app_true = A_davenport @ look_dir_eci_app
up_dir_app_true = A_davenport @ scope_up_dir_eci_app

erc = expected_star_centroids_in_fits(
    catalog,
    look_dir_app_true,
    up_dir_app_true,
    limiting_magnitude=limiting_magnitude,
    add_distortion=add_distortion,
)

img_prepared = mr.prepare_fits_for_plotting(img, background_method="parabola")
plt.figure()
plt.imshow(img_prepared, cmap="gray")
plt.scatter(
    erc[:, 0],
    erc[:, 1],
    c="y",
    marker="+",
    s=20,
)
plt.scatter(stars_found[:, 0], stars_found[:, 1], c="m", marker="o", s=10)


# %%
# Generating the synthetic image
station.telescope.fwhm = 4.0
mr.tic("Synthesizing CCD Image")
adu_grid_streaked_sampled = station.telescope.ccd.generate_ccd_image(
    date_mid,
    fits_info["integration_time"],
    station,
    look_dir_app_true,
    [fits_info["ra_rate"], fits_info["dec_rate"]],
    1e4,
    catalog,
    up_dir_eci=up_dir_app_true,
    limiting_magnitude=limiting_magnitude,
    add_distortion=add_distortion,
    sint_val=sint,
)
mr.toc()

adu_grid_streaked_sampled_prepared = mr.prepare_fits_for_plotting(
    adu_grid_streaked_sampled, background_method="naive"
)

# adu_grid_streaked_sampled = adu_grid_streaked_sampled / 10
# adu_grid_streaked_sampled += int(1e3)

# %%
# Overlaying the two images

plt.figure()
plt.imshow(img_prepared, cmap="gray")
plt.imshow(adu_grid_streaked_sampled_prepared, cmap="gray_r", alpha=0.5)
plt.scatter(
    erc[:, 0],
    erc[:, 1],
    c="y",
    marker="+",
    s=20,
)
plt.scatter(stars_found[:, 0], stars_found[:, 1], c="m", marker="o", s=10)

plt.figure()
plt.imshow(img_prepared, cmap="gray")
plt.colorbar()
plt.figure()
plt.imshow(adu_grid_streaked_sampled_prepared, cmap="gray")
plt.colorbar()
# plt.figure()
# plt.imshow(np.log10(np.clip(np.abs(img - adu_grid_streaked_sampled), 1, np.inf)), cmap="coolwarm")
# plt.colorbar()
plt.show()

# %%
# Saving the images to file

import imageio
from PIL import Image

imageio.imwrite("observed_log_adu.png", Image.fromarray(img_prepared).convert("L"))
imageio.imwrite("synthetic_log_adu.png", Image.fromarray(adu_grid_streaked_sampled_prepared).convert("L"))