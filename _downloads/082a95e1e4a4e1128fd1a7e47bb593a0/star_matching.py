"""
Star Matching
=============

Given star centroid locations and an initial estimate of the look direction and tracking rate, fit the catalog
"""

import datetime
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.linalg import orthogonal_procrustes

import mirage as mr


def update_refraction(
    station: mr.Station,
    look_dir_eci: np.ndarray,
    up_dir_eci: np.ndarray,
    date: datetime.datetime,
    p_mbar: float,
    t_k: float,
) -> np.ndarray:
    _, el_true = station.eci_to_az_el(date, look_dir_eci)
    el_app = mr.apparent_refacted_elevation(p_mbar, t_k, el_true)
    print(f"Applying {np.rad2deg(el_app - el_true):.2e} deg of refraction")

    dcm_tele = np.vstack((np.cross(up_dir_eci, look_dir_eci), up_dir_eci, look_dir_eci))
    dcm_true_to_app = mr.rv_to_dcm(dcm_tele[0, :] * (el_true - el_app)).T
    dcm_tele_app = dcm_true_to_app @ dcm_tele

    look_dir_eci_app = dcm_tele_app[2, :]
    up_dir_eci_app = dcm_tele_app[1, :]
    return look_dir_eci_app, up_dir_eci_app


def expected_star_centroids_in_fits(
    station: mr.Station,
    catalog: mr.StarCatalog,
    look_dir_eci: np.ndarray,
    up_dir_eci: np.ndarray,
    limiting_magnitude: float = 15.0,
    add_distortion: bool = True,
) -> np.ndarray:

    uvs_in_frame, vm_in_frame, inds_in_frame = catalog.in_fov(
        look_dir_eci,
        up_dir_eci,
        limiting_magnitude=limiting_magnitude,
        return_inds=True,
    )
    star_xs, star_ys = station.telescope.j2000_unit_vectors_to_pixels(
        look_dir_eci, up_dir_eci, uvs_in_frame, add_distortion=add_distortion
    )

    return np.vstack(
        (
            star_xs.flatten(),
            star_ys.flatten(),
            uvs_in_frame.T,
            vm_in_frame.flatten(),
            inds_in_frame,
        )
    ).T


def generate_matched_image(
    info_path: str,
    img_ind: int,
    station: mr.Station,
    catalog: mr.StarCatalog,
    add_distortion: bool = True,
    add_refraction: bool = True,
    limiting_magnitude: float = 12.0,
):
    data_mat = mr.load_obs_data(station, info_path, img_ind)
    date_mid = data_mat["date_mid"]

    look_dir_eci = data_mat["look_dir_eci_processed"]
    up_dir_eci = data_mat["up_dir_eci_processed"]
    p_mbar = data_mat["pressure_pa"] / 100
    t_k = data_mat["temp_k"]
    if add_refraction:
        look_dir_eci, up_dir_eci = update_refraction(
            station, look_dir_eci, up_dir_eci, date_mid, p_mbar, t_k
        )

    fits_path = os.path.join(os.path.split(info_path)[0], data_mat["fits_file"])
    fits_info = mr.info_from_fits(fits_path)

    sms_names = data_mat["_obs_mat"]["saveMatchedStars"][0][img_ind].dtype.names
    sms = dict(zip(sms_names, data_mat["_obs_mat"]["saveMatchedStars"][0][img_ind]))
    sms = {k: np.squeeze(v) for k, v in sms.items()}
    ex = sms["endpoints_x"].T
    ey = sms["endpoints_y"].T
    x = ex[:, 1] - ex[:, 0]
    y = ey[:, 1] - ey[:, 0]
    streak_dir = np.vstack((x, y)).T
    streak_dir_hat = mr.hat(streak_dir)
    ang = np.arctan2(streak_dir_hat[:, 1], streak_dir_hat[:, 0])

    stars_found = np.vstack((4096 - sms["x0"], 4096 - sms["y0"], sms["Gmag"])).T

    matched_ind = sms["idx_catMatched"] - 1
    matched_gmag = catalog._mags[matched_ind]
    matched_irrad = mr.apparent_magnitude_to_irradiance(matched_gmag)
    matched_adu = sms["brightness"]
    coefs = np.polyfit(np.log10(matched_irrad), np.log10(matched_adu), 1)
    fit_adu_of_irrad = lambda irrad: 10 ** np.polyval(coefs, np.log10(irrad))
    sint = lambda irrad: fit_adu_of_irrad(irrad) / fits_info["integration_time"] / irrad

    img = fits_info["ccd_adu"]
    img = np.fliplr(np.flipud(img))

    stars_expected = expected_star_centroids_in_fits(
        station,
        catalog,
        look_dir_eci,
        up_dir_eci,
        limiting_magnitude=limiting_magnitude,
        add_distortion=add_distortion,
    )
    _, expected_pair_inds, found_pair_inds = np.intersect1d(
        stars_expected[:, -1].astype(int), matched_ind, return_indices=True
    )
    uv_pairs_found = station.telescope.pixels_to_j2000_unit_vectors(
        look_dir_eci,
        up_dir_eci,
        stars_found[found_pair_inds, :2],
        input_is_distorted=True,
    )
    uv_pairs_expected = stars_expected[expected_pair_inds, 2:5]

    err_before = (
        stars_expected[expected_pair_inds, :2] - stars_found[found_pair_inds, :2]
    )
    print(f"BEFORE QUEST: median error {np.median(mr.vecnorm(err_before)):.2f} pixels")
    print(f"Performing the QUEST fit with {expected_pair_inds.size} stars")

    A = orthogonal_procrustes(uv_pairs_expected, uv_pairs_found)[0]
    print(
        f"Applying a {np.rad2deg(mr.wrap_to_pi(mr.vecnorm(mr.dcm_to_rv(A)))).squeeze():.2f} degree rotation to the telescope orientation"
    )

    look_dir_true = A @ look_dir_eci
    up_dir_true = A @ up_dir_eci

    expected_stars_corrected = expected_star_centroids_in_fits(
        station,
        catalog,
        look_dir_true,
        up_dir_true,
        limiting_magnitude=limiting_magnitude,
        add_distortion=add_distortion,
    )
    _, expected_pair_inds, found_pair_inds = np.intersect1d(
        expected_stars_corrected[:, -1].astype(int), matched_ind, return_indices=True
    )

    err_updated = (
        expected_stars_corrected[expected_pair_inds, :2]
        - stars_found[found_pair_inds, :2]
    )
    print(f"AFTER QUEST: median error {np.median(mr.vecnorm(err_updated)):.2f} pixels")

    # %%
    # Generating the synthetic image
    mr.tic("Synthesizing CCD Image")
    img_sym = station.telescope.ccd.generate_ccd_image(
        date_mid,
        fits_info["integration_time"],
        station,
        look_dir_true,
        [fits_info["ra_rate"], fits_info["dec_rate"]],
        1e4,
        catalog,
        up_dir_eci=up_dir_true,
        limiting_magnitude=limiting_magnitude,
        add_distortion=add_distortion,
        sint_val=sint,
        noise=False,
    )
    mr.toc()
    return dict(
        img_sym=img_sym,
        img=img,
        stars_found=stars_found,
        expected_stars_corrected=expected_stars_corrected,
        matched_irrad=matched_irrad,
        matched_adu=matched_adu,
        fit_adu_of_irrad=fit_adu_of_irrad,
        data_mat=data_mat,
        err_updated=err_updated,
    )


info_path = "/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/2022-09-18_GPS_PRN14/ObservationData.mat"
img_ind = 200
add_distortion = True
add_refraction = True
limiting_magnitude = 11.0
station = mr.Station()
station.telescope.fwhm = 3.0
mr.tic("Loading star catalog")
catalog = mr.StarCatalog("gaia", station, mr.now(), aberration=False)
mr.toc()

fig = plt.figure()
im = plt.imshow(np.eye(4096), cmap='gray')

def animate(i):
    res = generate_matched_image(info_path, i, station, catalog, add_distortion, add_refraction, limiting_magnitude)
    img = np.log10(res['img_sym'])
    im.set_data(img)
    plt.clim(img.min(), img.max())
    return im,

frames = 250
anim_time = 6
fps = frames / anim_time
interval = 1000 / fps
anim = FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True)
anim.save("out/synth_imgs.gif")


# for i in range(100):
#     n = SimpleNamespace(**res)
enddd

# img_sym_prepared = mr.prepare_fits_for_plotting(img_sym, background_method="naive")
img_sym_prepared = np.log10(n.img_sym)

data_mat = res["data_mat"]
sint_synth = mr.sint(station, np.pi / 2 - data_mat["el_rad_true"])

plt.figure()
plt.scatter(n.matched_irrad, n.matched_adu)
plt.plot(n.matched_irrad, n.fit_adu_of_irrad(n.matched_irrad), c="r", markersize=7)
plt.xlabel("Irradiance [W/m^2]")
plt.ylabel("ADU")
plt.grid()
plt.xscale("log")
plt.yscale("log")
plt.legend(["Data", "Best linear fit"])


# %%
# Overlaying the two images

n.img -= int(1e3)
n.img = np.clip(n.img, 1, np.inf)
img_prepared = np.log10(n.img)
print(np.median(img_prepared))

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
plt.colorbar()
plt.clim(np.min(img_sym_prepared), np.max(img_sym_prepared))
plt.figure()
plt.imshow(img_sym_prepared, cmap="gray")
plt.legend()
plt.colorbar()

# %%
# Subtracting the two images
adu_err = n.img_sym.astype(np.int64) - n.img.astype(np.int64)
adu_err_stdev = np.abs(adu_err) / np.sqrt(np.abs(n.img.astype(np.int64)))
plt.figure()
plt.imshow(adu_err_stdev, cmap="plasma")
plt.clim(0, 6)
plt.colorbar()
plt.xlim(2700, 2934)
plt.ylim(900, 1090)
plt.show()

# %%
# Saving the images to file

# import imageio
# from PIL import Image

# img_prepared = img_prepared / np.max(img_prepared) * 255
# img_sym_prepared = (
#     img_sym_prepared
#     / np.max(img_sym_prepared)
#     * 255
# )

# imageio.imwrite("observed_log_adu.png", Image.fromarray(img_prepared).convert("L"))
# imageio.imwrite(
#     "synthetic_log_adu.png",
#     Image.fromarray(img_sym_prepared).convert("L"),
# )
