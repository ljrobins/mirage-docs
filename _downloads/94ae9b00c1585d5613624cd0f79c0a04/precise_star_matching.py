"""
Precise Star Matching
=====================

Using processed data to match the pixel positions of stars in a CCD image
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr


# @mr.with_profiler
def main():
    # obs_mat_path = "/Users/liamrobinson/Documents/mirage/fits/precise_star_matching/ObservationData.mat"
    # fits_dir = os.dirname(obs_mat_path)

    obs_mat_path = "/Users/liamrobinson/Downloads/ObsData/ObservationData.mat"
    fits_dir = os.path.join(os.environ['SRCDIR'], "..", "2024_02_29")

    station = mr.Station()
    mr.tic("Loading Data")
    data_mat = mr.load_obs_data(station, obs_mat_path, 8)
    mr.toc()

    date_mid = data_mat["date_mid"] - mr.seconds(10)

    obs_look_dir_from_az_el = station.az_el_to_eci(
        data_mat["az_rad"], data_mat["el_rad_true"], date_mid
    )
    el_app = mr.apparent_refacted_elevation(
        data_mat["pressure_pa"] / 100, data_mat["temp_k"], data_mat["el_rad_true"]
    )
    obs_look_dir_from_az_el_app = station.az_el_to_eci(
        data_mat["az_rad"], el_app, date_mid
    )
    angle_rot = mr.angle_between_vecs(
        obs_look_dir_from_az_el, obs_look_dir_from_az_el_app
    )
    axis_rot = mr.hat(np.cross(obs_look_dir_from_az_el, obs_look_dir_from_az_el_app))
    dcm_app_to_true = mr.rv_to_dcm(axis_rot * angle_rot)

    fits_path = os.path.join(fits_dir, data_mat["fits_file"])
    fits_info = mr.info_from_fits(fits_path)
    adu_proc = mr.prepare_fits_for_plotting(fits_info["ccd_adu"])
    catalog = mr.StarCatalog("gaia", station, date_mid, aberration=True)

    look_dir_eci = data_mat["look_dir_eci_processed"]
    look_dir_eci_app = dcm_app_to_true @ look_dir_eci

    scope_up_dir_eci = data_mat["up_dir_eci_processed"]
    scope_up_dir_eci_app = dcm_app_to_true @ scope_up_dir_eci

    in_uvs, _ = catalog.in_fov(
        look_dir_eci_app, scope_up_dir_eci_app, limiting_magnitude=14
    )
    ys, xs = station.telescope.j2000_unit_vectors_to_pixels(
        look_dir_eci_app, scope_up_dir_eci_app, in_uvs, add_distortion=True
    )

    mr.tic("Synthesizing CCD Image")
    ccd_adu_synth = station.telescope.ccd.generate_ccd_image(
        date_mid,
        data_mat["integration_time"],
        station,
        look_dir_eci_app,
        [data_mat["ra_rate"], data_mat["dec_rate"]],
        1e5,
        catalog,
        scope_up_dir_eci_app,
    )
    mr.toc()
    adu_synth_proc = mr.prepare_fits_for_plotting(ccd_adu_synth)

    plt.figure(figsize=(4, 4))
    plt.imshow(adu_proc, cmap="gray")
    plt.imshow(adu_synth_proc, cmap="cool", alpha=0.5)
    plt.scatter(xs, ys, c="m", s=5)
    plt.show()


if __name__ == "__main__":
    main()
