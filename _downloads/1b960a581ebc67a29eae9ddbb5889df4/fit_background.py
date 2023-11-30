"""
Fitting the Background
======================

Reading real FITS images and using them to calibrate the background
"""

import datetime
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import mirage as mr
import mirage.vis as mrv

# def process_fits_file(fits_path: str) -> tuple:
#     print(fits_path)
#     with fits.open(fits_path) as hdul:
#         header = hdul[0].header
#         date = datetime.datetime.strptime(
#             header["DATE-OBS"], "%Y-%m-%dT%H:%M:%S.%f"
#         ).replace(tzinfo=datetime.timezone.utc)
#         ccd_temp = header["SET-TEMP"]
#         site_lat_geod_deg = header["OBSGEO-B"]
#         site_lon_deg = header["OBSGEO-L"]
#         site_alt_m = header["OBSGEO-H"]
#         center_azimuth_rad = np.deg2rad(header["CENTAZ"])
#         center_elevation_rad = np.deg2rad(header["CENTALT"])
#         airmass = header["AIRMASS"]
#         track_rate_rad_ra = mr.dms_to_rad(0, 0, header["TELTKRA"])  # rad/s
#         track_rate_rad_dec = mr.dms_to_rad(0, 0, header["TELTKDEC"])  # rad/s
#         ra = mr.hms_to_rad(*[float(x) for x in header["OBJCTRA"].split(" ")])
#         dec = mr.dms_to_rad(*[float(x) for x in header["OBJCTDEC"].split(" ")])
#         t_int = header["EXPTIME"]
#         ccd_adu = hdul[0].data

#         ccd_adu_minus_parabola = mr.ChargeCoupledDevice().subtract_parabola(ccd_adu)
#         _, background_mean = mr.image_background_naive(ccd_adu_minus_parabola)
#         return ra, dec, date, t_int, background_mean


# def main():
#     fits_dir = os.path.join(os.environ["SRCDIR"], "..", "data")
#     fit_paths = [
#         os.path.join(fits_dir, fit_file)
#         for fit_file in os.listdir(fits_dir)
#         if fit_file.endswith(".fit")
#     ]
#     valid_fits_paths = fit_paths[::50]

#     mr.tic()
#     with Pool(processes=8) as pool:
#         results = list(pool.map(process_fits_file, valid_fits_paths))
#     mr.toc()

#     ras = []
#     decs = []
#     dates = []
#     t_ints = []
#     background_means = []

#     for ra, dec, date, t_int, background_mean in results:
#         ras.append(ra)
#         decs.append(dec)
#         dates.append(date)
#         t_ints.append(t_int)
#         background_means.append(background_mean)

#     ras = np.array(ras)
#     decs = np.array(decs)
#     dates = np.array(dates)
#     t_ints = np.array(t_ints)
#     background_means = np.array(background_means)

#     mr.tic()
#     observing_station = mr.Station("pogs")
#     observing_station.telescope.integration_time = 1
#     obj_look_eci_initial = mr.ra_dec_to_eci(ras, decs)
#     moon = mr.moon(dates)
#     sun = mr.sun(dates)
#     sb = observing_station.sky_brightness(dates, obj_look_eci_initial)
#     sb = sb.flatten() * t_ints.flatten()
#     angle_to_moon = mr.angle_between_vecs(obj_look_eci_initial, moon)
#     z_obs = mr.angle_between_vecs(
#         obj_look_eci_initial, observing_station.j2000_at_dates(dates)
#     )
#     moon_phase_angle = mr.angle_between_vecs(obj_look_eci_initial - moon, sun - moon)
#     moon_phase_fraction = mr.moon_phase_fraction(moon_phase_angle)
#     mr.toc()
#     print(sb)

#     plt.subplot(2, 2, 1)
#     plt.plot(angle_to_moon, background_means, ".")
#     mrv.texit("", "Angle to Moon [rad]", "Background Mean [ADU]")

#     plt.subplot(2, 2, 2)
#     plt.plot(z_obs, background_means, ".")
#     mrv.texit("", "Zenith Angle [rad]", "Background Mean [ADU]")

#     plt.subplot(2, 2, 3)
#     plt.plot(moon_phase_fraction, background_means, ".")
#     mrv.texit("", "Moon Phase Fraction", "Background Mean [ADU]")

#     plt.subplot(2, 2, 4)
#     plt.plot(sb, background_means, ".")
#     mrv.texit("", "Synthetic Background Mean [ADU]", "Background Mean [ADU]")
#     plt.tight_layout()
#     plt.show()

#     import pyvista as pv

#     pl = pv.Plotter()
#     mrv.render_observation_scenario(
#         pl,
#         dates=dates,
#         station=observing_station,
#         look_dirs_eci=obj_look_eci_initial,
#         sensor_extent_km=40e3,
#         sensor_half_angle_deg=0.1,
#         night_lights=True,
#         atmosphere=True,
#         borders=True,
#     )
#     pl.show()

#     # %%
#     # Plotting the observations with the location of the Moon

#     print(np.min(background_means), np.max(background_means))
#     ra_moon, dec_moon = mr.eci_to_ra_dec(moon)
#     plt.scatter(ras, decs, c=background_means)
#     plt.scatter(ra_moon, dec_moon)
#     plt.xlim(0, 2 * np.pi)
#     plt.ylim(-np.pi / 2, np.pi / 2)
#     mrv.texit("", "RA [rad]", "Dec [rad]", ["Observations", "Moon"])
#     plt.colorbar()
#     plt.show()


# if __name__ == "__main__":
#     main()
