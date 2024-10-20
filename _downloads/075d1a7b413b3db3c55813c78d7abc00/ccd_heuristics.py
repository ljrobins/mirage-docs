"""
CCD Heuristics
==============

Examples to develop a better intuition for CCD counts from known sources
"""

import numpy as np

import mirage as mr

z_obs = 0.0  # Point the telescope towards zenith
station = mr.Station(preset='pogs')
projected_irrad_per_pixel_area = mr.dms_to_rad(
    0, 0, station.telescope.ccd.pixel_scale
) ** 2 * mr.mpsas_to_irradiance_per_steradian(22)
sint_val = mr.sint(station, z_obs)[0]
count_per_second_per_pixel = sint_val * projected_irrad_per_pixel_area
print(
    f'For a telescope pointed towards zenith of 22 MPSAS sky, each pixel counts on average {count_per_second_per_pixel:.2f} per second'
)

# %%
# We can also look at counts due to point sources. Note that these sources are actually spread across a few pixels, so the values are actually much lower on the CCD
total_star_counts = sint_val * mr.apparent_magnitude_to_irradiance(16)
print(
    f'A magnitude 16 star produces on average {total_star_counts:.2e} counts per second'
)

total_star_counts = sint_val * mr.apparent_magnitude_to_irradiance(8)
print(
    f'A magnitude 8 star produces on average {total_star_counts:.2e} counts per second'
)

irrad_sphere = (
    mr.normalized_light_curve_sphere(
        cd_sphere=1.0, r_sphere_m=10, phase_angle_rad=np.pi / 2
    )
    / (40e6) ** 2
)
print(
    f'A 10-meter diffuse sphere in GEO produces on average {irrad_sphere*sint_val:.2e} counts per second'
)

# %%
# The size in square pixels of a large GEO satellite when observed from the surface of the Earth by POGS
station.telescope.ccd.pixel_scale = 1
station.telescope.aperture_diameter = 0.5
sat_radius_m = 20
sat_dist_m = (36e3) * 1e3
pscale = station.telescope.ccd.pixel_scale  # arcseconds / pixel
p_area_sterad = mr.dms_to_rad(0, 0, pscale) ** 2  # sterad / pixel ** 2
angular_radius_of_sat_geo = np.arctan(sat_radius_m / sat_dist_m)
angular_radius_of_sat_geo_pix = angular_radius_of_sat_geo / mr.dms_to_rad(
    0, 0, station.telescope.ccd.pixel_scale
)

print(f'A GEO satellite is {2*angular_radius_of_sat_geo_pix:.1f} pixels wide from POGS')

sat_radius_m = 0.3
sat_dist_m = (1000) * 1e3
pscale = station.telescope.ccd.pixel_scale  # arcseconds / pixel
p_area_sterad = mr.dms_to_rad(0, 0, pscale) ** 2  # sterad / pixel ** 2
angular_radius_of_sat_leo = np.arctan(sat_radius_m / sat_dist_m)
angular_radius_of_sat_leo_pix = angular_radius_of_sat_leo / mr.dms_to_rad(
    0, 0, station.telescope.ccd.pixel_scale
)

print(f'A LEO satellite is {2*angular_radius_of_sat_leo_pix:.1f} pixels wide from POGS')

# %%
# Airy disk size for GEO objects

rayleigh_crit_rad = 1.22 * 550e-9 / station.telescope.aperture_diameter
rayleigh_crit_pix = rayleigh_crit_rad / mr.dms_to_rad(
    0, 0, station.telescope.ccd.pixel_scale
)
print(
    f'For GEO the Airy disk is {rayleigh_crit_pix/angular_radius_of_sat_geo_pix:.1f}x wider than the object itself'
)
print(
    f'For LEO the Airy disk is {rayleigh_crit_pix/angular_radius_of_sat_leo_pix:.1f}x wider than the object itself'
)
