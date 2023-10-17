"""
CCD Heuristics
==============

Examples to develop a better intuition for CCD counts from known sources
"""
import numpy as np

import mirage as mr

z_obs = np.pi / 4  # Point the telescope towards zenith
station = mr.Station(preset="pogs")
projected_irrad_per_pixel_area = mr.dms_to_rad(
    0, 0, station.telescope.pixel_scale
) ** 2 * mr.mpsas_to_irradiance_per_steradian(22)
sint_val = mr.sint(station, z_obs)[0]
count_per_second_per_pixel = sint_val * projected_irrad_per_pixel_area
print(
    f"For a telescope pointed towards zenith of 22 MPSAS sky, each pixel counts on average {count_per_second_per_pixel:.2f} per second"
)

# %%
# We can also look at counts due to point sources. Note that these sources are actually spread across a few pixels, so the values are actually much lower on the CCD
total_star_counts = sint_val * mr.apparent_magnitude_to_irradiance(16)
print(
    f"A magnitude 16 star produces on average {total_star_counts:.2e} counts per second"
)

total_star_counts = sint_val * mr.apparent_magnitude_to_irradiance(8)
print(
    f"A magnitude 8 star produces on average {total_star_counts:.2e} counts per second"
)

irrad_sphere = (
    mr.normalized_light_curve_sphere(
        cd_sphere=1.0, r_sphere_m=10, phase_angle_rad=np.pi / 2
    )
    / (40e6) ** 2
)
print(
    f"A 10-meter diffuse sphere in GEO produces on average {irrad_sphere*sint_val:.2e} counts per second"
)
