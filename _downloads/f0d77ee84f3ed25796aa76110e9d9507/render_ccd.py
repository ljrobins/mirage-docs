"""
CCD Rendering
=============

Renders a synthetic CCD image of an observation taken by the POGS telescope
"""

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
# %%
# Loading a fits image from the Purdue Optical Ground Station
from astropy.io import fits

import mirage as mr
import mirage.vis as mrv

with fits.open(os.path.join(os.environ['SRCDIR'], '..', 'examples/07-observer/00095337.fit')) as hdul:
    header = hdul[0].header
    for key in header.keys():
        print(key, header[key])

    date_obs_initial = datetime.datetime.strptime(header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc)
    ccd_temp = header['SET-TEMP']
    site_lat_geod_deg = header['OBSGEO-B']
    site_lon_deg = header['OBSGEO-L']
    site_alt_m = header['OBSGEO-H']
    center_azimuth_rad = np.deg2rad(header['CENTAZ'])
    center_elevation_rad = np.deg2rad(header['CENTALT'])
    airmass = header['AIRMASS']
    track_rate_rad_ra = mr.dms_to_rad(0, 0, header['TELTKRA']) # rad/s
    track_rate_rad_dec = mr.dms_to_rad(0, 0, header['TELTKDEC']) # rad/s
    obj_ra_rad_initial = mr.hms_to_rad(*[float(x) for x in header['OBJCTRA'].split(' ')])
    obj_dec_rad_initial = mr.dms_to_rad(*[float(x) for x in header['OBJCTDEC'].split(' ')])
    lst_deg_initial = np.rad2deg(mr.hms_to_rad(*[float(x) for x in header['LST'].split(' ')]))
    integration_time_seconds = header['EXPTIME']
    ccd_adu = hdul[0].data

date_obs_final = date_obs_initial + mr.seconds(integration_time_seconds)

observing_station = mr.Station(lat_deg=site_lat_geod_deg, lon_deg=site_lon_deg, alt_km=site_alt_m/1e3)

station_eci_initial = observing_station.j2000_at_dates(date_obs_initial)
station_eci_final = observing_station.j2000_at_dates(date_obs_final)

obj_ra_rad_final = obj_ra_rad_initial + integration_time_seconds * track_rate_rad_ra
obj_dec_rad_final = obj_dec_rad_initial + integration_time_seconds * track_rate_rad_dec

obj_look_eci_initial = mr.ra_dec_to_eci(obj_ra_rad_initial, obj_dec_rad_initial)
obj_look_eci_final = mr.ra_dec_to_eci(obj_ra_rad_final, obj_dec_rad_final)

eci_from_az_el = observing_station.az_el_to_eci(center_azimuth_rad, center_elevation_rad, date_obs_initial)
ra_dec_from_eci_from_az_el = mr.eci_to_ra_dec(eci_from_az_el)

obs_dates = np.array([date_obs_initial, date_obs_final])
obs_dirs_eci = np.vstack((obj_look_eci_initial, obj_look_eci_final))

# import pyvista as pv
# pl = pv.Plotter()
# mrv.render_observation_scenario(pl, dates=obs_dates, 
#                             station=observing_station, 
#                             look_dirs_eci=obs_dirs_eci,
#                             sensor_extent_km=20e3)
# pl.show()

# %%
# Synthesizing the same image

# %%
# Let's synthesize a CCD image for the same observation conditions

observing_station.telescope.fwhm = 2

obj = mr.SpaceObject("matlib_hylas4.obj", identifier=26853)
r_obj_eci = obj.propagate(obs_dates)

sv = mr.sun(obs_dates)
nadir = -mr.hat(r_obj_eci)
attitude = mr.AlignedAndConstrainedAttitude(
    v_align=nadir,
    v_const=sv,
    dates=obs_dates,
    axis_order=(1, 2, 0),
)
obj_lc_sampler, _ = observing_station.observe_light_curve(
    obj,
    attitude,
    mr.Brdf("phong"),
    obs_dates,
    use_engine=True,
    instance_count=1,
    model_scale_factor=1,
    rotate_panels=True,
)
lc_adu = obj_lc_sampler()

mr.tic()
adu_grid_streaked_sampled = observing_station.telescope.ccd.generate_ccd_image(
    obs_dates, observing_station, obs_dirs_eci, lc_adu, hot_pixel_probability=0, dead_pixel_probability=0,
)
mr.toc()

# %%
# Let's take a look at the real and synthetic CCD images

plt.figure(figsize=(8, 4))
plt.subplot(1,2,1)
plt.imshow(np.log10(ccd_adu), cmap="gist_stern")
mrv.texit(
    f'POGS CCD', "", "", grid=False
)
plt.colorbar(cax=mrv.get_cbar_ax(), label="$\log_{10}(ADU)$")

plt.subplot(1,2,2)
plt.imshow(np.log10(adu_grid_streaked_sampled), cmap="gist_stern")
mrv.texit(
    f'Synthetic CCD', "", "", grid=False
)
plt.colorbar(cax=mrv.get_cbar_ax(), label="$\log_{10}(ADU)$")
plt.tight_layout()
plt.show()

# %%
# Inspecting the backgrounds

frac_cuts = (1e-4, 5e-3)
thresh = slice(int(frac_cuts[0] * adu_grid_streaked_sampled.size), int((1-frac_cuts[1]) * adu_grid_streaked_sampled.size))
synth_br_data = np.sort(adu_grid_streaked_sampled.flatten())[thresh][::100]
real_br_data = np.sort(ccd_adu.flatten())[thresh][::100]

synth_br = np.mean(synth_br_data)
real_br = np.mean(real_br_data)

synth_br_poisson_samples = np.random.poisson(synth_br, synth_br_data.size)
real_br_poisson_samples = np.random.poisson(real_br, real_br_data.size)

plt.subplot(1,2,2)
bins = np.arange(np.min(synth_br_data), np.max(synth_br_data))
hist_args = dict(density=True, bins=bins, alpha=0.7)
plt.hist(synth_br_data, **hist_args)
plt.hist(synth_br_poisson_samples, **hist_args)
mrv.texit("Synthetic backgrounds", "ADU", "Density", ["Image", "Poisson fit"])

plt.subplot(1,2,1)
hist_args['bins'] = np.arange(np.min(real_br_poisson_samples), np.max(real_br_poisson_samples))
plt.hist(real_br_data, **hist_args)
plt.hist(real_br_poisson_samples, **hist_args)
mrv.texit("Real backgrounds", "ADU", "Density", ["Image", "Poisson fit"])

plt.tight_layout()
plt.gcf().set_size_inches(8, 4)
plt.show()