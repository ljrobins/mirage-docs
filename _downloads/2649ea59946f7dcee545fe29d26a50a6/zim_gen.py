"""
Zimmerwald Data
===============

Generating a synthetic image to compare with one taken by the Zimmerwald MAIN telescope
"""

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import mirage as mr
import mirage.photo as mrp

file = '/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/aiub/ZigZag/13021D_31082022_fullframe.fits'
header, data = mrp.load_fits(file)
intl_des = '2013-021D'

cycle_time = header['ACT']
i = 85
img = data[i].T  # Transpose to get its dimensions right
timedelta_to_i = mr.seconds(i * cycle_time)

date_obs_initial = (
    datetime.datetime.strptime(header['FRAME'], '%Y-%m-%dT%H:%M:%S.%f').replace(
        tzinfo=datetime.timezone.utc
    )
    + timedelta_to_i
)
integration_time_s = header['EXPOSURE']
date_mid = date_obs_initial + mr.seconds(integration_time_s / 2)

obj = mr.SpaceObject('cube.obj', identifier=intl_des)
station = mr.Station('zimmain')

station.telescope.fwhm = 4
catalog = mr.GaiaSpectralStarCatalog(station)

obj_pos, obj_vel = obj.propagate(date_mid, return_velocity=True)
station_pos, station_vel = station.j2000_at_dates(date_mid, return_velocity=True)
r_obs = mr.vecnorm(obj_pos - station_pos)[0]
look_dir_eci = mr.hat(obj_pos - station_pos)
up_dir_eci = mr.hat(station_pos)  # for the az/el mount

az, el = station.eci_to_az_el(date_mid, look_dir_eci)
theta = 90 - np.rad2deg(el)

dra, ddec = mr.rv_to_daz_del(obj_pos - station_pos, obj_vel - station_vel)
dra *= mr.AstroConstants.rad_to_arcsecond
ddec *= mr.AstroConstants.rad_to_arcsecond
print(
    f'RA tracking rate: {dra:.3f} arcsec/sec, Dec tracking rate: {ddec:.3f} arcsec/sec'
)
v_perp = mr.hat(np.cross(look_dir_eci, up_dir_eci))
up_dir_eci = np.cross(v_perp, look_dir_eci)

print(theta)
rotator_angle = np.deg2rad(theta - 205)
print(np.rad2deg(rotator_angle))

print(up_dir_eci)
up_dir_eci = mr.rv_to_dcm(look_dir_eci * rotator_angle) @ up_dir_eci

look_ra_app, look_dec_app = mr.eci_to_ra_dec(look_dir_eci)
print(f'Object identity: {intl_des}')
print(f'Observation midpoint: {date_mid} UTC')
print(f'Exposure time: {integration_time_s} seconds')
print(
    f'We are looking at RA/Dec (midpoint): {mr.rad_to_dms(look_ra_app)}, {mr.rad_to_dms(look_dec_app)}'
)
print(look_dir_eci)
print(up_dir_eci)

# import pyvista as pv
# import mirage.vis as mrv
# p = pv.Plotter()
# mrv.render_observation_scenario(
#     p,
#     np.array([date_mid]),
#     station=station,
#     look_dirs_eci=look_dir_eci,
#     sensor_extent_km=r_obs,
#     up_dirs_eci=up_dir_eci,
# )
# p.show()

phase_angle_rad = mr.angle_between_vecs(-look_dir_eci, mr.sun(date_mid))
s2 = mr.Station()
sint_val = mr.sint(s2, np.deg2rad(theta))[0]

print(r_obs)
print(phase_angle_rad)
print(sint_val)

r_sph_meter = 1
irrad_sphere = (
    mr.AstroConstants.sun_irradiance_vacuum
    * mr.normalized_light_curve_sphere(
        cd_sphere=0.5, r_sphere_m=r_sph_meter, phase_angle_rad=phase_angle_rad
    )
    / (r_obs * 1e3) ** 2
)[0, 0]
print(irrad_sphere)

obj_signal = sint_val * irrad_sphere * integration_time_s
print(obj_signal)


mr.tic('Synthesizing CCD Image')
img_syn = station.telescope.ccd.generate_ccd_image(
    date_mid,
    integration_time_s,
    station,
    look_dir_eci,
    [dra, ddec],
    obj_signal,
    catalog,
    up_dir_eci=up_dir_eci,
    add_distortion=False,
    noise=True,
    sky_brightness=True,
    binning=(2, 2),
)
flat_frame = np.load(os.path.join(os.environ['DATADIR'], 'zimmain_flat.nogit.npy'))
mr.toc()

# simulating binning
assert all([not x % 2 for x in img_syn.shape]), (
    "One or more of the image's dimensions sizes is odd, cannot bin"
)
img_syn = img_syn.reshape(img_syn.shape[0] // 2, 2, img_syn.shape[1] // 2, 2).sum(
    axis=(1, 3)
)
img_syn, _, _, img_syn_br_std = mrp.background_subtract(
    img_syn,
    reduction_factor=1,
    sigma=3.0,
    box_size=100,
)
img, _, _, img_br_std = mrp.background_subtract(
    img, reduction_factor=1, sigma=3.0, box_size=100
)

print(f'Background std of synthetic image: {img_syn_br_std:.2f}')
print(f'Background std of real image: {img_br_std:.2f}')

np.savez(
    '/Users/liamrobinson/Documents/maintained-research/mirage/examples/14-digital-twin-2024/data/zim.nogit.npz',
    img=img,
    img_syn=img_syn,
    br_std_img=img_br_std,
    br_std_synth=img_syn_br_std,
)

plt.figure()
mrp.plot_fits(img_syn)

plt.figure()
mrp.plot_fits(img)
plt.title(f'{i}')
plt.show()

r_sph_meter = 3
irrad_sphere = (
    mr.AstroConstants.sun_irradiance_vacuum
    * mr.normalized_light_curve_sphere(
        cd_sphere=1, r_sphere_m=r_sph_meter, phase_angle_rad=phase_angle_rad
    )
    / (r_obs * 1e3) ** 2
)[0, 0]
obj_signal = sint_val * irrad_sphere * integration_time_s

flat_frame_interp = RegularGridInterpolator(
    points=(np.arange(0, 2160, 2), np.arange(0, 2560, 2)),
    values=flat_frame,
    bounds_error=False,
    fill_value=1.0,
)

mr.tic('Synthesizing CCD Image')
img_syn = station.telescope.ccd.generate_ccd_image(
    date_mid,
    integration_time_s,
    station,
    look_dir_eci,
    [dra, ddec],
    obj_signal,
    catalog,
    up_dir_eci=up_dir_eci,
    add_distortion=False,
    noise=True,
    sky_brightness=True,
    binning=(2, 2),
    flat_field=None,
)
img_syn = img_syn.reshape(img_syn.shape[0] // 2, 2, img_syn.shape[1] // 2, 2).sum(
    axis=(1, 3)
)
img_syn, _, _, img_syn_br_std = mrp.background_subtract(
    img_syn,
    reduction_factor=1,
    sigma=3.0,
    box_size=100,
)

img_syn = img_syn.astype(np.float32)
img_syn = img_syn * 0.002 + 100  # Attenuating for clouds
img_syn = img_syn * flat_frame.T  # Apply the flat fielding effect

# Subtract the background again
img_syn, _, _, img_syn_br_std = mrp.background_subtract(
    img_syn,
    reduction_factor=1,
    sigma=3.0,
    box_size=100,
)

np.savez(
    '/Users/liamrobinson/Documents/maintained-research/mirage/examples/14-digital-twin-2024/data/zim_corr.nogit.npz',
    img=img,
    img_syn=img_syn,
    br_std_img=img_br_std,
    br_std_synth=img_syn_br_std,
)


plt.figure()
mrp.plot_fits(img_syn)

plt.figure()
mrp.plot_fits(img)
plt.title(f'{i}')
plt.show()
