"""
Parametric Box-Wing Inversion
=============================

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import mirage as mr

orbit_sol_path = '/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/pogs/2022/2022-09-18/ProcessedData/Fitted_Orbits/OrbitSolutions.mat'
orbit_mat = loadmat(orbit_sol_path)
sol_mat = {
    k: orbit_mat['CuratedObjects'][0][k][0].squeeze()
    for k in orbit_mat['CuratedObjects'][0].dtype.names
}
sol_mat['JD'] = np.sum(sol_mat['JD'], axis=0)
integration_time_s = 4.0
dates = mr.jd_to_date(sol_mat['JD'])
epsecs = mr.date_to_epsec(dates)
data_index = np.arange(epsecs.size)

station = mr.Station()

obj = mr.SpaceObject('matlib_gps_iii.obj', identifier='NAVSTAR 80 (USA 309)')
r_obj_j2k = obj.propagate(dates)

svi = mr.sun(dates)

brdf_name = 'cook-torrance'

brdf_wing = mr.Brdf(brdf_name, cd=0.0, cs=0.7, n=0.11)
brdf_box = mr.Brdf(brdf_name, cd=0.1, cs=0.1, n=0.06)

station_pos_eci = station.j2000_at_dates(dates)
object_pos_eci = obj.propagate(dates)
ovi = mr.hat(station_pos_eci - object_pos_eci)
z_obs = mr.angle_between_vecs(station_pos_eci, -ovi)

el = np.rad2deg(np.pi / 2 - z_obs).flatten()
print(el[0], el.max(), el.min())

n_hat_pan = svi
n_hat_box = -mr.hat(object_pos_eci)

rmag = mr.vecnorm(sol_mat['r'].T).flatten()


def controlled_box_wing_normalized_light_curve(
    svi: np.ndarray,
    ovi: np.ndarray,
    brdf_box: mr.Brdf,
    brdf_wing: mr.Brdf,
    a_pan: float,
    a_bus: float,
) -> np.ndarray:
    lc_norm_box = (
        a_bus * brdf_box.eval_normalized_brightness(L=svi, O=ovi, N=n_hat_box).flatten()
    )
    lc_norm_wings = (
        a_pan
        * brdf_wing.eval_normalized_brightness(L=svi, O=ovi, N=n_hat_pan).flatten()
    )
    lc_simp_norm = lc_norm_box + lc_norm_wings
    return lc_simp_norm


def normalized_brightness_to_adu(
    lc_norm: np.ndarray,
    station: mr.Station,
    dates: np.ndarray,
    z_obs: np.ndarray,
    rmag: np.ndarray,
    integration_time_s: float,
):
    isun = mr.total_solar_irradiance_at_dates(dates)
    lc_irrad = lc_norm / (rmag * 1e3) ** 2 * isun

    lambdas = np.linspace(300, 1000, 100)
    spec = mr.scaled_solar_spectrum(1, lambdas=lambdas)
    atmos_trans = mr.atmospheric_transmission(lambdas, z_obs, station.alt_km)

    spec_atmos = spec * atmos_trans
    lc_irrad_spec = lc_irrad.reshape(-1, 1) * spec_atmos
    g_mag = mr.apparent_magnitude_in_band(lambdas, lc_irrad_spec, band='G')

    count_rate = mr.integrated_spectrum(
        station,
        z_obs,
        lambdas=lambdas,
        spectrum=spec_atmos,
        include_atmospheric_transmission=False,
    )
    lc_adu = count_rate * integration_time_s * lc_irrad

    return g_mag, lc_adu


a_pan = 28.5212  # m^2
x_scale = 2.4638  # m
y_scale = 3.4036  # m
z_scale = 1.778  # m
a_bus = x_scale * z_scale

lc_simp_norm = controlled_box_wing_normalized_light_curve(
    svi, ovi, brdf_box, brdf_wing, a_bus=a_bus, a_pan=a_pan
)
lc_simp_g_mag, lc_simp_adu = normalized_brightness_to_adu(
    lc_simp_norm, station, dates, z_obs, rmag, integration_time_s
)

plt.figure(figsize=(4, 4))
plt.scatter(
    epsecs / 3600, sol_mat['brightness'], label='Observed: NAVSTAR 80 (USA 309)', s=2
)
plt.plot(epsecs / 3600, lc_simp_adu, c='k', label='Fit (Cook-Torrance)')
plt.title('Box-Wing Fit: ADU')
plt.xlabel(f'Hours past {dates[0].strftime("%Y-%m-%d %H:%M:%S")} UTC')
plt.ylabel('Flux [ADU]')
plt.grid()
plt.tight_layout()

plt.figure(figsize=(4, 4))
plt.scatter(
    epsecs / 3600,
    mr.irradiance_to_apparent_magnitude(sol_mat['flux']),
    label='Observed: NAVSTAR 80 (USA 309)',
    s=2,
)
plt.plot(
    epsecs / 3600,
    lc_simp_g_mag,
    c='k',
    label='Fit (Cook-Torrance)',
)
plt.gca().invert_yaxis()
plt.title('Box-Wing Fit: Apparent Magnitude')
plt.xlabel(f'Hours past {dates[0].strftime("%Y-%m-%d %H:%M:%S")} UTC')
plt.ylabel('Apparent Gaia G-Band Magnitude')
plt.grid()

plt.tight_layout()
plt.legend()
plt.show()
