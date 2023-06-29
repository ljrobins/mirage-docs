"""
Earth Albedo BRDF
=================

Modeling the incident radiation at a spacecraft due to reflected sunlight from the Earth
"""

# %%
# Useful papers not cited below:
# `Kuvyrkin 2016 <https://arc.aiaa.org/doi/pdf/10.2514/1.A33349>`_
# `Strahler 1999 <https://modis.gsfc.nasa.gov/data/atbd/atbd_mod09.pdf>`_

# %%
# Let's first load the coefficient arrays :math:`f_{iso}`, :math:`f_{geo}`, and :math:`f_{vol}` from file

import matplotlib.pyplot as plt
import numpy as np
import pyspaceaware as ps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json

save_dict = ps.load_albedo_file()
fiso_map = np.array(save_dict["fiso_map"])
fgeo_map = np.array(save_dict["fgeo_map"])
fvol_map = np.array(save_dict["fvol_map"])
lat_geod_grid = np.array(save_dict["lat_geod_grid"])
lon_grid = np.array(save_dict["lon_grid"])
lat_geod_space = lat_geod_grid[:, 0]
lon_space = lon_grid[0, :]
mapshape = lon_grid.shape

# %%
# The surface BRDF function in `Blanc 2014 <https://hal-mines-paristech.archives-ouvertes.fr/file/index/docid/1024989/filename/2014_igarss_albedo_blanc.pdf>`_

pws = (
    lambda ts, fiso, fgeo, fvol: fiso
    + fvol * (-0.007574 - 0.070987 * ts**2 + 0.307588 * ts**3)
    + fgeo * (-1.284909 - 0.166314 * ts**2 + 0.041840 * ts**3)
)
pbs = (
    lambda ts, fiso, fgeo, fvol: fiso
    + 0.189184 * fvol
    - 1.377622 * fgeo
)
albedo = lambda ts, fiso, fgeo, fvol: 0.5 * pws(
    ts, fiso, fgeo, fvol
) + 0.5 * pbs(ts, fiso, fgeo, fvol)

# %%
# Now we define the date to evaluate the reflected albedo irradiance at and the ECEF position of the satellite
date = ps.now()
datestr = f'{date.strftime("%Y-%m-%d %H:%M:%S")} UTC'
sat_pos_ecef = (6378 + 4e4) * ps.hat(np.array([[1, 1, 0]]))

# %%
# Now we identify all the useful geometry: the ECEF positions of the grid cells, the Sun vector, the solar zenith angle at each grid cell, and the albedo at each point
ecef_grid = ps.lla_to_itrf(
    lat_geod=lat_geod_grid.flatten(),
    lon=lon_grid.flatten(),
    a=0 * lon_grid.flatten(),
)
eci_to_ecef_rotm = ps.ecef_to_eci(date).T
sun_ecef_hat = (
    eci_to_ecef_rotm @ ps.hat(ps.sun(ps.date_to_jd(date))).T
).T
sun_dir = np.tile(sun_ecef_hat, (ecef_grid.shape[0], 1))
solar_zenith = np.arccos(ps.dot(ps.hat(ecef_grid), sun_dir))
solar_zenith_grid = solar_zenith.reshape(mapshape)
albedo_grid = albedo(solar_zenith_grid, fiso_map, fgeo_map, fvol_map)

# %%
# For fun, let's classify the types of twilight to plot later
solar_type_grid = np.zeros_like(solar_zenith_grid)
solar_type_grid[
    (solar_zenith_grid > np.pi / 2)
    & (solar_zenith_grid < np.pi / 2 + np.deg2rad(18))
] = 3
# Astronomical twilight
solar_type_grid[
    (solar_zenith_grid > np.pi / 2)
    & (solar_zenith_grid < np.pi / 2 + np.deg2rad(12))
] = 2
# Nautical twilight
solar_type_grid[
    (solar_zenith_grid > np.pi / 2)
    & (solar_zenith_grid < np.pi / 2 + np.deg2rad(6))
] = 1
# Civil twilight
solar_type_grid[solar_zenith_grid > np.pi / 2 + np.deg2rad(16)] = 4
# Night

# %%
# Computing which grid cells are visible from the satellite
surf_to_sat = sat_pos_ecef - ecef_grid
surf_to_sat_dir = ps.hat(surf_to_sat)
surf_to_sat_rmag_m_grid = 1e3 * ps.vecnorm(surf_to_sat).reshape(
    mapshape
)
tosat_to_normal_ang = np.arccos(
    ps.dot(ps.hat(ecef_grid), surf_to_sat_dir)
)
tosat_to_normal_grid = tosat_to_normal_ang.reshape(mapshape)
pt_visible_from_sat = tosat_to_normal_grid < np.pi / 2

# Visible and illuminated points
ill_and_vis = pt_visible_from_sat & (solar_type_grid == 0)
loss_at_surf = (
    np.cos(solar_zenith_grid) * np.cos(tosat_to_normal_grid)
) * ill_and_vis
is_ocean = np.abs(albedo_grid - albedo_grid[0, 0]) < 1e-8
loss_at_surface_specular = (
    ps.brdf_phong(
        sun_dir, surf_to_sat_dir, ps.hat(ecef_grid), 0, 0.4, 10
    ).reshape(mapshape)
    * is_ocean
)

obs_type_grid = np.zeros_like(solar_zenith_grid)
obs_type_grid[pt_visible_from_sat] = 1
obs_type_grid[(solar_type_grid == 0) & ~pt_visible_from_sat] = 2
obs_type_grid[ill_and_vis] = 3

# %%
# Computes the areas of each grid cell
dp, dt = (
    lat_geod_space[1] - lat_geod_space[0],
    lon_space[1] - lon_space[0],
)
cell_area_grid = np.tile(
    np.array(
        [
            ps.lat_lon_cell_area((p + dp, p), (0, dt))
            for p in lat_geod_space
        ]
    ).reshape(-1, 1),
    (1, lon_space.size),
)

# %%
# Computing Lambertian reflection (for the land) and Phong reflection (for the ocean) from each grid cell
rmag_loss_grid = 1 / surf_to_sat_rmag_m_grid**2
irrad_from_surf = (
    1361
    * rmag_loss_grid
    * cell_area_grid
    * (loss_at_surf * albedo_grid + loss_at_surface_specular)
)
print(f"{np.sum(irrad_from_surf):.2e}")

# %%
# Let's compare with the implementation in the pyspaceaware package
ps.tic()
alb_irrad = ps.albedo_irradiance(date, sat_pos_ecef)
ps.toc()
print(f"{alb_irrad:.2e}")

# %%
# Defining a few useful functions to simplify the plotting process
bcmap = "PiYG"


def plt_map_under(ax):
    ax.imshow(
        albedo_grid, cmap="gray", alpha=0.3, extent=[-180, 180, -90, 90]
    )


def get_cbar_ax(ax):
    return make_axes_locatable(ax).append_axes(
        "right", size="5%", pad=0.05
    )


def label_map_axes(ax):
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")


# %%
# Plotting the albedo across the grid
fig, ax = plt.subplots()
plt.imshow(albedo_grid, cmap="PuBuGn_r", extent=[-180, 180, -90, 90])
ax.set_title("March Mean Albedo")
cb = plt.colorbar(label="Surface Albedo", cax=get_cbar_ax(ax))
label_map_axes(ax)
plt.tight_layout()
plt.show()

# %%
# Plotting the solar zenith angle
fig, ax = plt.subplots()
plt.colorbar(
    ax.imshow(
        solar_zenith_grid, cmap="Blues", extent=[-180, 180, -90, 90]
    ),
    label="Solar zenith angle [rad]",
    cax=get_cbar_ax(ax),
)
plt_map_under(ax)
ax.set_title(f"Solar Zenith Angles: {datestr}")
label_map_axes(ax)
plt.tight_layout()
plt.show()

# %%
# Plotting the twilight types
fig, ax = plt.subplots()
cb = plt.colorbar(
    ax.imshow(
        solar_type_grid, cmap="Blues", extent=[-180, 180, -90, 90]
    ),
    cax=get_cbar_ax(ax),
)
cb.set_ticks(range(5))
cb.set_ticklabels(
    [
        "Day",
        "Civil twilight",
        "Nautical twilight",
        "Astronomical twilight",
        "Night",
    ]
)
plt_map_under(ax)
ax.set_title(f"Twilight Types: {datestr}")
label_map_axes(ax)
plt.tight_layout()
plt.show()

# %%
# Plotting grid cell visibility and illumination conditions
fig, ax = plt.subplots()
cb = plt.colorbar(
    ax.imshow(
        obs_type_grid,
        cmap=plt.cm.get_cmap("Paired", 4),
        interpolation="nearest",
        extent=[-180, 180, -90, 90],
    ),
    cax=get_cbar_ax(ax),
)
cb.set_ticks(range(4))
cb.set_ticklabels(
    [
        "Not visible or illum.",
        "Visible not illum.",
        "Illum. not visible",
        "Illum. and visible",
    ]
)
plt_map_under(ax)
ax.set_title(f"Observation Conditions: {datestr}")
label_map_axes(ax)
plt.tight_layout()
plt.show()

# %%
# BRDF kernel values at each point
fig, ax = plt.subplots()
plt.colorbar(
    ax.imshow(
        loss_at_surf + loss_at_surface_specular,
        cmap="Blues",
        extent=[-180, 180, -90, 90],
    ),
    label="",
    cax=get_cbar_ax(ax),
)
plt_map_under(ax)
ax.set_title(f"BRDF Kernel: {datestr}")
label_map_axes(ax)
plt.tight_layout()
plt.show()

# %%
# Plotting the areas of each grid cell
fig, ax = plt.subplots()
plt.colorbar(
    ax.imshow(
        cell_area_grid, cmap="Blues", extent=[-180, 180, -90, 90]
    ),
    label="$[m^2]$",
    cax=get_cbar_ax(ax),
)
plt_map_under(ax)
ax.set_title(f"Cell Areas: {datestr}")
label_map_axes(ax)
plt.tight_layout()
plt.show()

# %%
# Plotting the irradiance from each grid cell
fig, ax = plt.subplots()
plt_map_under(ax)
plt.colorbar(
    ax.imshow(irrad_from_surf, cmap="hot", extent=[-180, 180, -90, 90]),
    label=r"$\left[W/m^2\right]$",
    cax=get_cbar_ax(ax),
)
ax.set_title(rf"Reflected Irradiance: {datestr}")
label_map_axes(ax)
plt.tight_layout()
plt.show()
