"""
Horizon Masked Observations
===========================
"""

import datetime

import numpy as np
import pyvista as pv
import terrainman as tm

import mirage as mr
import mirage.vis as mrv

# %%
# Let's define an observation station right before an ISS pass
date_start = datetime.datetime(2023, 5, 12, 0, 37, 0, tzinfo=datetime.timezone.utc)
dates = date_start + mr.minutes(np.linspace(0, 11, 100))
station = mr.Station(
    preset="pogs",
    lat_deg=43.65311150689344,
    lon_deg=-70.19252101245867,
    alt_km=0.0,
    name="Peaks_Island_Maine",
    altitude_reference="terrain",
)

# %%
# And grab the ISS, which will propagate using the closest available TLEs for accuracy
obj = mr.SpaceObject("tess.obj", identifier=25544)
brdf = mr.Brdf("phong")

# %%
# We can now apply a bunch of constraints to the observation, including a horizon mask for the local terrain
station.constraints = [
    mr.HorizonMaskConstraint(station),
]
tile = tm.TerrainDataHandler().load_tiles_containing(
    station.lat_geod_deg, station.lon_deg
)
mask = mr.HorizonMask(
    station.lat_geod_rad,
    station.lon_rad,
    station.name,
)
sz, deg_radius = 3000, 1.0
lat_space = (station.lat_geod_deg + deg_radius) - np.linspace(0, 2 * deg_radius, sz)
lon_space = (station.lon_deg - deg_radius) + np.linspace(0, 2 * deg_radius, sz)
lat_grid, lon_grid = np.meshgrid(lat_space, lon_space)
elev_grid = tile.interpolate(lat_grid, lon_grid) / 1e3
elev_grid += mr.geoid_height_at_lla(station.lat_geod_rad, station.lon_rad)
itrf_terrain = mr.lla_to_itrf(
    np.deg2rad(lat_grid).flatten(),
    np.deg2rad(lon_grid).flatten(),
    elev_grid.flatten(),
)

# %%
# We can now define the object's attitude profile and observe a light curve

obj_attitude = mr.RbtfAttitude(
    w0=0.000 * np.array([0, 1, 1]),
    q0=mr.hat(np.array([0, 0, 0, 1])),
    itensor=obj.principal_itensor,
)

obj_eci = obj.propagate(dates)
station_eci = station.j2000_at_dates(dates)
look_dir_eci = mr.hat(obj_eci - station_eci)
horizon_constraint = station.eval_constraints(look_dir_eci=look_dir_eci, dates=dates)

# %%
# We can now plot an animation of the pass with the horizon mask superimposed on the local terrain
enu_terrain = (mr.ecef_to_enu(station.itrf) @ (itrf_terrain - station.itrf).T).T
dem = pv.StructuredGrid(
    enu_terrain[:, 0].reshape(elev_grid.shape),
    enu_terrain[:, 1].reshape(elev_grid.shape),
    enu_terrain[:, 2].reshape(elev_grid.shape),
)
dem["Elevation [km]"] = elev_grid.flatten(order="F")
dem["Latitude"] = lat_grid.flatten(order="F")
dem["Longitude"] = lon_grid.flatten(order="F")

enu_rays = mr.az_el_to_enu(mask.az, mask.el)

pre_render_fcn = lambda pl: (
    pl.add_mesh(
        dem,
        smooth_shading=True,
        scalars="Elevation [km]",
        opacity=0.5,
        show_scalar_bar=False,
    ),
    mrv.plot3(pl, enu_rays, color="c", line_width=5),
    mrv.plot3(
        pl,
        mr.az_el_to_enu(*station.eci_to_az_el(dates, look_dir_eci)),
        line_width=5,
    ),
)


def render_fcn(pl: pv.Plotter, i: int, dates=None, horizon_constraint=None):
    mrv.scatter3(
        pl,
        obj_enu[i, :].reshape((1, 3)),
        point_size=40,
        color="g" if horizon_constraint[i] else "r",
        name="obj_pos",
        lighting=False,
    )
    pl.camera.focal_point = obj_enu[i, :].flatten()
    pl.camera.position = (0.0, 0.0, 0.0)
    pl.camera.clipping_range = (0.01, 50e3)
    pl.camera.up = (0.0, 0.0, 1.0)
    pl.add_text(
        f'Observing {obj.satnum}\n{dates[i].strftime("%m/%d/%Y, %H:%M:%S")} UTC\nAZ = {np.rad2deg(az[i]):.2f} deg\nEL = {np.rad2deg(el[i]):.2f} deg',
        name="utc_str",
        font="courier",
        color="white",
    )


az, el = station.eci_to_az_el(dates, look_dir_eci)
obj_enu = mr.az_el_to_enu(az, el)

mrv.render_video(
    pre_render_fcn,
    lambda pl, i: render_fcn(pl, i, dates, horizon_constraint),
    lambda pl, i: None,
    dates.size,
    "maine_iss_pass.gif",
    background_color="k",
)
