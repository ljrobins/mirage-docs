"""
Horizon Masks
=================

Builds a terrain-driven horizon mask for a given station and displays the result
"""


import numpy as np
import pyvista as pv
import terrainman as tm

# %%
# Defining the station at Katmandu, where ``station.name`` informs the name of the resulting mask file
import pyspaceaware as ps
import pyspaceaware.vis as psv

station = ps.Station(
    preset="pogs",
    lat_deg=27.7172,
    lon_deg=85.3240,
    alt_km=0,
    name="Katmandu",
    altitude_reference="terrain",
)

# %%
# Loads a terrain tile containing the station and create a mask for the station location

tile = tm.TerrainDataHandler().load_tiles_containing(
    station.lat_geod_deg, station.lon_deg
)
mask = ps.HorizonMask(
    station.lat_geod_rad,
    station.lon_rad,
    station.name,
    mask_resolution=2000,
)

# %%
# Build a tile from the raw tile data
lat_grid, lon_grid = tile.lat_grid, tile.lon_grid
elev_grid = tile.elev_grid / 1e3
itrf_terrain = ps.lla_to_itrf(
    np.deg2rad(lat_grid).flatten(),
    np.deg2rad(lon_grid).flatten(),
    elev_grid.flatten() + ps.geoid_height_at_lla(station.lat_geod_rad, station.lon_rad),
)

# %%
# Convert the terrain data into East North Up (ENU) coordinates and plot the result
enu_terrain = (ps.ecef_to_enu(station.itrf) @ (itrf_terrain - station.itrf).T).T
dem = pv.StructuredGrid(
    enu_terrain[:, 0].reshape(elev_grid.shape),
    enu_terrain[:, 1].reshape(elev_grid.shape),
    enu_terrain[:, 2].reshape(elev_grid.shape),
)
dem["Elevation [km]"] = elev_grid.flatten(order="F")

enu_rays = ps.az_el_to_enu(mask.az, mask.el)

pl = pv.Plotter()
pl.set_background("black")
pl.add_mesh(
    dem,
    smooth_shading=True,
    scalars="Elevation [km]",
    cmap="terrain",
    opacity=1.0,
    show_scalar_bar=True,
)

psv.scatter3(pl, enu_rays, color="w", show_scalar_bar=False)
pl.add_text("Katmandu Horizon Mask", font="courier")

path = pv.Polygon(
    center=(0.0, 0.0, 0.0),
    radius=0.0001,
    normal=(0.0, 0.0, 1.0),
    n_sides=200,
)
pl.open_gif("orbit_horizon.gif", fps=30)
for campos in path.points:
    pl.camera.position = campos
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.write_frame()
pl.close()
