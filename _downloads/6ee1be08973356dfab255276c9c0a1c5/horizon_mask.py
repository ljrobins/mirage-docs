"""
Horizon Masks
=================

Builds a terrain-driven horizon mask for a given station and displays the result
"""


# %%
# Defining the station at Katmandu, where ``station.name`` informs the name of the resulting mask file
import pyspaceaware as ps
import numpy as np
import pyvista as pv
import terrainman as tm

station = ps.Station(
    preset="pogs",
    lat_deg=27.7172,
    lon_deg=85.3240,
    alt_km=2.5,
    name="Katmandu",
    use_terrain_data=True,
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
)

# %%
# Build an interpolated from from the raw tile data
sz, deg_radius = 3000, 1.0
lat_space = (station.lat_geod_deg + deg_radius) - np.linspace(0, 2 * deg_radius, sz)
lon_space = (station.lon_deg - deg_radius) + np.linspace(0, 2 * deg_radius, sz)
lat_grid, lon_grid = np.meshgrid(lat_space, lon_space)
elev_grid = tile.interpolate(lat_grid, lon_grid) / 1e3
itrf_terrain = ps.lla_to_itrf(
    np.deg2rad(lat_grid).flatten(),
    np.deg2rad(lon_grid).flatten(),
    elev_grid.flatten(),
)

# %%
# Convert the terrain data into East North Up (ENU) coordinates and plot the result
enu_terrain = (ps.ecef_to_enu(station.ecef) @ (itrf_terrain - station.ecef).T).T
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
    opacity=0.6,
    show_scalar_bar=False,
)

ps.scatter3(pl, enu_rays, color="w", show_scalar_bar=False)
pl.add_text("Katmandu Horizon Mask", font="courier")

# pl.show(auto_close=False)
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
