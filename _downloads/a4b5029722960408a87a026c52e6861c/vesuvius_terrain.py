"""
Terrain Tiles
=============
"""


import numpy as np
import pyvista as pv
import terrainman as tm

import pyspaceaware as ps
import pyspaceaware.vis as psv

lat_deg, lon_deg = 40.8224, 14.4289
lat_rad, lon_rad = np.deg2rad(lat_deg), np.deg2rad(lon_deg)

tile = tm.TerrainDataHandler().load_tiles_containing(lat_deg, lon_deg)

elev_grid = tile.elev_grid / 1e3 + ps.geoid_height_at_lla(lat_rad, lon_rad)
itrf_terrain = ps.lla_to_itrf(
    np.deg2rad(tile.lat_grid).flatten(),
    np.deg2rad(tile.lon_grid).flatten(),
    elev_grid.flatten(),
)

dem = pv.StructuredGrid(
    itrf_terrain[:, 0].reshape(elev_grid.shape),
    itrf_terrain[:, 1].reshape(elev_grid.shape),
    itrf_terrain[:, 2].reshape(elev_grid.shape),
)
dem["Elevation [km]"] = elev_grid.flatten(order="f")
dem["Latitude"] = tile.lat_grid.flatten(order="f")
dem["Longitude"] = tile.lon_grid.flatten(order="f")

pl = pv.Plotter()
pl.add_mesh(
    dem,
    smooth_shading=True,
    scalars="Elevation [km]",
    opacity=0.6,
    show_scalar_bar=True,
)
psv.plot_earth(
    pl, mode="ecef", date=ps.utc(2023, 12, 9, 12), high_def=True, ocean_floor=False
)
pl.camera.focal_point = np.mean(itrf_terrain, axis=0)
pl.camera.position = 6800 * ps.hat(np.mean(itrf_terrain, axis=0))

pl.show()
