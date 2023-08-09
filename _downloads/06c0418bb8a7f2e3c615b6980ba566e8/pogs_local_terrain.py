"""
POGS Local Terrain
==================
Plots a map of the local terrain around the Purdue Optical Ground Station
"""

import pygmt

import pyspaceaware as ps

projection = "Cyl_stere/30/-20/12c"
station = ps.Station(preset="pogs")
dl = 1.2
dlbig = 10
sm_region = [
    station.lon_deg - dl,
    station.lon_deg + dl,
    station.lat_geod_deg - dl,
    station.lat_geod_deg + dl,
]
lg_region = [
    station.lon_deg - dlbig,
    station.lon_deg + dlbig,
    station.lat_geod_deg - dlbig,
    station.lat_geod_deg + dlbig,
]
lg_region = [round(x) for x in lg_region]


grid = pygmt.datasets.load_earth_relief(resolution="15s", region=sm_region)
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])

fig = pygmt.Figure()
fig.grdimage(
    grid=dgrid,
    cmap="gray",
    projection=projection,
    transparency=50,
)
fig.grdimage(
    grid=grid,
    cmap="elevation",
    projection=projection,
    frame=True,
    transparency=50,
)
fig.grdcontour(
    annotation=500,
    interval=250,
    grid=grid,
)

fig.plot(
    x=station.lon_deg, y=station.lat_geod_deg, pen="1p", style="c0.2c", fill="red3"
)
fig.text(
    text="POGS",
    x=station.lon_deg,
    y=station.lat_geod_deg,
    font="12p,Courier-Bold,red3",
    justify="BL",
    offset="0.15/0.15",
    fill="gray",
)

fig.coast(
    borders=[1, 2],
    shorelines="3,gray",
)
fig.colorbar(frame=["x+lElevation [m]"])

with fig.inset(
    position="jBL+o0.5c/0.2c", box="+pblack", region=lg_region, projection="M3c"
):
    # Use a plotting method to create a figure inside the inset.
    fig.coast(
        land="gray",
        borders=[1, 2],
        shorelines="1/thin",
        water="white",
        # Use dcw to selectively highlight an area
        dcw="US.NM+gred",
    )
    fig.plot(
        x=station.lon_deg,
        y=station.lat_geod_deg,
        pen="0.5p",
        style=f"s{dl/2}c",
        fill="white",
    )

fig.show()
