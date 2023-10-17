"""
Signed Distance Fields
======================

Plotting 2D and 3D Signed Distance Fields
"""

import matplotlib.pyplot as plt
import numpy as np
import pysdf
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

obj = mr.SpaceObject("stanford_bunny.obj").clean()
obj.v -= np.mean(obj.v, axis=0)

f = pysdf.SDF(obj.v, obj.f)

grid_width = 1.3 * np.max(mr.vecnorm(obj.v))
grid_density = 150
grid = pv.ImageData(
    spacing=(
        2 * grid_width / grid_density,
        2 * grid_width / grid_density,
        2 * grid_width / grid_density,
    ),
    origin=(-grid_width, -grid_width, -grid_width),
    dimensions=(grid_density, grid_density, grid_density),
)

sdf_vals = -f(grid.points)

# %%
# Plotting a 2D slide of the SDF

sdf_slice = sdf_vals.reshape(grid.dimensions)[:, grid_density // 2 + 10, :]

plt.figure()
plt.contour(
    sdf_slice,
    levels=np.linspace(np.min(sdf_slice), np.max(sdf_slice), 10),
    colors="k",
    extent=[-grid_width, grid_width, -grid_width, grid_width],
)
plt.imshow(
    np.flipud(sdf_slice),
    extent=[-grid_width, grid_width, -grid_width, grid_width],
    cmap="RdBu",
)
mrv.texit("Signed Distance Field", "x", "y", grid=False)
plt.colorbar(label="Signed Distance", cax=mrv.get_cbar_ax())
plt.tight_layout()
plt.show()


# %%
# Plotting the full 3D SDF

pl = pv.Plotter(window_size=(1000, 1000))

for cval in np.linspace(np.min(sdf_vals), -np.min(sdf_vals), 7):
    mesh1, mesh2 = grid.contour([cval], sdf_vals, method="flying_edges").clip(
        "y", origin=(0.0, 0.0, 0.0), return_clipped=True
    )
    if mesh1.points.shape[0] > 0:
        pl.add_mesh(
            mesh1,
            opacity=1.0,
            scalars=cval * np.ones(mesh1.n_points),
            cmap="coolwarm",
            scalar_bar_args=dict(title="SDF Value"),
            clim=[np.min(sdf_vals), -np.min(sdf_vals)],
            smooth_shading=True,
        )

pl.camera.position = (0.0, 0.6, 0.0)
pl.show()

# %%
# Animating an orbital path around the SDF

pl = pv.Plotter()

for cval in np.linspace(np.min(sdf_vals), -np.min(sdf_vals), 10):
    mesh = grid.contour([cval], sdf_vals, method="flying_edges")
    if mesh.points.shape[0] > 0:
        pl.add_mesh(
            mesh,
            opacity=0.1,
            scalars=cval * np.ones(mesh.n_points),
            cmap="coolwarm",
            scalar_bar_args=dict(title="SDF Value"),
        )

pl.open_gif("sdf_orbit.gif")
path = pl.generate_orbital_path(n_points=36, shift=mesh.length / 3)
pl.orbit_on_path(path, write_frames=True)
