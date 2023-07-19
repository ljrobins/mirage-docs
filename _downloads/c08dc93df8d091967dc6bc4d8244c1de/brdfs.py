"""
BRDFs on the local hemisphere
=============================

BRDFs on a hemisphere centered around the surface normal vector
"""

import numpy as np
import pyspaceaware as ps
import vtk
import pyvista as pv

# %%
# Let's set up grids on the upper hemisphere of a unit sphere to compute the BRDF value at all those unit vectors
num = 200
el_space, az_space = np.linspace(0, np.pi / 2, num), np.linspace(0, 2 * np.pi, num)
el_grid, az_grid = np.meshgrid(el_space, az_space)

# %%
# Now we can iterate through a range of specular exponents and BRDFs to visualize how the BRDF varies
pl = pv.Plotter(shape=(2, 2))
pl.set_background("white")
for i, n in enumerate([2, 8]):
    for j, name in enumerate(ps.lighting._SUPPORTED_BRDFS):
        brdf = ps.Brdf(name, cd=0.8, cs=0.2, n=n)
        (xx, yy, zz) = ps.sph_to_cart(az_grid, el_grid, 0 * el_grid + 1)
        O = np.hstack(
            (
                xx.reshape(((num**2, 1))),
                yy.reshape(((num**2, 1))),
                zz.reshape(((num**2, 1))),
            )
        )
        L = ps.hat(np.tile(np.array([[0, 1, 1]]), (num**2, 1)))
        N = ps.hat(np.tile(np.array([[0, 0, 1]]), (num**2, 1)))
        b = brdf.eval(L, O, N).reshape(xx.shape)
        mesh = pv.StructuredGrid(xx * b, yy * b, zz * b)
        pl.subplot(i, j)
        pl.add_text(
            f"{name.capitalize()}: ${n=}$", font_size=12, font="arial", color="black"
        )
        pl.add_mesh(mesh, scalars=b.T, show_scalar_bar=False, cmap="isolum")
        ps.plot_basis(pl, np.eye(3), color="gray")
        ps.plot_arrow(
            pl,
            origin=[0, 0, 0],
            direction=L[0, :],
            scale=1,
            color="yellow",
            label="L",
        )
        ps.plot_arrow(
            pl,
            origin=[0, 0, 0],
            direction=N[0, :],
            scale=1,
            color="red",
            label="N",
        )

pl.link_views()
pl.view_isometric()
pl.show()
