"""
BRDFs on the local hemisphere
=============================

BRDFs on a hemisphere centered around the surface normal vector
"""

import numpy as np
import pyvista as pv
import vtk

import mirage as mr
import mirage.vis as mrv

# %%
# Let's set up grids on the upper hemisphere of a unit sphere to compute the BRDF value at all those unit vectors
num = 200
el_space, az_space = np.linspace(0, np.pi / 2, num), np.linspace(0, 2 * np.pi, num)
el_grid, az_grid = np.meshgrid(el_space, az_space)

(xx, yy, zz) = mr.sph_to_cart(az_grid, el_grid, 0 * el_grid + 1)
O = np.hstack(
    (
        xx.reshape(((num**2, 1))),
        yy.reshape(((num**2, 1))),
        zz.reshape(((num**2, 1))),
    )
)
L = mr.hat(np.tile(np.array([[0, 1, 1]]), (num**2, 1)))
N = mr.hat(np.tile(np.array([[0, 0, 1]]), (num**2, 1)))

# %%
# Now we can iterate through a range of specular exponents and reflection of coeffients to visualize how the BRDF varies
pl = pv.Plotter(shape=(3, 3))
pl.set_background("white")
name = "phong"
for i, n in enumerate([2, 8, 20]):
    for j, cd in enumerate(np.linspace(0, 1, 3)):
        brdf = mr.Brdf(name, cd=cd, cs=1 - cd, n=n)
        b = brdf.eval(L, O, N).reshape(xx.shape)
        mesh = pv.StructuredGrid(xx * b, yy * b, zz * b)
        pl.subplot(i, j)
        pl.add_text(
            f"{name.capitalize()}: $cd={cd}$, $cs={1-cd}$, ${n=}$",
            font_size=16,
            font="courier",
            color="black",
        )
        pl.add_mesh(mesh, scalars=b.T, show_scalar_bar=False, cmap="isolum")
        mrv.plot_basis(pl, np.eye(3), color="gray")
        mrv.plot_arrow(
            pl,
            origin=[0, 0, 0],
            direction=L[0, :],
            scale=1,
            color="yellow",
            label="L",
        )
        mrv.plot_arrow(
            pl,
            origin=[0, 0, 0],
            direction=N[0, :],
            scale=1,
            color="k",
            label="N",
        )

pl.link_views()
pl.view_isometric()
pl.show()
