# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "colorcet",
#     "mirage",
# ]
#
# [tool.uv.sources]
# mirage = { path = "../../", editable = true }
# ///
"""
Blinn-Phong BRDF Animated
=========================

Animating how the reflected radiance fraction changes
"""

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

brdf = mr.Brdf('Phong', 0.5, 0.2, 10)

num = 200
arrow_scale = 0.5
el_space, az_space = np.linspace(0.1, np.pi / 2, num), np.linspace(0, 2 * np.pi, num)
el_grid, az_grid = np.meshgrid(el_space, az_space)

pl = pv.Plotter()

pl.open_gif('out/brdf_anim.gif', fps=20)
nframes = 150
t = np.linspace(0, 2 * np.pi, nframes, endpoint=False)

mrv.plot_basis(
    pl, np.eye(3), color='cornflowerblue', labels=['U', 'V', 'N'], scale=arrow_scale
)

(xx, yy, zz) = mr.sph_to_cart(az_grid, el_grid, 0 * el_grid + 1)
O = np.hstack(
    (
        xx.reshape(((num**2, 1))),
        yy.reshape(((num**2, 1))),
        zz.reshape(((num**2, 1))),
    )
)
N = mr.hat(np.tile(np.array([[0, 0, 1]]), (num**2, 1)))

theta = np.linspace(0, 2 * np.pi, nframes, endpoint=False)

for i in range(nframes):
    L = mr.hat(
        np.tile(
            np.array(
                [
                    [
                        np.sin(theta[i]),
                        np.cos(theta[i]),
                        1 + 0.2 * np.cos(theta[i]) + 0.3 * np.sin(3 * theta[i]),
                    ]
                ]
            ),
            (num**2, 1),
        )
    )
    b = (brdf.eval(L, O, N) * mr.dot(L, N) * mr.dot(O, N)).reshape(xx.shape)
    mesh = pv.StructuredGrid(xx * b, yy * b, zz * b)
    pl.add_text(f'{brdf.name.upper()}', font_size=18, font='courier', color='black')
    pl.add_mesh(
        mesh,
        scalars=b.T.flatten(),
        show_scalar_bar=False,
        cmap='isolum',
        smooth_shading=True,
        name='brdf_vals',
    )

    mrv.plot_arrow(
        pl,
        origin=[0, 0, 0],
        direction=L[0, :],
        scale=arrow_scale,
        color='yellow',
        label='L',
        name='L',
    )

    pl.camera.position = (2.0, 0.0, 3.0)
    pl.camera.focal_point = (0.0, 0.0, 0.0)

    pl.write_frame()

pl.close()
