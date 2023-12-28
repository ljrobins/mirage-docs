"""
Brightness Isosurfaces
======================

Plotting all orientations of a cube that produce a certian brightness value at a given phase angle
"""
import numpy as np
import pyvista as pv

import mirage as mr
import mirage.sim as mrs
import mirage.vis as mrv

# path = "/Users/liamrobinson/Documents/PyLightCurves/mlp_model_train_on_irradiance_cube_brdf_phong_cd_0.5_cs_0.5_n_5.0_layers_50_50_50_50_50_50_50_50_50_50.onnx"
# mdl = mrs.MLPBrightnessModel(path=path)
obj = mrv.SpaceObject("cube.obj")
mdl = lambda svb, ovb: obj.convex_light_curve(
    mr.Brdf("phong", cd=0.5, cs=0.5, n=5.0), svb, ovb
)
n = 150
d_min = -np.pi
grid = pv.ImageData(
    dimensions=(n, n, n),
    spacing=(abs(d_min) / n * 2, abs(d_min) / n * 2, abs(d_min) / n * 2),
    origin=(d_min, d_min, d_min),
)
rv = grid.points

li = np.tile(np.array([[1, 0, 0]]), (n**3, 1))
oi = mr.hat(np.tile(np.array([[1, 1, 0]]), (n**3, 1)))

dcms = mr.rv_to_dcm(rv)
lb = mr.stack_mat_mult_vec(dcms, li)
ob = mr.stack_mat_mult_vec(dcms, oi)
mr.tic()
bv = mdl(lb, ob)  # Brightness values at sampled orientations
mr.toc()

print(f"Mean brightness value: {np.mean(bv)}")

mesh = grid.contour([np.mean(bv)], bv, method="marching_cubes")
mtri = mesh.triangulate()
inds = np.tile([False, True, True, True], (mtri.faces.size // 4,))
mesh.smooth(n_iter=100, inplace=True)
F = mtri.faces[inds].reshape(-1, 3)
V = mtri.points
v2v = (V.shape[0] // 3, V.shape[0] // 9)
dist = mr.vecnorm(mesh.points)
mesh.points[dist.flatten() > np.pi, :] = np.nan

pl = pv.Plotter(lighting=None)
pl.set_background("black")

pl.add_mesh(
    mesh,
    color="linen",
    clim=(0, np.pi),
    show_scalar_bar=False,
    pbr=True,
    metallic=0.2,
    roughness=0.5,
    diffuse=1,
)

light = pv.Light((-2, 2, 0), (0, 0, 0), "white")
pl.add_light(light)

light = pv.Light((2, 0, 0), (0, 0, 0), (0.7, 0.0862, 0.0549))
pl.add_light(light)

light = pv.Light((0, 0, 10), (0, 0, 0), "white")
pl.add_light(light)

mrv.two_sphere(pl, np.pi, color="linen", opacity=0.1)

pl.show()

# %%
# Making a simpler render

pl = pv.Plotter()

pl.add_mesh(
    mesh,
    color="cornflowerblue",
    show_scalar_bar=False,
    pbr=True,
    metallic=0.2,
    roughness=0.5,
    diffuse=1,
)

mrv.two_sphere(pl, np.pi, color="linen", opacity=0.1)
pl.add_text("Rotation vector space", font_size=14, font="courier")
pl.show_bounds(
    grid="front",
    location="outer",
    all_edges=True,
)
pl.show()
