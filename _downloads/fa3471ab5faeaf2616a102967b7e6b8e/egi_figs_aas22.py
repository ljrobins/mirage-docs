"""
EGI Optimization
================
EGI optimization figures recreated in Python, originally published in :cite:p:robinson2022:.
"""

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

# %%
# Plotting the EGI of a cube
obj = mr.SpaceObject("cube.obj")
cpos = [7, 7, 5]

pl = pv.Plotter()
mrv.render_spaceobject(pl, obj, opacity=1, color="linen")
mrv.plot_egi(pl, obj, obj.egi, scale_opacity=False)
pl.camera.position = cpos
pl.show()

# %%
# Plotting the initial optimized EGI
n = 500
brdf = mr.Brdf("phong", cd=0.5, cs=0.5, n=10)
normal_candidates = mr.fibonacci_sample_sphere(n)
svb = mr.rand_unit_vectors(n)
ovb = mr.rand_unit_vectors(n)
g_candidates = brdf.compute_reflection_matrix(svb, ovb, normal_candidates)
lc = obj.convex_light_curve(brdf, svb, ovb)
a_candidates = np.expand_dims(mr.vendrow_fnnls(g_candidates, lc.flatten())[0], axis=1)
valid = a_candidates.flatten() > np.sum(a_candidates) / 100
egi_candidate = normal_candidates[valid, :] * a_candidates[valid, :]

pl = pv.Plotter()
mrv.render_spaceobject(pl, obj, opacity=0.5, color="linen")
mrv.plot_egi(pl, obj, egi_candidate, scale_opacity=True)
pl.camera.position = cpos
pl.show()

# %%
# Plotting the resampled EGI

resampled_n_candidates = []
nc = 100
for n_cand in mr.hat(egi_candidate):
    resampled_n_candidates.append(mr.rand_cone_vectors(n_cand, np.pi / 20, nc))
resampled_n_candidates = np.vstack(resampled_n_candidates)

g_candidates = brdf.compute_reflection_matrix(svb, ovb, resampled_n_candidates)
lc = obj.convex_light_curve(brdf, svb, ovb)
a_candidates = np.expand_dims(mr.vendrow_fnnls(g_candidates, lc.flatten())[0], axis=1)
valid = a_candidates.flatten() > np.sum(a_candidates) / 100
egi_candidate_resampled = resampled_n_candidates[valid, :] * a_candidates[valid, :]

pl = pv.Plotter()
mrv.render_spaceobject(pl, obj, color="linen", opacity=0.5)
mrv.scatter3(pl, 1.2 * resampled_n_candidates, color="r", point_size=3, opacity=0.5)
mrv.plot_egi(pl, obj, egi_candidate_resampled, scale_opacity=True)
pl.camera.position = cpos
pl.show()

# %%
# Plotting merged EGI

egi_merged = mr.merge_clusters(egi_candidate_resampled, np.pi / 10)

pl = pv.Plotter()
mrv.render_spaceobject(pl, obj, color="linen", opacity=0.5)
mrv.plot_egi(pl, obj, egi_merged)
pl.camera.position = cpos
pl.show()

# %%
# Plotting the reconstructed object
egi_merged -= np.sum(egi_merged, axis=0) / egi_merged.shape[0]
mr.tic()
h = mr.optimize_supports_little(egi_merged)
dt = mr.toc(return_elapsed_seconds=True)
obj_reconstructed = mr.construct_from_egi_and_supports(egi_merged, h)

pl = pv.Plotter(shape=(1, 2), window_size=(1000, 500))
pl.subplot(0, 0)
mrv.render_spaceobject(pl, obj, opacity=1, color="linen")
pl.add_text("Original", font="courier", position="upper_left")
pl.subplot(0, 1)
mrv.render_spaceobject(pl, obj_reconstructed, opacity=1, color="linen")
pl.add_text("Reconstructed", font="courier", position="upper_left")
pl.link_views()
pl.camera.position = cpos
pl.camera.zoom(1.4)
pl.show()


# %%
# Reconstructing with the non-sparse EGI

pl = pv.Plotter(shape=(1,4), window_size=(2000, 500))
pl.subplot(0, 0)
mrv.render_spaceobject(pl, obj, opacity=1, color="linen")
pl.add_text("Truth", font="courier", position="upper_left")

egi_candidate = mr.close_egi(egi_candidate)
egi_candidate_resampled = mr.close_egi(egi_candidate_resampled)

pl.subplot(0, 1)
mr.tic()
h_initial = mr.optimize_supports_little(egi_candidate)
dt_initial = mr.toc(return_elapsed_seconds=True)
obj_reconstructed_initial = mr.construct_from_egi_and_supports(egi_candidate, h_initial)
mrv.render_spaceobject(pl, obj_reconstructed_initial, opacity=1, color="linen")
pl.add_text(f"Initial EGI: {dt_initial:.2f}s", font="courier", position="upper_left")

pl.subplot(0, 2)
mr.tic()
h_resampled = mr.optimize_supports_little(egi_candidate_resampled)
dt_resampled = mr.toc(return_elapsed_seconds=True)
obj_reconstructed_resampled = mr.construct_from_egi_and_supports(egi_candidate_resampled, h_resampled)
mrv.render_spaceobject(pl, obj_reconstructed_resampled, opacity=1, color="linen")
pl.add_text(f"Resampled EGI: {dt_resampled:.2f}s", font="courier", position="upper_left")

pl.subplot(0, 3)
mrv.render_spaceobject(pl, obj_reconstructed, opacity=1, color="linen")
pl.add_text(f"Merged EGI: {dt:.2f}s", font="courier", position="upper_left")

pl.show()