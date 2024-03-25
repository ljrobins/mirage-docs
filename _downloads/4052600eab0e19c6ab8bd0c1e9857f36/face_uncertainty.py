"""
Face Uncertainty
================

More rigorously defining the uncertainty in the faces of an estimated object
"""

# %%
# Setting up the observation conditions

# isort: off

import matplotlib.pyplot as plt
import numpy as np
import vtk
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

itensor = np.diag([1.0, 2.0, 3.0])
w0 = 9e-2 * mr.hat(np.array([[1.0, 2.0, 0.0]]))
idate = mr.utc(2023, 1, 1, 5)
obs_time = mr.hours(4)
obs_dt = mr.seconds(10)

obj_file = "collapsed_cube.obj"

station = mr.Station(preset="pogs")
brdf = mr.Brdf(name="phong", cd=0.5, cs=0.3, n=10)
obj = mr.SpaceObject(obj_file, identifier="goes 15")

dates, epsecs = mr.date_arange(idate, idate + obs_time, obs_dt, return_epsecs=True)

attitude = mr.RbtfAttitude(w0=w0, q0=np.array([[0.0, 0.0, 0.0, 1.0]]), itensor=itensor)
q_of_t, w_of_t = attitude.propagate(epsecs)

# %%
# Computing the light curve

lc_ccd_signal_sampler, aux_data = station.observe_light_curve(
    obj, attitude, brdf, dates, use_engine=False, model_scale_factor=1
)

# %%
# Inversion from the noisy light curve

sun_body = aux_data["sun_vector_object_body"]
obs_body = aux_data["observer_vector_object_body"]

lc_ccd_signal = lc_ccd_signal_sampler()
lc_noisy_irrad = lc_ccd_signal / (aux_data["sint"] * station.telescope.integration_time)
lc_noisy_unit_irrad = (
    lc_noisy_irrad
    * (aux_data["rmag_station_to_sat"] * 1e3) ** 2
    / mr.AstroConstants.sun_irradiance_vacuum
)

egi_opt_initial, egi_opt = mr.optimize_egi(
    lc_noisy_unit_irrad[~lc_noisy_unit_irrad.mask],
    sun_body[~lc_noisy_unit_irrad.mask, :],
    obs_body[~lc_noisy_unit_irrad.mask, :],
    brdf,
    merge_iter=1,
    merge_angle=np.pi / 6,
    return_all=True,
    num_candidates=1000,
)
rec_obj = mr.construct_mesh_from_egi(mr.close_egi(egi_opt))

# %%
# Plotting the reconstructed and truth objects

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
mrv.render_spaceobject(pl, obj)
pl.subplot(0, 1)
mrv.render_spaceobject(pl, rec_obj)
pl.show()

# %%
# Plotting the reflection matrix of the reconstructed object

G_rec = brdf.compute_reflection_matrix(
    L=sun_body[~lc_noisy_unit_irrad.mask, :],
    O=obs_body[~lc_noisy_unit_irrad.mask, :],
    N=rec_obj.unique_normals,
)

is_g_full_rank = np.linalg.matrix_rank(G_rec) == G_rec.shape[1]
print(f"Is G full rank? {is_g_full_rank}")

plt.imshow(G_rec, aspect="auto", cmap="plasma")
mrv.texit(
    "Reconstructed Object Reflection Matrix $G$",
    "Normal index",
    "Time index",
    grid=False,
)
plt.clim([0, 1])
plt.colorbar(cax=mrv.get_cbar_ax(), label="Normalized irradiance per unit area")
plt.show()

# %%
# Plotting the expected normalized irradiance from each facet at each time

total_expected_norm_irrad = np.sum(G_rec * rec_obj.unique_areas, axis=0)
u_quantity = 1 - (total_expected_norm_irrad - np.min(total_expected_norm_irrad)) / (
    np.max(total_expected_norm_irrad) - np.min(total_expected_norm_irrad)
)

plt.bar(np.arange(len(total_expected_norm_irrad)), total_expected_norm_irrad)
mrv.texit(
    "Expected Normalized Irradiance $a_j\sum_{i}{G_{ij}}$",
    "Normal index",
    "Total normalized irradiance",
    grid=False,
)
plt.show()

# %%
# Plotting the light curve error at each timestep

if hasattr(obj, "file_name"):
    delattr(obj, "file_name")
lc_rec = mr.run_light_curve_engine(
    brdf,
    obj,
    sun_body[~lc_noisy_unit_irrad.mask, :],
    obs_body[~lc_noisy_unit_irrad.mask, :],
)
lc_err = np.abs(lc_rec - lc_noisy_unit_irrad[~lc_noisy_unit_irrad.mask])
plt.figure(figsize=(7, 5))
plt.plot(epsecs[~lc_noisy_unit_irrad.mask], lc_err, c="k")
plt.xlabel("Epoch seconds")
plt.ylabel("Normalized irradiance [W/m$^2$]")
plt.legend(["Noisy", "Reconstructed"])
plt.tight_layout()
plt.show()

# %%
# Attributing that light curve error to each face and plotting
total_err_per_face = np.sum(
    lc_err.reshape(-1, 1) * (rec_obj.unique_areas * G_rec), axis=0
)
u_quality = (total_err_per_face - np.min(total_err_per_face)) / (
    np.max(total_err_per_face) - np.min(total_err_per_face)
)

# %%
# This has all been wrapped in a single function:
fu = mr.face_uncertainty(
    rec_obj,
    sun_body[~lc_noisy_unit_irrad.mask, :],
    obs_body[~lc_noisy_unit_irrad.mask, :],
    brdf,
    lc_noisy_unit_irrad[~lc_noisy_unit_irrad.mask],
)


# %%
# Plotting various uncertainties

pl = pv.Plotter()
mrv.render_spaceobject(pl, rec_obj, scalars=u_quantity[rec_obj.unique_to_all])
pl.add_text("$u_{quantity}$")
pl.show()
pl = pv.Plotter()
mrv.render_spaceobject(pl, rec_obj, scalars=u_quality[rec_obj.unique_to_all])
pl.add_text("$u_{quality}$")
pl.show()

pl = pv.Plotter()
mrv.render_spaceobject(pl, rec_obj, scalars=fu)
pl.add_text("$u_j$")
pl.show()
