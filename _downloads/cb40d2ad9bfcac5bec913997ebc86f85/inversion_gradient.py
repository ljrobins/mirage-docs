"""
EGI Jacobian
============

Computing the change in the convex object guess (via its EGI) due to a change in the light curve
"""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

itensor = np.diag([1.0, 2.0, 3.0])
w0 = 1e-2 * mr.hat(np.array([[1.0, 5.0, 0.0]]))
# w0 = 1e-2 * mr.rand_unit_vectors(1)
q0 = np.array([[0.0, 0.0, 0.0, 1.0]])
# q0 = mr.rand_quaternions(1)
idate = mr.utc(2023, 1, 1, 5)
obs_time = mr.hours(3)
obs_dt = mr.seconds(10)
pl_shape = (3, 3)
inversions = pl_shape[0] * pl_shape[1]

obj_file = "cylinder.obj"

station = mr.Station(preset="pogs")
brdf = mr.Brdf(name="phong", cd=0.5, cs=0.0, n=10)
brdf_inversion = brdf
attitude = mr.RbtfAttitude(w0=w0, q0=q0, itensor=itensor)

dates, epsecs = mr.date_arange(idate, idate + obs_time, obs_dt, return_epsecs=True)

q_of_t, w_of_t = attitude.propagate(epsecs)
dcms_of_t = mr.quat_to_dcm(q_of_t)

obj = mr.SpaceObject(obj_file, identifier="goes 15")
lc_ccd_signal_sampler, aux_data = station.observe_light_curve(
    obj, attitude, brdf, dates, use_engine=False, model_scale_factor=4
)

sun_body = aux_data["sun_vector_object_body"]
obs_body = aux_data["observer_vector_object_body"]
rmag = aux_data["rmag_station_to_sat"]

sint = aux_data["sint"]
lc_hat = aux_data["lc_clean_norm"]
constr = aux_data["all_constraints_satisfied"]
br_mean = aux_data["background_mean"]
airy_disk_pixels = aux_data["airy_disk_pixels"]
obs_to_moon = aux_data["obs_to_moon"]
lc_clean = aux_data["lc_clean"]
snr = aux_data["snr"]


mr.tic()
lc_sampled = lc_ccd_signal_sampler()
mr.toc()

# plt.plot(epsecs, lc_clean)
# plt.scatter(epsecs, lc_sampled, s=2, c="r")
# plt.show()


lc_normalized = (
    lc_sampled
    / (sint * station.telescope.integration_time)
    * (rmag * 1e3) ** 2
    / mr.AstroConstants.sun_irradiance_vacuum
)

egi = mr.optimize_egi(lc_normalized, sun_body, obs_body, brdf)

# G_actual = brdf.compute_reflection_matrix(sun_body, obs_body, egi)

# plt.imshow(G_actual, extent=[-1,1,-1,1])
# plt.show()
# endd

# print(G.shape, G_deep.shape, gel.shape)
# endd

# %%
# Expected error in each light curve data point
import pyvista as pv

h = mr.optimize_supports_little(egi)
rec_obj = mr.construct_from_egi_and_supports(egi, h)
rec_obj.shift_to_center_of_mass()
fu = mr.face_uncertainty(rec_obj, sun_body, obs_body, brdf, lc_sampled)
pl = pv.Plotter()
pv.plotting.opts.InterpolationType(0)
mrv.render_spaceobject(pl, rec_obj, scalars=fu[rec_obj.unique_to_all])
mrv.render_spaceobject(pl, obj, style="wireframe", color="r")
mrv.plot_basis(pl, np.eye(3), ["x", "y", "z"])
# mrv.scatter3(pl, mr.hat(egi), , cmap="plasma", point_size=30)
pl.show()
