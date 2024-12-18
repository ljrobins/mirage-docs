"""
Multiple Hypothesis Inversion
=============================
Inverting convex shapes from light curves with uncertainty
"""

import numpy as np

import mirage as mr

itensor = np.diag([1.0, 2.0, 3.0])
w0 = 1e-1 * mr.hat(np.array([[0.1, 2.0, 0.1]]))
q0 = np.array([[0.0, 0.0, 0.0, 1.0]])
idate = mr.utc(2023, 1, 1, 5)
obs_time = mr.hours(3)
obs_dt = mr.seconds(10)
pl_shape = (2, 1)
nights = np.prod(pl_shape)
lw = 3


# station = mr.Station(preset="pogs")
# brdf = mr.Brdf(name="phong", cd=0.5, cs=0.5, n=5)
# brdf_inversion = brdf
# # brdf_inversion = mr.Brdf(name='phong', cd=1, cs=0.0, n=5)
# attitude = mr.RbtfAttitude(w0=w0, q0=q0, itensor=itensor)

# dates = []
# epsecs = []
# for i in range(nights):
#     d = mr.date_arange(idate, idate + obs_time, obs_dt)
#     dates.append(d)
#     idate += mr.days(1)
# dates = np.concatenate(dates)
# epsecs = np.array([(d - dates[0]).total_seconds() for d in dates])

# q_of_t, w_of_t = attitude.propagate(epsecs)
# model_scale_factor = 10.0

# for obj_file in ['collapsed_cube.obj']:
#     obj = mr.SpaceObject(obj_file).clean()
#     obj.sat.satnum = 36411 # goes 15
#     obj.shift_to_center_of_mass()

#     # mrv.vis_attitude_motion(obj, q_of_t)
#     # endd

#     lc_ccd_signal_sampler, aux_data = station.observe_light_curve(
#         obj, attitude, brdf, dates, use_engine=True, model_scale_factor=model_scale_factor
#     )

#     sun_body = aux_data["sun_vector_object_body"]
#     obs_body = aux_data["observer_vector_object_body"]

#     sint = aux_data["sint"]
#     lc_hat = aux_data["lc_clean_norm"]
#     constr = aux_data["all_constraints_satisfied"]
#     br_mean = aux_data["background_mean"]
#     airy_disk_pixels = aux_data["airy_disk_pixels"]
#     obs_to_moon = aux_data["obs_to_moon"]
#     lc_clean = aux_data["lc_clean"]
#     snr = aux_data["snr"]

#     # plt.figure(figsize=(7, 5))
#     lcs_noisy_adu = np.array([lc_ccd_signal_sampler() for _ in range(1000)])
#     lcs_noisy_irrad = lcs_noisy_adu / (
#         aux_data["sint"] * station.telescope.integration_time
#     )
#     lcs_noisy_unit_irrad = (
#         lcs_noisy_irrad
#         * (aux_data["rmag_station_to_sat"] * 1e3) ** 2
#         / mr.AstroConstants.sun_irradiance_vacuum
#     )

#     pl = pv.Plotter(shape=pl_shape)
#     rec_objs = []
#     lc_sampled = lcs_noisy_adu[0, :]
#     lc = lcs_noisy_unit_irrad[0, :]
#     for i in range(nights):
#         inds = (np.array([i, i + 1]) / nights * lc.size).astype(int)

#         lc_this = lc[inds[0] : inds[1]]
#         sun_body_this = sun_body[inds[0] : inds[1], :]
#         obs_body_this = obs_body[inds[0] : inds[1], :]

#         egi_opt_initial, egi_opt = mr.optimize_egi(
#             # lc_this / np.max(lc_this),
#             lc_this,
#             sun_body_this,
#             obs_body_this,
#             brdf_inversion,
#             merge_iter=2,
#             merge_angle=np.pi / 8,
#             return_all=True,
#         )

#         # Inversion
#         rec_obj = mr.construct_mesh_from_egi(mr.close_egi(egi_opt))
#         # rec_obj = mr.SpaceObject(vertices_and_faces=(rec_obj.v / model_scale_factor, rec_obj.f))
#         rec_obj = rec_obj.introduce_concavity(
#             -mr.hat(np.sum(egi_opt_initial, axis=0)),
#             np.random.uniform(0, 60) if np.random.rand() > 0.5 else 0,
#             # 60,
#             linear_iter=3,
#             loop_iter=0,
#         )
#         rec_obj.shift_to_center_of_mass()
#         rec_objs.append(rec_obj)

#     err = []
#     for i,rec_obj in enumerate(rec_objs):
#         lc_hat = (
#             mr.run_light_curve_engine(brdf, rec_obj, sun_body, obs_body)
#         )
#         err.append(np.linalg.norm(lc_hat - lc))
#         print(f"Error for night {i+1}: {err[-1]:.2e}")
#     err = np.array(err)
#     weights = (1 - (err - np.min(err)) / (np.max(err) - np.min(err))) ** 2

#     fu_lambdas = []
#     for i, rec_obj in enumerate(rec_objs):
#         inds = (np.array([i, i + 1]) / nights * lc.size).astype(int)

#         lc_this = lc[inds[0] : inds[1]]
#         sun_body_this = sun_body[inds[0] : inds[1], :]
#         obs_body_this = obs_body[inds[0] : inds[1], :]
#         snr_this = snr[inds[0] : inds[1]]

#         print(f"Computing face uncertainty {i+1}/{len(rec_objs)}")

#         fu = mr.face_uncertainty(rec_obj, sun_body_this, obs_body_this, brdf, lc_this)
#         fu_lambdas.append(mr.SphericalWeight(rec_obj.unique_normals, (1-fu[rec_obj.all_to_unique]) * weights[i]))
#         az,el,_ = mr.cart_to_sph(*rec_obj.face_normals.T)
#         pl.subplot(i // pl_shape[0], i % pl_shape[1])
#         # pv.plotting.opts.InterpolationType(0)
#         mrv.render_spaceobject(pl, rec_obj, opacity=0.7)
#         mrv.render_spaceobject(pl, obj, style="wireframe", color="r", line_width=lw)
#         pl.add_text(
#             f"Night {i+1}",
#             font="courier",
#         )

#         # az,el = np.meshgrid(np.linspace(0,2*np.pi,1000), np.linspace(-np.pi/2,np.pi/2,1000))
#         # mr.tic()
#         # vals = fu_lambdas[-1](az.flatten(), el.flatten()).reshape(az.shape)
#         # plt.imshow(vals, extent=[0,2*np.pi,-np.pi/2,np.pi/2], origin='lower', cmap='cividis', aspect='auto')
#         # mrv.texit(f"Face Uncertainty Map", "Azimuth (rad)", "Elevation (rad)", grid=True)
#         # plt.colorbar()
#         # mr.toc()
#         # plt.show()

#     pl.show()

#     # %%
#     # Merging the guesses and testing new error

#     merged_obj = mr.merge_shapes(rec_objs, fu_lambdas)

#     pl = pv.Plotter()
#     mrv.render_spaceobject(pl, merged_obj, opacity=0.7)
#     mrv.render_spaceobject(pl, obj, style="wireframe", color="r", line_width=lw)
#     pl.add_text(
#         f"Merged Guess",
#         font="courier",
#     )
#     # mrv.render_spaceobject(pl, rec_objs[0], style="wireframe", color="b")
#     pl.show()

#     lc_hat = (
#         mr.run_light_curve_engine(brdf, merged_obj, sun_body, obs_body)
#         * model_scale_factor**2
#     )
#     err = np.linalg.norm(lc_hat - lc)
#     print(f"New Error: {err:.2e}")
